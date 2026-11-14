// Run against a local server with Playwright and Chromium (see arcade_multiplayer.md).
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');
const artifacts=fs.mkdtempSync(path.join(os.tmpdir(),'kart-paddock-'));
const url=process.env.KART_URL || 'http://localhost:8031/viewer/?game&arcade&ws=8796';

async function driver(page){
  // Synthetic key presses only. Each client reads the public centreline and the normal broadcast.
  await page.addInitScript(()=>{
    const Native=window.WebSocket;
    const map=fetch('../kart_rl/data/roggwil_arcade.json').then(r=>r.json());
    window.WebSocket=class extends Native {
      constructor(...args){
        super(...args);
        let id,keys={},track;
        let length;map.then(data=>{track=data.centerline;length=data.length_m;});
        let prevS=null; window.lineCrossings=[];
        this.addEventListener('message',event=>{
          const m=JSON.parse(event.data);if(m.type==='hello')id=m.player_id;
          if(m.type!=='states'||!track)return;
          const st=m.states.find(s=>s.id===id);if(!st)return;
          if(['racing','finished'].includes(m.game.phase) && prevS!==null && prevS-st.s>length/2)window.lineCrossings.push(st.lap);
          prevS=st.s;
          const driving=m.game.phase==='racing';
          const pos=Math.round(st.s*track.length/length);
          const target=track[(pos+Math.round((4+st.vx*.40)*track.length/length))%track.length];
          const delta=Math.atan2(target[1]-st.y,target[0]-st.x)-st.yaw;
          const error=Math.atan2(Math.sin(delta),Math.cos(delta));
          const goal=11/(1+1.6*Math.abs(error));
          const next={ArrowUp:driving&&st.vx<goal,ArrowDown:driving&&st.vx>goal+1,
                      ArrowLeft:driving&&error>.085,ArrowRight:driving&&error<-.085};
          for(const [key,value] of Object.entries(next))if(!!keys[key]!==value){
            document.body.dispatchEvent(new KeyboardEvent(value?'keydown':'keyup',{key,bubbles:true}));
          }
          keys=next;
        });
      }
    };
  });
}
async function enter(page,link,name,avatar,car,tyre){
  await driver(page);await page.goto(link);
  await page.click('#openingFilm button');
  await page.getByRole('button',{name:'Überspringen'}).click();
  await page.fill('#pilotName',name);await page.click('[data-avatar="'+avatar+'"]');
  await page.click('#nextGarage');await page.click('[data-car="'+car+'"]');
  await page.click('[data-tyre="'+tyre+'"]');
  if(name==='Kath')await page.screenshot({path:path.join(artifacts,'garage.png')});
  await page.click('#joinGrid');await page.waitForSelector('#clubReady:visible');
}
(async()=>{
  const browser=await chromium.launch({executablePath:process.env.CHROME_BIN,headless:true,
    args:['--enable-unsafe-swiftshader','--disable-renderer-backgrounding','--disable-background-timer-throttling']});
  try{
    const errors=[];
    const pages=await Promise.all([0,1].map(async()=>{const p=await browser.newPage({viewport:{width:1024,height:640},ignoreHTTPSErrors:new URL(url).hostname==='localhost'});
      p.on('pageerror',e=>errors.push(e.message));
      if(url.startsWith('https:'))p.on('websocket',ws=>{
        if(ws.url()!=='wss://'+new URL(url).host+'/socket') errors.push('unexpected socket '+ws.url());
      });return p;}));
    const [a,b]=pages;
    await enter(a,url,'Kath','Fuchs','Club','Standard');
    await a.context().grantPermissions(['clipboard-read','clipboard-write']);
    await a.click('#copyLink');
    const invite=await a.evaluate(()=>navigator.clipboard.readText());
    assert.equal(invite,a.url());
    await enter(b,invite,'Gast','Panda','GT','Grip');
    await a.click('#plusBot');await a.click('#plusBot');await a.click('#plusBot');
    await a.waitForFunction(()=>window.kartViewer.gameState().players.length===5);
    assert(await a.locator('#greenFlag').isDisabled());
    await b.click('#clubReady');await a.click('#clubReady');
    await a.click('#greenFlag');
    const state=p=>p.evaluate(()=>window.kartViewer.gameState());
    await Promise.all(pages.map(p=>p.waitForFunction(()=>window.kartViewer.gameState()?.phase==='countdown')));
    const first=await state(a);assert.equal(first.elapsed,0);assert.deepEqual(first.grid,(await state(b)).grid);
    await Promise.all(pages.map(p=>p.waitForFunction(()=>{
      const rendered=window.kartViewer.karts();
      return window.kartViewer.gameState().grid.every(slot=>{
        const kart=rendered.find(k=>k.name===slot.name);
        return kart && Math.hypot(kart.x-slot.x,kart.y-slot.y)<.015;
      });
    },null,{timeout:5000})));
    for(const p of pages){
      const view=await p.evaluate(()=>window.kartViewer.gridView());
      for(const slot of first.grid){
        const [n,x,y,yaw]=view.boxes[slot.position-1];
        assert.equal(n,slot.position);assert(Math.hypot(x-slot.x,y-slot.y)<1e-6);assert.equal(yaw,slot.yaw);
        assert((view.finish[0]-x)*Math.cos(yaw)+(view.finish[1]-y)*Math.sin(yaw)>6);
      }
    }
    await a.screenshot({path:path.join(artifacts,'grid.png')});
    await a.waitForSelector('.lights',{state:'attached'});await b.waitForSelector('.lights',{state:'attached'});
    const camera=await a.evaluate(()=>window.kartViewer.gridView());
    assert.equal(camera.camera,'onboard');assert.equal(camera.fov,54);
    const ownSlot=first.grid.find(slot=>slot.name==='Kath');
    assert(Math.abs(camera.eye[1]-.72)<1e-6);
    assert(Math.hypot(camera.eye[0]-ownSlot.x+.35*Math.cos(ownSlot.yaw),
                      -camera.eye[2]-ownSlot.y+.35*Math.sin(ownSlot.yaw))<.015);
    await a.screenshot({path:path.join(artifacts,'cockpit.png')});
    await a.waitForFunction(()=>window.kartViewer.gameState()?.elapsed>4);
    await a.screenshot({path:path.join(artifacts,'driving.png')});
    for(let i=0;i<58;i++){
      await a.waitForTimeout(10000);const g=await state(a);
      console.log(g.phase,g.elapsed,g.players.map(p=>[p.name,p.lap,Math.round(p.progress)]));
      if(g.phase==='finished')break;
    }
    const result=await state(a);
    assert.equal(result.phase,'finished');assert(result.results.every(p=>p.lap===3 && p.finish_time>0));
    await b.waitForSelector('#podium:visible');assert.deepEqual(result.results,(await state(b)).results);
    for(const p of pages)assert.deepEqual(await p.evaluate(()=>window.lineCrossings),[0,1,2,3]);
    await a.click('#saveLap');
    assert.match(await a.locator('#personalRecord').innerText(),/Club \/ Standard \u00b7 KartArcade-v2/);
    await a.screenshot({path:path.join(artifacts,'result.png')});
    await b.click('#againReady');await a.click('#againReady');await a.click('#againStart');
    await a.waitForFunction(()=>window.kartViewer.gameState().phase==='countdown');
    const rematch=await state(a);assert.notEqual(first.race_id,rematch.race_id);
    await b.waitForFunction(()=>window.kartViewer.gameState().phase==='countdown');
    assert.deepEqual(rematch.grid,(await state(b)).grid);
    assert(rematch.players.every(p=>!p.lap && p.finish_time===null));assert.deepEqual(errors,[]);
    console.log('PASS',artifacts,first.grid.map(p=>p.name),rematch.grid.map(p=>p.name));
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
