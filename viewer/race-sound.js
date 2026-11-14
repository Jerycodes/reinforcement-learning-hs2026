import {musicPlayer, soundtrackUrl} from './music-player.js';
export const profiles={Club:['Club-Kart',.94],Sprint:['Sprint-Kart',1.09],GT:['Sport-Kart',1.01]};
export function raceSound(){
  const panel=document.createElement('details');
  panel.id='raceSound';
  panel.innerHTML='<summary>Sound</summary><label><input type="checkbox"> Stumm</label>';
  document.body.append(panel);
  let ctx, master, phase='', lastCue='', lastContact=false, lastBoost=false;
  const voices=new Map(), buses={}, settings={Motor:.65,Musik:.32,Effekte:.6,Stumm:false};
  try{Object.assign(settings,JSON.parse(localStorage.kartMix||'{}'));}catch{}
  function mix(){
    if(ctx){
      master.gain.setTargetAtTime(settings.Stumm?0:.65,ctx.currentTime,.02);
      for(const key of Object.keys(buses)) buses[key].gain.setTargetAtTime(settings[key],ctx.currentTime,.02);
    }
    try{localStorage.kartMix=JSON.stringify(settings);}catch{}
  }
  const mute=panel.querySelector('input');mute.checked=settings.Stumm;
  mute.onchange=()=>{settings.Stumm=mute.checked;mix();};
  const channels=soundtrackUrl ? ['Motor','Musik','Effekte'] : ['Motor','Effekte'];
  for(const key of channels){
    const label=document.createElement('label');label.textContent=key;
    const input=document.createElement('input');input.type='range';input.min=0;input.max=1;input.step=.01;input.value=settings[key];
    input.oninput=()=>{settings[key]=Number(input.value);mix();};label.append(input);panel.append(label);
  }
  let music, samples, garage, request=0;
  function unlock(){
    if(!window.AudioContext)return;
    if(!ctx){
      ctx=new AudioContext();master=ctx.createGain();const limiter=ctx.createDynamicsCompressor();
      limiter.threshold.value=-9;limiter.ratio.value=12;master.connect(limiter).connect(ctx.destination);
      samples=Promise.all(['idle','load','motor-garage'].map(async name=>{
        const response=await fetch('./audio/'+name+'.wav');
        if(!response.ok)throw Error(name);
        return ctx.decodeAudioData(await response.arrayBuffer());
      })).catch(()=>null);
      for(const key of channels){buses[key]=ctx.createGain();buses[key].connect(master);}
      music=musicPlayer(ctx,buses.Musik);
      samples.then(decoded=>{bank=decoded;});
      mix();
    }
    ctx.resume().catch(()=>{});
  }
  function note(bus,t,f,end,duration,volume,type='sine'){
    const osc=ctx.createOscillator(), gain=ctx.createGain();osc.type=type;
    osc.frequency.setValueAtTime(f,t);osc.frequency.exponentialRampToValueAtTime(end,t+duration);
    gain.gain.setValueAtTime(.0001,t);gain.gain.exponentialRampToValueAtTime(volume,t+.008);
    gain.gain.exponentialRampToValueAtTime(.0001,t+duration);
    osc.connect(gain).connect(buses[bus]);osc.start(t);osc.stop(t+duration+.01);
  }
  let noise;
  function friction(type, intensity=1){
    if(!noise){
      noise=ctx.createBuffer(1,ctx.sampleRate,ctx.sampleRate);
      const data=noise.getChannelData(0);
      for(let i=0;i<data.length;i++) data[i]=Math.random()*2-1;
    }
    const [frequency,end,duration,level]={
      flag:[1450,580,.34,.35], skid:[1200,850,.19,.20],boost:[700,2200,.32,.17],impact:[460,100,.13,.6]
    }[type];
    const src=ctx.createBufferSource(),filter=ctx.createBiquadFilter(),gain=ctx.createGain(),t=ctx.currentTime;
    src.buffer=noise;filter.type=type==='impact'?'lowpass':'bandpass';
    filter.Q.value=type==='skid'?3:.8;
    filter.frequency.setValueAtTime(frequency,t);filter.frequency.exponentialRampToValueAtTime(end,t+duration);
    gain.gain.setValueAtTime(.0001,t);gain.gain.exponentialRampToValueAtTime(level*intensity,t+.012);
    gain.gain.exponentialRampToValueAtTime(.0001,t+duration);
    src.connect(filter).connect(gain).connect(buses.Effekte);src.start(t,Math.random()*.5);src.stop(t+duration);
  }
  let bank;
  function motor(id,car,rev,volume,pan, throttle){
    if(!bank)return;
    const t=ctx.currentTime;
    let voice=voices.get(id);
    if(!voice){
      const stereo=ctx.createStereoPanner();stereo.connect(buses.Motor);
      const layers=bank.slice(0,2).map(buffer=>{
        const src=ctx.createBufferSource(),gain=ctx.createGain();
        src.buffer=buffer;src.loop=true;gain.gain.value=0;
        src.connect(gain).connect(stereo);src.start();
        return {src,gain};
      });
      voice={stereo,layers};voices.set(id,voice);
    }
    const blend=Math.min(1,rev*.82+.18*Math.max(0,throttle));
    voice.stereo.pan.setTargetAtTime(pan,t,.09);
    voice.layers.forEach(({src,gain},i)=>{
      src.playbackRate.setTargetAtTime(profiles[car][1]*(i?.73+.52*rev:.88+.64*rev),t,.12);
      gain.gain.setTargetAtTime(volume*(i?Math.sqrt(blend):Math.sqrt(1-blend))*(.42+.58*Math.max(0,throttle)),t,.09);
    });
  }
  function clear(keep=[]){
    for(const [id,v] of voices)if(!keep.includes(id)){
      for(const {gain,src} of v.layers){gain.gain.setTargetAtTime(0,ctx.currentTime,.012);src.stop(ctx.currentTime+.07);}
      voices.delete(id);
    }
  }
  function stopGarage(){
    request++;
    if(garage){garage.gain.gain.setTargetAtTime(0,ctx.currentTime,.012);garage.src.stop(ctx.currentTime+.07);garage=null;}
  }
  async function preview(car){
    unlock();if(!ctx)return;
    stopGarage();const token=request;bank=await samples;
    if(!bank||request!==token)return;
    const src=ctx.createBufferSource(),gain=ctx.createGain();
    src.buffer=bank[2];src.playbackRate.value=profiles[car][1];gain.gain.value=.75;
    src.connect(gain).connect(buses.Motor);garage={src,gain};src.start();
  }
  function update(g,states){
    if(!ctx)return;
    const t=ctx.currentTime,own=states.find(s=>s.id===g.player_id);
    if(phase!==g.phase){music(g.phase);stopGarage();lastCue='';lastContact=lastBoost=false;
      }
    if(own?.parked && !parked) friction('flag');
    parked=Boolean(own?.parked);
    phase=g.phase;
    const cue=phase==='countdown'?Math.ceil(g.countdown):phase==='racing'&&g.elapsed<.5?'go':'';
    if(cue!==lastCue && (cue==='go'||(cue>0&&cue<=3)))note('Effekte',t,cue==='go'?880:440,cue==='go'?880:440,.17,.18);
    lastCue=cue;
    const active=own && ['racing','countdown','lobby'].includes(phase)?states.filter(s=>!s.parked):[];
    clear(active.map(s=>s.id));
    for(const s of active){
      const dx=s.x-own.x,dy=s.y-own.y,dist=Math.hypot(dx,dy),me=s.id===own.id;
      motor(s.id,s.car,Math.min(1,Math.abs(s.vx)/28+.16*Math.max(0,s.throttle)),(me?.83:.16/(1+dist*.28))*(phase==='lobby'?.22:1),
        Math.max(-1,Math.min(1,(dx*Math.sin(own.yaw)-dy*Math.cos(own.yaw))/6)),s.throttle);
    }
    if(own?.boost&&!lastBoost)friction('boost');
    const contact=own?.contact || own?.off_track;
    if(contact&&!lastContact)friction('impact');
    const scrub=Math.min(1,Math.abs((own?.steer || 0)*(own?.vx || 0))/8);
    if(phase==='racing' && own && !own.parked && (own.drifting || scrub>.42) && t-lastSkid>.16){
      lastSkid=t;
      friction('skid',own.drifting ? .85 : scrub*.36);
    }
    lastContact=contact;lastBoost=own?.boost;
  }
  let parked=false,lastSkid=-1;
  return {unlock,preview,update,menu(){phase='menu';music?.(phase);},stop(){phase='';music?.(phase);stopGarage();clear();}};
}
