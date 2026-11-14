export const profiles = {Club:['V8 Crossplane',42,.55],Sprint:['V10 Hochdrehzahl',63,.82],GT:['V12 Grand Prix',51,1.25]};
export function raceSound(){
  const panel=document.createElement('details');
  panel.id='raceSound';
  panel.innerHTML='<summary>♪ Sound</summary><label><input type="checkbox"> Stumm</label>';
  document.body.append(panel);
  let ctx, master, phase='', beat=0, next=0, previewUntil=0, lastCue='', lastContact=false, lastBoost=false;
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
  for(const key of ['Motor','Musik','Effekte']){
    const label=document.createElement('label');label.textContent=key;
    const input=document.createElement('input');input.type='range';input.min=0;input.max=1;input.step=.01;input.value=settings[key];
    input.oninput=()=>{settings[key]=Number(input.value);mix();};label.append(input);panel.append(label);
  }
  function unlock(){
    if(!window.AudioContext)return;
    if(!ctx){
      ctx=new AudioContext();master=ctx.createGain();master.connect(ctx.destination);
      for(const key of ['Motor','Musik','Effekte']){buses[key]=ctx.createGain();buses[key].connect(master);}
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
  function motor(id,car,rev,volume,pan){
    let v=voices.get(id);
    if(!v){
      const osc=ctx.createOscillator(), overtone=ctx.createOscillator(), filter=ctx.createBiquadFilter(), gain=ctx.createGain(), stereo=ctx.createStereoPanner();
      osc.type='sawtooth';overtone.type='triangle';osc.connect(filter);overtone.connect(filter);
      filter.connect(gain).connect(stereo).connect(buses.Motor);gain.gain.value=0;
      osc.start();overtone.start();v={osc,overtone,filter,gain,stereo};voices.set(id,v);
    }
    const [,base,harmonic]=profiles[car]||profiles.Club,t=ctx.currentTime,f=base*(1+rev*2.9);
    for(const [param,value] of [[v.osc.frequency,f],[v.overtone.frequency,f*(2+harmonic)],[v.filter.frequency,240+rev*1200],[v.gain.gain,volume],[v.stereo.pan,pan]])param.setTargetAtTime(value,t,.065);
  }
  function clear(keep=[]){
    for(const [id,v] of voices)if(!keep.includes(id)){
      v.gain.gain.setTargetAtTime(0,ctx.currentTime,.012);v.osc.stop(ctx.currentTime+.07);v.overtone.stop(ctx.currentTime+.07);voices.delete(id);
    }
  }
  function preview(car){unlock();if(!ctx)return;previewUntil=ctx.currentTime+1.65;previewCar=car;}
  let previewCar='Club';
  setInterval(()=>{
    if(!ctx||ctx.state!=='running')return;
    const t=ctx.currentTime;
    if(t<previewUntil){const a=(1.65-previewUntil+t)/1.65;motor('preview',previewCar,.08+.85*Math.sin(Math.PI*a)**2,.17,0);}
    else if(voices.has('preview')) clear([...voices.keys()].filter(k=>k!=='preview'));
    if(!['menu','lobby','racing'].includes(phase))return;
    const intensity=phase==='racing'?1:.24;
    if(next<t)next=t+.01;
    while(next<t+.12){
      if(beat%2===0)note('Musik',next,125,44,.19,.28*intensity);
      else note('Musik',next,3200,1600,.035,.037*intensity,'triangle');
      const riff=[110,0,165,0,130.81,0,98,146.83][beat%8];
      if(riff)note('Musik',next, riff,riff,.16,.105*intensity,'triangle');
      next+=60/124/2;beat++;
    }
  },50);
  function update(g,states){
    if(!ctx)return;
    const t=ctx.currentTime,own=states.find(s=>s.id===g.player_id);
    if(phase!==g.phase){beat=0;next=t+.02;lastCue='';lastContact=lastBoost=false;
      if(g.phase==='finished')[392,494,587].forEach((f,i)=>note('Effekte',t+i*.13,f,f,.32,.16));}
    phase=g.phase;
    const cue=phase==='countdown'?Math.ceil(g.countdown):phase==='racing'&&g.elapsed<.5?'go':'';
    if(cue!==lastCue && (cue==='go'||(cue>0&&cue<=3)))note('Effekte',t,cue==='go'?880:440,cue==='go'?880:440,.17,.18);
    lastCue=cue;
    const active=own && ['racing','countdown','lobby'].includes(phase)?states.filter(s=>!s.parked):[];
    clear([...active.map(s=>s.id),...(t<previewUntil?['preview']:[])]);
    for(const s of active){
      const dx=s.x-own.x,dy=s.y-own.y,dist=Math.hypot(dx,dy),me=s.id===own.id;
      motor(s.id,s.car,Math.min(1,Math.abs(s.vx)/28+.16*Math.max(0,s.throttle)),(me?.13:.065/(1+dist*.28))*(phase==='lobby'?.22:1),
        Math.max(-1,Math.min(1,(dx*Math.sin(own.yaw)-dy*Math.cos(own.yaw))/6)));
    }
    if(own?.boost&&!lastBoost)note('Effekte',t,150,650,.27,.10,'triangle');
    if(own?.off_track&&!lastContact)note('Effekte',t,110,32,.12,.18,'sawtooth');
    if(own?.drifting && Math.floor(t*8)!==lastSkid){lastSkid=Math.floor(t*8);note('Effekte',t,950,730,.13,.038,'sawtooth');}
    lastContact=own?.off_track;lastBoost=own?.boost;
  }
  let lastSkid=-1;
  return {unlock,preview,update,menu(){phase='menu';},stop(){phase='';previewUntil=0;clear();}};
}
