import {profiles} from './race-sound.js';
import {pilots, showroom} from './paddock-model.js';

const by=id=>document.getElementById(id);
const escape=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const cars={Club:[22,12,1],Sprint:[20,14,1.08],GT:[24,10,.92]};
const tyres={Standard:[1,1],Grip:[1.16,.92],Speed:[.89,1.08]};
const choices=(group,values)=>values.map((v,i)=>`<button data-${group}="${v}" aria-pressed="${i===0}">${group==='avatar'?pilots[v].icon+' ':''}${v}</button>`).join('');
const time=s=>`${Math.floor(s/60)}:${(s%60).toFixed(2).padStart(5,'0')}`;
function markup(){
return `<main id="paddock">
<header><strong><i>◩</i> KART CLUB</strong><span>Race-Inn · Roggwil, CH</span></header>
<section id="entry" class="split"><article><p class="kicker">ROGGWIL / KART GRAND PRIX</p>
<h1 id="pageTitle">Fahrerprofil</h1>
<div id="profile"><label for="pilotName">Dein Fahrername</label><input id="pilotName" maxlength="24" autocomplete="off" placeholder="Name eingeben">
<fieldset><legend>Avatar</legend><div class="options">${choices('avatar',Object.keys(pilots))}</div></fieldset>
<button id="nextGarage" class="primary">Zur Garage →</button></div>
<div id="loadout" hidden><fieldset><legend>Kart</legend><div class="options">${choices('car',Object.keys(cars))}</div></fieldset>
<fieldset><legend>Reifen</legend><div class="options">${choices('tyre',Object.keys(tyres))}</div></fieldset>
<button id="listenMotor" type="button"></button><div id="ratings"></div><div class="actions"><button id="backProfile">Zurück</button><button id="joinGrid" class="primary">In die Lobby →</button></div></div></article>
<figure><div id="displayKart"></div><figcaption id="caption"></figcaption></figure></section>
<section id="clubLobby" hidden><p class="kicker">PRIVATER RENNRAUM</p><h1>Fahrerlager</h1><p id="seats"></p><div id="cards"></div>
<div class="actions"><button id="copyLink">Einladungslink kopieren</button><label id="difficulty">Bot-Stufe <select id="botLevel"><option>Club</option><option selected>Sport</option><option>Pro</option></select></label><button id="plusBot">+ Bot</button><button id="minusBot">− Bot</button></div>
<div class="actions"><button id="clubReady">Bereit</button><button id="greenFlag" class="primary">Rennen starten →</button></div>
<p class="keys">WASD / Pfeile fahren · Shift driften · Leertaste boosten · R zurücksetzen (+3 s)</p></section>
<section id="podium" hidden><p class="kicker">ZIEL / ROGGWIL</p><h1>Rennergebnis</h1><ol id="results"></ol><p id="flying"></p><button id="saveLap">Als persönliche Bestzeit merken</button><div class="actions"><button id="againReady">Bereit zur Revanche</button><button id="againStart" class="primary">Neues Rennen →</button></div></section>
<p id="personalRecord"></p><p id="raceNotice" role="status"></p></main><div id="startSignal" aria-live="polite"></div>`;
}
export function createPaddock(connect,send,sound){
  document.body.insertAdjacentHTML('beforeend',markup());
  const profile={name:'',avatar:'Fuchs',car:'Club',tyre:'Standard'};
  let state, lastCards='',lastResults='',lastSignal='',lastRank=0,passUntil=0,phase='';
  const preview=showroom(by('displayKart'));
  function spec(){
    const [speed,accel,turn]=cars[profile.car],[grip,drive]=tyres[profile.tyre];
    const rows=[['Tempo',Math.round(speed*3.6)+' km/h',speed/24],['Antritt',(accel*drive).toFixed(1)+' m/s²',accel*drive/15.2],['Handling',Math.round(turn*grip*100)+' %',turn*grip/1.26]];
    by('ratings').innerHTML=rows.map(([k,v,f])=>`<div>${k}<b>${v}</b><meter min="0" max="1" value="${f}"></meter></div>`).join('');
    by('caption').textContent=`${profile.avatar} / ${profile.car} · ${profile.tyre}`;
    by('listenMotor').textContent='▶ Motor anhören · '+profiles[profile.car][0];
    preview(profile.car,profile.tyre,profile.avatar);
  }
  for(const key of ['car','tyre','avatar']){
    const buttons=[...document.querySelectorAll('[data-'+key+']')];
    buttons.forEach(b=>b.onclick=()=>{profile[key]=b.dataset[key];if(key==='car')sound.preview(profile.car);buttons.forEach(o=>o.setAttribute('aria-pressed',o===b));spec();});
  }
  spec();
  by('listenMotor').onclick=()=>sound.preview(profile.car);
  function notice(msg){by('raceNotice').textContent=msg;}
  by('nextGarage').onclick=()=>{
    profile.name=by('pilotName').value.trim();
    if(!profile.name){notice('Gib zuerst deinen Fahrernamen ein.');by('pilotName').focus();return;}
    sound.preview(profile.car);notice('');by('profile').hidden=true;by('loadout').hidden=false;by('pageTitle').textContent='Deine Garage';
  };
  by('backProfile').onclick=()=>{by('profile').hidden=false;by('loadout').hidden=true;by('pageTitle').textContent='Fahrerprofil';notice('');};
  by('pilotName').onkeydown=e=>{if(e.key==='Enter')by('nextGarage').click();};
  by('joinGrid').onclick=()=>{notice('Verbinde …');by('joinGrid').disabled=true;connect();};
  by('plusBot').onclick=()=>send({type:'bot',add:true,level:by('botLevel').value});by('minusBot').onclick=()=>send({type:'bot',add:false});
  by('greenFlag').onclick=by('againStart').onclick=()=>send({type:'control',action:'start_race'});
  by('clubReady').onclick=by('againReady').onclick=()=>send({type:'ready',value:!state.players.find(p=>p.id===state.player_id).ready});
  by('copyLink').onclick=async()=>{try{await navigator.clipboard.writeText(location.href);notice('Einladungslink kopiert.');}catch{notice(location.href);}};
  function recordKey(me) {
    return 'kart.personal.' + JSON.stringify([me.name.toLowerCase(), me.car, me.tyre, state.rules]);
  }
  function showRecord(me) {
    try {
      const saved = Number(localStorage.getItem(recordKey(me)));
      by('personalRecord').textContent = saved > 0
        ? `Persönliche Bestzeit · ${me.name} · ${time(saved)} · ${me.car} / ${me.tyre} · ${state.rules}` : '';
    } catch { by('personalRecord').textContent = ''; }
  }
  by('saveLap').onclick = () => {
    const me = state.results.find(p => p.id === state.player_id);
    if (!me?.best_flying || me.bot) return;
    try {
      const key = recordKey(me);
      const previous = Number(localStorage.getItem(key)) || Infinity;
      localStorage.setItem(key, String(Math.min(previous, me.best_flying)));
      showRecord(me);
      notice('Bestzeit in diesem Browser gespeichert.');
    } catch { notice('Der Browserspeicher ist nicht verfügbar.'); }
  };
  function signal(html){if(lastSignal!==html){by('startSignal').innerHTML=html;lastSignal=html;}}
  return {profile,notice,closed(msg){by('joinGrid').disabled=false;by('paddock').hidden=false;notice(msg);if(state){by('clubLobby').hidden=true;by('podium').hidden=true;by('entry').hidden=false;state=null;phase='';signal('');}},
    render(game){
      state=game;
      const me=game.players.find(p=>p.id===game.player_id);if(!me)return;
      const host=game.host_id===me.id,lobby=game.phase==='lobby'||(me.waiting&&game.phase!=='finished'),finished=game.phase==='finished';
      if(phase!==game.phase){showRecord(me);notice('');phase=game.phase;lastRank=0;passUntil=0;}
      by('entry').hidden=true;by('clubLobby').hidden=!lobby;by('podium').hidden=!finished;
      by('paddock').hidden=!(lobby||finished);
      document.body.classList.toggle('on-track',!lobby&&!finished);
      if(lobby||finished){
        signal('');
        by('seats').textContent=me.waiting?'Du fährst ab dem nächsten Start mit.':`${game.players.length} / 5 Startplätze · ${game.race_laps} Runden`;
        const cards=game.players.map(p=>`<div class="seat"><span class="face">${pilots[p.avatar].icon}</span><div><b>${escape(p.name)}</b><small>${p.avatar} · ${p.car} / ${p.tyre}</small></div><em>${p.bot?'Bot '+p.level+' · ':p.id===game.host_id?'Host · ':''}${p.ready?'Bereit':'Wartet'}</em></div>`).join('')+'<div class="seat free">Freier Startplatz</div>'.repeat(5-game.players.length);
        if(cards!==lastCards){by('cards').innerHTML=cards;lastCards=cards;}
        for(const id of ['greenFlag','againStart','plusBot','minusBot','difficulty','copyLink']) by(id).hidden=!host;
        by('plusBot').disabled=game.players.length>=5;by('minusBot').disabled=!game.players.some(p=>p.bot);
        by('clubReady').disabled=me.waiting;by('againReady').disabled=false;
        by('greenFlag').disabled=by('againStart').disabled=!game.players.every(p=>p.ready);
        by('clubReady').textContent=by('againReady').textContent=me.ready?'Bereit ✓':finished?'Bereit zur Revanche':'Bereit melden';
        if(finished){
          const own = game.results.find(p=>p.id===me.id);
          by('saveLap').hidden = !own?.best_flying;
          by('flying').textContent = own?.best_flying
            ? `Beste fliegende Runde: ${time(own.best_flying)} · ${own.car} / ${own.tyre} · ${game.rules}`
            : 'Keine fliegende Runde ohne Rücksetzen.';
          const results=game.results.map(p=>`<li><b>${p.rank}</b><span>${pilots[p.avatar].icon} ${escape(p.name)}</span><strong>${p.finish_time==null?'DNF':time(p.finish_time)}</strong></li>`).join('');if(results!==lastResults){by('results').innerHTML=results;lastResults=results;}}
      }else if(game.phase==='countdown'){
        if(game.countdown>3)signal('<div class="grid-list"><b>Startaufstellung</b>'+game.grid.map(p=>`<span>P${p.position}　${escape(p.name)}</span>`).join('')+'</div>');
        else{const n=Math.ceil(game.countdown);signal('<div class="lights">'+[3,2,1].map(i=>`<i class="${i>=n?'lit':''}"></i>`).join('')+'</div>');}
      }else{
        if(lastRank && me.rank<lastRank)passUntil=game.elapsed+1.1;
        lastRank=me.rank;
        signal(me.finish_time!=null?`Ziel! <small>${time(me.finish_time)}</small>`:game.elapsed<1?'<div class="lights green"><i></i><i></i><i></i></div>LOS!':game.elapsed<passUntil?'Überholt!': '');
      }
    }
  };
}
