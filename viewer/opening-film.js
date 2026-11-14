import {paddockKart} from './paddock-model.js';
export function openingFilm(track,scene,camera,renderer,sound){
  const overlay=document.createElement('section');overlay.id='openingFilm';
  overlay.innerHTML='<strong>ROGGWIL<br>GRAND PRIX</strong><button type="button">Spiel starten ›</button>';
  document.body.append(overlay);
  const button=overlay.querySelector('button'), title=overlay.querySelector('strong');
  let running=false,frame=0;
  const cast=[];
  function finish(){
    cancelAnimationFrame(frame);sound.stop();sound.menu();document.body.classList.remove('opening-shot');
    for(const kart of cast){scene.remove(kart);kart.traverse(obj=>{obj.geometry?.dispose();obj.material?.dispose?.();});}
    overlay.remove();document.getElementById('pilotName')?.focus();
  }
  button.onclick=()=>{
    sound.unlock();
    if(running||!track||matchMedia('(prefers-reduced-motion: reduce)').matches){finish();return;}
    running=true;document.body.classList.add('opening-shot');button.textContent='Überspringen';
    for(const [car,avatar] of [['Sprint','Fuchs'],['GT','Panda']]){
      const kart=paddockKart(car,'Grip',avatar);scene.add(kart);cast.push(kart);
    }
    const start=performance.now(),pts=track.centerline;
    let shot=-1;
    function render(now){
      const t=Math.max(0,now-start)/1000;
      if(t>=8){finish();return;}
      const cut=Math.min(3,Math.floor(t/2));
      if(cut!==shot){shot=cut;sound.preview(cut%2?'GT':'Sprint');}
      title.hidden=t>.9&&t<6.8;
      for(const [i,kart] of cast.entries()){
        const pos=(26+t*13+i*(5-t*.8))%pts.length,n=Math.floor(pos),a=pts[n],b=pts[(n+1)%pts.length],u=pos-n;
        const yaw=Math.atan2(b[1]-a[1],b[0]-a[0]),lane=i?1:-1;
        kart.position.set(a[0]+u*(b[0]-a[0])-lane*Math.sin(yaw),0,-a[1]-u*(b[1]-a[1])-lane*Math.cos(yaw));
        kart.rotation.y=yaw;
        kart.userData.sparks.visible=cut===2;
        kart.userData.boostFlame.visible=cut===3;
      }
      const k=cast[0],p=k.position,c=Math.cos(k.rotation.y),s=Math.sin(k.rotation.y);
      const [along,side,h]=[[.7,1.6,.32],[-3,1,.75],[4,3,1.2],[-6,1.6,2.3]][cut];
      camera.position.set(p.x+c*along+s*side,h,p.z-s*along+c*side);
      camera.lookAt(p.x+c*.5,.35,p.z-s*.5);renderer.render(scene,camera);
      frame=requestAnimationFrame(render);
    }
    render(start);
  };
}
