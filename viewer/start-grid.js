import * as THREE from 'three';

const paint = new THREE.MeshBasicMaterial({color:0xfff4c7,side:THREE.DoubleSide});
function asphaltStripe(parent,x,z,length,width) {
  const mark=new THREE.Mesh(new THREE.PlaneGeometry(length,width),paint);
  mark.rotation.x=-Math.PI/2;
  mark.position.set(x,.022,z);
  parent.add(mark);
}

export function startBoxes(slots) {
  const field=new THREE.Group();
  for(const slot of slots){
    const box=new THREE.Group();
    box.position.set(slot.x,0,-slot.y);
    box.rotation.y=slot.yaw;
    box.userData.position=slot.position;
    for(const z of [-.82,.82]) asphaltStripe(box,0,z,2.45,.095);
    asphaltStripe(box,1.18,0,.10,1.72);
    const canvas=document.createElement('canvas');
    canvas.width=canvas.height=128;
    const ctx=canvas.getContext('2d');
    ctx.fillStyle='#fff4c7';ctx.font='bold 110px Arial';ctx.textAlign='center';
    ctx.fillText(String(slot.position),64,105);
    const texture=new THREE.CanvasTexture(canvas);
    const number=new THREE.Mesh(new THREE.PlaneGeometry(.86,.86),
      new THREE.MeshBasicMaterial({map:texture,transparent:true,depthWrite:false}));
    number.rotation.set(-Math.PI/2,0,-Math.PI/2);
    number.position.set(1.74,.025,0);
    box.add(number);field.add(box);
  }
  return field;
}

export function gantryLights(scene,left,right) {
  const dx=right[0]-left[0],dy=right[1]-left[1],length=Math.hypot(dx,dy);
  const bulbs=[];
  const center=new THREE.Vector3((left[0]+right[0])/2,2.8,-(left[1]+right[1])/2);
  const frame=new THREE.Mesh(new THREE.BoxGeometry(2.3,.60,.48),
    new THREE.MeshStandardMaterial({color:0x14181c}));
  frame.position.copy(center);frame.rotation.y=Math.atan2(dy,dx);scene.add(frame);
  for(const side of [-.72,.72]){
    const hanger=new THREE.Mesh(new THREE.BoxGeometry(.09,.68,.09),frame.material);
    hanger.position.copy(center);hanger.position.y=3.31;
    hanger.position.x+=side*dx/length;hanger.position.z-=side*dy/length;scene.add(hanger);
  }
  for(const offset of [-.72,0,.72]){
    const lamp=new THREE.Mesh(new THREE.SphereGeometry(.235,20,12),
      new THREE.MeshBasicMaterial({color:0x381b1a}));
    lamp.position.copy(center);
    lamp.position.x+=(offset*dx+.27*dy)/length;
    lamp.position.z-=(offset*dy-.27*dx)/length;
    scene.add(lamp);bulbs.push(lamp);
  }
  return (phase,remaining,elapsed)=>{
    const count=Math.ceil(remaining);
    bulbs.forEach((lamp,i)=>lamp.material.color.setHex(
      phase==='racing' && elapsed<4 ? 0x62ff65 :
      phase==='countdown' && count<=3 && i<4-count ? 0xff342a : 0x402022));
  };
}
