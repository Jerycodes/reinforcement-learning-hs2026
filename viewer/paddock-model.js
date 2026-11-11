import * as T from 'three';

export const pilots = {
  Fuchs: {color:0xfa622e, accent:0xffe2bc, ears:'point', icon:'🦊'},
  Panda: {color:0xf1eee3, accent:0x242437, ears:'round', icon:'🐼'},
  Luchs: {color:0xeeb936, accent:0x55394f, ears:'tuft', icon:'🐱'},
  Drache: {color:0x41d4a2, accent:0x876aff, ears:'horn', icon:'🐲'},
};
export const paints = {Club:0x18beb6, Sprint:0xff6044, GT:0x8d73ed};
const bands = {Standard:0xe8e4d2, Grip:0xf2a83e, Speed:0x5ed4ff};

export function paddockKart(car, tyre, avatar, length=1.55, width=1.02) {
  const root = new T.Group();
  const material = color => new T.MeshStandardMaterial({color, roughness:.48, metalness:.14});
  const paint=material(paints[car]), dark=material(0x20212d), alloy=material(0xcbd4ca);
  const pilot=pilots[avatar], fur=material(pilot.color), trim=material(pilot.accent);
  function part(geometry, mat, x,y,z) {
    const obj=new T.Mesh(geometry,mat); obj.position.set(x,y,z); obj.castShadow=true; root.add(obj); return obj;
  }
  function box(x,y,z,w,h,d,mat) {return part(new T.BoxGeometry(w,h,d),mat,x,y,z);}
  function oval(x,y,z,w,h,d,mat) {
    const m=part(new T.SphereGeometry(1,18,12),mat,x,y,z); m.scale.set(w,h,d); return m;
  }
  box(0,.18,0,1.62,.14,.88,dark);
  const nose=oval(.62,.27,0,car==='GT'?.43:.35,.20,car==='Sprint'?.32:.44,paint);
  nose.rotation.z=-.17;
  box(.69,.425,0,.35,.035,.11,alloy);
  box(-.83,.27,0,.12,.18,1.16,dark);
  for (const z of [-.48,.48]) oval(-.12,.28,z,.54,.18,.15,paint);
  oval(-.24,.46,0,.29,.34,.29,dark);
  oval(-.15,.59,0,.21,.27,.23,fur);
  oval(-.17,.96,0,.275,.26,.265,fur);
  const visor=material(0x152b41);
  oval(.015,.98,0,.155,.112,.255,visor);
  for (const side of [-1,1]) {
    if (pilot.ears==='round') oval(-.21,1.16,side*.20,.099,.095,.08,trim);
    else {
      const ear=part(new T.ConeGeometry(.10,pilot.ears==='horn'?.34:.22,4),trim,-.19,1.21,side*.16);
      ear.rotation.x=side*.32;
      if(pilot.ears==='tuft') part(new T.ConeGeometry(.043,.15,4),dark,-.19,1.34,side*.19);
    }
    oval(.15,.59,side*.21,.14,.076,.07,fur);
  }
  // Collar, muzzle and visor reflection remain readable in the chase camera.
  oval(-.16,.746,0,.197,.065,.206,trim);
  oval(.06,.873,0,.10,.071,.15,trim);
  box(.162,.988,-.045,.01,.018,.19,alloy);
  const steering=part(new T.TorusGeometry(.146,.027,8,20),dark,.17,.59,0);
  steering.rotation.y=Math.PI/2-.35;
  if(car==='GT') {box(-.72,.40,0,.09,.37,.08,dark);box(-.76,.58,0,.25,.075,1.0,paint);}
  if(car==='Sprint') box(.32,.39,0,.17,.17,.48,paint);
  const front=[];
  for(const x of [-.57,.59]) for(const side of [-1,1]) {
    const pivot=new T.Group();pivot.position.set(x,.205,side*.53);root.add(pivot);
    const wheel=new T.Mesh(new T.CylinderGeometry(.205,.205,.19,20),dark);
    wheel.rotation.x=Math.PI/2;pivot.add(wheel);wheel.castShadow=true;
    const ring=new T.Mesh(new T.TorusGeometry(.133,.018,6,20),material(bands[tyre]));
    ring.position.z=side*.099;pivot.add(ring);
    if(x>0) front.push(pivot);
  }
  root.userData.frontWheels=front;
  const bounds = new T.Box3().setFromObject(root, true);
  const size = bounds.getSize(new T.Vector3());
  const center = bounds.getCenter(new T.Vector3());
  const flame=part(new T.ConeGeometry(.14,.72,12),new T.MeshBasicMaterial({color:0x5de8ff}),-1.08,.26,0);
  flame.rotation.z=Math.PI/2;flame.visible=false;root.userData.boostFlame=flame;
  const sparks=new T.Group(), sparkMaterial=new T.MeshBasicMaterial({color:0xffcc45});
  for (const side of [-1,1]) for(let n=0;n<3;n++) {
    const spark=new T.Mesh(new T.ConeGeometry(.023,.22+n*.06,4),sparkMaterial);
    spark.rotation.z=1.8+n*.18;spark.position.set(-.69-n*.13,.10+n*.045,side*(.51+n*.045));
    sparks.add(spark);
  }
  sparks.visible=false;root.add(sparks);root.userData.sparks=sparks;
  for (const part of root.children) {
    part.position.x -= center.x;
    part.position.z -= center.z;
  }
  root.scale.set(length/size.x, width/size.z, width/size.z);
  return root;
}

export function disposeModel(root) {
  const materials=new Set();
  root.traverse(m=>{if(m.isMesh){m.geometry.dispose();materials.add(m.material);}});
  for(const mat of materials) mat.dispose();
}

export function showroom(container) {
  const renderer=new T.WebGLRenderer({alpha:true,antialias:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio,1.6));container.append(renderer.domElement);
  const scene=new T.Scene(),camera=new T.PerspectiveCamera(30,1,.1,20);
  scene.add(new T.HemisphereLight(0xfff2de,0x65719e,2));
  const light=new T.DirectionalLight(0xffe1c7,3);light.position.set(2,5,3);scene.add(light);
  camera.position.set(2.65,1.95,3.05);camera.lookAt(0,.52,0);
  let kart;
  function draw(){
    const w=container.clientWidth,h=container.clientHeight;if(!w||!h)return;
    renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();renderer.render(scene,camera);
  }
  new ResizeObserver(draw).observe(container);
  return (car,tyre,avatar)=>{
    if(kart){scene.remove(kart);disposeModel(kart);}
    kart=paddockKart(car,tyre,avatar);scene.add(kart);draw();
  };
}
