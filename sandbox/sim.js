// AirDreamer 导航沙盒：深度相机、动力学、奖励、控制器和 3D 显示
(()=>{
// ---------- constants from the paper ----------
const W=80,H=48, DMIN=0.6, DMAX=6.0, HFOV=87*Math.PI/180, VFOV=58*Math.PI/180;
const CTRL_DT=0.05, PHYS_DT=0.005;          // 20 Hz policy, 200 Hz physics
const LAMBDA=[1.5,1.5,0.5];                  // m/s
const GOAL_R=1.0, COLL_R=0.3, Z_GOAL=1.5, Z_MIN=0.3, Z_MAX=3.2, T_MAX=60;
const OBS_H=3.5;

// ---------- RNG ----------
function rng(seed){return function(){seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
const U=(r,a,b)=>a+(b-a)*r();
const box=(x0,y0,x1,y1,h=OBS_H)=>({min:[Math.min(x0,x1),Math.min(y0,y1),0],max:[Math.max(x0,x1),Math.max(y0,y1),h]});

// ---------- scenes (ENU, metres) ----------
const SCENES={
  wall:{name:'大墙',build(){return{start:[0,0],goal:[9,0],boxes:[box(4.3,-4,4.7,4),box(6.3,1.0,7.1,1.8)]}}},
  passage:{name:'1 m 窄缝',build(){return{start:[0,0],goal:[9,0],boxes:[box(4.3,-7,4.7,1.5),box(4.3,2.5,4.7,7)]}}},
  ushape:{name:'U 形陷阱',build(){return{start:[0,0],goal:[8,0],boxes:[box(4.5,-2.7,4.9,2.7),box(1.5,-2.7,4.9,-2.3),box(1.5,2.3,4.9,2.7)]}}},
  cshape:{name:'C 形陷阱',build(){return{start:[0,0],goal:[6,0],startYaw:0,boxes:[box(1.5,-2.2,1.9,2.2),box(-1.6,-2.2,1.9,-1.8),box(-1.6,1.8,1.9,2.2),box(-1.6,-2.2,-1.2,-0.9),box(-1.6,0.9,-1.2,2.2)]}}},
  forest:{name:'树林',build(seed){const r=rng(seed),b=[];let tries=0;
    while(b.length<34&&tries++<2000){const x=U(r,2,9),y=U(r,-5,5),s=U(r,0.3,0.7);
      if(b.some(o=>Math.hypot((o.min[0]+o.max[0])/2-x,(o.min[1]+o.max[1])/2-y)<1.35))continue;b.push(box(x-s/2,y-s/2,x+s/2,y+s/2))}
    return{start:[0,0],goal:[11,0],boxes:b}}},
  train:{name:'训练分布',build(seed){const r=rng(seed),b=[];
    // U trap with random orientation, a couple of walls, scattered cuboids — like Fig. 3 of the paper
    const ux=U(r,3.5,5.5),uy=U(r,-1.5,1.5),w=U(r,2.0,3.0),d=U(r,2.2,3.2);
    b.push(box(ux,uy-w,ux+0.4,uy+w),box(ux-d,uy-w,ux+0.4,uy-w+0.4),box(ux-d,uy+w-0.4,ux+0.4,uy+w));
    b.push(box(U(r,-3,0),7,U(r,4,9),7.4), box(U(r,7,9),-7,U(r,7,9)+0.4,U(r,-4,-2)));
    let tries=0;while(b.length<22&&tries++<3000){const x=U(r,-3,13),y=U(r,-8,8),sx=U(r,0.4,1.4),sy=U(r,0.4,1.4);
      if(Math.hypot(x,y)<1.8||Math.hypot(x-10,y)<1.8)continue;
      const nb=box(x-sx/2,y-sy/2,x+sx/2,y+sy/2);
      if(b.some(o=>nb.min[0]<o.max[0]+0.9&&nb.max[0]>o.min[0]-0.9&&nb.min[1]<o.max[1]+0.9&&nb.max[1]>o.min[1]-0.9))continue;b.push(nb)}
    return{start:[0,0],goal:[10,U(r,-2,2)],boxes:b}}}
};

// ---------- geometry ----------
function rayHit(o,d,boxes){ // returns t (z-depth because body-forward component of d is 1)
  let best=Infinity;
  if(d[2]<-1e-6){const t=-o[2]/d[2];if(t>0)best=t}
  for(const b of boxes){let t0=-Infinity,t1=Infinity,miss=false;
    for(let i=0;i<3;i++){const di=d[i],oi=o[i];
      if(Math.abs(di)<1e-9){if(oi<b.min[i]||oi>b.max[i]){miss=true;break}continue}
      let a=(b.min[i]-oi)/di,c=(b.max[i]-oi)/di;if(a>c){const s=a;a=c;c=s}
      if(a>t0)t0=a;if(c<t1)t1=c;if(t0>t1){miss=true;break}}
    if(miss||t1<0)continue;const t=t0>0?t0:0;if(t<best)best=t}
  return best}
function boxDist(p,b){let s=0;for(let i=0;i<3;i++){const v=Math.max(b.min[i]-p[i],0,p[i]-b.max[i]);s+=v*v}return Math.sqrt(s)}
// per-pixel ray directions in body FLU with x=1
const RAYS=new Float32Array(W*H*2);
for(let v=0;v<H;v++)for(let u=0;u<W;u++){const k=(v*W+u)*2;RAYS[k]=Math.tan(HFOV/2)*(1-2*(u+0.5)/W);RAYS[k+1]=Math.tan(VFOV/2)*(1-2*(v+0.5)/H)}
const wrap=a=>{while(a>Math.PI)a-=2*Math.PI;while(a<-Math.PI)a+=2*Math.PI;return a};

// ---------- occupancy grid (for the mapping baseline and as a "memory" overlay) ----------
const G={x0:-12,y0:-14,res:0.2,n:150};
let occ=new Int8Array(G.n*G.n); // 0 unknown, 1 free, 2 occupied
const cell=(x,y)=>{const i=Math.floor((x-G.x0)/G.res),j=Math.floor((y-G.y0)/G.res);return(i<0||j<0||i>=G.n||j>=G.n)?-1:j*G.n+i};

// ---------- state ----------
let S=null, sceneKey='ushape', ctrlKey='astar', running=true, epSeed=1, stats={};
const keys={};
function sampleDR(r){return{mass:U(r,.9,1.1),inertia:[U(r,.85,1.15),U(r,.85,1.15),U(r,.85,1.15)],twr:U(r,.85,1.15),f2m:U(r,.85,1.15),drag:U(r,.75,1.25),up:[0,0,0,0].map(()=>U(r,.37,.49)),down:[0,0,0,0].map(()=>U(r,.37,.49))}}
function reset(){
  epSeed++;const r=rng(epSeed*7919);const sc=SCENES[sceneKey].build(epSeed);
  const g=[sc.goal[0],sc.goal[1],Z_GOAL];
  const yaw0=document.getElementById('randYaw').checked?U(r,-Math.PI,Math.PI):(sc.startYaw??Math.atan2(g[1]-sc.start[1],g[0]-sc.start[0]));
  const dr=sampleDR(r);
  // simplified mapping of the randomised parameters onto the velocity loop
  const spin=(dr.up.reduce((a,b)=>a+b)+dr.down.reduce((a,b)=>a+b))/8;
  S={sc,boxes:sc.boxes,goal:g,p:[sc.start[0],sc.start[1],U(r,1.2,1.8)],v:[0,0,0],yaw:yaw0,yawRate:0,
     dr,tauV:0.18*dr.mass/dr.twr*(0.43/spin),tauYaw:0.25*(dr.inertia[2]),dragK:0.05*dr.drag,
     cmdB:[0,0,0],cmdYaw:yaw0,act:[0,0,0,1,0],t:0,step:0,ret:0,terms:{prog:0,safety:0,height:0,goal:0,coll:0},
     depth:new Float32Array(W*H),dxyPrev:null,done:null,path:[],trail:[],planT:0,stuckT:0};
  occ=new Int8Array(G.n*G.n);
  S.dxyPrev=Math.hypot(g[0]-S.p[0],g[1]-S.p[1]);
  renderDepth();buildScene();updateDR();hideBanner();
}

// ---------- sensing ----------
function renderDepth(){
  const c=Math.cos(S.yaw),s=Math.sin(S.yaw),o=S.p,d=[0,0,0];
  for(let i=0;i<W*H;i++){const by=RAYS[2*i],bz=RAYS[2*i+1];
    d[0]=c-s*by;d[1]=s+c*by;d[2]=bz;
    let t=rayHit(o,d,S.boxes);if(!(t>=DMIN&&t<=DMAX))t=DMAX; // invalid pixels filled with 6 m, as in the paper
    S.depth[i]=t}
  // integrate centre rows into the occupancy grid
  for(let u=0;u<W;u++){const by=RAYS[2*(24*W+u)];const dep=Math.min(S.depth[23*W+u],S.depth[24*W+u]);
    const dx=c-s*by,dy=s+c*by,range=dep*Math.hypot(1,by),nx=dx/Math.hypot(dx,dy),ny=dy/Math.hypot(dx,dy);
    for(let q=0.1;q<range-0.1;q+=0.1){const k=cell(o[0]+nx*q,o[1]+ny*q);if(k>=0&&occ[k]!==2)occ[k]=1}
    if(dep<DMAX-0.05){const k=cell(o[0]+nx*range,o[1]+ny*range);if(k>=0)occ[k]=2}}
}
function obsDist(){let m=Infinity;for(const b of S.boxes)m=Math.min(m,boxDist(S.p,b));return m}

// ---------- controllers (output: body-frame velocity cmd in m/s and world yaw cmd) ----------
function centreHits(){const out=[],c=Math.cos(S.yaw),s=Math.sin(S.yaw);
  for(let u=0;u<W;u+=2){const by=RAYS[2*(24*W+u)],dep=S.depth[24*W+u];if(dep>=DMAX-0.05)continue;
    const r=dep*Math.hypot(1,by),dx=c-s*by,dy=s+c*by,n=Math.hypot(dx,dy);out.push([dx/n,dy/n,r])}return out}
function toBody(vx,vy){const c=Math.cos(S.yaw),s=Math.sin(S.yaw);return[c*vx+s*vy,-s*vx+c*vy]}
const CTRL={
  greedy:{name:'贪心势场',desc:'反应式避障：朝目标的吸引 + 深度图斥力（带切向分量），没有记忆，偏航锁定朝向目标',
    act(){const g=S.goal,p=S.p;let ax=g[0]-p[0],ay=g[1]-p[1];const n=Math.hypot(ax,ay)||1;ax=ax/n*1.2;ay=ay/n*1.2;
      for(const[hx,hy,r]of centreHits()){if(r<1.8){const k=1.0*(1/r-1/1.8);ax-=hx*k-0.8*hy*k;ay-=hy*k+0.8*hx*k}}
      const sp=Math.hypot(ax,ay);if(sp>1.5){ax*=1.5/sp;ay*=1.5/sp}
      const[bx,by]=toBody(ax,ay);return{vb:[bx,by,0.8*(Z_GOAL-p[2])],yaw:Math.atan2(g[1]-p[1],g[0]-p[0])}}},
  astar:{name:'建图 + A*',desc:'把深度图融合进占据栅格，未知区域当作可通行，A* 重规划；偏航跟随运动方向，转弯前先转头观察',
    act(){if(S.t-S.planT>0.25||!S.path.length){S.path=plan();S.planT=S.t}
      const p=S.p;let tgt=S.goal;
      if(S.path.length){let acc=0,prev=[p[0],p[1]];tgt=S.path[S.path.length-1];
        for(const q of S.path){acc+=Math.hypot(q[0]-prev[0],q[1]-prev[1]);prev=q;if(acc>1.2){tgt=q;break}}}
      let dx=tgt[0]-p[0],dy=tgt[1]-p[1];const n=Math.hypot(dx,dy)||1;const want=Math.atan2(dy,dx);
      const err=Math.abs(wrap(want-S.yaw));let dmin=DMAX;for(const h of centreHits())dmin=Math.min(dmin,h[2]);
      let sp=Math.min(1.3,0.3+0.5*Math.max(0,dmin-0.5));sp*=err>0.6?0.05:Math.cos(err);
      const[bx,by]=toBody(dx/n*sp,dy/n*sp);return{vb:[bx,by,0.8*(Z_GOAL-p[2])],yaw:want}}},
  manual:{name:'手动',desc:'键盘控制：WASD 平移，QE 转头，RF 升降',
    act(){const f=(keys.w?1:0)-(keys.s?1:0),l=(keys.a?1:0)-(keys.d?1:0),u=(keys.r?1:0)-(keys.f?1:0),y=(keys.q?1:0)-(keys.e?1:0);
      return{vb:[f*1.2,l*1.2,u*0.5],yaw:S.cmdYaw+y*1.6*CTRL_DT}}}
};
function plan(){ // A* on an inflated copy of the occupancy grid
  const n=G.n,R=5,cost=new Uint8Array(n*n); // 255 = blocked (within 0.4 m), otherwise soft penalty up to 1.0 m
  for(let j=0;j<n;j++)for(let i=0;i<n;i++)if(occ[j*n+i]===2)
    for(let dj=-R;dj<=R;dj++)for(let di=-R;di<=R;di++){const r2=di*di+dj*dj;if(r2>R*R)continue;const ii=i+di,jj=j+dj;if(ii<0||jj<0||ii>=n||jj>=n)continue;
      const c=r2<=4?255:Math.round(12*(1-Math.sqrt(r2)/R))+1;const kk=jj*n+ii;if(c>cost[kk])cost[kk]=c}
  const s=cell(S.p[0],S.p[1]),g=cell(S.goal[0],S.goal[1]);if(s<0||g<0)return[];
  const gi=g%n,gj=(g/n)|0,gs=new Float32Array(n*n).fill(Infinity),from=new Int32Array(n*n).fill(-1),closed=new Uint8Array(n*n);
  const heap=[];const push=(k,f)=>{heap.push([f,k]);let i=heap.length-1;while(i>0){const pa=(i-1)>>1;if(heap[pa][0]<=heap[i][0])break;[heap[pa],heap[i]]=[heap[i],heap[pa]];i=pa}};
  const pop=()=>{const top=heap[0],last=heap.pop();if(heap.length){heap[0]=last;let i=0;for(;;){const l=2*i+1,r=l+1;let m=i;if(l<heap.length&&heap[l][0]<heap[m][0])m=l;if(r<heap.length&&heap[r][0]<heap[m][0])m=r;if(m===i)break;[heap[m],heap[i]]=[heap[i],heap[m]];i=m}}return top};
  gs[s]=0;push(s,0);const D=[[1,0,1],[-1,0,1],[0,1,1],[0,-1,1],[1,1,1.414],[1,-1,1.414],[-1,1,1.414],[-1,-1,1.414]];
  while(heap.length){const[,k]=pop();if(closed[k])continue;closed[k]=1;if(k===g)break;const i=k%n,j=(k/n)|0;
    for(const[di,dj,w]of D){const ii=i+di,jj=j+dj;if(ii<0||jj<0||ii>=n||jj>=n)continue;const kk=jj*n+ii;const near=Math.abs(ii-s%n)<=1&&Math.abs(jj-((s/n)|0))<=1;if(cost[kk]===255&&kk!==g&&!near)continue;
      const ng=gs[k]+w*(1+(cost[kk]===255?20:cost[kk]));if(ng<gs[kk]){gs[kk]=ng;from[kk]=k;push(kk,ng+Math.hypot(ii-gi,jj-gj))}}}
  if(from[g]<0)return[];const out=[];for(let k=g;k!==s&&k>=0;k=from[k])out.push([G.x0+((k%n)+0.5)*G.res,G.y0+(((k/n)|0)+0.5)*G.res]);return out.reverse()}

// ---------- step ----------
function controlStep(){
  const c=CTRL[ctrlKey].act();
  const vb=[0,1,2].map(i=>Math.max(-LAMBDA[i],Math.min(LAMBDA[i],c.vb[i])));
  S.cmdB=vb;S.cmdYaw=wrap(c.yaw);
  S.act=[vb[0]/LAMBDA[0],vb[1]/LAMBDA[1],vb[2]/LAMBDA[2],Math.cos(S.cmdYaw),Math.sin(S.cmdYaw)];
  // physics @200 Hz: first-order velocity + yaw tracking (stand-in for the Lee controller + randomised motors)
  for(let k=0;k<CTRL_DT/PHYS_DT;k++){
    const cy=Math.cos(S.yaw),sy=Math.sin(S.yaw);
    const vw=[cy*vb[0]-sy*vb[1],sy*vb[0]+cy*vb[1],vb[2]];
    for(let i=0;i<3;i++){S.v[i]+=(vw[i]-S.v[i])*PHYS_DT/S.tauV-S.dragK*S.v[i]*Math.abs(S.v[i])*PHYS_DT;S.p[i]+=S.v[i]*PHYS_DT}
    const yr=Math.max(-2.5,Math.min(2.5,wrap(S.cmdYaw-S.yaw)/S.tauYaw));S.yawRate+=(yr-S.yawRate)*PHYS_DT/0.05;S.yaw=wrap(S.yaw+S.yawRate*PHYS_DT)}
  S.t+=CTRL_DT;S.step++;
  renderDepth();
  // reward, exactly the paper's terms
  const g=S.goal,dxy=Math.hypot(g[0]-S.p[0],g[1]-S.p[1]),dob=obsDist();
  const tr={prog:S.dxyPrev-dxy,safety:-Math.pow(Math.max(0,1-dob),2),height:-Math.pow(g[2]-S.p[2],2),goal:0,coll:0};
  S.dxyPrev=dxy;
  if(dob<COLL_R){tr.coll=-5;S.done='碰撞'}
  else if(Math.hypot(dxy,g[2]-S.p[2])<GOAL_R){tr.goal=100;S.done='到达'}
  else if(S.p[2]<Z_MIN||S.p[2]>Z_MAX){S.done='高度越界'}
  else if(S.t>=T_MAX){S.done='超时'}
  const r=10*tr.prog+5*tr.safety+2*tr.height+tr.goal+tr.coll;
  S.terms=tr;S.r=r;S.ret+=r;
  if(S.step%2===0){S.trail.push([S.p[0],S.p[1],S.p[2]]);if(S.trail.length>4000)S.trail.shift()}
  if(S.done)finish();
}
function finish(){
  const k=sceneKey+'/'+ctrlKey;stats[k]=stats[k]||{n:0,ok:0};stats[k].n++;if(S.done==='到达')stats[k].ok++;
  const li=document.createElement('li');const cls=S.done==='到达'?'ok':S.done==='超时'?'to':'fail';
  li.innerHTML=`<span>${SCENES[sceneKey].name} · ${CTRL[ctrlKey].name}</span><span class="${cls}">${S.done}</span><span>G=${S.ret.toFixed(1)}</span>`;
  const log=document.getElementById('log');log.prepend(li);while(log.children.length>40)log.lastChild.remove();
  showBanner(S.done,cls);updateSR();
  if(document.getElementById('auto').checked&&ctrlKey!=='manual')setTimeout(()=>{if(S.done)reset()},1400);
}

// ---------- three.js ----------
const vp=document.getElementById('vp');
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(2,window.devicePixelRatio||1));vp.prepend(renderer.domElement);
const scene=new THREE.Scene();const cam=new THREE.PerspectiveCamera(50,1.6,0.1,200);
const orbit=new THREE.OrbitControls(cam,renderer.domElement);orbit.enableDamping=true;orbit.target.set(4,0,0);cam.position.set(-6,11,10);
scene.add(new THREE.HemisphereLight(0xffffff,0x667788,0.9));const sun=new THREE.DirectionalLight(0xffffff,0.6);sun.position.set(5,12,4);scene.add(sun);
const E=(x,y,z)=>new THREE.Vector3(x,z,-y); // ENU -> three (y-up)
let world=new THREE.Group();scene.add(world);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function themeColors(){scene.background=new THREE.Color(css('--scene-bg'))}
// drone
const drone=new THREE.Group();scene.add(drone);
(()=>{const m=new THREE.MeshStandardMaterial({color:0x2b3640,roughness:.6}),acc=new THREE.MeshStandardMaterial({color:0xe4572e});
  drone.add(new THREE.Mesh(new THREE.BoxGeometry(0.22,0.07,0.16),m));
  for(const a of[1,-1]){const arm=new THREE.Mesh(new THREE.BoxGeometry(0.5,0.02,0.03),m);arm.rotation.y=a*Math.PI/4;drone.add(arm)}
  for(const[x,z]of[[.18,.18],[.18,-.18],[-.18,.18],[-.18,-.18]]){const r=new THREE.Mesh(new THREE.CylinderGeometry(.09,.09,.01,20),new THREE.MeshStandardMaterial({color:0x8aa0ad,transparent:true,opacity:.6}));r.position.set(x,.04,z);drone.add(r)}
  const cam=new THREE.Mesh(new THREE.BoxGeometry(.05,.04,.1),acc);cam.position.set(.13,0,0);drone.add(cam)})();
// camera frustum in drone frame (three: x forward, y up, -z = left? we map body y(left) -> three -z)
const fr=(()=>{const L=DMAX,a=Math.tan(HFOV/2)*L,b=Math.tan(VFOV/2)*L,c=[[L,b,-a],[L,b,a],[L,-b,a],[L,-b,-a]];const pts=[];
  for(const q of c)pts.push(0,0,0,...q);for(let i=0;i<4;i++)pts.push(...c[i],...c[(i+1)%4]);
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(pts,3));
  return new THREE.LineSegments(g,new THREE.LineBasicMaterial({color:0xe4572e,transparent:true,opacity:.55}))})();drone.add(fr);
// trail, path, goal
const trailGeo=new THREE.BufferGeometry();trailGeo.setAttribute('position',new THREE.Float32BufferAttribute(new Float32Array(4000*3),3));
const trail=new THREE.Line(trailGeo,new THREE.LineBasicMaterial({color:0x0b7a8a}));scene.add(trail);
const pathGeo=new THREE.BufferGeometry();pathGeo.setAttribute('position',new THREE.Float32BufferAttribute(new Float32Array(2000*3),3));
const pathLine=new THREE.Line(pathGeo,new THREE.LineDashedMaterial({color:0x7b5cd6,dashSize:.25,gapSize:.15}));scene.add(pathLine);
// occupancy overlay
const occTex=new THREE.DataTexture(new Uint8Array(G.n*G.n*4),G.n,G.n,THREE.RGBAFormat);occTex.magFilter=THREE.NearestFilter;
const occMesh=new THREE.Mesh(new THREE.PlaneGeometry(G.n*G.res,G.n*G.res),new THREE.MeshBasicMaterial({map:occTex,transparent:true,depthWrite:false}));
occMesh.rotation.x=-Math.PI/2;occMesh.position.copy(E(G.x0+G.n*G.res/2,G.y0+G.n*G.res/2,0.02));scene.add(occMesh);
function buildScene(){
  scene.remove(world);world=new THREE.Group();scene.add(world);themeColors();
  const grid=new THREE.GridHelper(40,40,new THREE.Color(css('--grid')),new THREE.Color(css('--grid')));grid.position.copy(E(4,0,0));world.add(grid);
  const om=new THREE.MeshStandardMaterial({color:0x9aa7b0,roughness:.85,transparent:true,opacity:.92});
  for(const b of S.boxes){const sx=b.max[0]-b.min[0],sy=b.max[1]-b.min[1],sz=b.max[2]-b.min[2];
    const m=new THREE.Mesh(new THREE.BoxGeometry(sx,sz,sy),om);m.position.copy(E((b.min[0]+b.max[0])/2,(b.min[1]+b.max[1])/2,sz/2));world.add(m);
    const e=new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry),new THREE.LineBasicMaterial({color:0x56636c}));e.position.copy(m.position);world.add(e)}
  const gm=new THREE.Mesh(new THREE.SphereGeometry(.25,24,16),new THREE.MeshStandardMaterial({color:0x2f9e44,emissive:0x1b5e20}));gm.position.copy(E(...S.goal));world.add(gm);
  const ring=new THREE.Mesh(new THREE.RingGeometry(GOAL_R-.04,GOAL_R,48),new THREE.MeshBasicMaterial({color:0x2f9e44,side:THREE.DoubleSide,transparent:true,opacity:.6}));ring.rotation.x=-Math.PI/2;ring.position.copy(E(S.goal[0],S.goal[1],.03));world.add(ring);
  const st=new THREE.Mesh(new THREE.RingGeometry(.25,.32,32),new THREE.MeshBasicMaterial({color:0x56636c,side:THREE.DoubleSide}));st.rotation.x=-Math.PI/2;st.position.copy(E(S.sc.start[0],S.sc.start[1],.03));world.add(st);
}
function syncThree(){
  drone.position.copy(E(...S.p));drone.rotation.set(0,S.yaw,0);
  const arr=trailGeo.attributes.position.array;S.trail.forEach((q,i)=>{const v=E(...q);arr[3*i]=v.x;arr[3*i+1]=v.y;arr[3*i+2]=v.z});
  trailGeo.setDrawRange(0,S.trail.length);trailGeo.attributes.position.needsUpdate=true;
  const pa=pathGeo.attributes.position.array,pts=ctrlKey==='astar'?S.path.slice(0,1999):[];pts.forEach((q,i)=>{const v=E(q[0],q[1],S.p[2]);pa[3*i]=v.x;pa[3*i+1]=v.y;pa[3*i+2]=v.z});
  pathGeo.setDrawRange(0,pts.length);pathGeo.attributes.position.needsUpdate=true;pathLine.computeLineDistances();
  const show=document.getElementById('showMap').checked;occMesh.visible=show;
  if(show){const d=occTex.image.data;for(let j=0;j<G.n;j++)for(let i=0;i<G.n;i++){const k=j*G.n+i,o=k*4,v=occ[k];
      if(v===2){d[o]=200;d[o+1]=60;d[o+2]=40;d[o+3]=230}else if(v===1){d[o]=60;d[o+1]=170;d[o+2]=190;d[o+3]=70}else d[o+3]=0}
    occTex.flipY=false;occTex.needsUpdate=true}
}
// views
let view='orbit';
const VIEWS={orbit:'自由',chase:'跟随',top:'俯视'};
for(const[k,n]of Object.entries(VIEWS)){const b=document.createElement('button');b.textContent=n;b.dataset.v=k;b.setAttribute('aria-pressed',k===view);b.onclick=()=>{view=k;document.querySelectorAll('#views button').forEach(x=>x.setAttribute('aria-pressed',x.dataset.v===view));orbit.enabled=view==='orbit'};document.getElementById('views').append(b)}
function updateCam(){
  if(view==='chase'){const c=Math.cos(S.yaw),s=Math.sin(S.yaw);cam.position.lerp(E(S.p[0]-3.2*c,S.p[1]-3.2*s,S.p[2]+1.6),.12);cam.lookAt(E(S.p[0]+2*c,S.p[1]+2*s,S.p[2]))}
  else if(view==='top'){cam.position.lerp(E(S.p[0]*.5+S.goal[0]*.5,(S.p[1]+S.goal[1])*.5,20),.1);cam.lookAt(E(S.p[0]*.5+S.goal[0]*.5,(S.p[1]+S.goal[1])*.5,0))}
  else orbit.update()}
function resize(){const r=vp.getBoundingClientRect();renderer.setSize(r.width,r.height,false);cam.aspect=r.width/r.height;cam.updateProjectionMatrix()}
new ResizeObserver(resize).observe(vp);

// ---------- panels ----------
const dctx=document.getElementById('depth').getContext('2d'),dimg=dctx.createImageData(W,H);
function turbo(x){ // x in [0,1], 0 near (red) -> 1 far (dark blue), polynomial turbo approximation
  x=1-Math.max(0,Math.min(1,x));
  const r=34.61+x*(1172.33-x*(10793.56-x*(33300.12-x*(38394.49-x*14825.05))));
  const g=23.31+x*(557.33+x*(1225.33-x*(3574.96-x*(1073.77+x*707.56))));
  const b=27.2+x*(3211.1-x*(15327.97-x*(27814-x*(22569.18-x*6838.66))));
  return[Math.max(0,Math.min(255,r)),Math.max(0,Math.min(255,g)),Math.max(0,Math.min(255,b))]}
function drawDepth(){for(let i=0;i<W*H;i++){const[r,g,b]=turbo((S.depth[i]-DMIN)/(DMAX-DMIN));dimg.data[4*i]=r;dimg.data[4*i+1]=g;dimg.data[4*i+2]=b;dimg.data[4*i+3]=255}dctx.putImageData(dimg,0,0)}
const TERMS=[['prog','10·r_prog',10],['safety','5·r_safety',5],['height','2·r_height',2],['goal','r_goal',1],['coll','r_collision',1]];
document.getElementById('rewards').innerHTML=TERMS.map(([k,l])=>`<div class="rw"><span>${l}</span><div class="bar"><b id="rb-${k}"></b></div><span class="num" id="rn-${k}">0.000</span></div>`).join('');
function drawRewards(){for(const[k,,w]of TERMS){const v=w*S.terms[k],el=document.getElementById('rb-'+k),sc=k==='goal'||k==='coll'?100:1.2;
    const f=Math.max(-1,Math.min(1,v/sc))*50;el.style.left=(f<0?50+f:50)+'%';el.style.width=Math.abs(f)+'%';el.style.background=v<0?'var(--bad)':'var(--good)';
    document.getElementById('rn-'+k).textContent=v.toFixed(3)}
  document.getElementById('ret').textContent=S.ret.toFixed(2);document.getElementById('step').textContent=`t = ${S.t.toFixed(2)} s · step ${S.step}`}
const f3=v=>(v>=0?' ':'')+v.toFixed(2);
function drawObs(){
  const g=S.goal,p=S.p,c=Math.cos(S.yaw),s=Math.sin(S.yaw),dx=g[0]-p[0],dy=g[1]-p[1],dz=g[2]-p[2],n=Math.hypot(dx,dy,dz)||1;
  const ng=[(c*dx+s*dy)/n,(-s*dx+c*dy)/n,dz/n],vb=[c*S.v[0]+s*S.v[1],-s*S.v[0]+c*S.v[1],S.v[2]];
  const q=[Math.cos(S.yaw/2),0,0,Math.sin(S.yaw/2)];
  const rows=[['n̂_g^b (3)',ng.map(f3).join(' ')],['d_xy, d_z',f3(Math.hypot(dx,dy))+'  '+f3(dz)],['v^b (3)',vb.map(f3).join(' ')],['Ω (3)',[0,0,S.yawRate].map(f3).join(' ')],['q (4)',q.map(f3).join(' ')],
    ['a: v_cmd/Λ',S.act.slice(0,3).map(f3).join(' ')],['a: (cΨ, sΨ)',S.act.slice(3).map(f3).join(' ')],['离障碍最近',obsDist().toFixed(2)+' m']];
  document.getElementById('obs').innerHTML=rows.map(r=>`<tr><td class="k">${r[0]}</td><td class="v">${r[1]}</td></tr>`).join('')}
function updateDR(){const d=S.dr,f=v=>v.toFixed(2);
  const rows=[['质量 scale',f(d.mass)],['惯量 scale',d.inertia.map(f).join(' ')],['推重比 scale',f(d.twr)],['F2M scale',f(d.f2m)],['阻力 scale',f(d.drag)],['spin-up',d.up.map(f).join(' ')],['spin-down',d.down.map(f).join(' ')],['→ 速度环 τ',S.tauV.toFixed(3)+' s']];
  document.getElementById('dr').innerHTML=rows.map(r=>`<tr><td class="k">${r[0]}</td><td class="v">${r[1]}</td></tr>`).join('')}
function updateSR(){const st=stats[sceneKey+'/'+ctrlKey];document.getElementById('sr').textContent=st?`本组合成功率 ${st.ok}/${st.n}`:''}
function drawHud(){document.getElementById('hud').innerHTML=`${SCENES[sceneKey].name} · ${CTRL[ctrlKey].name}<br>|v| ${Math.hypot(...S.v).toFixed(2)} m/s · ψ ${(S.yaw*180/Math.PI).toFixed(0)}°<br>到目标 ${S.dxyPrev.toFixed(2)} m · z ${S.p[2].toFixed(2)} m`}
const banner=document.getElementById('banner');function showBanner(t,c){banner.textContent=t;banner.className='banner '+c;banner.hidden=false}function hideBanner(){banner.hidden=true}

// ---------- UI wiring ----------
function seg(id,obj,get,set){const el=document.getElementById(id);for(const[k,v]of Object.entries(obj)){const b=document.createElement('button');b.textContent=v.name;b.dataset.k=k;if(v.desc)b.title=v.desc;b.onclick=()=>{set(k);el.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed',x.dataset.k===get()));reset();updateSR()};b.setAttribute('aria-pressed',k===get());el.append(b)}}
seg('scenes',SCENES,()=>sceneKey,k=>sceneKey=k);
seg('ctrls',CTRL,()=>ctrlKey,k=>ctrlKey=k);
const playBtn=document.getElementById('play');playBtn.onclick=()=>{running=!running;playBtn.textContent=running?'暂停':'继续'};
document.getElementById('reset').onclick=reset;
addEventListener('keydown',e=>{const k=e.key.toLowerCase();if('wasdqerf'.includes(k)&&k.length===1){keys[k]=true;if(ctrlKey==='manual')e.preventDefault()}});
addEventListener('keyup',e=>{keys[e.key.toLowerCase()]=false});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>S&&buildScene());
new MutationObserver(()=>S&&buildScene()).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});

// ---------- loop ----------
let last=performance.now(),acc=0,panelT=0;
function frame(now){const dt=Math.min(0.1,(now-last)/1000);last=now;
  if(running&&!S.done){acc+=dt*(+document.getElementById('speed').value);let n=0;while(acc>=CTRL_DT&&n<12&&!S.done){controlStep();acc-=CTRL_DT;n++}if(n>=12)acc=0}
  syncThree();updateCam();
  if(now-panelT>60){drawDepth();drawRewards();drawObs();drawHud();panelT=now}
  renderer.render(scene,cam);requestAnimationFrame(frame)}
reset();resize();requestAnimationFrame(frame);
window.__sim={get S(){return S},step:controlStep,reset,setScene:k=>{sceneKey=k;reset()},setCtrl:k=>{ctrlKey=k;reset()}};
})();
