'use strict';
const $ = id => document.getElementById(id);
const robot = $('robot'), cameras = [$('camera1'), $('camera2')];
const colors = ['#73bfee','#78d6a9','#caa7f3','#efc37d','#b1c4ea','#d8a1b7','#6db6b1','#cad286'];
let catalog = [], data = null, frame = 50, playing = false, ready = false;
let scenario = new URLSearchParams(location.search).get('scenario');
if(scenario)$('scenario').value=scenario;
let generation = 0, controller = null, pollTimer = null, animation = 0;
let inferenceGeneration=0,inferenceController=null,inferenceTimer=null,inferenceBusy=false,predictionTemplate=null;
const jointSignals = {q:{title:'Joint Angle',unit:'rad'}, qdot:{title:'Joint Velocity',unit:'rad/s'}, tau:{title:'Joint Torque',unit:'N·m'}};
const plots = ['force','joint','imu'].map(name => ({name,canvas:$(name+'Plot')}));
for (const plot of plots) plot.context = plot.canvas.getContext('2d');
const finite = value => typeof value === 'number' && Number.isFinite(value);
const hasForce = value => finite(value) && value > 1e-6;
const hasPlotForce = value => hasForce(value) && Number(value.toFixed($('wrenchSignal').value==='torque'?2:1)) > 0;
const fmt = (value,digits=2) => finite(value) ? value.toFixed(digits) : '—';
const signed = (value,digits=2) => finite(value) ? (value>=0?'+':'')+value.toFixed(digits) : '—';

function layoutCameras() {
  const hasBanner=data?.scenario==='lie'||
    (data?.scenario==='stand'&&data.number>=1&&data.number<=286);
  const cropTop=hasBanner?16/660:0;
  for(const video of cameras) {
    const viewport=video.parentElement,figure=viewport.parentElement;
    const width=video.videoWidth,height=video.videoHeight;
    if(!width||!height)continue;
    const top=Math.ceil(height*cropTop),visibleHeight=height-top;
    const scale=Math.min(figure.clientWidth/width,figure.clientHeight/visibleHeight);
    viewport.style.width=`${width*scale}px`;
    viewport.style.height=`${visibleHeight*scale}px`;
    video.style.width=`${width*scale}px`;
    video.style.height=`${height*scale}px`;
    video.style.transform=`translateY(${-top*scale}px)`;
  }
}
const cameraResize=new ResizeObserver(layoutCameras);
cameras.forEach(video=>cameraResize.observe(video.parentElement.parentElement));

function interpolate(xs,ys,x) {
  if (x<=xs[0]) return ys[0];
  if (x>=xs[xs.length-1]) return ys[ys.length-1];
  let low=0,high=xs.length-1;
  while(high-low>1){const middle=(low+high)>>1;if(xs[middle]<=x)low=middle;else high=middle;}
  return ys[low]+(ys[high]-ys[low])*(x-xs[low])/(xs[high]-xs[low]);
}
function cameraTime(index, seconds) {
  const mapping=data.videos[index];
  if(mapping.available===false)return null;
  return interpolate(mapping.event_times,mapping.media_times,seconds+data.event_time_s[0]);
}
function setMediaTime(media,time) {
  if(media.readyState>=1 && finite(time)) media.currentTime=Math.max(0,Math.min(time,media.duration-.00001));
}
function synchronizeCameras(seconds,force=false) {
  if(!data)return;
  cameras.forEach((video,index)=>{
    const target=cameraTime(index,seconds);
    if(force||Math.abs(video.currentTime-target)>.055)setMediaTime(video,target);
  });
}
function pause() {
  playing=false;cancelAnimationFrame(animation);
  robot.pause();cameras.forEach(video=>video.pause());
  $('play').textContent='▶ Play';
}
function seekFrame(next) {
  if(!data)return;
  frame=Math.max(0,Math.min(data.frames-1,Math.round(next)));
  const seconds=data.time_s[frame];
  if(ready)setMediaTime(robot,seconds);
  synchronizeCameras(seconds,true);update();
}
function frameAt(seconds) {
  let low=0,high=data.frames-1;
  while(low<high){const mid=Math.ceil((low+high)/2);if(data.time_s[mid]<=seconds+1e-5)low=mid;else high=mid-1;}
  return low;
}
async function play() {
  if(!ready||!data)return;
  if(playing){pause();return;}
  if(frame>=data.frames-1)seekFrame(0);
  const current=generation;
  const media=[robot,...cameras.filter((_,index)=>data.videos[index].available!==false)];
  synchronizeCameras(robot.currentTime,true);
  playing=true;$('play').textContent='Ⅱ Pause';
  try {
    await Promise.all(media.filter(video=>!video.error).map(video=>video.play()));
    if(current!==generation||!playing)return;
    tick();
  } catch(error) {
    if(current!==generation)return;
    pause();showLoading('Video is not ready','Reload this sample to retry.',true);
  }
}
function tick() {
  if(!playing||!data)return;
  const next=frameAt(robot.currentTime);
  if(next!==frame){frame=next;update();}
  synchronizeCameras(robot.currentTime);
  animation=requestAnimationFrame(tick);
}
robot.addEventListener('ended',()=>{
  if(!data)return;
  pause();seekFrame(data.frames-1);
});

function range(series,includeZero=false) {
  let lo=includeZero?0:Infinity,hi=includeZero?0:-Infinity;
  for(const values of series)for(const v of values)if(finite(v)){lo=Math.min(lo,v);hi=Math.max(hi,v);}
  if(!finite(lo)||!finite(hi))return {lo:0,hi:1};
  if(hi-lo<1e-8){lo-=.5;hi+=.5;}
  const margin=(hi-lo)*.1;
  return {lo:includeZero&&lo===0?0:lo-margin,hi:hi+margin};
}
function plotSeries(name) {
  if(name==='force') {
    const pred=data.prediction;
    if($('wrenchSignal').value==='torque') {
      const magnitude=wrench=>wrench?.slice(3).every(finite)?Math.hypot(...wrench.slice(3)):null;
      return {values:[data.arrow_context.reference_wrench_pelvis_N_Nm.map((v,i)=>
                       data.force_labels_available&&data.force_valid[i]?magnitude(v):null),
                     pred.wrench_pelvis_N_Nm.map((v,i)=>pred.valid[i]?magnitude(v):pred.origin==='model'?0:null)],
              colors:['#ff9559','#33e0f5'],dashes:[[5,3],[5,3]],zero:true};
    }
    return {values:[data.force_magnitude_N.map((v,i)=>data.force_valid[i]?v:null),
                    pred.magnitude_N.map((v,i)=>pred.valid[i]?v:pred.origin==='model'?0:null)],
            colors:['#ff9559','#33e0f5'],dashes:[[],[]],zero:true};
  }
  if(name==='joint') {
    const rows=data[$('joint').value];
    return {values:data.joint_names.map((_,ch)=>rows.map(row=>row[ch])),colors,zero:false};
  }
  const start=$('imuMode').value==='gyro'?4:7;
  return {values:[0,1,2].map(ch=>data.imu.map(row=>row[start+ch])),colors,zero:false};
}
function drawPlot(plot) {
  if(!data)return;
  const {canvas,context:ctx,name}=plot;
  if(name==='force'&&inferenceBusy)return;
  const width=canvas.clientWidth,height=canvas.clientHeight,dpr=window.devicePixelRatio||1;
  if(width<1||height<1)return;
  const pixelWidth=Math.round(width*dpr),pixelHeight=Math.round(height*dpr);
  if(canvas.width!==pixelWidth||canvas.height!==pixelHeight){canvas.width=pixelWidth;canvas.height=pixelHeight;}
  ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,height);
  const spec=plotSeries(name),bounds=range(spec.values,spec.zero);
  const left=43,right=15,top=8,bottom=23,w=width-left-right,h=Math.max(1,height-top-bottom);
  const t0=data.event_time_s[0],t1=data.event_time_s[data.frames-1];
  const x=time=>left+(time-t0)/(t1-t0)*w;
  const y=value=>top+(bounds.hi-value)/(bounds.hi-bounds.lo)*h;
  ctx.lineWidth=1;ctx.strokeStyle='#293241';ctx.fillStyle='#72849b';ctx.font='9px ui-monospace, monospace';
  const digits=bounds.hi-bounds.lo>15?0:bounds.hi-bounds.lo>2?1:2;
  ctx.textAlign='right';
  for(let k=0;k<3;k++) {
    const yy=top+h*k/2,value=bounds.hi-(bounds.hi-bounds.lo)*k/2;
    ctx.beginPath();ctx.moveTo(left,yy+.5);ctx.lineTo(width-right,yy+.5);ctx.stroke();
    ctx.fillText(value.toFixed(digits),left-7,yy+3);
  }
  ctx.save();ctx.beginPath();ctx.rect(left-1,top-2,w+2,h+4);ctx.clip();
  for(let ch=0;ch<spec.values.length;ch++) {
    const values=spec.values[ch];
    ctx.setLineDash(spec.dashes?.[ch]||[]);
    for(const [end,alpha] of [[data.frames-1,.24],[frame,spec.values.length>3?.7:1]]) {
      ctx.beginPath();let continuing=false;
      for(let i=0;i<=end;i++) {
        if(!finite(values[i])){continuing=false;continue;}
        const xx=x(data.event_time_s[i]),yy=y(values[i]);
        if(continuing)ctx.lineTo(xx,yy);else ctx.moveTo(xx,yy);
        continuing=true;
      }
      ctx.globalAlpha=alpha;ctx.strokeStyle=spec.colors[ch%spec.colors.length];
      ctx.lineWidth=name==='force'?1.8:1.1;ctx.stroke();
    }
  }
  ctx.globalAlpha=1;ctx.lineWidth=1;ctx.setLineDash([]);
  ctx.strokeStyle='#d9e8f7';const currentX=x(data.event_time_s[frame]);
  ctx.beginPath();ctx.moveTo(currentX,top);ctx.lineTo(currentX,top+h);ctx.stroke();
  if(name==='force')spec.values.forEach((values,ch)=>{
    if(!hasPlotForce(values[frame]))return;
    ctx.fillStyle=spec.colors[ch%spec.colors.length];ctx.beginPath();ctx.arc(currentX,y(values[frame]),2.7,0,Math.PI*2);ctx.fill();
  });
  ctx.restore();ctx.fillStyle='#7a8ca3';ctx.font='9px ui-monospace, monospace';
  if(name==='force'){
    const pred=data.prediction;
    const links=[data.link,pred.valid[frame]&&pred.link_idx[frame]>=0?pred.target_links[pred.link_idx[frame]]:null];
    const labels=links.flatMap((link,ch)=>link&&link!=='unknown'&&hasPlotForce(spec.values[ch][frame])?
      [{link,color:spec.colors[ch],dotY:y(spec.values[ch][frame])}]:[]).sort((a,b)=>a.dotY-b.dotY);
    labels.forEach((label,i)=>{
      label.labelY=Math.max(top+8,Math.min(top+h-8,label.dotY));
      if(i)label.labelY=Math.max(label.labelY,labels[i-1].labelY+16);
    });
    const overflow=labels.length?Math.max(0,labels.at(-1).labelY-(top+h-8)):0;
    for(const label of labels){
      const labelY=label.labelY-overflow,boxWidth=Math.min(w-6,ctx.measureText(label.link).width+8);
      const labelX=Math.max(left+3,Math.min(width-right-boxWidth,currentX+boxWidth+12<width-right?currentX+8:currentX-boxWidth-8));
      ctx.strokeStyle=label.color;ctx.lineWidth=1;ctx.setLineDash([]);
      ctx.beginPath();ctx.moveTo(currentX,label.dotY);ctx.lineTo(labelX>currentX?labelX:labelX+boxWidth,labelY);ctx.stroke();
      ctx.fillStyle='#151a23ee';ctx.fillRect(labelX,labelY-7,boxWidth,14);
      ctx.textAlign='left';ctx.fillStyle=label.color;ctx.fillText(label.link,labelX+4,labelY+3,boxWidth-8);
    }
    ctx.fillStyle='#7a8ca3';
  }
  for(const [fraction,align] of [[0,'left'],[.5,'center'],[1,'right']]){
    const elapsed=fraction*(t1-t0);
    ctx.textAlign=align;ctx.fillText(elapsed.toFixed(2)+'s',x(t0+elapsed),height-6);
  }
}
function showWalkLocalization(pred) {
  return data.scenario==='walk'&&pred.origin==='model'&&data.unified_model&&Array.isArray(pred.localization_link_idx);
}
function predictedForces(pred,index) {
  if(!pred.valid[index])return [];
  return pred.force_field_pelvis_N[index].flatMap((force,link)=>
    pred.link_contact_active[index][link]?[{link,force}]:[]);
}
function hasReferenceWrench() {
  return data?.scenario==='stand'&&data.force_labels_available&&
    data.arrow_context.reference_link_origin_world_m[frame].every(finite);
}
function cross(a,b) {
  return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
}
function drawTorque(ctx,project,center,torque,color,scale) {
  const magnitude=Math.hypot(...torque);
  if(!$('showTorque').checked||!Number.isFinite(magnitude)||magnitude<1e-6)return;
  const axis=torque.map(v=>v/magnitude);
  const basis=Math.abs(axis[2])<.9?[0,0,1]:[0,1,0];
  let u=cross(basis,axis);
  const norm=Math.hypot(...u);u=u.map(v=>v/norm);
  const v=cross(axis,u),radius=Math.min(.22,.055+.045*Math.sqrt(magnitude));
  const points=Array.from({length:41},(_,i)=>{
    const angle=i/40*Math.PI*1.6;
    return project(center.map((c,j)=>c+radius*(u[j]*Math.cos(angle)+v[j]*Math.sin(angle))));
  });
  if(points.some(p=>!p))return;
  ctx.save();ctx.globalAlpha=1;ctx.strokeStyle=ctx.fillStyle=color;
  ctx.lineWidth=Math.max(1.5,2.5*scale);ctx.lineCap='round';
  ctx.setLineDash([5*scale,3*scale]);
  ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.stroke();
  const tip=points.at(-1),previous=points.at(-3),dx=tip[0]-previous[0],dy=tip[1]-previous[1];
  const length=Math.hypot(dx,dy);
  if(length>1e-6){
    const ux=dx/length,uy=dy/length,head=Math.max(5,9*scale);
    ctx.beginPath();ctx.moveTo(...tip);
    ctx.lineTo(tip[0]-ux*head-uy*head*.45,tip[1]-uy*head+ux*head*.45);
    ctx.lineTo(tip[0]-ux*head+uy*head*.45,tip[1]-uy*head-ux*head*.45);
    ctx.closePath();ctx.fill();
  }
  ctx.restore();
}
function replayArrows(payload,index) {
  const geometry=payload.arrow_context,pred=payload.prediction;
  const realWrench=geometry.reference_wrench_pelvis_N_Nm[index];
  const arrows=[{point:geometry.reference_link_origin_world_m[index],
    force:payload.force_pelvis_N[index],torque:realWrench.slice(3),
    valid:payload.force_labels_available&&payload.force_valid[index],color:'#ff9559'}];
  for(const contact of predictedForces(pred,index))
    arrows.push({point:geometry.link_origin_world_m[index][contact.link],
      force:contact.force,
      torque:pred.wrench_field_pelvis_N_Nm?.[index]?.[contact.link]?.slice(3),
      valid:true,color:'#33e0f5'});
  return arrows;
}
function drawRobotViews() {
  const geometry=data?.arrow_context,pred=data?.prediction;
  const dot=(a,b)=>a.reduce((sum,v,i)=>sum+v*b[i],0);
  for(const [view,id] of ['robotFront','robotSide'].entries()) {
    const canvas=$(id),ctx=canvas.getContext('2d');
    const width=canvas.clientWidth,height=canvas.clientHeight,dpr=window.devicePixelRatio||1;
    if(width<1||height<1)continue;
    if(canvas.width!==Math.round(width*dpr)||canvas.height!==Math.round(height*dpr)){
      canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);
    }
    ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,height);
    if(!ready||!geometry||robot.readyState<2)continue;
    const viewWidth=geometry.width/2,viewHeight=geometry.height;
    const scale=Math.min(width/viewWidth,height/viewHeight);
    const ox=(width-viewWidth*scale)/2,oy=(height-viewHeight*scale)/2;
    ctx.drawImage(robot,view*robot.videoWidth/2,0,robot.videoWidth/2,robot.videoHeight,
                  ox,oy,viewWidth*scale,viewHeight*scale);
    const R=geometry.rotation_world_pelvis[frame],world=force=>R.map(row=>dot(row,force));
    const arrows=replayArrows(data,frame).map(arrow=>({...arrow,
      force:world(arrow.force),torque:arrow.torque?world(arrow.torque):null}));
    const link=pred.link_idx[frame];
    const localization=showWalkLocalization(pred)&&pred.valid[frame]&&link>=0?pred.localization_link_idx[frame]:-1;
    const camera=geometry.cameras[view];
    const project=point=>{
      const delta=point.map((v,i)=>v-camera.pos[i]),depth=dot(delta,camera.forward);
      if(depth<=.02)return null;
      return [ox+scale*(viewWidth/2+camera.focal_px*dot(delta,camera.right)/depth),
              oy+scale*(viewHeight/2-camera.focal_px*dot(delta,camera.up)/depth)];
    };
    ctx.save();ctx.beginPath();ctx.rect(ox,oy,viewWidth*scale,viewHeight*scale);ctx.clip();
    if(localization>=0){
      const point=project(geometry.link_origin_world_m[frame][localization]);
      if(point){
        ctx.strokeStyle='#33e0f5';ctx.lineWidth=Math.max(1.5,2*scale);
        ctx.beginPath();ctx.arc(point[0],point[1],Math.max(5,8*scale),0,2*Math.PI);ctx.stroke();
      }
    }
    for(const arrow of arrows){
      if(arrow.valid&&arrow.torque)drawTorque(ctx,project,arrow.point,arrow.torque,arrow.color,scale);
      const magnitude=Math.hypot(...arrow.force);
      if(!arrow.valid||!hasForce(magnitude))continue;
      const tail=project(arrow.point);if(!tail)continue;
      ctx.fillStyle=arrow.color;
      ctx.beginPath();ctx.arc(tail[0],tail[1],Math.max(2,3*scale),0,2*Math.PI);ctx.fill();
      const length=Math.min(1.2,magnitude*.015);
      const tip=project(arrow.point.map((v,i)=>v+arrow.force[i]*length/magnitude));if(!tip)continue;
      const dx=tip[0]-tail[0],dy=tip[1]-tail[1],pixels=Math.hypot(dx,dy);if(pixels<1)continue;
      const head=Math.min(pixels*.5,Math.max(5,11*scale)),ux=dx/pixels,uy=dy/pixels;
      ctx.strokeStyle=arrow.color;ctx.lineWidth=Math.max(1.5,3*scale);ctx.lineCap='round';
      ctx.setLineDash([]);
      ctx.beginPath();ctx.moveTo(...tail);ctx.lineTo(tip[0]-ux*head*.7,tip[1]-uy*head*.7);ctx.stroke();
      ctx.beginPath();ctx.moveTo(...tip);ctx.lineTo(tip[0]-ux*head-uy*head*.4,tip[1]-uy*head+ux*head*.4);
      ctx.lineTo(tip[0]-ux*head+uy*head*.4,tip[1]-uy*head-ux*head*.4);ctx.closePath();ctx.fill();
    }
    ctx.restore();
  }
}
robot.addEventListener('seeked',drawRobotViews);
function draw(){plots.forEach(drawPlot);drawRobotViews();}
function updateWrenchReadout() {
  const pred=data.prediction,valid=pred.valid[frame];
  const reference=hasReferenceWrench();
  $('wrenchControls').hidden=false;
  $('wrenchReadout').hidden=false;
  $('realForceRow').hidden=!reference;
  $('realTorqueRow').hidden=!reference;
  const vectorText=(v,digits,unit)=>`[${v.map(value=>fmt(value,digits)).join(', ')}] ${unit}`;
  const real=reference&&data.force_valid[frame]?data.arrow_context.reference_wrench_pelvis_N_Nm[frame]:null;
  const estimate=valid&&pred.wrench_pelvis_N_Nm?pred.wrench_pelvis_N_Nm[frame]:
    pred.origin==='model'?[0,0,0,0,0,0]:null;
  for(const [prefix,wrench] of [['real',real],['estimated',estimate]]){
    $(prefix+'ForceValue').textContent=wrench?vectorText(wrench.slice(0,3),1,'N'):'—';
    $(prefix+'TorqueValue').textContent=wrench?vectorText(wrench.slice(3),2,'N·m'):'—';
  }
  $('wrenchReadout').title=reference?
    `Real origin: ${data.link} body origin. Estimate origins: each predicted link body origin. XYZ components in pelvis coordinates; force and torque share the same target time.`:
    'XYZ components in pelvis coordinates. Wrench origin: each predicted link body origin; force and torque share the same target time.';
}
function update() {
  if(!data)return;
  $('seek').value=frame;
  $('clock').replaceChildren(document.createTextNode(signed(data.event_time_s[frame],3)+' s'));
  const small=document.createElement('small');small.textContent=`${frame} / ${data.frames-1}`;$('clock').append(small);
  const pred=data.prediction,valid=pred.valid[frame];
  const torquePlot=$('wrenchSignal').value==='torque',unit=torquePlot?'N·m':'N',digits=torquePlot?2:1;
  $('wrenchWaiting').hidden=!inferenceBusy;
  $('forcePlot').style.visibility=inferenceBusy?'hidden':'visible';
  $('forcePlot').setAttribute('aria-busy',String(inferenceBusy));
  $('wrenchPlotUnit').textContent=unit;
  $('forcePlot').setAttribute('aria-label',`${torquePlot?'Torque':'Force'} magnitude (${unit}); click to seek`);
  const currentValues=plotSeries('force').values;
  const refText=finite(currentValues[0][frame])?`${fmt(currentValues[0][frame],digits)} ${unit}`:'Missing';
  const predText=finite(currentValues[1][frame])?`${fmt(currentValues[1][frame],digits)} ${unit}`:inferenceBusy?'Waiting':'Unavailable';
  $('forceNow').textContent=(data.force_labels_available?`Real ${refText} · `:'')+`Estimate ${predText}`;
  $('forceLegend').replaceChildren();
  $('robotRealKey').hidden=!data.force_labels_available;
  const legend=data.force_labels_available?[['reference-key','● Real'],['prediction-key','● Estimate']]:[['prediction-key','● Estimate']];
  legend.forEach(([style,label])=>{const span=document.createElement('span');span.className=style;span.textContent=label;$('forceLegend').append(span);});
  updateWrenchReadout();
  const selected=$('joint').value,signal=jointSignals[selected],row=data[selected][frame];
  $('jointTitle').textContent=signal.title;$('jointUnit').textContent=signal.unit;
  $('jointHint').textContent=selected==='tau'?'Recorded motor torque · Click the plot to seek':`All ${row.length} joints · Click the plot to seek`;
  $('jointPlot').setAttribute('aria-label',`${signal.title}, all ${row.length} joints; click to seek`);
  const gyro=$('imuMode').value==='gyro',values=data.imu[frame].slice(gyro?4:7,gyro?7:10);
  $('imuNow').textContent=values.map(v=>signed(v)).join(' / ');
  $('imuUnit').textContent=gyro?'rad/s':'m/s²';
  $('imuHint').textContent=gyro?'Recorded IMU · xyz angular velocity':'Recorded IMU · xyz acceleration, including gravity';
  draw();
}
$('wrenchSignal').addEventListener('change',update);

function showLoading(text,hint,error=false) {
  $('loading').hidden=false;$('loading').classList.toggle('error',error);
  $('loadingText').textContent=text;$('loadingHint').textContent=hint||'';$('retry').hidden=!error;
}
function videoTimingLabel(video) {
  return ({frame_timestamp:'Frame timestamps',log_anchored:'Log anchored',estimated_dropped_frames:'Estimated timing',estimated_affine:'Log estimated',estimated_motion_aligned:'Motion aligned'})[video.status]||video.status;
}
async function getJSON(url,signal,method='GET') {
  const endpoint=new URL(url,location.origin);
  if(scenario)endpoint.searchParams.set('scenario',scenario);
  const response=await fetch(endpoint,{signal,method});
  const value=await response.json();
  if(!response.ok)throw new Error(value.error||`Request failed ${response.status}`);
  return value;
}
function cancelInference() {
  inferenceGeneration++;inferenceController?.abort();clearTimeout(inferenceTimer);inferenceBusy=false;
}
async function startInference() {
  if(!data)return;
  cancelInference();
  const current=generation,request=++inferenceGeneration;
  inferenceController=new AbortController();inferenceBusy=true;
  $('inferenceError').hidden=true;
  const signal=inferenceController.signal;
  data.prediction={...predictionTemplate,origin:'pending',valid:Array(data.frames).fill(false),event_valid:Array(data.frames).fill(false)};
  update();
  const currentRequest=()=>current===generation&&request===inferenceGeneration;
  const failed=error=>{
    if(!currentRequest()||error.name==='AbortError')return;
    inferenceBusy=false;$('inferenceError').hidden=false;
    $('inferenceError').textContent='Estimation failed. Reload this sample to retry.';
    $('inferenceError').title=englishError(error);update();
  };
  try {
    const job=await getJSON(`/api/inference/${data.number}`,signal,'POST');
    if(!currentRequest())return;
    async function poll() {
      try {
        const status=await getJSON(`/api/inference/${job.id}`,signal);
        if(!currentRequest())return;
        if(status.state==='error'||status.state==='cancelled')throw new Error(status.message||'Estimation cancelled');
        if(status.state==='ready'){
          data.prediction=status.prediction;
          inferenceBusy=false;update();return;
        }
        inferenceTimer=setTimeout(poll,100);
      } catch(error){failed(error);}
    }
    poll();
  } catch(error){failed(error);}
}
function englishError(error) {
  return /\p{Script=Han}/u.test(error.message)?'The request failed. Please reload this sample.':error.message;
}
function fail(error,current) {
  if(current!==generation||error.name==='AbortError')return;
  pause();ready=false;$('play').disabled=true;
  cancelInference();drawRobotViews();
  showLoading('Unable to load replay',englishError(error),true);
}
async function pollRender(current,retry=false) {
  if(current!==generation||!data)return;
  try {
    const status=await getJSON(`/api/render/${data.number}?base=${data.base_mode}${retry?'&retry=1':''}`,controller.signal);
    if(current!==generation)return;
    if(status.key!==data.render_key)throw new Error('Replay data changed. Please reload this sample.');
    if(status.state==='error')throw new Error(status.message);
    if(status.state==='ready') {
      showLoading('Loading replay video','Playback will be available when the video is ready.');
      robot.onloadeddata=()=>{
        if(current!==generation)return;
        ready=true;$('loading').hidden=true;$('play').disabled=false;seekFrame(frame);
      };
      robot.onerror=()=>fail(new Error('Unable to play the MuJoCo video. Please reload this sample.'),current);
      robot.src=status.url;robot.load();return;
    }
    showLoading(status.state==='rendering'?`Rendering MuJoCo replay · ${Math.round(status.progress*100)}%`:'Preparing MuJoCo replay','The first replay is rendered and cached automatically.');
    pollTimer=setTimeout(()=>pollRender(current),700);
  } catch(error){fail(error,current);}
}
async function loadPair(number,{keepFrame=false,retry=false}={}) {
  const desiredFrame=keepFrame?frame:null;
  pause();ready=false;$('play').disabled=true;
  cancelInference();drawRobotViews();
  if(controller)controller.abort();clearTimeout(pollTimer);
  controller=new AbortController();const current=++generation;
  $('pair').value=number;showLoading('Loading sample','Loading robot data, forces and both cameras.');
  try {
    const value=await getJSON(`/api/pair/${number}?base=fixed`,controller.signal);
    if(current!==generation)return;
    data=value;predictionTemplate=value.prediction;frame=Math.min(desiredFrame??data.peak_frame,data.frames-1);
    history.replaceState(null,'',`/?scenario=${scenario}&pair=${data.number}`);
    $('pairCount').textContent=`${catalog.findIndex(p=>p.number===number)+1} / ${catalog.length}`;
    $('previous').disabled=number===catalog[0].number;$('next').disabled=number===catalog[catalog.length-1].number;
    $('seek').max=data.frames-1;
    $('timeStart').textContent=signed(data.event_time_s[0])+' s';$('timeEnd').textContent=signed(data.event_time_s.at(-1))+' s';
    data.videos.forEach((mapping,index)=>{
      const video=cameras[index],label=$('timing'+(index+1));
      const missing=$('cameraMissing'+(index+1)),available=mapping.available!==false;
      video.hidden=!available;video.parentElement.hidden=!available;missing.hidden=available;
      if(!available){
        video.onloadedmetadata=null;video.onerror=null;video.pause();video.removeAttribute('src');video.load();
        label.textContent='Unavailable';label.title="The recording does not cover this event.";label.classList.remove('estimated');
        missing.textContent=`Camera ${mapping.camera} is unavailable for this sample`;missing.title="The recording does not cover this event.";
        return;
      }
      label.textContent=videoTimingLabel(mapping);label.classList.toggle('estimated',mapping.status.includes('estimated'));
      label.title=finite(mapping.max_frame_error_ms)?`Video frame sampling error ≤ ${mapping.max_frame_error_ms.toFixed(2)} ms; not a measure of hardware synchronization.`:'Local timing error cannot be independently verified.';
      video.onloadedmetadata=()=>{if(current===generation){layoutCameras();setMediaTime(video,cameraTime(index,data.time_s[frame]));}};
      video.onerror=()=>{if(current===generation)label.textContent='Video failed to load';};
      if(!video.currentSrc.endsWith(mapping.url)){video.src=mapping.url;video.load();}
      else setMediaTime(video,cameraTime(index,data.time_s[frame]));
    });
    layoutCameras();
    startInference();update();await pollRender(current,retry);
  } catch(error){fail(error,current);}
}
function changePair(direction) {
  if(!catalog.length)return;
  const index=catalog.findIndex(p=>p.number===Number($('pair').value));
  const next=Math.max(0,Math.min(catalog.length-1,index+direction));
  if(next!==index)loadPair(catalog[next].number);
}
$('play').onclick=play;
$('seek').oninput=()=>{pause();seekFrame(Number($('seek').value));};
$('pair').onchange=()=>loadPair(Number($('pair').value));
$('scenario').onchange=()=>{
  pause();cancelInference();controller?.abort();clearTimeout(pollTimer);
  location.assign(`/?scenario=${encodeURIComponent($('scenario').value)}`);
};
$('previous').onclick=()=>changePair(-1);$('next').onclick=()=>changePair(1);
$('retry').onclick=()=>loadPair(Number($('pair').value),{keepFrame:true,retry:true});
$('showTorque').onchange=drawRobotViews;
for(const id of ['joint','imuMode'])$(id).onchange=update;
plots.forEach(plot=>{
  plot.canvas.onclick=event=>{
    if(!data)return;
    const rect=plot.canvas.getBoundingClientRect(),fraction=Math.max(0,Math.min(1,(event.clientX-rect.left-43)/(rect.width-58)));
    pause();seekFrame(fraction*(data.frames-1));
  };
  plot.canvas.onmousemove=event=>{
    if(!data)return;
    const rect=plot.canvas.getBoundingClientRect(),fraction=Math.max(0,Math.min(1,(event.clientX-rect.left-43)/(rect.width-58)));
    const elapsed=data.event_time_s[Math.round(fraction*(data.frames-1))]-data.event_time_s[0];
    plot.canvas.title=`Click to seek to ${elapsed.toFixed(3)} s`;
  };
});
window.addEventListener('resize',draw);
document.addEventListener('keydown',event=>{
  if(['INPUT','SELECT','TEXTAREA'].includes(event.target.tagName))return;
  if(event.code==='Space'){event.preventDefault();play();}
  else if(event.code==='PageUp'){event.preventDefault();changePair(-1);}
  else if(event.code==='PageDown'){event.preventDefault();changePair(1);}
});
async function init() {
  try {
    const result=await getJSON('/api/catalog');catalog=result.pairs;
    scenario=result.scenario;
    $('scenario').replaceChildren(...result.scenarios.map(name=>new Option(name[0].toUpperCase()+name.slice(1),name)));
    $('scenario').value=scenario;
    document.title=result.model_display_name || 'Sixthsense';
    $('pair').replaceChildren(...catalog.map(pair=>{
      const link=typeof pair.contact_link==='string'?pair.contact_link.trim():'';
      const known=link&&!['unknown','none','null','n/a','—','-'].includes(link.toLowerCase());
      return new Option(`${pair.number} · ${result.scenario}${known?' · '+link:''}`,pair.number);
    }));
    const query=new URLSearchParams(location.search),requested=Number(query.get('pair')||1);
    loadPair(catalog.some(pair=>pair.number===requested)?requested:catalog[0].number);
  } catch(error){fail(error,generation);}
}
init();
