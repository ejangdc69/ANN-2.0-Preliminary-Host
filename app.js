import { predictFromModel } from './Assets/predict_exact.js';
const $=s=>document.querySelector(s);
let model,last,lastInput;
let signalPhase=0,signalMode='idle',signalRun=0,signalTimers=[],soundEnabled=false,audioContext=null;
const reducedMotion=window.matchMedia('(prefers-reduced-motion: reduce)');
const SIGNAL_STEPS=['Ready · run an automatic animation or step through the layers','01 / Active one-hot input features','02 / Positive ReLU hidden activations','03 / Contributions converge at output'];
function cancelSignal(){signalRun++;signalTimers.forEach(clearTimeout);signalTimers=[];}
function signalQueue(fn,ms){signalTimers.push(setTimeout(fn,ms));}
function signalStatus(){const label=$('#signal-stage');if(label)label.textContent=SIGNAL_STEPS[signalPhase];$('#signal-counter').textContent=`${signalPhase} / 3`;$('#signal-next').disabled=signalMode!=='manual'||signalPhase>=3;$('#signal-replay').disabled=!lastInput;$('#demo-forward').setAttribute('aria-busy',String(signalMode==='auto'));}
function unlockSound(){if(!soundEnabled)return;try{const Context=window.AudioContext||window.webkitAudioContext;if(!Context)return;if(!audioContext)audioContext=new Context();audioContext.resume().catch(()=>{});}catch(e){}}
function signalChime(){if(!soundEnabled||reducedMotion.matches)return;try{const ctx=audioContext;if(!ctx)return;const start=ctx.currentTime;[660,880].forEach((freq,i)=>{const osc=ctx.createOscillator(),gain=ctx.createGain(),t=start+i*.12;osc.type='sine';osc.frequency.value=freq;gain.gain.setValueAtTime(.0001,t);gain.gain.exponentialRampToValueAtTime(.075,t+.018);gain.gain.exponentialRampToValueAtTime(.0001,t+.22);osc.connect(gain).connect(ctx.destination);osc.start(t);osc.stop(t+.24)});/* Reuse the unlocked context for subsequent runs. */}catch(e){/* Audio is optional. */}}
function advanceSignal(n,run){if(run!==signalRun)return;signalPhase=n;signalStatus();drawNetwork();if(n===3)signalChime();}
function prepareSignal(){cancelSignal();infer();signalPhase=0;signalMode='idle';signalStatus();drawNetwork();}
function startSignal(mode){if(!model)return;unlockSound();prepareSignal();signalMode=mode;signalStatus();if(mode==='auto'){const run=signalRun;if(reducedMotion.matches){advanceSignal(3,run);signalMode='idle';signalStatus();return;}[1,2,3].forEach((stage,i)=>signalQueue(()=>{advanceSignal(stage,run);if(stage===3){signalMode='idle';signalStatus()}},[130,860,1620][i]));}else{advanceSignal(1,signalRun)}}

const banner=$('#load-banner'), loadMessage=$('#load-message');
function setLoadState(state,msg){banner.dataset.state=state;loadMessage.textContent=msg;const status=$('#model-status');if(status)status.textContent=state==='ready'?'MODEL READY':state==='error'?'MODEL ERROR':'LOADING MODEL';}
window.addEventListener('error', e=>{if(!model)setLoadState('error','Startup failed: '+e.message+' — confirm all files and assets/ are uploaded.');});
const names={calculator:'DELAY CALCULATOR',network:'NEURAL ARCHITECTURE',performance:'PERFORMANCE',methodology:'METHODOLOGY'};
for(const btn of document.querySelectorAll('[data-page]'))btn.addEventListener('click',()=>{const name=btn.dataset.page;document.querySelectorAll('.page').forEach(x=>x.classList.toggle('active',x.id===`page-${name}`));document.querySelectorAll('.navitem').forEach(x=>x.classList.toggle('active',x===btn));$('#breadcrumb-name').textContent=names[name];if(name!=='network'&&signalMode==='auto'){cancelSignal();signalMode='idle';signalStatus()}if(name==='network')drawNetwork();if(name==='performance')drawPerformance();window.scrollTo({top:0,behavior:'smooth'});});
function fillSelect(id,options){const node=$('#'+id);node.replaceChildren();for(const name of options){const el=document.createElement('option');el.value=name;el.textContent=name;node.append(el);}}
function syncActivityReference(){const chosen=document.querySelector('#activity')?.value;document.querySelectorAll('[data-activity-code]').forEach(el=>{const selected=el.dataset.activityCode===chosen;el.classList.toggle('active',selected);el.setAttribute('aria-pressed',String(selected));});}
function infer(){
 const input={month:$('#month').value,activity:$('#activity').value,location:$('#location').value};
 last=predictFromModel(model,input);lastInput={...input};
 $('#prediction').textContent=last.days.toFixed(2);
 $('#result-caption').textContent=`${input.month} · Activity ${input.activity} · ${input.location} — unrounded estimate: ${last.days.toFixed(6)} days.`;
 renderForwardPass(input);drawNetwork();
}
function renderForwardPass(input){
 if(!last||!model)return;
 const featureIndices=model.one_hot_features.map((f,i)=>String(input[f.input])===String(f.category)?i:-1).filter(i=>i>=0);
 const rows=last.activations.map((v,j)=>{
   const pre=last.preActivations[j],w=model.W_hidden_output[j],c=last.contributions[j];
   return `<tr><td>H${j+1}</td><td>${pre.toFixed(4)}</td><td>${v.toFixed(4)}</td><td>${w.toFixed(4)}</td><td class="${c>=0?'positive':'negative'}">${c>=0?'+':''}${c.toFixed(4)}</td></tr>`;
 }).join('');
 $('#activation-data').innerHTML=`<div class="forward-summary">Result: <strong>${last.days.toFixed(6)} days</strong> · Output bias: ${last.outputBias.toFixed(6)}</div><div class="tablewrap"><table><thead><tr><th>Neuron</th><th>Pre-ReLU</th><th>ReLU</th><th>Output weight</th><th>Contribution (days)</th></tr></thead><tbody>${rows}</tbody></table></div>`;
 const features=featureIndices.map(i=>model.one_hot_features[i]);
 const eq=$('#equation-inspector');
 if(eq){eq.replaceChildren();
  const summary=document.createElement('p');summary.className='helper';summary.textContent=`Inputs: ${input.month} / Activity ${input.activity} / ${input.location}. Active one-hot features: ${features.map(f=>f.input+': '+f.category).join(' | ')}.`;eq.append(summary);
  const h=document.createElement('div');h.className='formula-step';h.textContent='hⱼ = max(0, bⱼ + W(month,j) + W(activity,j) + W(location,j))';eq.append(h);
  const sample=document.createElement('div');sample.className='formula-step';sample.textContent=`H1 = max(0, ${model.b_hidden[0].toFixed(6)} + ${featureIndices.map(i=>model.W_input_hidden[i][0].toFixed(6)).join(' + ')}) = ${last.activations[0].toFixed(6)}`;eq.append(sample);
  const total=document.createElement('div');total.className='formula-step strong-step';total.textContent=`D̂ = ${last.outputBias.toFixed(6)} + (${last.contributions.map(x=>x.toFixed(6)).join(' + ')}) = ${last.days.toFixed(6)} days`;eq.append(total);
  const notice=document.createElement('p');notice.className='helper';notice.textContent='Full numerical precision is retained internally; displayed coefficients are rounded. This is the exact deployed ANN, not a separate regression fit.';eq.append(notice);
 }
}
function drawPerformance(){const node=$('#performance-chart');if(!node)return;
 const vals=[['V2.3 · validation',1.0482],['V2.2 ANN · historical test',1.1019],['V2.2 month mean · historical test',1.1153],['V2.2 month majority · historical test',1.3621]];
 node.innerHTML=vals.map(([label,val],i)=>`<div class="bar-row"><div class="bar-label"><span>${label}</span><strong>${val.toFixed(4)} days</strong></div><div class="bar-track"><div class="bar-fill ${i===0?'selected':''}" style="width:${(val/1.5*100).toFixed(2)}%"></div></div></div>`).join('');
}
function drawNetwork(){
 const target=$('#network-viz');if(!model||!target)return;
 const width=880,height=340,inputY=i=>11+i*9.35,hiddenY=j=>22+j*26;
 const activeInput=i=>!!lastInput&&String(lastInput[model.one_hot_features[i].input])===String(model.one_hot_features[i].category);
 const maxA=last?Math.max(...last.activations,0.001):1;
 const maxW=Math.max(...model.W_input_hidden.flat().map(Math.abs),.001);
 const maxC=last?Math.max(...last.contributions.map(Math.abs),.001):1;
 const stage=signalPhase;
 let svg=`<svg class="signal-svg phase-${stage}" viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="linkgrad"><stop stop-color="#2287ba"/><stop offset="1" stop-color="#22a7cf"/></linearGradient></defs>`;
 for(let i=0;i<35;i++)for(let j=0;j<12;j++){
   const active=activeInput(i),w=model.W_input_hidden[i][j];
   const opacity=stage>=1&&active?.18+.57*Math.abs(w)/maxW:.045;
   svg+=`<line pathLength="1" class="edge edge-input ${active&&stage>=1?'signal-active':''}" x1="190" y1="${inputY(i)}" x2="448" y2="${hiddenY(j)}" stroke="${w<0?'#c78aff':'#38cbe9'}" stroke-opacity="${opacity.toFixed(3)}" stroke-width="${active&&stage>=1?1.25:.6}"/>`;
 }
 for(let j=0;j<12;j++){
   const active=stage>=2&&last?.activations[j]>0,contrib=last?.contributions[j]||0;
   const opacity=stage>=3&&active?.25+.7*Math.abs(contrib)/maxC:.10;
   svg+=`<line pathLength="1" class="edge edge-output ${active&&stage>=3?'signal-active':''}" x1="452" y1="${hiddenY(j)}" x2="693" y2="170" stroke="${contrib<0?'#c78aff':'#31d7c6'}" stroke-opacity="${opacity.toFixed(3)}" stroke-width="${active&&stage>=3?1.4+2*Math.abs(contrib)/maxC:.7}"/>`;
 }
 for(let i=0;i<35;i++){const active=stage>=1&&activeInput(i);svg+=`<circle class="signal-node ${active?'node-lit':''}" cx="190" cy="${inputY(i)}" r="${active?4.5:2.8}" fill="${active?'#6ee7f5':'#356585'}"/>`;}
 for(let j=0;j<12;j++){const ratio=last?last.activations[j]/maxA:0,active=stage>=2&&ratio>0;svg+=`<circle class="signal-node ${active?'node-lit':''}" cx="450" cy="${hiddenY(j)}" r="${active?6+3*ratio:6}" fill="${active?'#37d5f2':'#204660'}" stroke="#87ecff" stroke-opacity="${active?.9:.25}"/>`;}
 if(stage>=3)svg+=`<circle class="output-halo" cx="694" cy="170" r="22" fill="none" stroke="#5df0bd" stroke-width="2" opacity=".7"/>`;
 svg+=`<circle class="signal-node ${stage>=3?'node-lit':''}" cx="694" cy="170" r="16" fill="${stage>=3?'#238775':'#1b4a51'}" stroke="${stage>=3?'#65ecc6':'#438b82'}" stroke-width="2"/><g font-family="system-ui" font-size="12" fill="#a9c1d5" text-anchor="middle"><text x="190" y="330">35 ENCODED INPUTS</text><text x="450" y="330">12 RELU NEURONS</text><text x="694" y="207">1 OUTPUT</text><text x="694" y="237" fill="#57d9ef">${stage>=3&&last?last.days.toFixed(3)+' days':'Awaiting output'}</text></g></svg>`;
 target.innerHTML=svg;
}
async function init(){try{const res=await fetch('./Assets/ann_v23_exact_coefficients.json');if(!res.ok)throw Error(`Model HTTP ${res.status}`);model=await res.json();for(const name of ['month','activity','location'])if(!Array.isArray(model.categories?.[name]))throw Error('Missing category: '+name);fillSelect('month',['March','April','May','June','July'].filter(x=>model.categories.month.includes(x)));fillSelect('activity',model.categories.activity);fillSelect('location',model.categories.location.slice().sort());$('#month').value='June';$('#activity').value='1';syncActivityReference();$('#activity').addEventListener('change',syncActivityReference);document.querySelectorAll('[data-activity-code]').forEach(el=>el.addEventListener('click',()=>{if($('#activity').disabled)return;$('#activity').value=el.dataset.activityCode;syncActivityReference();}));document.querySelectorAll('.navigate-calculator').forEach(el=>el.addEventListener('click',()=>document.querySelector('[data-page=calculator]').click()));$('#location').value=model.categories.location.find(x=>x.toUpperCase()==='APARRI') || model.categories.location[0];for(const id of ['month','activity','location','calculate-button'])$('#'+id).disabled=false;setLoadState('ready','Model loaded. Select inputs and calculate; other tabs are ready.');$('#prediction-form').addEventListener('submit',e=>{e.preventDefault();try{infer()}catch(err){$('#result-caption').textContent=err.message;}});drawNetwork();drawPerformance();$('#demo-forward').addEventListener('click',()=>startSignal('auto'));
$('#signal-step').addEventListener('click',()=>startSignal('manual'));
$('#signal-next').addEventListener('click',()=>{if(signalMode==='manual'&&signalPhase<3){advanceSignal(signalPhase+1,signalRun);if(signalPhase===3){signalMode='idle';signalStatus()}}});
$('#signal-replay').addEventListener('click',()=>startSignal('auto'));
$('#signal-audio').addEventListener('click',()=>{soundEnabled=!soundEnabled;unlockSound();$('#signal-audio').textContent=soundEnabled?'♪ Sound on':'♪ Sound off';$('#signal-audio').setAttribute('aria-pressed',String(soundEnabled))});
signalStatus();}catch(e){$('#result-caption').textContent='Model loading failed: '+e.message;setLoadState('error','Unable to initialize ANN: '+e.message+'. Confirm app.js and assets/ were uploaded.');$('.live').textContent='MODEL UNAVAILABLE';}}
init();
