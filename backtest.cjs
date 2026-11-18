const fs=require('node:fs');
const path=require('node:path');
const {teamFeatures,features,runStage,sum,dot,sigmoid}=require('./engine.cjs');
const history=require('./history.json');
const players=require('./players.json');
// No held-out event is used to select these priors or penalty strengths.
// Legacy coefficients for the common features, neutral prior for the new ADR signal.
const prior=[.53,.57,.60,.35,1.70,0];
const priorSD=[.45,.45,1,.85,1,.40];
const labels=['HLTV rank','VRS rank','recent form','H2H','map edge','recent player ADR'];
function makeTeams(s){
  return s.names.map(n=>teamFeatures(n,s.teams[n],players[s.name][n]));
}
function trainingRows(s){
  const teams=makeTeams(s), rows=[];
  for(const game of s.results){
    const a=teams[s.names.indexOf(game.a)], b=teams[s.names.indexOf(game.b)];
    for(const [map,y] of game.maps){
      rows.push({x:features(a,b,s.pool.indexOf(map)),y:+y});
    }
  }
  return rows;
}
function fit(stages){
  const rows=stages.flatMap(trainingRows);
  const weights=prior.slice();
  // Cyclic Newton updates for penalized binomial negative log likelihood.
  for(let iter=0;iter<400;iter++){
    let maxMove=0;
    for(let j=0;j<weights.length;j++){
      let g=(weights[j]-prior[j])/priorSD[j]**2, h=1/priorSD[j]**2;
      for(const {x,y} of rows){
        const p=sigmoid(dot(weights,x));
        g+=x[j]*(p-y); h+=x[j]**2*p*(1-p);
      }
      const move=Math.max(-.5,Math.min(.5,g/h));
      maxMove=Math.max(maxMove,Math.abs(move));
      weights[j]-=move;
    }
    if(maxMove<1e-9) return weights;
  }
  throw Error('Fit did not converge');
}
function predictMatches(s,run){
  return s.results.map(r=>{
    const index=s.names.indexOf(r.a), other=s.names.indexOf(r.b);
    return {id:r.id,bo:r.bo,y:r.win,p:run.odds[+(r.bo==='bo3')][index][other]};
  });
}
function metrics(rows){
  return {n:rows.length,brier:sum(rows.map(({p,y})=>(p-y)**2))/rows.length,
    logLoss:-sum(rows.map(({p,y})=>y?Math.log(p):Math.log(1-p)))/rows.length,
    correct:sum(rows.map(({p,y})=>Number((p>=.5)===Boolean(y))))};
}
function ticket(s,run){
  return [1,2,3].map(c=>s.names.filter((_,i)=>run.ticket[i]===c));
}
// Fit a lower-variance alternative with a single common slope.
function scaling(stages){
  const rows=stages.flatMap(trainingRows).map(({x,y})=>({z:dot(prior,x),y}));
  let k=1;
  for(let i=0;i<30;i++){
    let g=(k-1)*4,h=4;
    for(const {z,y} of rows){
      const p=sigmoid(k*z);
      g+=(p-y)*z; h+=p*(1-p)*z*z;
    }
    k-=g/h;
  }
  return k;
}
const report={labels,prior,priorSD,folds:[]};
for(let i=2;i<history.length;i++){
  const s=history[i],teams=makeTeams(s),fitted=fit(history.slice(0,i));
  const scale=scaling(history.slice(0,i)), fold={stage:s.name,fitted,scale};
  for(const [model,w] of Object.entries({baseline:prior,scaled:prior.map(w=>w*scale),fittedModel:fitted})){
    const run=runStage(s,teams,w);
    const predictions=predictMatches(s,run);
    fold[model]={metrics:metrics(predictions),predictions,ticket:ticket(s,run),
      hits:run.actualHits,modelPass:run.pass};
  }
  report.folds.push(fold);
  console.log(s.name,...['baseline','scaled','fittedModel'].map(k=>[fold[k].metrics,fold[k].hits]));
}
report.aggregate={};
for(const k of ['baseline','scaled','fittedModel']){
  const rows=report.folds.flatMap(f=>f[k].predictions);
  report.aggregate[k]={all:metrics(rows),bo1:metrics(rows.filter(r=>r.bo==='bo1')),bo3:metrics(rows.filter(r=>r.bo==='bo3'))};
}
report.finalWeights=fit(history);
report.finalScale=scaling(history);
report.mapCount=history.flatMap(trainingRows).length;
fs.writeFileSync(path.join(__dirname,'backtest.json'),JSON.stringify(report,null,2));
console.log(JSON.stringify({aggregate:report.aggregate,weights:report.finalWeights,mapCount:report.mapCount}));

