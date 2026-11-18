// Feature units are shared by historical fitting and Singapore predictions.
const sum = xs=>xs.reduce((a,b)=>a+b,0);
const dot = (a,b)=>sum(a.map((v,i)=>v*b[i]));
const sigmoid = x=>1/(1+Math.exp(-x));
function teamFeatures(name,t,playerMatches) {
  const form=sum(t.recent.map(([,w],i)=>(w-.5)*.9**i))/10;
  const mapValues=t.maps.slice(0,7).map(([pct,n])=>n<3?0:(pct/100-.5)*n/(n+8));
  const mean=sum(mapValues)/7;
  // Average of the two highest ADRs per prior series. Two neutral series shrink noisy samples.
  const observed=playerMatches.filter(x=>x.adr.length===5);
  const adr=sum(observed.map(x=>{
    const [one,two]=[...x.adr].sort((a,b)=>b-a);
    return ((one+two)/2-85)/10;
  }))/(2+observed.length);
  return {...t,name,form,adr,edges:mapValues.map(x=>x-mean)};
}
function features(a,b,map){
  function record(t,opp){return t.recent.filter(([n])=>n===opp).map(([,w])=>2*w-1);}
  const direct=record(a,b.name),reverse=record(b,a.name);
  const h=direct.length>=reverse.length?sum(direct):-sum(reverse);
  return [Math.log(b.rank/a.rank),Math.log(b.vrs/a.vrs),a.form-b.form,
    h/(4+Math.max(direct.length,reverse.length)),a.edges[map]-b.edges[map],a.adr-b.adr];
}
function createStageSampler(stage,teams,weights){
  const names=stage.names;
  let seed=20261118;
  function random(){
    seed=(Math.imul(seed,1664525)+1013904223)>>>0;
    return seed/4294967296;
  }
  function matchup(a,b,bo3) {
    const A=teams[a], B=teams[b];
    const probs=A.edges.map((_,m)=>sigmoid(dot(weights,features(A,B,m))));
    let total=0;
    for(let run=0;run<500;run++){
      const pool=[0,1,2,3,4,5,6], picks=[];
      const first=random()<.5?0:1;
      function move(side,pick){
        const t=side===0?A:B;
        const weights=pool.map(m=>{
          const [pct,n]=t.maps[(pick?7:14)+m];
          const pref=(pct*n/100+1)/(n+7);
          const advantage=(probs[m]-.5)*(side===0?1:-1);
          return (pref+.025)**1.6*Math.exp((pick?1:-1)*5*advantage);
        });
        let target=random()*sum(weights), k=0;
        while(k<weights.length-1 && (target-=weights[k])>0) k++;
        const [m]=pool.splice(k,1);
        if(pick) picks.push(m);
      }
      if(bo3){
        move(first,false); move(1-first,false); move(first,true);
        move(1-first,true); move(1-first,false); move(first,false);
        const [x,y,z]=[...picks,...pool].map(m=>probs[m]);
        total+=x*y+(x+y-2*x*y)*z;
      }else{
        for(const s of [first,first,1-first,1-first,1-first,first]) move(s,false);
        total+=probs[pool[0]];
      }
    }
    return total/500;
  }
  const odds=[false,true].map(bo3=>{
    const matrix=names.map(()=>[]);
    for(let a=0;a<16;a++) for(let b=a+1;b<16;b++){
      matrix[a][b]=matchup(a,b,bo3); matrix[b][a]=1-matrix[a][b];
    }
    return matrix;
  });
  const priorities='162534 162435 152634 152436 142635 142536 162345 152346 132645 132546 142356 132456 123645 123546 123456'.split(' ');
  function pair(group,met,late=false){
    if(group.length===0) return [];
    if(group.length===6 && late){
      for(const row of priorities){
        const t=[...row].map(i=>group[i-1]);
        if([0,2,4].every(i=>!met[t[i]].includes(t[i+1]))) return t;
      }
    }
    for(let k=group.length-1;k>0;k--){
      if(met[group[0]].includes(group[k])) continue;
      const rest=pair(group.filter((_,i)=>i!==0&&i!==k),met);
      if(rest) return [group[0],group[k],...rest];
    }
    return null;
  }
  function tournament(actual=null,onClosingMatch=null){
    const w=Array(16).fill(0),l=w.slice(),met=names.map(()=>[]);
    for(let round=0;round<5;round++){
      const groups={};
      for(let i=0;i<16;i++) if(w[i]<3&&l[i]<3) (groups[[w[i],l[i]]]??=[]).push(i);
      const bh=met.map(opps=>sum(opps.map(j=>w[j]-l[j])));
      for(const g of Object.values(groups)){
        g.sort((a,b)=>bh[b]-bh[a]||a-b);
        const pairs=round?pair(g,met,round>=3):g.slice(0,8).flatMap(i=>[i,i+8]);
        for(let k=0;k<pairs.length;k+=2){
          const a=pairs[k],b=pairs[k+1],long=w[a]===2||l[a]===2;
          if(onClosingMatch && round===2 && long) onClosingMatch(w[a]===2 ? 1 : 3,a,b);
          let win;
          if(actual){
            const recorded=actual.find(r=>(r.a===names[a]&&r.b===names[b])||(r.b===names[a]&&r.a===names[b]));
            if(!recorded) throw Error('Pairing replay mismatch: '+names[a]+' vs '+names[b]);
            win=recorded.a===names[a]?recorded.win:!recorded.win;
          } else win=random()<odds[+long][a][b];
          w[win?a:b]++; l[win?b:a]++; met[a].push(b); met[b].push(a);
        }
      }
    }
    const result=w.map((n,i)=>n===3?(l[i]===0?1:2):n===0?3:0);
    if([0,1,2,3].some((c)=>result.filter(x=>x===c).length!==[6,2,6,2][c])) throw Error('invalid stage');
    return result;
  }
  return {odds,random,tournament};
}
function runStage(stage,teams,weights,others=[]){
  const names=stage.names;
  const {odds,random,tournament}=createStageSampler(stage,teams,weights);
  if(stage.results) tournament(stage.results);
  const train=Array.from({length:stage.searchSize??12000},()=>tournament());
  const counts=teams.map((_,i)=>[0,1,2,3].map(c=>sum(train.map(r=>+(r[i]===c)))));
  function passes(ticket,r){
    let hits=0;
    for(let i=0;i<16;i++) if(ticket[i] && ticket[i]===r[i]) hits++;
    return hits>=5;
  }
  const score=t=>sum(train.map(r=>+passes(t,r)));
  let best, high=-1;
  for(let start=0;start<8;start++){
    let t=Array(16).fill(0);
    for(const c of [1,3,2]){
      const ranks=names.map((_,i)=>counts[i][c]+random()*1000);
      const free=names.map((_,i)=>i).filter(i=>!t[i]).sort((a,b)=>ranks[b]-ranks[a]);
      for(const i of free.slice(0,c===2?6:2)) t[i]=c;
    }
    let value=score(t);
    while(true){
      let next=t, top=value;
      for(let i=0;i<16;i++) for(let j=i+1;j<16;j++) if(t[i]!==t[j]){
        const u=t.slice(); [u[i],u[j]]=[u[j],u[i]];
        const s=score(u); if(s>top){top=s;next=u;}
      }
      if(top===value) break;
      t=next; value=top;
    }
    if(value>high){best=t;high=value;}
  }
  
  const validation=Array.from({length:stage.validationSize??30000},()=>tournament());
  function hits(ticket,outcomes){return sum(ticket.map((c,i)=>Number(c!==0&&outcomes[i]===c)));}
  function evaluate(ticket){return sum(validation.map(r=>+passes(ticket,r)))/validation.length;}
  const marginals=teams.map((_,i)=>[1,2,3].map(c=>sum(validation.map(r=>+(r[i]===c)))/validation.length));
  const actual=stage.results?tournament(stage.results):null;
  return {ticket:best,pass:evaluate(best),otherPass:others.map(evaluate),marginals,
    comparisons:others.map(t=>({
      newOnly:sum(validation.map(r=>Number(passes(best,r)&&!passes(t,r)))),
      otherOnly:sum(validation.map(r=>Number(!passes(best,r)&&passes(t,r)))),
      n:validation.length
    })),actualHits:actual?hits(best,actual):null,actual,odds};
}
module.exports={teamFeatures,features,createStageSampler,runStage,sum,dot,sigmoid};

