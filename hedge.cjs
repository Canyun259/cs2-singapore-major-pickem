// Compare opening-stage Pick'Em hedges on the frozen Singapore data.
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const {teamFeatures, createStageSampler} = require('./engine.cjs');
const base = require('./singapore-result.json');
const teamData = require('./data.json');
const playerData = require('./players.json').Singapore;

// Seed order determines round one (i vs i + 8) and breaks Buchholz ties.
const names = ['Ninjas in Pyjamas','PARIVISION','HEROIC','FaZe','Liquid','Eternal Fire','100 Thieves','9z','M80','DENDELE','Imperial','Wildcard','BC.Game','FlyQuest','Chinggis Warriors','Lynn Vision'];
const indexes = names.map((_, i) => i);
const sizes = {search: 60000, validation: 500000};
const bits = new Uint8Array(1 << 16);
for (let mask=1; mask<bits.length; mask++) bits[mask] = bits[mask >>> 1] + (mask & 1);
function maskOf(group) {return group.reduce((mask,i) => mask | (1<<i), 0);}
function asNames(mask) {return indexes.filter(i => mask & (1<<i)).map(i => names[i]);}
const original = base.picks.map(group => maskOf(group.map(n => names.indexOf(n))));
const teamPairs = indexes.flatMap(i => indexes.slice(i+1).map(j => (1<<i) | (1<<j)));
const teams = names.map(n => teamFeatures(n,teamData[n],playerData[n]));
const simulator = createStageSampler({names},teams,base.weights);

function drawSample(n, sampler=simulator) {
  const results = [new Uint16Array(n), new Uint16Array(n), new Uint16Array(n)];
  const meetingCounts = [new Uint32Array(bits.length), new Uint32Array(bits.length)];
  const partnerAdvanced = [new Uint32Array(bits.length), new Uint32Array(bits.length)];
  const closings = [[],[]];
  function remember(category,a,b) {
    closings[category===1 ? 0 : 1].push((1<<a) | (1<<b));
  }
  const ordinaryCounts = Array(16).fill(0);
  for(let k=0;k<n;k++){
    closings[0].length = 0;
    closings[1].length = 0;
    const final = sampler.tournament(null,remember);
    for(let i=0;i<16;i++){
      if(final[i]) results[final[i]-1][k] |= 1<<i;
      if(final[i]===2) ordinaryCounts[i]++;
    }
    for(let side=0;side<2;side++){
      if(closings[side].length!==2) throw Error('Expected two closing matches per side');
      const special=results[side===0?0:2][k];
      for(const pair of closings[side]){
        if(bits[pair & special]!==1) throw Error('A closing pair must score exactly once');
        meetingCounts[side][pair]++;
        if(pair & results[1][k]) partnerAdvanced[side][pair]++;
      }
    }
  }
  return {n, results, ordinaryCounts, meetingCounts, partnerAdvanced};
}

// The three numbers in a ticket are team bitmasks for 3-0, ordinary, 0-3.
function checkTicket([up,middle,down]){
  if(bits[up]!==2 || bits[middle]!==6 || bits[down]!==2 || (up&middle) || (middle&down) || (up&down)) throw Error('Invalid ticket');
}
function score(sample,ticket) {
  const [u,m,d]=ticket, [us,ms,ds]=sample.results;
  let official=0, capped=0;
  for(let k=0;k<sample.n;k++){
    const a=bits[u&us[k]], b=bits[m&ms[k]], c=bits[d&ds[k]];
    official += Number(a+b+c>=5);
    capped += Number(b+Number(a>0)+Number(c>0)>=5);
  }
  return {official,capped};
}
function pairStats(sample,pair,side) {
  const special=sample.results[side===0?0:2], middle=sample.results[1];
  let any=0,both=0,clean=0,oneAndPartnerOrdinary=0;
  for(let k=0;k<sample.n;k++){
    const hits=bits[special[k]&pair];
    any+=Number(hits>0);
    both+=Number(hits===2);
    clean+=Number(hits>0 && !(middle[k]&pair));
    oneAndPartnerOrdinary+=Number(hits===1 && Boolean(middle[k]&pair));
  }
  const meet=sample.meetingCounts[side][pair];
  return {any:any/sample.n,both:both/sample.n,none:1-any/sample.n,
    clean:clean/sample.n,onePartnerOrdinary:oneAndPartnerOrdinary/(any-both),
    meet:meet/sample.n,partnerOrdinaryIfMeet:meet?sample.partnerAdvanced[side][pair]/meet:null};
}

function selectTickets(search,original) {
  const ranked = indexes.slice().sort((a,b) => search.ordinaryCounts[b]-search.ordinaryCounts[a] || a-b);
  const protectedSix = maskOf(ranked.slice(0,6));
  const priorEight = original[0] | original[1];
  const upStats = new Map(teamPairs.map(p=>[p,pairStats(search,p,0)]));
  const downStats = new Map(teamPairs.map(p=>[p,pairStats(search,p,1)]));
  const weakPairs = teamPairs.filter(p=>!(p & (protectedSix | priorEight)));
  // Choose the 0-3 pair first by at-least-one coverage. Both must be outside the
  // original eight advancement picks and outside the six ordinary priority teams.
  const down = weakPairs.slice().sort((a,b)=>downStats.get(b).any-downStats.get(a).any)[0];
  const mostDownMeeting = weakPairs.slice().sort((a,b)=>downStats.get(b).meet-downStats.get(a).meet)[0];
  
  const winners = {};
  const candidateCounts = {};
  function nominate(label,ticket,primary,secondary){
    candidateCounts[label]=(candidateCounts[label]||0)+1;
    const last=winners[label];
    if(!last || primary>last.primary || (primary===last.primary && secondary>last.secondary))
      winners[label]={ticket,primary,secondary};
  }
  const memo = new Map();
  function getScore(ticket){
    const key=ticket.join('/');
    if(!memo.has(key)){checkTicket(ticket); memo.set(key,score(search,ticket));}
    return memo.get(key);
  }
  // Keep each protected ordinary team unless it is itself the 3-0 anchor.
  // If all six are protected, neither 3-0 selection may use one.
  for(const up of teamPairs){
    if(up&down) continue;
    const anchors=bits[up&protectedSix];
    if(anchors>1) continue;
    const middleBase=protectedSix & ~up;
    const fills=anchors===0 ? [0] : indexes.map(i=>1<<i).filter(bit=>!(bit & (up|middleBase|down)));
    for(const fill of fills){
      const ticket=[up,middleBase|fill,down], r=getScore(ticket), upper=upStats.get(up);
      if(anchors===0){nominate('protectAllSix',ticket,r.capped,r.official);continue;}
      nominate('oneAnchor',ticket,r.capped,r.official);
      // A genuine new long shot: the second choice was absent from all eight
      // previous advancement picks. The other five ordinary priorities are kept.
      if((up & ~protectedSix) & priorEight) continue;
      nominate('newOutsider',ticket,r.capped,r.official);
      nominate('newOutsiderOfficial',ticket,r.official,r.capped);
      nominate('newOutsiderCoverage',ticket,upper.any,r.capped);
      nominate('newOutsiderMeeting',ticket,upper.meet,r.capped);
      nominate('newOutsiderClean',ticket,upper.clean,r.capped);
      if(indexes.slice(0,8).some(i=>up===((1<<i)|(1<<(i+8)))))
        nominate('newOutsiderSameOpener',ticket,upper.any,r.capped);
    }
  }
  // Evaluate the alternative reading: both picks are opponents in round one.
  const openingPairs=indexes.slice(0,8).map(i=>(1<<i)|(1<<(i+8)));
  for(const up of openingPairs){
    if(up&down) continue;
    const eligible=ranked.filter(i=>!((1<<i) & (up|down)));
    const ticket=[up,maskOf(eligible.slice(0,6)),down];
    const r=getScore(ticket);
    nominate('sameOpenerUp',ticket,upStats.get(up).any,r.capped);
  }
  const upMeet=teamPairs.filter(p=>!(p&mostDownMeeting)).sort((a,b)=>upStats.get(b).meet-upStats.get(a).meet)[0];
  const normalForMeeting=ranked.filter(i=>!((1<<i)&(upMeet|mostDownMeeting))).slice(0,6);
  winners.closingPairs={ticket:[upMeet,maskOf(normalForMeeting),mostDownMeeting]};
  return {ranked,protectedSix,priorEight,upStats,downStats,down,mostDownMeeting,winners,candidateCounts};
}
const search=drawSample(sizes.search);
const {protectedSix,upStats,downStats,down,mostDownMeeting,winners,candidateCounts}=selectTickets(search,original);

// Same validation population for every ticket; no search on these results.
const heldout = drawSample(sizes.validation);
function examine(ticket){
  checkTicket(ticket);
  const n=heldout.n, [u,m,d]=ticket, [us,ms,ds]=heldout.results;
  const dist=Array(11).fill(0);
  let middleThree=0,middleFour=0,middleTotal=0,bothSides=0,anySpecial=0;
  let eachPlusThree=0,capped=0,newOnly=0,originalOnly=0,passWithNeither=0;
  for(let k=0;k<n;k++){
    const a=bits[u&us[k]],b=bits[m&ms[k]],c=bits[d&ds[k]],hits=a+b+c;
    dist[hits]++;middleThree+=Number(b>=3);middleFour+=Number(b>=4);middleTotal+=b;
    bothSides+=Number(a>0&&c>0);anySpecial+=Number(a+c>0);
    eachPlusThree+=Number(a>0&&c>0&&b>=3);
    capped+=Number(Number(a>0)+b+Number(c>0)>=5);
    if(a+c===0 && hits>=5)passWithNeither++;
    const before=bits[original[0]&us[k]]+bits[original[1]&ms[k]]+bits[original[2]&ds[k]]>=5;
    newOnly+=Number(hits>=5 && !before);originalOnly+=Number(hits<5 && before);
  }
  return {picks:ticket.map(asNames), pass:dist.slice(5).reduce((a,b)=>a+b,0)/n,
    cappedPass:capped/n, eachSpecial:bothSides/n, anySpecial:anySpecial/n,
    middleMean:middleTotal/n,middleThree:middleThree/n,middleFour:middleFour/n,
    eachPlusThree:eachPlusThree/n,passWithNeither:passWithNeither/n,
    up:pairStats(heldout,u,0),down:pairStats(heldout,d,1),newOnly,originalOnly,distribution:dist};
}
const chosen={original, ...Object.fromEntries(Object.entries(winners).map(([label,v])=>[label,v.ticket]))};
const reports={};
const evaluated=new Map();
for(const [label,ticket] of Object.entries(chosen)){
  if(!evaluated.has(ticket.join('/'))) evaluated.set(ticket.join('/'),examine(ticket));
  reports[label]=evaluated.get(ticket.join('/'));
}
const topUpMeeting=teamPairs.slice().sort((a,b)=>upStats.get(b).meet-upStats.get(a).meet)[0];
const output={snapshot:base.snapshot,search:sizes.search,validation:sizes.validation,
  protectedSix:asNames(protectedSix),candidateCounts, pairCoverageDown:asNames(down),
  unrestrictedCoverage: [upStats,downStats].map((stats,side)=>{
    const pair=teamPairs.slice().sort((a,b)=>stats.get(b).any-stats.get(a).any)[0];
    return {names:asNames(pair),...pairStats(heldout,pair,side)};
  }),
  mostDownMeeting:{names:asNames(mostDownMeeting),...pairStats(heldout,mostDownMeeting,1)},
  mostUpMeeting:{names:asNames(topUpMeeting),...pairStats(heldout,topUpMeeting,0)},reports};
// Replay the same ticket-selection rules on the prior, temporally held-out stages.
// The frozen baseline ticket and fold-specific scale were made without that stage's results.
const archive=require('./history.json');
const folds=require('./backtest.json');
const pastPlayers=require('./players.json');
output.history=folds.folds.map(fold=>{
  const stage=archive.find(s=>s.name===fold.stage);
  const oldTicket=fold.scaled.ticket.map(group=>maskOf(group.map(n=>stage.names.indexOf(n))));
  checkTicket(oldTicket);
  const historicTeams=stage.names.map(n=>teamFeatures(n,stage.teams[n],pastPlayers[stage.name][n]));
  const historicSampler=createStageSampler(stage,historicTeams,folds.prior.map(w=>w*fold.scale));
  const pastSearch=drawSample(sizes.search,historicSampler);
  const choices=selectTickets(pastSearch,oldTicket);
  const actual=historicSampler.tournament(stage.results);
  const realMasks=[1,2,3].map(c=>maskOf(indexes.filter(i=>actual[i]===c)));
  const grading=(ticket)=>{
    const points=ticket.map((p,i)=>bits[p&realMasks[i]]);
    return {picks:ticket.map(p=>indexes.filter(i=>p&(1<<i)).map(i=>stage.names[i])),
      points,hits:points.reduce((a,b)=>a+b,0)};
  };
  const baseline=grading(oldTicket);
  if(baseline.hits!==fold.scaled.hits) throw Error('Historical baseline replay mismatch '+stage.name);
  const tested={baseline};
  for(const label of ['newOutsiderSameOpener','newOutsiderCoverage','newOutsiderClean','closingPairs']){
    const choice=choices.winners[label];
    // A missing valid opening pair falls back to the out-of-eight coverage pair.
    const selected=choice || choices.winners.newOutsiderCoverage;
    tested[label]={...grading(selected.ticket),fallback:!choice};
  }
  return {stage:stage.name,scale:fold.scale,tested};
});
fs.writeFileSync(path.join(__dirname,'hedge-result.json'),JSON.stringify(output,null,2)+'\n');
for(const [label,r] of Object.entries(reports)) console.log(label,JSON.stringify({
  up:r.picks[0],middle:r.picks[1],down:r.picks[2],pass:r.pass,capped:r.cappedPass,
  upOne:r.up.any,downOne:r.down.any,either:r.anySpecial,each:r.eachSpecial,
  upperMeet:r.up.meet,ordinary:r.middleMean,ordinaryThree:r.middleThree,
  upperClean:r.up.clean,otherOrdinary:r.up.onePartnerOrdinary}));
console.log('most likely closings',JSON.stringify([output.mostUpMeeting,output.mostDownMeeting]));

console.log('Historical points',JSON.stringify(output.history.map(row=>[row.stage,...['baseline','newOutsiderSameOpener','newOutsiderCoverage','newOutsiderClean','closingPairs'].map(k=>[row.tested[k].hits,row.tested[k].fallback||false])])));

