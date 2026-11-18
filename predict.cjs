// Singapore Stage 1, inputs frozen on 18 November 2026. Run with Node.js.
const fs=require('node:fs');
const path=require('node:path');
const report=require('./backtest.json');
const data=require('./data.json');
const playerData=require('./players.json').Singapore;
const {runStage,teamFeatures}=require('./engine.cjs');
const names=['Ninjas in Pyjamas','PARIVISION','HEROIC','FaZe','Liquid','Eternal Fire','100 Thieves','9z','M80','DENDELE','Imperial','Wildcard','BC.Game','FlyQuest','Chinggis Warriors','Lynn Vision'];
const stage={names,name:'Singapore',searchSize:60000,validationSize:100000};
const teams=names.map(n=>teamFeatures(n,data[n],playerData[n]));
const previous=[2,1,2,2,1,2,2,0,2,0,0,0,3,0,3,0];
const weights=report.prior.map(w=>w*report.finalScale);
const selected=runStage(stage,teams,weights,[previous]);
const alternative=runStage(stage,teams,report.finalWeights,[previous,selected.ticket]);
function picks(t){return [1,2,3].map(c=>names.filter((n,i)=>t[i]===c));}
const output={snapshot:'2026-11-18',scale:report.finalScale,weights,
  estimatedPass:selected.pass,previousPassOnSameRuns:selected.otherPass[0],
  headToHead:selected.comparisons[0],picks:picks(selected.ticket),
  marginals:Object.fromEntries(names.map((n,i)=>[n,selected.marginals[i]])),
  sensitivity:{model:'six separately fitted weights',picks:picks(alternative.ticket),
    estimatedPass:alternative.pass,selectedPass:alternative.otherPass[1],previousPass:alternative.otherPass[0]}};
fs.writeFileSync(path.join(__dirname,'singapore-result.json'),JSON.stringify(output,null,2));
console.log(JSON.stringify(output,null,2));

