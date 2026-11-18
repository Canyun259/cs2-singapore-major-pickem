'use strict';
const fs=require('node:fs'), path=require('node:path');
const root=path.resolve(__dirname,'..');
const {teamFeatures,createStageSampler}=require(path.join(root,'engine.cjs'));
const base=require(path.join(root,'singapore-result.json'));
const names=Object.keys(base.marginals), data=require(path.join(root,'data.json')),players=require(path.join(root,'players.json')).Singapore;
const teams=names.map(n=>teamFeatures(n,data[n],players[n]));
const sim=createStageSampler({names},teams,base.weights);
const skip=60000, total=500000, extra=200000;
for(let i=0;i<skip;i++) sim.tournament();
function save(n,file) {
  const buf=Buffer.alloc(n*6);
  for(let i=0;i<n;i++){
    const outcome=sim.tournament(), masks=[0,0,0];
    for(let j=0;j<16;j++) if(outcome[j]) masks[outcome[j]-1]|=1<<j;
    for(let k=0;k<3;k++)buf.writeUInt16LE(masks[k],i*6+k*2);
  }
  fs.writeFileSync(path.join(__dirname,file),buf);
}
save(total,'stage.bin');save(extra,'independent.bin');
fs.writeFileSync(path.join(__dirname,'teams.json'),JSON.stringify(names));
console.log(total,extra);

