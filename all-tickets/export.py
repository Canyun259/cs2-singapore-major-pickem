"""Run in the all-tickets directory after scoring. Python 3 + NumPy."""
from pathlib import Path
from itertools import combinations
import base64, csv, io, json, math, random, zipfile, zlib
import numpy as np

root=Path(__file__).resolve().parent
names=json.loads((root/'teams.json').read_text(encoding='utf-8-sig'))
display=[{'PARIVISION':'PVISION','Ninjas in Pyjamas':'NIP','BC.Game':'GOSH'}.get(n,n) for n in names]
n=500000
pairs=sorted((1<<i)|(1<<j) for i,j in combinations(range(16),2))
mids=sorted(sum(1<<i for i in group) for group in combinations(range(16),6))
def members(m): return [i for i in range(16) if m>>i&1]
def mask(group): return sum(1<<names.index(nm) for nm in group)
def encode(ticket):
    u,m,d=ticket
    return (pairs.index(u)*3003+[v for v in mids if not v&u].index(m))*28+[v for v in pairs if not v&(u|m)].index(d)
def decode(index):
    ui,local=divmod(int(index),84084)
    mi,di=divmod(local,28)
    u=pairs[ui];m=[v for v in mids if not v&u][mi]
    return u,m,[v for v in pairs if not v&(u|m)][di]
def groupnames(ticket):return [[display[i] for i in members(m)] for m in ticket]
lookup=np.array([i.bit_count() for i in range(65536)],dtype=np.uint8)
counts=np.fromfile(root/'counts.bin',dtype='<u4')
assert len(counts)==math.comb(16,2)*math.comb(14,6)*math.comb(8,2)==10090080
trials=np.fromfile(root/'stage.bin',dtype='<u2').reshape(-1,3)
test=np.fromfile(root/'independent.bin',dtype='<u2').reshape(-1,3)
assert len(trials)==n and len(test)==200000
def direct(ticket,sim):
    hits=sum(lookup[sim[:,c]&g] for c,g in enumerate(ticket))
    return hits>=5
hist=np.bincount(counts.astype('i4'),minlength=n+1)
rank=(len(counts)-hist.cumsum()+1).astype('<u4')
bestIndexes=np.flatnonzero(counts>=np.partition(counts,-20)[-20])
bestIndexes=sorted(bestIndexes,key=lambda i:(-int(counts[i]),int(i)))[:20]
ref=json.loads((root.parent/'hedge-result.json').read_text(encoding='utf-8-sig'))
for r in ref['reports'].values():
    idx=encode(tuple(map(mask,r['picks'])))
    assert counts[idx]==round(r['pass']*n)
for idx in list(bestIndexes)+random.Random(114).sample(range(len(counts)),128):
    t=decode(idx)
    assert encode(t)==idx and direct(t,trials).sum()==counts[idx]
top=[]
for i in bestIndexes:
    t=decode(i)
    top.append(dict(index=int(i),teams=groupnames(t),masks=t,count=int(counts[i]),rank=int(rank[counts[i]]),holdout=int(direct(t,test).sum())))
baseline=tuple(map(mask,ref['reports']['original']['picks']))
hedge=tuple(map(mask,ref['reports']['newOutsiderCoverage']['picks']))
def describe(t):
    i=encode(t);return dict(index=i,teams=groupnames(t),masks=t,count=int(counts[i]),rank=int(rank[counts[i]]),holdout=int(direct(t,test).sum()))
out={'n':n,'holdoutN':len(test),'total':len(counts),'min':int(counts.min()),'max':int(counts.max()),'tiedFirst':int(hist[counts.max()]),'mean':float(counts.mean()/n),'median':float(np.median(counts)/n),'above30':int(hist[150000:].sum()),'top':top,'original':describe(baseline),'hedge':describe(hedge)}
# Count qualifying tickets for one fixed stage. Relabeling teams preserves the
# number of passing tickets for every stage. This checks the full score sum.
fixed=trials[0]
fixedPass=0
packs=[]
header=['并列排名','3-0 队伍1','3-0 队伍2']+[f'普通晋级{i+1}' for i in range(6)]+['0-3 队伍1','0-3 队伍2','过关次数','模拟次数','通过率']
pairArray=np.array(pairs,dtype=np.uint16)
midsArray=np.array(mids,dtype=np.uint16)
pack=lambda raw:base64.b64encode(zlib.compress(raw,6)).decode('ascii')
dest=root/'全部填法-按3比0组合分表.zip'
with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4,allowZip64=True) as z:
    for ui,u in enumerate(pairs):
        legalM=midsArray[(midsArray&u)==0]
        allowed=(pairArray[None,:] & (legalM[:,None]|u))==0
        downs=np.broadcast_to(pairArray,allowed.shape)[allowed]
        middle=np.repeat(legalM,28)
        fixedPass+=int(np.sum((int(lookup[int(fixed[0])&u])+lookup[middle&fixed[1]]+lookup[downs&fixed[2]])>=5))
        rows=counts[ui*84084:(ui+1)*84084]
        packs.append(pack(rows.tobytes()))
        order=np.argsort(-rows.astype(np.int32),kind='stable')
        # Team names contain no CSV metacharacters. Precompute group fields.
        byM={int(m):','.join(display[i] for i in members(int(m))) for m in legalM}
        byD={d:','.join(display[i] for i in members(d)) for d in pairs}
        pref=byD[u]; f=io.StringIO()
        csv.writer(f).writerow(header)
        texts={int(v):f'{int(v)*100/n:.4f}%' for v in np.unique(rows)}
        for j in order:
            v=int(rows[j])
            f.write(f'{rank[v]},{pref},{byM[int(middle[j])]},{byD[int(downs[j])]},{v},{n},{texts[v]}\r\n')
        a,b=[display[i] for i in members(u)]
        z.writestr(f'{ui+1:03d}_{a}+{b}.csv',f.getvalue().encode('utf-8-sig'))
    z.writestr('说明.txt',('数据截止 2026-11-18。16 支队伍，2 个全胜、6 个普通晋级、2 个全败。每队最多填一次，同栏无顺序。\r\n'
      '120 张分表按 3-0 二人组命名。每张表 84,084 种填法，按全局过关次数降序，同次数并列。\r\n'
      '通过率 = 十格至少命中五格的模拟次数 / 500,000。所有填法共用同一批赛果。PVISION 即 PARIVISION，GOSH 即 BC.Game。\r\n').encode('utf-8-sig'))
assert int(counts.sum(dtype=np.int64))==n*fixedPass
out['fixedPassingTickets']=fixedPass
(root/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),'utf-8')
template=(root/'viewer.template.html').read_text(encoding='utf-8-sig')
logo_dir=root/'logos'
logo_sources=json.loads((logo_dir/'sources.json').read_text(encoding='utf-8'))
logos=['data:image/webp;base64,'+base64.b64encode((logo_dir/logo_sources[name]['file']).read_bytes()).decode('ascii') for name in names]

for key,val in [('@@LOGOS@@',logos),('@@NAMES@@',display),('@@TOP@@',top[:10]),('@@HEDGE@@',list(hedge)),('@@RANK@@',pack(rank.tobytes())),('@@DATA@@',packs)]:
    template=template.replace(key,json.dumps(val,ensure_ascii=False,separators=(',',':')))
(root/'任意十格通过率查询.html').write_text(template,encoding='utf-8')
with zipfile.ZipFile(dest) as z:
    assert len(z.namelist())==121 and z.testzip() is None
print(json.dumps({k:v for k,v in out.items() if k not in ('top',)},ensure_ascii=False))
print(json.dumps(top[:5],ensure_ascii=False))

