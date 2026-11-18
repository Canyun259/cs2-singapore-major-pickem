"""Build stage-opening snapshots from the Aokua/csgopred archive (data/csgopred.db).
Usage: python prepare.py PATH_TO_DB. Requires only the Python standard library.
Rankings and opening pairings are frozen in events.json, sourced per team from HLTV.
"""
import json
import sqlite3
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
db = sqlite3.connect(f"{Path(sys.argv[1]).resolve().as_uri()}?mode=ro", uri=True)
db.row_factory = sqlite3.Row
events = json.loads((ROOT / "events.json").read_text(encoding="utf-8-sig"))
short = dict(anc="Ancient", anb="Anubis", d2="Dust2", inf="Inferno", mrg="Mirage",
             nuke="Nuke", ovp="Overpass", trn="Train", vtg="Vertigo", cch="Cache")
pools = [
    ["Ancient","Anubis","Inferno","Mirage","Nuke","Overpass","Vertigo"],
    ["Ancient","Anubis","Dust2","Inferno","Mirage","Nuke","Vertigo"],
    ["Ancient","Anubis","Dust2","Inferno","Mirage","Nuke","Train"],
    ["Ancient","Dust2","Inferno","Mirage","Nuke","Overpass","Train"],
    ["Ancient","Anubis","Dust2","Inferno","Mirage","Nuke","Overpass"],
]

def games(where, params):
    return db.execute("SELECT * FROM matches WHERE source='hltv' AND " + where +
                      " ORDER BY start_date DESC,match_id DESC", params).fetchall()

def played(row):
    """Read played maps only. Drop series with inconsistent map-to-series orientation."""
    if row["format"] == "bo1":
        m = short.get(json.loads(row["raw"]).get("format"))
        if m and max(row["score_a"],row["score_b"]) >= 13:
            return [(m, row["score_a"] > row["score_b"])]
    maps = []
    for s in db.execute("SELECT * FROM maps WHERE match_id=? ORDER BY map_order", (row["match_id"],)):
        if min(s["score_a"],s["score_b"]) < 0 or max(s["score_a"],s["score_b"]) < 13 or s["score_a"]==s["score_b"]:
            continue
        maps.append((s["map_name"].replace("Dust II","Dust2"), s["score_a"] > s["score_b"]))
    if row["format"] == "bo1":
        return maps if len(maps)==1 and maps[0][1]==(row["winner"]==row["team_a"]) else []
    if sum(win for _,win in maps)!=row["score_a"] or len(maps)!=row["score_a"]+row["score_b"]:
        return []
    return maps

corrections = json.loads((ROOT/"map-corrections.json").read_text(encoding="utf-8-sig"))
snapshots = []
for index, event in enumerate(events):
    stage = games("event=?", (event["name"],))
    start = min(row["start_date"] for row in stage)
    since = (date.fromisoformat(start)-timedelta(days=90)).isoformat()
    pool = pools[index]
    pairs = event["opening"]
    if index==3:
        order = ["Legacy","FaZe Clan","B8","GamerLegion","fnatic","PARIVISION","Ninjas in Pyjamas","Imperial"]
        pairs = sorted(pairs,key=lambda p: order.index(p["teams"][0]))
    names = [p["teams"][side] for side in (0,1) for p in pairs]
    teams = {}
    for p in pairs:
        for name, (rank,vrs) in zip(p["teams"],p["ranks"]):
            prior = games("start_date>=? AND start_date<? AND (team_a=? OR team_b=?) AND score_a!=score_b",
                          (since,start,name,name))
            recent, appearances, wins = [], Counter(), Counter()
            for row in prior:
                is_a = row["team_a"] == name
                opp = row["team_b"] if is_a else row["team_a"]
                recent.append([opp,int(row["winner"]==name)])
                for m, a_won in played(row):
                    appearances[m]+=1
                    wins[m]+=int(is_a==a_won)
            ns = [appearances[m] for m in pool]
            winrates = [[100*wins[m]/max(1,n),n] for m,n in zip(pool,ns)]
            # Archived veto opportunity counts are absent. Approximate preferences from
            # pre-stage map appearances. Smoothing keeps unseen maps possible.
            picks = [[100*(n+1)/(sum(ns)+7),sum(ns)] for n in ns]
            inverse = [1/(n+1) for n in ns]
            bans = [[100*x/sum(inverse),sum(ns)] for x in inverse]
            teams[name] = dict(rank=rank,vrs=vrs,recent=recent[:20],maps=winrates+picks+bans)
    results = [dict(id=r["match_id"],a=r["team_a"],b=r["team_b"],win=int(r["winner"]==r["team_a"]),
                    bo=r["format"],maps=played(r)) for r in reversed(stage)]
    for r in results:
        if r["id"] in corrections:
            r["maps"] = corrections[r["id"]]["maps"]
        assert r["maps"]
    assert len(names)==len(set(names))==16 and len(results)==33
    assert all(m in pool for r in results for m,w in r["maps"])
    snapshots.append(dict(name=event["name"],start=start,names=names,pool=pool,teams=teams,results=results))
(ROOT/"history.json").write_text(json.dumps(snapshots,ensure_ascii=False,indent=1),encoding="utf-8")
for s in snapshots:
    print(s["name"], len(s["results"]), sum(len(r["maps"]) for r in s["results"]))

