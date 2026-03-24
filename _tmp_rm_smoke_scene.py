import sqlite3, os, shutil
p='/home/debian/lokalny_obiektyw/data/app.db'
base='/home/debian/lokalny_obiektyw/data/processed_galleries'
raw='/home/debian/lokalny_obiektyw/data/raw_uploads'
con=sqlite3.connect(p)
con.row_factory=sqlite3.Row
rows=list(con.execute("select id,tour_id,title,panorama_path,preview_path from scenes where title like '%Smoke Finalize Guard%'").fetchall())
print('matches', len(rows))
for r in rows:
    print(dict(r))
if rows:
    scene_ids=[r['id'] for r in rows]
    tour_scene=[(r['tour_id'], r['id']) for r in rows]
    con.executemany("delete from hotspots where from_scene_id=? or to_scene_id=?", [(sid,sid) for sid in scene_ids])
    con.executemany("delete from jobs where scene_id=?", [(sid,) for sid in scene_ids])
    con.executemany("delete from scenes where id=?", [(sid,) for sid in scene_ids])
    con.commit()
    for tid,sid in tour_scene:
        for root in (base, raw):
            d=os.path.join(root, tid, sid)
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
print('done')
