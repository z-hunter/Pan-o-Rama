import sqlite3
p='/home/debian/lokalny_obiektyw/data/app.db'
con=sqlite3.connect(p)
con.row_factory=sqlite3.Row
queries=[
("tours", "select id,title,slug,created_at from tours where title like '%Smoke Finalize Guard%'"),
("jobs_payload", "select id,kind,status,tour_id,scene_id,created_at from jobs where payload_json like '%Smoke Finalize Guard%'"),
("events", "select id,event_type,tour_id,created_at from analytics_events where meta_json like '%Smoke Finalize Guard%' limit 20"),
]
for name,q in queries:
    rows=list(con.execute(q).fetchall())
    print(name, len(rows))
    for r in rows:
        print(dict(r))
