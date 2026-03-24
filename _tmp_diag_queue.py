import sqlite3
from datetime import datetime, timezone
p='/home/debian/lokalny_obiektyw/data/app.db'
con=sqlite3.connect(p)
con.row_factory=sqlite3.Row
print('=== JOB COUNTS ===')
for r in con.execute("select status,count(*) c from jobs group by status order by status"):
    print(dict(r))
print('\n=== SCENE COUNTS ===')
for r in con.execute("select processing_status,count(*) c from scenes group by processing_status order by processing_status"):
    print(dict(r))
print('\n=== RUNNING JOBS ===')
for r in con.execute("select id,kind,status,stage,progress_pct,message,tour_id,scene_id,created_at,updated_at from jobs where status='running' order by created_at asc"):
    print(dict(r))
print('\n=== LATEST QUEUED JOBS ===')
for r in con.execute("select id,kind,status,stage,progress_pct,message,tour_id,scene_id,created_at,updated_at from jobs where status='queued' order by created_at desc limit 20"):
    print(dict(r))
print('\n=== LATEST FAILED JOBS ===')
for r in con.execute("select id,kind,status,stage,progress_pct,message,tour_id,scene_id,created_at,updated_at from jobs where status='failed' order by updated_at desc limit 20"):
    print(dict(r))
print('\n=== LATEST QUEUED SCENES ===')
for r in con.execute("select id,tour_id,title,processing_status,processing_error,job_id,created_at,updated_at from scenes where processing_status='queued' order by created_at desc limit 20"):
    print(dict(r))
