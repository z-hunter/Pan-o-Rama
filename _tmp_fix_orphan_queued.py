import sqlite3
p='/home/debian/lokalny_obiektyw/data/app.db'
con=sqlite3.connect(p)
cur=con.cursor()
cur.execute("""
UPDATE scenes
SET processing_status='failed',
    processing_error='Orphan queued scene (job missing). Please re-upload scene.',
    updated_at=datetime('now')
WHERE processing_status='queued' AND (job_id IS NULL OR job_id='')
""")
print('updated_scenes', cur.rowcount)
con.commit()
for r in con.execute("select id,tour_id,processing_status,processing_error,job_id from scenes where processing_status='queued' order by created_at desc limit 10"):
    print(r)
