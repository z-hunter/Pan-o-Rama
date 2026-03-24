import sqlite3
p='/home/debian/lokalny_obiektyw/data/app.db'
con=sqlite3.connect(p)
con.row_factory=sqlite3.Row
for r in con.execute("select id,tour_id,title,processing_status,processing_error,job_id,updated_at from scenes where processing_status='failed' order by updated_at desc limit 20"):
    print(dict(r))
