import sqlite3
p='/home/debian/lokalny_obiektyw/data/app.db'
con=sqlite3.connect(p)
print('jobs', con.execute("select status,count(*) from jobs group by status order by status").fetchall())
print('scenes', con.execute("select processing_status,count(*) from scenes group by processing_status order by processing_status").fetchall())
print('recent queued', con.execute("select id,created_at from jobs where status='queued' order by created_at desc limit 5").fetchall())
