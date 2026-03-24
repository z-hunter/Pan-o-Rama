import sqlite3, datetime
pid='2c616bff-3a31-42b2-b4c6-5856aab5f88a'
con=sqlite3.connect('/home/debian/lokalny_obiektyw/data/app.db')
con.row_factory=sqlite3.Row
r=con.execute('select id,slug,visibility,status from tours where id=?',(pid,)).fetchone()
print('before', dict(r) if r else None)
ts=datetime.datetime.utcnow().replace(microsecond=0).isoformat()+'Z'
con.execute('update tours set visibility=?, updated_at=? where id=?',('public',ts,pid))
con.commit()
r2=con.execute('select id,slug,visibility,status from tours where id=?',(pid,)).fetchone()
print('after', dict(r2) if r2 else None)
