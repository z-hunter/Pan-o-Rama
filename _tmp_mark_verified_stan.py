import sqlite3, datetime
p='/home/debian/lokalny_obiektyw/data/app.db'
email='stan@sakartvelo360.com'
con=sqlite3.connect(p)
con.row_factory=sqlite3.Row
row=con.execute("select id,email,email_verified,email_verified_at,status from users where email=?", (email,)).fetchone()
print('before', dict(row) if row else None)
if row:
    ts=datetime.datetime.utcnow().replace(microsecond=0).isoformat()+'Z'
    con.execute("update users set email_verified=1, email_verified_at=?, status='active', updated_at=? where email=?", (ts, ts, email))
    con.commit()
row2=con.execute("select id,email,email_verified,email_verified_at,status from users where email=?", (email,)).fetchone()
print('after', dict(row2) if row2 else None)
