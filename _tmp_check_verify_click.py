import sqlite3
EMAIL='zx.hunter@gmail.com'
DB='/home/debian/lokalny_obiektyw/data/app.db'
con=sqlite3.connect(DB)
con.row_factory=sqlite3.Row
u=con.execute('select id,email,email_verified,email_verified_at,updated_at from users where email=?',(EMAIL,)).fetchone()
print('user', dict(u) if u else None)
if u:
    rows=con.execute('select id,expires_at,used_at,created_at from email_verification_tokens where user_id=? order by created_at desc limit 5',(u['id'],)).fetchall()
    print('recent_tokens', [dict(r) for r in rows])
