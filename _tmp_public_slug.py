import sqlite3
con=sqlite3.connect('/home/debian/lokalny_obiektyw/data/app.db')
con.row_factory=sqlite3.Row
r=con.execute("select slug,visibility,status from tours where deleted_at is null and visibility='public' and status='published' order by created_at desc limit 1").fetchone()
print(dict(r) if r else None)
