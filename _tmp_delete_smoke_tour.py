import sqlite3, os, shutil, datetime
p='/home/debian/lokalny_obiektyw/data/app.db'
proc='/home/debian/lokalny_obiektyw/data/processed_galleries'
raw='/home/debian/lokalny_obiektyw/data/raw_uploads'
TID='92e802c1-91bf-462a-b1da-ed46f08857dc'
con=sqlite3.connect(p)
con.row_factory=sqlite3.Row
tour=con.execute("select id,title,deleted_at from tours where id=?",(TID,)).fetchone()
print('tour before', dict(tour) if tour else None)
if tour:
    ts=datetime.datetime.utcnow().replace(microsecond=0).isoformat()+'Z'
    con.execute("update tours set deleted_at=?, updated_at=? where id=?", (ts, ts, TID))
    con.commit()
for root in (proc, raw):
    d=os.path.join(root, TID)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
        print('removed dir', d)
chk=con.execute("select id,title,deleted_at from tours where id=?",(TID,)).fetchone()
print('tour after', dict(chk) if chk else None)
