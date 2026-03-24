import sqlite3, re
from pathlib import Path
log=Path('/tmp/gunicorn_access.log')
lines=log.read_text(errors='ignore').splitlines()[-5000:]
rows=[]
for ln in lines:
    if '" 403 ' in ln and (' GET /t/' in ln or ' GET /galleries/' in ln):
        rows.append(ln)
print('403_count', len(rows))
for ln in rows[-40:]:
    print(ln)

slugs=[]
pids=[]
for ln in rows:
    m=re.search(r'GET /t/([^\s\"]+)', ln)
    if m:
        slugs.append(m.group(1))
    m2=re.search(r'GET /galleries/([0-9a-f\-]{36})/', ln)
    if m2:
        pids.append(m2.group(1))
slugs=sorted(set(slugs))
pids=sorted(set(pids))
print('slugs', slugs)
print('pids', pids)

con=sqlite3.connect('/home/debian/lokalny_obiektyw/data/app.db')
con.row_factory=sqlite3.Row
for s in slugs:
    r=con.execute('select id,slug,title,visibility,status,deleted_at from tours where slug=?',(s,)).fetchone()
    print('tour_by_slug', s, dict(r) if r else None)
for pid in pids:
    r=con.execute('select id,slug,title,visibility,status,deleted_at from tours where id=?',(pid,)).fetchone()
    print('tour_by_id', pid, dict(r) if r else None)
