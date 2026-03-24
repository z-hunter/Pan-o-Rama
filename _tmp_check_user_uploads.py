import sqlite3, os
email='stan@sakartvelo360.com'
base='/home/debian/lokalny_obiektyw/data'
con=sqlite3.connect(base+'/app.db')
con.row_factory=sqlite3.Row
u=con.execute('select id,email,created_at from users where email=?',(email,)).fetchone()
print('user', dict(u) if u else None)
if not u:
    raise SystemExit(0)
uid=u['id']
tours=list(con.execute('select id,title,visibility,status,created_at,deleted_at from tours where owner_id=? order by created_at',(uid,)).fetchall())
print('tours_count', len(tours))
for t in tours:
    tid=t['id']
    print('tour', dict(t))
    scenes=list(con.execute('select id,title,panorama_path,preview_path,processing_status,created_at from scenes where tour_id=? order by created_at',(tid,)).fetchall())
    print(' scenes_count', len(scenes))
    raw_tour=os.path.join(base,'raw_uploads',tid)
    proc_tour=os.path.join(base,'processed_galleries',tid)
    print(' raw_dir_exists', os.path.isdir(raw_tour), 'proc_dir_exists', os.path.isdir(proc_tour))
    if os.path.isdir(raw_tour):
        c=0
        for _,_,files in os.walk(raw_tour):
            c+=len(files)
        print(' raw_files', c)
    if os.path.isdir(proc_tour):
        c=0
        for _,_,files in os.walk(proc_tour):
            c+=len(files)
        print(' proc_files', c)
    for s in scenes:
        print('  scene', dict(s))
