import sqlite3, datetime, uuid, secrets, hashlib, json, urllib.request, urllib.error
EMAIL='zx.hunter@gmail.com'
API_KEY='re_gVqdDM1r_KnBPu8Jb1jaDy1aifKmBQQGC'
FROM='Pan-o-Rama <noreply@pan-o-rama.online>'
BASE='https://pan-o-rama.online'
DB='/home/debian/lokalny_obiektyw/data/app.db'

con=sqlite3.connect(DB)
con.row_factory=sqlite3.Row
u=con.execute('select id,email from users where email=?',(EMAIL,)).fetchone()
if not u:
    print('user_not_found')
    raise SystemExit(1)

token=secrets.token_urlsafe(48)
token_hash=hashlib.sha256(token.encode('utf-8')).hexdigest()
now=datetime.datetime.utcnow().replace(microsecond=0)
now_iso=now.isoformat()+'Z'
exp=(now+datetime.timedelta(hours=24)).isoformat()+'Z'
con.execute('insert into email_verification_tokens (id,user_id,token_hash,expires_at,used_at,created_at) values (?,?,?,?,NULL,?)',
            (str(uuid.uuid4()), u['id'], token_hash, exp, now_iso))
con.commit()
verify_url=f'{BASE}/auth/verify?token={token}'

payload={
  'from': FROM,
  'to': [EMAIL],
  'subject': 'Confirm your Pan-o-Rama account',
  'html': (
      '<p>Welcome to Pan-o-Rama.</p>'
      '<p>Confirm your email to activate your account:</p>'
      f'<p><a href="{verify_url}">{verify_url}</a></p>'
      '<p>This link expires in 24 hours.</p>'
  )
}
req=urllib.request.Request('https://api.resend.com/emails', data=json.dumps(payload).encode('utf-8'), method='POST', headers={
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'User-Agent': 'Pan-o-Rama/1.0 (+https://pan-o-rama.online)'
})
try:
    with urllib.request.urlopen(req, timeout=20) as r:
        print('send_status', r.status)
        print('send_body', r.read().decode('utf-8','ignore'))
except urllib.error.HTTPError as e:
    print('send_status', e.code)
    print('send_body', e.read().decode('utf-8','ignore'))
print('verify_url', verify_url)
