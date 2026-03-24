import json, time, sqlite3, urllib.request, urllib.error
base='http://127.0.0.1:8000'
email=f'smoke_verify_{int(time.time())}@example.net'
payload=json.dumps({'email':email,'password':'Passw0rd!123','display_name':'Smoke'}).encode('utf-8')
req=urllib.request.Request(base+'/auth/register', data=payload, method='POST', headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req, timeout=15) as r:
    reg_status=r.status
    reg_body=r.read().decode('utf-8','ignore')
print('register', reg_status, reg_body)
# login before verify
payload=json.dumps({'email':email,'password':'Passw0rd!123'}).encode('utf-8')
req=urllib.request.Request(base+'/auth/login', data=payload, method='POST', headers={'Content-Type':'application/json'})
try:
    with urllib.request.urlopen(req, timeout=15) as r:
        print('login_before', r.status, r.read().decode('utf-8','ignore'))
except urllib.error.HTTPError as e:
    print('login_before', e.code, e.read().decode('utf-8','ignore'))
# token count
con=sqlite3.connect('/home/debian/lokalny_obiektyw/data/app.db')
uid=con.execute('select id,email_verified from users where email=?',(email,)).fetchone()
print('user', uid)
cnt=con.execute('select count(*) from email_verification_tokens where user_id=?',(uid[0],)).fetchone()[0]
print('tokens', cnt)
# invalid verify link behavior
req=urllib.request.Request(base+'/auth/verify?token=invalid', method='GET')
opener=urllib.request.build_opener(urllib.request.HTTPRedirectHandler())
try:
    resp=opener.open(req, timeout=15)
    print('verify_invalid_final', resp.geturl(), resp.status)
except urllib.error.HTTPError as e:
    print('verify_invalid', e.code, e.headers.get('Location'))
