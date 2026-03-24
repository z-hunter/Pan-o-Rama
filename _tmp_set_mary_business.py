import sqlite3, datetime, uuid
email='maryvoit@hotmail.com'
uid='931bdcca-5a52-490b-89c3-b86a4bd88bf2'
db='/home/debian/lokalny_obiektyw/data/app.db'
con=sqlite3.connect(db)
con.row_factory=sqlite3.Row
ts=datetime.datetime.utcnow().replace(microsecond=0).isoformat()+'Z'
# cancel active subscriptions
con.execute("update subscriptions set status='canceled', updated_at=? where user_id=? and status='active'", (ts, uid))
# insert new business active
con.execute(
    """
    insert into subscriptions (id,user_id,plan_id,status,billing_provider,provider_customer_id,provider_subscription_id,current_period_end,created_at,updated_at)
    values (?,?,?,?,?,?,?,?,?,?)
    """,
    (str(uuid.uuid4()), uid, 'business', 'active', 'mock', None, None, None, ts, ts)
)
con.commit()
row=con.execute("select plan_id,status,billing_provider,created_at,updated_at from subscriptions where user_id=? order by created_at desc limit 1", (uid,)).fetchone()
print('latest_subscription', dict(row) if row else None)
