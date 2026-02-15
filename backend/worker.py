import os
import time

# Runs the local DB-backed job worker loop.
#
# This is the simplest way to test async processing locally without Redis.
# For production scaling, swap to Redis/RQ (or run multiple workers) later.


def main():
    # Ensure backend/app.py can be imported when run from repo root.
    try:
        from backend.app import worker_loop  # type: ignore
    except Exception:
        from app import worker_loop  # type: ignore

    poll = float(os.getenv("JOB_POLL_INTERVAL_SEC") or "0.75")
    print(f"[worker] starting (poll_interval={poll}s)")
    while True:
        try:
            worker_loop(poll_interval_sec=poll)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[worker] crashed: {type(e).__name__}: {e}")
            time.sleep(1.0)


if __name__ == "__main__":
    main()

