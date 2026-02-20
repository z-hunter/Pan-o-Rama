#!/usr/bin/env python3
import argparse
import csv
import sqlite3
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="List users from Pan-o-Rama app.db"
    )
    parser.add_argument(
        "--db",
        default="data/app.db",
        help="Path to SQLite DB (default: data/app.db)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit rows (0 means no limit)",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Print CSV instead of table",
    )
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="Show only users with status='active'",
    )
    return parser.parse_args()


def has_column(con: sqlite3.Connection, table: str, column: str) -> bool:
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == column for r in rows)


def build_query(active_only: bool, limit: int, include_email_verified: bool):
    where = "WHERE status = 'active'" if active_only else ""
    limit_sql = f"LIMIT {int(limit)}" if limit and limit > 0 else ""
    email_verified_expr = "email_verified" if include_email_verified else "0 AS email_verified"
    return f"""
        SELECT
            id,
            email,
            display_name,
            status,
            is_admin,
            {email_verified_expr},
            created_at
        FROM users
        {where}
        ORDER BY created_at DESC
        {limit_sql}
    """


def print_table(rows):
    if not rows:
        print("No users found.")
        return
    headers = [
        "created_at",
        "email",
        "display_name",
        "status",
        "admin",
        "verified",
        "id",
    ]
    widths = {h: len(h) for h in headers}
    mapped_rows = []
    for r in rows:
        mapped = {
            "created_at": str(r["created_at"] or ""),
            "email": str(r["email"] or ""),
            "display_name": str(r["display_name"] or ""),
            "status": str(r["status"] or ""),
            "admin": str(int(r["is_admin"] or 0)),
            "verified": str(int(r["email_verified"] or 0)),
            "id": str(r["id"] or ""),
        }
        mapped_rows.append(mapped)
        for h in headers:
            widths[h] = max(widths[h], len(mapped[h]))

    header_line = " | ".join(h.ljust(widths[h]) for h in headers)
    separator = "-+-".join("-" * widths[h] for h in headers)
    print(header_line)
    print(separator)
    for r in mapped_rows:
        print(" | ".join(r[h].ljust(widths[h]) for h in headers))
    print(f"\nTotal: {len(mapped_rows)}")


def print_csv(rows):
    writer = csv.writer(sys.stdout)
    writer.writerow(
        ["id", "email", "display_name", "status", "is_admin", "email_verified", "created_at"]
    )
    for r in rows:
        writer.writerow(
            [
                r["id"],
                r["email"],
                r["display_name"],
                r["status"],
                int(r["is_admin"] or 0),
                int(r["email_verified"] or 0),
                r["created_at"],
            ]
        )


def main():
    args = parse_args()
    db_path = Path(args.db).expanduser().resolve()
    if not db_path.exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        return 1
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        include_email_verified = has_column(con, "users", "email_verified")
        query = build_query(args.active_only, args.limit, include_email_verified)
        rows = con.execute(query).fetchall()
        if args.csv:
            print_csv(rows)
        else:
            print_table(rows)
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
