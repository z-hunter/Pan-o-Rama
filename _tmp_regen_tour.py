import os
import sys
import json
import sqlite3

# Add current dir to path
sys.path.append(os.getcwd())

from backend.app import create_app
from backend.services.tour_service import generate_tour, load_tour_scenes_and_hotspots
from backend.core.database import get_db

def main():
    app = create_app()
    tid = '03db9028-5877-4e0d-951f-bc5794159d76'
    
    with app.app_context():
        db = get_db()
        tour = db.execute('SELECT * FROM tours WHERE id = ?', (tid,)).fetchone()
        if not tour:
            print(f"Tour {tid} not found")
            return
            
        t_dict = {k: tour[k] for k in tour.keys()}
        scenes = load_tour_scenes_and_hotspots(tid)
        
        print(f"Regenerating tour {tid}...")
        url = generate_tour(tid, scenes, watermark_enabled=True, tour_settings=t_dict)
        print(f"Success! View at: {url}")

if __name__ == "__main__":
    main()
