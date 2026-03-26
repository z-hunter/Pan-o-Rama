import os
import sys
import json

# Add current dir to path
sys.path.append(os.getcwd())

from backend.app import create_app
from backend.services.tour_service import generate_tour, load_tour_scenes_and_hotspots
from backend.core.database import get_db

def main():
    app = create_app()
    
    with app.app_context():
        db = get_db()
        tours = db.execute('SELECT * FROM tours WHERE id IN (SELECT DISTINCT tour_id FROM scenes)').fetchall()
        print(f"Found {len(tours)} tours with scenes. Updating to Three.js engine...")
        
        for tour in tours:
            tid = tour['id']
            title = tour['title']
            t_dict = {k: tour[k] for k in tour.keys()}
            scenes = load_tour_scenes_and_hotspots(tid)
            
            try:
                generate_tour(tid, scenes, watermark_enabled=True, tour_settings=t_dict)
                print(f"  [+] Updated: {title}")
            except Exception as e:
                print(f"  [!] Failed {title}: {e}")
        
        print("All tours updated successfully.")

if __name__ == "__main__":
    main()
