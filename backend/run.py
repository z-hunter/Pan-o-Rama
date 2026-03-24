from backend.app import app
import os

if __name__ == '__main__':
    # Ensure we are in the right directory context for data/ folders if needed
    # though core/config.py should handle BASE_DIR relative to itself.
    print("Starting Lokalny Obiektyw local server on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
