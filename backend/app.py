from flask import Flask, request, jsonify, send_from_directory, send_file, g, uuid
import os
import logging
from flask_cors import CORS

# Import modular core components
from backend.core.config import (
    UPLOAD_FOLDER, PROCESSED_FOLDER, VISITOR_COOKIE_NAME, FRONTEND_FOLDER, IMG_FOLDER
)
from backend.core.database import close_db, init_db, get_db
from backend.core.auth import get_current_user
from backend.core.models import now_iso

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

    # Setup Logging
    log_file = os.path.join(os.path.dirname(__file__), 'flask_app.log')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.DEBUG)

    # DB Teardown
    app.teardown_appcontext(close_db)

    # Initialize DB
    with app.app_context():
        init_db()

    # Register Blueprints
    from backend.blueprints.auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint, url_prefix='/auth')

    from backend.blueprints.users import users as users_blueprint
    app.register_blueprint(users_blueprint)

    from backend.blueprints.tours import tours as tours_blueprint
    app.register_blueprint(tours_blueprint, url_prefix='/tours')

    from backend.blueprints.scenes import scenes as scenes_blueprint
    app.register_blueprint(scenes_blueprint)

    from backend.blueprints.analytics import analytics as analytics_blueprint
    app.register_blueprint(analytics_blueprint, url_prefix='/api/analytics')

    # Global Hooks
    @app.before_request
    def attach_current_user():
        g.current_user = get_current_user()
        g.visitor_id = request.cookies.get(VISITOR_COOKIE_NAME) or str(uuid.uuid4())
        g.set_visitor_cookie = (request.cookies.get(VISITOR_COOKIE_NAME) is None)

    @app.after_request
    def persist_visitor_cookie(resp):
        try:
            if getattr(g, "set_visitor_cookie", False):
                resp.set_cookie(
                    VISITOR_COOKIE_NAME,
                    g.visitor_id,
                    max_age=60 * 60 * 24 * 730,
                    httponly=True,
                    samesite="Lax",
                )
        except Exception:
            pass
        return resp

    # Core Page Routes
    @app.route('/')
    def index():
        return send_from_directory(FRONTEND_FOLDER, 'index.html')

    @app.route('/login')
    @app.route('/login.html')
    def login_page():
        return send_from_directory(FRONTEND_FOLDER, 'login.html')

    @app.route('/dashboard')
    @app.route('/dashboard.html')
    def dashboard_page():
        return send_from_directory(FRONTEND_FOLDER, 'dashboard.html')

    @app.route('/account')
    @app.route('/account.html')
    def account_page():
        return send_from_directory(FRONTEND_FOLDER, 'account.html')

    @app.route('/projects.html')
    def projects_page():
        return send_from_directory(FRONTEND_FOLDER, 'projects.html')

    # Static Assets
    @app.route('/css/<path:filename>')
    def static_css(filename):
        return send_from_directory(os.path.join(FRONTEND_FOLDER, 'css'), filename)

    @app.route('/js/<path:filename>')
    def static_js(filename):
        return send_from_directory(os.path.join(FRONTEND_FOLDER, 'js'), filename)

    @app.route('/img/<path:filename>')
    def static_img(filename):
        return send_from_directory(IMG_FOLDER, filename)

    @app.route('/galleries/<project_id>/<path:filename>')
    def serve_gallery_files(project_id, filename):
        return send_from_directory(os.path.join(PROCESSED_FOLDER, project_id), filename)

    return app

app = create_app()

def analytics_track(event_type, visitor_id=None, user_id=None, tour_id=None, amount_cents=None, currency=None, meta=None):
    try:
        db = get_db()
        db.execute(
            """
            INSERT INTO analytics_events (id, event_type, visitor_id, user_id, tour_id, amount_cents, currency, meta_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (str(uuid.uuid4()), event_type, visitor_id, user_id, tour_id, amount_cents, currency, json.dumps(meta or {}), now_iso())
        )
        db.commit()
    except Exception:
        pass

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
