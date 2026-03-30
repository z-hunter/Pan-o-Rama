from flask import Blueprint, jsonify, g
from backend.core.auth import require_auth, current_user_is_admin
from backend.services.job_service import get_job
from backend.core.models import serialize_job

jobs = Blueprint("jobs", __name__)

@jobs.route("/<job_id>", methods=["GET"])
@require_auth
def get_job_status(job_id):
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
        
    if job["owner_id"] != g.current_user["id"] and not current_user_is_admin():
        return jsonify({"error": "Forbidden"}), 403
        
    return jsonify({"job": serialize_job(job)}), 200
