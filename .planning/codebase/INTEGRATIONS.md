# External Integrations

## Core Infrastructure
- **Redis**: Used as the backend for the RQ (Redis Queue) job queuing system for async image processing.
- **OVHcloud VPS**: Deployment host for the service with Debian os, running Nginx and Gunicorn.

## APIs & Services
- **Stripe**: Payment processing library (`stripe` python package) integrated within `app.py`.
