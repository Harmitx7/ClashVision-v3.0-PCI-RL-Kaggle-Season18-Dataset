web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
worker: python -m app.ml.trainer
release: alembic upgrade head
