.PHONY: run db-start db-stop db-status migrate

run:
	poetry run uvicorn app.main:app --reload --port 8000

db-start:
	pg_ctl -D $(PGDATA) -o "-k $(PGHOST) -p $(PGPORT)" start

db-status:
	pg_ctl -D $(PGDATA) status

migrate:
	poetry run alembic upgrade head