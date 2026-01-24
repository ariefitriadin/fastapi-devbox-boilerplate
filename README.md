# 🚀 FastAPI Boilerplate (Devbox + Postgres 17)

A production-ready, highly opinionated FastAPI boilerplate designed for local development using **Devbox** for environment isolation and **Postgres 17** running on a custom port via Unix Sockets.

## 🛠 Tech Stack

- **Runtime:** Python 3.13
- **Framework:** [FastAPI](https://fastapi.tiangolo.com/)
- **Environment:** [Devbox](https://www.jetpack.io/devbox/) (Nix-based isolation)
- **Database:** PostgreSQL 17 (Custom port 5433)
- **ORM:** SQLAlchemy 2.0 (Async)
- **Migrations:** Alembic
- **Dependency Management:** Poetry

## 🏗 Project Structure

```
.
├── app/
│   ├── core/           # Database engine & Config logic
│   ├── models/         # SQLAlchemy models (Declarative Base)
│   ├── schemas/        # Pydantic validation models
│   └── main.py         # Entry point & Health checks
├── migrations/         # Alembic versioning & env.py
├── pyproject.toml      # Poetry & PEP 621 dependencies
├── devbox.json         # Nix environment & Services
└── .env                # Local environment variables (not committed)
```

## 🚦 Getting Started

### 1. Prerequisites

Ensure you have Devbox and Nix installed on your system.

### 2. Enter the Environment

Open your terminal in the project root and run:

```bash
devbox shell
```

Wait for the initial setup to complete. If prompted to overwrite the `.venv`, type `y`.

### 3. Initialize the Database (First Time Only)

Since Postgres is isolated within Devbox, you must initialize the data directory manually if `devbox services` is unavailable:

```bash
# Create the missing folder structure
mkdir -p $PGHOST && mkdir -p $PGDATA

# Initialize database files
initdb -D $PGDATA
```

### 4. Start the Services

Because of system conflicts on port 5432, this boilerplate uses port 5433 and Unix sockets:

```bash
# Start Postgres manually on the specific port and socket path
pg_ctl -D $PGDATA -o "-k $PGHOST -p 5433" start
```

### 5. Setup API & Migrations

```bash
# Apply initial database schema
poetry run alembic upgrade head

# Launch the FastAPI server
poetry run uvicorn app.main:app --reload --port 8000
```

## 🩺 Health Check

Verify the stack is 100% operational by visiting:  
`GET http://127.0.0.1:8000/health`

**Expected Response:**
```json
{
  "status": "online",
  "database": "connected",
  "environment": "devbox",
  "version": "0.1.0"
}
```

## 🔧 Environment Configuration (.env)

Use the following format in your `.env` to ensure asyncpg connects correctly via the Unix socket, bypassing DNS/hostname issues:

```bash
# Change 'ari3' to your system username
DATABASE_URL=postgresql+asyncpg://ari3@:5433/postgres?host=/home/ari3/petprojects/python/apiboilerplate/.devbox/virtenv/postgresql
APP_NAME="My FastAPI Boilerplate"
DEBUG=True
```

## 📜 Useful Commands

| Action | Command |
|--------|---------|
| Check DB Status | `pg_ctl -D $PGDATA status` |
| Stop DB | `pg_ctl -D $PGDATA stop` |
| Generate Migration | `poetry run alembic revision --autogenerate -m "description"` |
| Apply Migrations | `poetry run alembic upgrade head` |
| Run API Server | `poetry run uvicorn app.main:app --reload` |

## ⚠️ Troubleshooting

### "No address associated with hostname"

This occurs when the connection string uses `localhost` instead of a socket path. Ensure your URL uses the `@:5433` format and includes the `?host=` query parameter pointing to the `.devbox/virtenv` directory.

### Nix Version Error

If `devbox run` fails due to a Nix version check, bypass it by running the commands directly (e.g., `poetry run...` or `pg_ctl...`) as long as you are inside the `devbox shell`.