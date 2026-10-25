# AeroServe PaaS

A lightweight Platform-as-a-Service (PaaS) designed to serve machine learning models efficiently in production environments.

## Quickstart Local Setup

Start the entire local infrastructure in under 5 commands[cite: 1]:

```bash
# 1. Clone the repository and navigate into it
git clone <your-repo-url> && cd aeroserve

# 2. Build and start the services (Gateway, Redis, Postgres)
docker compose up --build -d

# 3. Verify all services are healthy
docker compose ps

# 4. Test the API Gateway liveness endpoint
curl http://localhost:8000/health