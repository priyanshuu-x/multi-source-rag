FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (separate layer - only re-runs when requirements.txt changes,
# not on every code change, so rebuilds are much faster during development)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]