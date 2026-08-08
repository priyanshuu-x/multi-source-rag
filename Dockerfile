FROM python:3.11-slim

WORKDIR /app

# build-essential: chromadb depends on hnswlib, a C++ extension that sometimes
# needs to compile from source if no prebuilt wheel matches this exact platform.
# python:3.11-slim strips compiler tools by default - this avoids a build that
# fails partway through with a compiler error.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (separate layer - only re-runs when requirements.txt changes,
# not on every code change, so rebuilds are much faster during development)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]