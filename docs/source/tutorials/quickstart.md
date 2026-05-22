# Quick Start

This tutorial walks you through installing the package and running your first API call.

## 1. Install

```bash
git clone https://github.com/DaScient/ai-in-biology.git
cd ai-in-biology
pip install -e ".[dev]"
```

## 2. Start the API

```bash
make api-run
# or directly:
uvicorn api.src.main:app --reload
```

## 3. Health Check

```bash
curl http://localhost:8000/health
# {"status":"healthy","version":"0.1.0"}
```

## 4. Predict Protein Structure

```python
import httpx

response = httpx.post(
    "http://localhost:8000/api/v1/protein/fold",
    json={"sequence": "MKTIIALSYIFCLVFA"},
)
print(response.json())
```

## 5. Open Interactive Docs

Navigate to [http://localhost:8000/docs](http://localhost:8000/docs) to explore all endpoints.
