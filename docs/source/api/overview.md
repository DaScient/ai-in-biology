# API Overview

The AI in Biological Sciences API exposes biological AI models via a RESTful HTTP interface.

## Base URL

```
https://api.bioai.dascient.com
```

For local development:

```
http://localhost:8000
```

## Authentication

Currently no authentication is required for the development API.
Production endpoints require a Bearer token in the `Authorization` header.

## Available Endpoint Groups

| Group | Prefix | Description |
|-------|--------|-------------|
| Protein | `/api/v1/protein` | Structure prediction and design |
| Genomics | `/api/v1/genomics` | Variant analysis and regulatory elements |
| Single-Cell | `/api/v1/scell` | Embeddings and trajectory inference |
| Ecology | `/api/v1/ecology` | Species distribution and tipping points |
| Drug Discovery | `/api/v1/drug` | Molecular docking and generation |

## Interactive Docs

Visit `/docs` for the Swagger UI or `/redoc` for ReDoc when the API server is running.
