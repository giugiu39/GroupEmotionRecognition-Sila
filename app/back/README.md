# Group Emotion Recognition — Backend API

FastAPI backend for the Group Emotion Recognition system. Receives image frames from Raspberry Pi nodes, classifies group emotions via an external VLM server, persists results in MySQL, and serves aggregated data to the mobile/web application.

## Endpoints

| Method | Path | Caller | Description |
|--------|------|--------|-------------|
| `POST` | `/emonodes/sendmessage` | Raspberry Pi | Submit a frame for emotion classification |
| `GET` | `/app/data/getbetweendates` | App | Query emotion counts for a node within a time range |
| `POST` | `/app/askagent` | App | Send a conversation to the chatbot |

---

### POST `/emonodes/sendmessage`

**Content-Type:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `foto` | file | Captured image from the node |
| `node_name` | string | Identifier of the sending node (e.g. `raspi-01`) |
| `num_persone` | integer | Number of people detected in the frame |
| `timestamp` | integer | Unix timestamp of the capture |

**Response:** `200 OK` (empty body)

The row is written immediately with `emotion = NULL`. The VLM is then called asynchronously; if it succeeds the row is updated with the predicted emotion. If the VLM is unreachable the row keeps `emotion = NULL` and the endpoint still returns `200 OK`.

---

### GET `/app/data/getbetweendates`

**Query parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `start` | integer | Start unix timestamp (inclusive) |
| `end` | integer | End unix timestamp (inclusive) |
| `nodename` | string | Node identifier |

**Response:**

```json
{
  "raspi-01": {
    "happiness": 2,
    "neutral": 3,
    "surprise": 1,
    "distress": 0
  }
}
```

---

### POST `/app/askagent`

**Content-Type:** `application/json`

```json
{
  "messages": [
    {"role": "user", "content": "What emotions were detected today?"}
  ],
  "foto": "optional — base64 string or https:// URL"
}
```

**Response:**

```json
{
  "response": "Based on the detected emotions..."
}
```

Returns `500` if the language model call fails.

---

## Setup

### Prerequisites

- Python 3.11+
- MySQL running on `localhost:3306` with the `emotion_db` database and `detections` table already provisioned

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

| Variable | Description |
|----------|-------------|
| `DB_HOST` | MySQL host |
| `DB_PORT` | MySQL port |
| `DB_NAME` | Database name |
| `DB_USER` | Database user |
| `DB_PASSWORD` | Database password |
| `VLM_URL` | Full URL of the VLM inference endpoint (e.g. `http://localhost:8001/predict`) |
| `ANTHROPIC_API_KEY` | API key for the language model provider |

### Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

For development with auto-reload:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation is available at `http://localhost:8000/docs` once the server is running.
