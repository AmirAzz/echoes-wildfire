# ECHOES-Wildfire Prototype

Standalone MVP for turning NASA FIRMS active-fire detections into wildfire event candidates.

## Run

```powershell
python server.py
```

Open:

```text
http://127.0.0.1:8787
```

## Current Workflow

1. Select a country, region, date range, and NASA FIRMS source.
2. Enter a NASA FIRMS map key, or enable demo mode.
3. The backend fetches FIRMS hotspot CSV data or uses bundled mock detections.
4. The event builder clusters nearby detections in space and time.
5. The UI shows event candidates with confidence, provenance, limitations, and a lightweight map view.

## Next Modules

- GDELT/news collector for public wildfire narratives.
- Copernicus/EFFIS context connector.
- OpenStreetMap nearby asset extraction.
- LLM-based `Build Digital Memory` endpoint with citations and confidence scoring.
