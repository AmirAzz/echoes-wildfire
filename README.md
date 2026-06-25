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

## Deploy on Streamlit Community Cloud

Use this main file path:

```text
streamlit_app.py
```

The Streamlit version supports the MVP flow: country/region/date input, NASA FIRMS or demo detections, event candidate clustering, GDELT/news evidence retrieval, provenance, confidence scoring, and a first digital memory preview.

Add the NASA FIRMS key in Streamlit secrets, not in the public UI:

```toml
NASA_FIRMS_MAP_KEY = "your_firms_map_key_here"
```

`FIRMS_MAP_KEY` is also accepted as a backwards-compatible alias.

## Current Workflow

1. Select a country and either a preset region or a custom city/area, plus date range and NASA FIRMS source.
2. Enter a NASA FIRMS map key, or enable demo mode.
3. The backend fetches FIRMS hotspot CSV data or uses bundled mock detections.
4. The event builder clusters nearby detections in space and time.
5. The UI shows event candidates with confidence, provenance, limitations, and a map view.
6. Select an event and optionally fetch GDELT/news evidence for the event window.
7. The digital memory preview combines satellite evidence with a preliminary public narrative.

Note: GDELT can rate-limit public cloud apps with HTTP 429. The Streamlit UI therefore fetches GDELT only after pressing the dedicated fetch button.

Note: preset regions are examples only. Use `Custom city/area` for any city or municipality not listed. The app geocodes custom areas with OpenStreetMap Nominatim.

Note: demo mode uses synthetic sample detections and intentionally returns a small fixed set of event candidates. Turn off demo mode to query real NASA FIRMS detections from the key stored in Streamlit secrets.

## Next Modules

- Copernicus/EFFIS context connector.
- OpenStreetMap nearby asset extraction.
- LLM-based `Build Digital Memory` endpoint with citations and confidence scoring.
