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

The Streamlit version supports the MVP flow: country/region/date input, NASA FIRMS or demo detections, event candidate clustering, GDELT and Google News RSS evidence retrieval, provenance, confidence scoring, and a first digital memory preview.

Add the NASA FIRMS key in Streamlit secrets, not in the public UI:

```toml
NASA_FIRMS_MAP_KEY = "your_firms_map_key_here"
GEMINI_API_KEY = "your_gemini_api_key_here"
```

`FIRMS_MAP_KEY` is also accepted as a backwards-compatible alias.

`GEMINI_MODEL` is optional. If it is not set, the app uses `gemini-3.5-flash` and falls back to other Flash models if the selected model is unavailable, rate-limited, or temporarily overloaded.

## Current Workflow

1. Select a country and either a preset region or a custom city/area, plus date range and NASA FIRMS source.
2. Enter a NASA FIRMS map key, or enable demo mode.
3. The backend fetches FIRMS hotspot CSV data or uses bundled mock detections.
4. The event builder clusters nearby detections in space and time.
5. The UI shows event candidates with confidence, provenance, limitations, and a map view.
6. Select an event and optionally fetch GDELT/news evidence for the event window.
7. Generate Gemini memory analysis to extract reported impacts, response actions, vulnerable groups, preparedness gaps, lessons learned, early-action recommendations, and proposal value.
8. The digital memory report combines satellite evidence, public narrative, Gemini insights, limitations, and the raw JSON record for export/debugging.

Note: GDELT can rate-limit public cloud apps with HTTP 429. The Streamlit UI therefore fetches GDELT only after pressing the dedicated fetch button.

If GDELT is rate-limited, the app can optionally show clearly marked demo fallback news rows. These rows are not real articles and must not be cited as external evidence.

Google News RSS is also available as a no-key fallback news source. It is easier to use for demos, but it should still be treated as public media evidence rather than official confirmation.

Note: preset regions are examples only. Use `Custom city/area` for any city or municipality not listed. The app geocodes custom areas with OpenStreetMap Nominatim.

Note: demo mode uses synthetic sample detections and intentionally returns a small fixed set of event candidates. Turn off demo mode to query real NASA FIRMS detections from the key stored in Streamlit secrets.

Note: NASA FIRMS Area API accepts a day range of 1-5 days per request, so longer date windows are split into 5-day chunks.

Note: use `SP` products for historical dates and `NRT` products for recent near-real-time detections. The app automatically maps common NRT sources to SP for date ranges older than 30 days.

## Next Modules

- Copernicus/EFFIS context connector.
- OpenStreetMap nearby asset extraction.
- Full-article retrieval/RAG so Gemini can cite source passages, not only news titles and metadata.
