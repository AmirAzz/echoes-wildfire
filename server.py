from __future__ import annotations

import csv
import json
import math
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parent
PUBLIC = ROOT / "public"
REGIONS_PATH = ROOT / "data" / "regions.json"
NASA_FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{bbox}/{days}/{start}"
MAX_FIRMS_DAYS_PER_CALL = 10


@dataclass
class Detection:
    lat: float
    lon: float
    acquired_at: datetime
    confidence_raw: str
    confidence_score: float
    frp: float
    brightness: float | None
    sensor: str
    satellite: str
    source: str


def load_regions() -> dict[str, Any]:
    return json.loads(REGIONS_PATH.read_text(encoding="utf-8"))


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_acquired(row: dict[str, str]) -> datetime:
    day = row.get("acq_date") or row.get("ACQ_DATE") or ""
    time_text = row.get("acq_time") or row.get("ACQ_TIME") or "0000"
    time_text = time_text.zfill(4)[:4]
    return datetime.strptime(f"{day} {time_text}", "%Y-%m-%d %H%M")


def to_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def confidence_to_score(value: str | None) -> float:
    if value is None:
        return 0.5
    text = str(value).strip().lower()
    if text in {"h", "high"}:
        return 0.9
    if text in {"n", "nominal", "medium"}:
        return 0.65
    if text in {"l", "low"}:
        return 0.35
    try:
        numeric = float(text)
    except ValueError:
        return 0.5
    if numeric > 1:
        return max(0.0, min(1.0, numeric / 100.0))
    return max(0.0, min(1.0, numeric))


def haversine_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    radius = 6371.0
    d_lat = math.radians(b_lat - a_lat)
    d_lon = math.radians(b_lon - a_lon)
    lat1 = math.radians(a_lat)
    lat2 = math.radians(b_lat)
    h = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(h))


def get_bbox(params: dict[str, str]) -> tuple[list[float], str]:
    if params.get("bbox"):
        values = [float(v.strip()) for v in params["bbox"].split(",")]
        if len(values) != 4:
            raise ValueError("Custom bbox must be west,south,east,north")
        return values, "custom bounding box"

    country = params.get("country", "")
    region = params.get("region", "")
    regions = load_regions()
    if country not in regions:
        raise ValueError(f"Unknown country: {country}")
    if region in regions[country].get("regions", {}):
        return regions[country]["regions"][region], f"{country} / {region}"
    return regions[country]["bbox"], country


def each_date_chunk(start: date, end: date) -> list[tuple[date, int]]:
    chunks: list[tuple[date, int]] = []
    cursor = start
    while cursor <= end:
        days = min(MAX_FIRMS_DAYS_PER_CALL, (end - cursor).days + 1)
        chunks.append((cursor, days))
        cursor += timedelta(days=days)
    return chunks


def parse_firms_csv(text: str, source: str) -> list[Detection]:
    rows = list(csv.DictReader(text.splitlines()))
    detections: list[Detection] = []
    for row in rows:
        if "latitude" not in row or "longitude" not in row:
            continue
        detections.append(
            Detection(
                lat=to_float(row.get("latitude")),
                lon=to_float(row.get("longitude")),
                acquired_at=parse_acquired(row),
                confidence_raw=row.get("confidence", ""),
                confidence_score=confidence_to_score(row.get("confidence")),
                frp=to_float(row.get("frp")),
                brightness=to_float(row.get("bright_ti4") or row.get("brightness") or row.get("bright_t31"), default=math.nan),
                sensor=row.get("instrument") or row.get("sensor") or source,
                satellite=row.get("satellite") or "",
                source=source,
            )
        )
    return detections


def fetch_firms(params: dict[str, str]) -> tuple[list[Detection], dict[str, Any]]:
    key = params.get("nasaKey", "").strip()
    source = params.get("source", "VIIRS_SNPP_NRT").strip()
    start = parse_date(params.get("startDate", ""))
    end = parse_date(params.get("endDate", ""))
    bbox, bbox_label = get_bbox(params)

    if end < start:
        raise ValueError("End date must be after start date")
    if not key:
        raise ValueError("NASA FIRMS map key is required unless demo mode is enabled")

    all_detections: list[Detection] = []
    requests_made = []
    ssl_context = ssl.create_default_context()

    for chunk_start, days in each_date_chunk(start, end):
        bbox_text = ",".join(str(v) for v in bbox)
        url = NASA_FIRMS_AREA_URL.format(
            key=urllib.parse.quote(key),
            source=urllib.parse.quote(source),
            bbox=urllib.parse.quote(bbox_text),
            days=days,
            start=chunk_start.isoformat(),
        )
        requests_made.append({"start": chunk_start.isoformat(), "days": days, "source": source})
        with urllib.request.urlopen(url, timeout=30, context=ssl_context) as response:
            text = response.read().decode("utf-8", errors="replace")
        all_detections.extend(parse_firms_csv(text, source))

    return all_detections, {
        "source": "NASA FIRMS",
        "source_endpoint": "area/csv",
        "sensor_source": source,
        "bbox": bbox,
        "bbox_label": bbox_label,
        "requests": requests_made,
        "retrieved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "limitations": [
            "FIRMS reports active-fire and thermal-anomaly detections, not confirmed wildfire perimeters.",
            "Cloud cover, sensor overpass timing, false positives, and missing contextual reports can affect interpretation.",
            "Hotspot clusters are event candidates generated by this prototype and require expert or official validation."
        ],
    }


def demo_detections(params: dict[str, str]) -> tuple[list[Detection], dict[str, Any]]:
    start = parse_date(params.get("startDate", "2024-07-01"))
    bbox, bbox_label = get_bbox(params)
    west, south, east, north = bbox
    center_lat = (south + north) / 2
    center_lon = (west + east) / 2
    seeds = [
        (center_lat + 0.04, center_lon - 0.05, 0, 14),
        (center_lat + 0.06, center_lon - 0.02, 0, 19),
        (center_lat + 0.01, center_lon + 0.03, 1, 11),
        (center_lat - 0.14, center_lon + 0.11, 5, 9),
        (center_lat - 0.12, center_lon + 0.15, 5, 17),
        (center_lat - 0.17, center_lon + 0.08, 6, 8),
        (center_lat + 0.22, center_lon + 0.18, 11, 12),
    ]
    detections = []
    for idx, (lat, lon, day_offset, hour) in enumerate(seeds):
        for j in range(4 + (idx % 3)):
            acquired = datetime.combine(start + timedelta(days=day_offset), datetime.min.time()) + timedelta(hours=hour, minutes=j * 11)
            detections.append(
                Detection(
                    lat=lat + j * 0.008,
                    lon=lon + j * 0.007,
                    acquired_at=acquired,
                    confidence_raw="h" if idx % 2 == 0 else "n",
                    confidence_score=0.88 if idx % 2 == 0 else 0.66,
                    frp=38 + idx * 17 + j * 4,
                    brightness=332 + idx * 3 + j,
                    sensor="VIIRS demo",
                    satellite="NPP",
                    source="DEMO",
                )
            )
    return detections, {
        "source": "Demo data",
        "bbox": bbox,
        "bbox_label": bbox_label,
        "retrieved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "limitations": ["Demo mode uses synthetic detections for interface testing only."],
    }


def cluster_events(detections: list[Detection], radius_km: float = 12.0, max_gap_hours: float = 36.0) -> list[dict[str, Any]]:
    sorted_detections = sorted(detections, key=lambda item: item.acquired_at)
    clusters: list[list[Detection]] = []

    for detection in sorted_detections:
        best_index = None
        best_distance = float("inf")
        for idx, cluster in enumerate(clusters):
            center_lat = sum(item.lat for item in cluster) / len(cluster)
            center_lon = sum(item.lon for item in cluster) / len(cluster)
            last_time = max(item.acquired_at for item in cluster)
            distance = haversine_km(detection.lat, detection.lon, center_lat, center_lon)
            gap_hours = abs((detection.acquired_at - last_time).total_seconds()) / 3600
            if distance <= radius_km and gap_hours <= max_gap_hours and distance < best_distance:
                best_index = idx
                best_distance = distance
        if best_index is None:
            clusters.append([detection])
        else:
            clusters[best_index].append(detection)

    events = []
    for idx, cluster in enumerate(clusters, start=1):
        center_lat = sum(item.lat for item in cluster) / len(cluster)
        center_lon = sum(item.lon for item in cluster) / len(cluster)
        start_time = min(item.acquired_at for item in cluster)
        end_time = max(item.acquired_at for item in cluster)
        max_frp = max(item.frp for item in cluster)
        avg_confidence = sum(item.confidence_score for item in cluster) / len(cluster)
        spread = max(haversine_km(item.lat, item.lon, center_lat, center_lon) for item in cluster) if cluster else 0
        detection_score = min(1.0, len(cluster) / 20)
        frp_score = min(1.0, max_frp / 250)
        event_confidence = round((avg_confidence * 0.55 + detection_score * 0.25 + frp_score * 0.20) * 100)
        if len(cluster) >= 8 and event_confidence >= 75:
            status = "confirmed candidate"
        elif len(cluster) >= 3 and event_confidence >= 55:
            status = "probable candidate"
        else:
            status = "possible thermal anomaly"
        events.append(
            {
                "event_id": f"WF-{idx:03d}",
                "center": {"lat": round(center_lat, 5), "lon": round(center_lon, 5)},
                "start": start_time.isoformat(timespec="minutes"),
                "end": end_time.isoformat(timespec="minutes"),
                "detections": len(cluster),
                "max_frp": round(max_frp, 2),
                "avg_confidence": round(avg_confidence, 2),
                "event_confidence": event_confidence,
                "status": status,
                "spatial_spread_km": round(spread, 2),
                "evidence_type": "observed satellite detections + prototype clustering",
                "requires_validation": status != "confirmed candidate",
                "detections_sample": [
                    {
                        "lat": item.lat,
                        "lon": item.lon,
                        "acquired_at": item.acquired_at.isoformat(timespec="minutes"),
                        "confidence": item.confidence_raw,
                        "frp": item.frp,
                        "sensor": item.sensor,
                    }
                    for item in cluster[:80]
                ],
            }
        )
    return sorted(events, key=lambda item: (item["event_confidence"], item["detections"]), reverse=True)


def json_response(handler: SimpleHTTPRequestHandler, data: Any, status: int = 200) -> None:
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        parsed = urllib.parse.urlparse(path)
        clean_path = parsed.path
        if clean_path == "/":
            return str(PUBLIC / "index.html")
        return str(PUBLIC / clean_path.lstrip("/"))

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/regions":
            json_response(self, load_regions())
            return
        if parsed.path == "/api/fires/search":
            params = {key: values[-1] for key, values in urllib.parse.parse_qs(parsed.query).items()}
            try:
                if params.get("demo") == "true":
                    detections, provenance = demo_detections(params)
                else:
                    detections, provenance = fetch_firms(params)
                events = cluster_events(
                    detections,
                    radius_km=to_float(params.get("radiusKm"), 12.0),
                    max_gap_hours=to_float(params.get("maxGapHours"), 36.0),
                )
                json_response(
                    self,
                    {
                        "events": events,
                        "detections_count": len(detections),
                        "provenance": provenance,
                        "methodology": {
                            "event_builder": "Spatio-temporal clustering of NASA FIRMS hotspots",
                            "radius_km": to_float(params.get("radiusKm"), 12.0),
                            "max_gap_hours": to_float(params.get("maxGapHours"), 36.0),
                            "confidence_formula": "55% average FIRMS confidence + 25% detection count score + 20% max FRP score",
                        },
                    },
                )
            except (ValueError, urllib.error.URLError, TimeoutError, csv.Error) as error:
                json_response(self, {"error": str(error)}, status=400)
            return
        super().do_GET()

    def log_message(self, format: str, *args: Any) -> None:
        sys.stdout.write("%s - %s\n" % (self.log_date_time_string(), format % args))


def main() -> None:
    port = int(os.environ.get("PORT", "8787"))
    os.chdir(PUBLIC)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"ECHOES-Wildfire running at http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
