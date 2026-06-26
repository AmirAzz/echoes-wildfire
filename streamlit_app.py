from __future__ import annotations

import csv
import html
import re
import json
import math
import socket
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


ROOT = Path(__file__).parent
REGIONS_PATH = ROOT / "data" / "regions.json"
NASA_FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{bbox}/{days}/{start}"
GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
GEMINI_MODEL_FALLBACKS = ["gemini-3.5-flash", "gemini-3-flash", "gemini-3.1-flash-lite", "gemini-2.5-flash"]
GEMINI_RETRYABLE_HTTP_CODES = {404, 429, 503}
ARTICLE_MAX_CHARS = 1800
ARTICLE_FETCH_LIMIT = 5
MAX_FIRMS_DAYS_PER_CALL = 5
FIRMS_SOURCES = [
    "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_SP",
    "MODIS_SP",
    "VIIRS_SNPP_NRT",
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
    "MODIS_NRT",
]
HISTORICAL_SOURCE_MAP = {
    "VIIRS_SNPP_NRT": "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_NRT": "VIIRS_NOAA20_SP",
    "MODIS_NRT": "MODIS_SP",
}
WILDFIRE_TERMS = ["wildfire", "forest fire", "bushfire", "firefighters", "evacuation", "burned area", "smoke"]
IMPACT_KEYWORDS = {
    "evacuation": ["evacuat", "shelter"],
    "road disruption": ["road", "highway", "traffic", "closure"],
    "property damage": ["home", "house", "property", "village", "damage"],
    "smoke exposure": ["smoke", "air quality", "respiratory"],
    "firefighting deployment": ["firefighter", "aircraft", "helicopter", "water bomber", "civil protection"],
}


@dataclass
class Detection:
    lat: float
    lon: float
    acquired_at: datetime
    confidence_raw: str
    confidence_score: float
    frp: float
    sensor: str
    satellite: str


@st.cache_data
def load_regions() -> dict[str, Any]:
    return json.loads(REGIONS_PATH.read_text(encoding="utf-8"))


def to_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def confidence_to_score(value: str | None) -> float:
    text = str(value or "").strip().lower()
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
    return max(0.0, min(1.0, numeric / 100 if numeric > 1 else numeric))


def haversine_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    radius = 6371.0
    d_lat = math.radians(b_lat - a_lat)
    d_lon = math.radians(b_lon - a_lon)
    lat1 = math.radians(a_lat)
    lat2 = math.radians(b_lat)
    h = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(h))


def parse_acquired(row: dict[str, str]) -> datetime:
    day = row.get("acq_date", "")
    time_text = row.get("acq_time", "0000").zfill(4)[:4]
    return datetime.strptime(f"{day} {time_text}", "%Y-%m-%d %H%M")


def parse_firms_csv(text: str, source: str) -> list[Detection]:
    detections: list[Detection] = []
    for row in csv.DictReader(text.splitlines()):
        if not row.get("latitude") or not row.get("longitude"):
            continue
        detections.append(
            Detection(
                lat=to_float(row.get("latitude")),
                lon=to_float(row.get("longitude")),
                acquired_at=parse_acquired(row),
                confidence_raw=row.get("confidence", ""),
                confidence_score=confidence_to_score(row.get("confidence")),
                frp=to_float(row.get("frp")),
                sensor=row.get("instrument") or source,
                satellite=row.get("satellite", ""),
            )
        )
    return detections


def date_chunks(start: date, end: date) -> list[tuple[date, int]]:
    chunks = []
    cursor = start
    while cursor <= end:
        days = min(MAX_FIRMS_DAYS_PER_CALL, (end - cursor).days + 1)
        chunks.append((cursor, days))
        cursor += timedelta(days=days)
    return chunks


def effective_firms_source(source: str, end: date) -> tuple[str, str | None]:
    if source in HISTORICAL_SOURCE_MAP and end < (date.today() - timedelta(days=30)):
        historical_source = HISTORICAL_SOURCE_MAP[source]
        return historical_source, f"Historical date range detected; using {historical_source} instead of {source}."
    return source, None


def fetch_firms(key: str, source: str, bbox: list[float], start: date, end: date) -> list[Detection]:
    detections: list[Detection] = []
    bbox_text = ",".join(str(v) for v in bbox)
    for chunk_start, days in date_chunks(start, end):
        url = NASA_FIRMS_AREA_URL.format(
            key=urllib.parse.quote(key.strip()),
            source=urllib.parse.quote(source),
            bbox=bbox_text,
            days=days,
            start=chunk_start.isoformat(),
        )
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                detections.extend(parse_firms_csv(response.read().decode("utf-8", errors="replace"), source))
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                raise RuntimeError(
                    "NASA FIRMS rejected the request. Check that the Streamlit secret key is valid and active."
                ) from exc
            if exc.code == 404:
                raise RuntimeError(
                    "NASA FIRMS endpoint or source was not found. Try another FIRMS source such as VIIRS_SNPP_NRT."
                ) from exc
            if exc.code == 429:
                raise RuntimeError("NASA FIRMS rate limit reached. Wait a few minutes and search again.") from exc
            if exc.code == 400:
                raise RuntimeError(
                    "NASA FIRMS rejected the request as invalid. The app now splits requests into 5-day chunks; "
                    "if this persists, check the date range, selected source, and bounding box."
                ) from exc
            raise RuntimeError(f"NASA FIRMS returned HTTP {exc.code}. Try demo mode or check the selected source/date range.") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach NASA FIRMS: {exc.reason}") from exc
    return detections


def get_nasa_firms_key() -> str:
    try:
        return str(st.secrets.get("NASA_FIRMS_MAP_KEY", "") or st.secrets.get("FIRMS_MAP_KEY", "")).strip()
    except Exception:
        return ""


def get_gemini_api_key() -> str:
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        return ""


def get_gemini_model() -> str:
    try:
        return normalize_gemini_model(str(st.secrets.get("GEMINI_MODEL", "") or DEFAULT_GEMINI_MODEL))
    except Exception:
        return DEFAULT_GEMINI_MODEL


def normalize_gemini_model(model: str) -> str:
    text = model.strip()
    if text.startswith("models/"):
        return text.removeprefix("models/")
    return text


@st.cache_data(ttl=86400)
def geocode_area(area: str, country: str) -> dict[str, Any]:
    query = f"{area}, {country}".strip(", ")
    params = {
        "q": query,
        "format": "jsonv2",
        "limit": "1",
        "polygon_geojson": "0",
    }
    url = f"{NOMINATIM_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "ECHOES-Wildfire/0.1"})
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace"))
    if not payload:
        raise ValueError(f"Could not geocode area: {query}")
    item = payload[0]
    south, north, west, east = [float(value) for value in item["boundingbox"]]
    return {
        "label": item.get("display_name", query),
        "bbox": [west, south, east, north],
        "lat": float(item.get("lat", 0)),
        "lon": float(item.get("lon", 0)),
    }


def gdelt_datetime(value: datetime) -> str:
    return value.strftime("%Y%m%d%H%M%S")


def event_date_window(event: dict[str, Any], padding_days: int = 2) -> tuple[datetime, datetime]:
    start = datetime.fromisoformat(event["start"]) - timedelta(days=padding_days)
    end = datetime.fromisoformat(event["end"]) + timedelta(days=padding_days)
    return start, end


def article_relevance(article: dict[str, Any], country: str, region: str) -> float:
    text = f"{article.get('title', '')} {article.get('domain', '')}".lower()
    score = 0.25
    if country.lower() in text:
        score += 0.18
    if region.lower() in text:
        score += 0.25
    score += min(0.35, sum(0.08 for term in WILDFIRE_TERMS if term in text))
    return round(min(score, 1.0), 2)


@st.cache_data(ttl=1800)
def fetch_gdelt_articles(country: str, region: str, start: datetime, end: datetime, max_records: int) -> list[dict[str, Any]]:
    query = f'("{region}" OR "{country}") ("wildfire" OR "forest fire" OR "firefighters" OR "evacuation")'
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(max_records),
        "sort": "HybridRel",
        "startdatetime": gdelt_datetime(start),
        "enddatetime": gdelt_datetime(end),
    }
    url = f"{GDELT_DOC_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "ECHOES-Wildfire/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise RuntimeError(
                "GDELT rate limit reached (HTTP 429). Wait a few minutes, lower Max GDELT articles, then fetch again."
            ) from exc
        raise

    articles = []
    seen_urls = set()
    for item in payload.get("articles", []):
        article_url = item.get("url", "")
        if not article_url or article_url in seen_urls:
            continue
        seen_urls.add(article_url)
        article = {
            "title": item.get("title", "Untitled"),
            "url": article_url,
            "domain": item.get("domain", ""),
            "language": item.get("language", ""),
            "source_country": item.get("sourceCountry", ""),
            "seen_date": item.get("seendate", ""),
        }
        article["relevance"] = article_relevance(article, country, region)
        articles.append(article)

    return sorted(articles, key=lambda row: row["relevance"], reverse=True)


@st.cache_data(ttl=1800)
def fetch_google_news_articles(country: str, region: str, max_records: int) -> list[dict[str, Any]]:
    query = f'{region} {country} wildfire OR "forest fire"'
    params = {
        "q": query,
        "hl": "en",
        "gl": "US",
        "ceid": "US:en",
    }
    url = f"{GOOGLE_NEWS_RSS_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "ECHOES-Wildfire/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        xml_text = response.read().decode("utf-8", errors="replace")

    root = ET.fromstring(xml_text)
    articles = []
    for item in root.findall(".//item")[:max_records]:
        title = item.findtext("title", default="Untitled")
        link = item.findtext("link", default="")
        pub_date = item.findtext("pubDate", default="")
        source_el = item.find("source")
        domain = source_el.text if source_el is not None and source_el.text else "Google News RSS"
        article = {
            "title": title,
            "url": link,
            "domain": domain,
            "language": "English",
            "source_country": "",
            "seen_date": pub_date,
            "source_type": "Google News RSS",
        }
        article["relevance"] = article_relevance(article, country, region)
        articles.append(article)
    return sorted(articles, key=lambda row: row["relevance"], reverse=True)


def clean_html_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<(script|style|noscript).*?</\1>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_meta_description(html_text: str) -> str:
    patterns = [
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']description["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:description["\']',
    ]
    for pattern in patterns:
        match = re.search(pattern, html_text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return clean_html_text(match.group(1))
    return ""


def resolve_google_news_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if "news.google." not in parsed.netloc:
        return url
    params = urllib.parse.parse_qs(parsed.query)
    for key in ("url", "q"):
        if params.get(key):
            return params[key][0]
    return url


def extract_article_text_from_html(html_text: str) -> str:
    meta_description = extract_meta_description(html_text)
    paragraphs = re.findall(r"<p\b[^>]*>(.*?)</p>", html_text, flags=re.IGNORECASE | re.DOTALL)
    paragraph_texts = [clean_html_text(paragraph) for paragraph in paragraphs]
    paragraph_texts = [text for text in paragraph_texts if len(text) >= 80]
    combined = " ".join([meta_description, *paragraph_texts[:8]]).strip()
    if not combined:
        combined = clean_html_text(html_text)
    return combined[:ARTICLE_MAX_CHARS]


@st.cache_data(ttl=86400)
def fetch_article_excerpt(url: str) -> dict[str, Any]:
    if not url or url.startswith("demo://"):
        return {"status": "skipped", "resolved_url": url, "excerpt": "", "error": "No fetchable public URL."}

    resolved_url = resolve_google_news_url(url)
    request = urllib.request.Request(
        resolved_url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; ECHOES-Wildfire/0.1; +https://streamlit.app)",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            content_type = response.headers.get("Content-Type", "")
            if "html" not in content_type.lower():
                return {
                    "status": "unsupported",
                    "resolved_url": resolved_url,
                    "excerpt": "",
                    "error": f"Unsupported content type: {content_type}",
                }
            raw_html = response.read(600_000).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return {"status": "failed", "resolved_url": resolved_url, "excerpt": "", "error": f"HTTP {exc.code}"}
    except urllib.error.URLError as exc:
        return {"status": "failed", "resolved_url": resolved_url, "excerpt": "", "error": str(exc.reason)}

    excerpt = extract_article_text_from_html(raw_html)
    if not excerpt:
        return {"status": "empty", "resolved_url": resolved_url, "excerpt": "", "error": "No readable text found."}
    return {"status": "ok", "resolved_url": resolved_url, "excerpt": excerpt, "error": ""}


def attach_article_excerpts(articles: list[dict[str, Any]], limit: int = ARTICLE_FETCH_LIMIT) -> list[dict[str, Any]]:
    enriched = []
    for idx, article in enumerate(articles):
        enriched_article = dict(article)
        if idx < limit and not article.get("is_demo_fallback"):
            evidence = fetch_article_excerpt(article.get("url", ""))
            enriched_article["resolved_url"] = evidence["resolved_url"]
            enriched_article["article_excerpt"] = evidence["excerpt"]
            enriched_article["article_fetch_status"] = evidence["status"]
            enriched_article["article_fetch_error"] = evidence["error"]
        else:
            enriched_article.setdefault("resolved_url", article.get("url", ""))
            enriched_article.setdefault("article_excerpt", "")
            enriched_article.setdefault("article_fetch_status", "not requested")
            enriched_article.setdefault("article_fetch_error", "")
        enriched.append(enriched_article)
    return enriched


def build_public_narrative(articles: list[dict[str, Any]]) -> dict[str, Any]:
    if not articles:
        return {
            "source": "GDELT Doc API",
            "articles_found": 0,
            "confidence": 0.0,
            "status": "no public narrative evidence found",
            "limitations": [
                "No GDELT articles were found for the selected event window and query.",
                "Absence of media evidence does not mean absence of a wildfire event.",
            ],
        }

    titles = [article["title"] for article in articles]
    title_text = " ".join(titles).lower()
    reported_impacts = [
        label
        for label, keywords in IMPACT_KEYWORDS.items()
        if any(keyword in title_text for keyword in keywords)
    ]
    domains = sorted({article["domain"] for article in articles if article["domain"]})
    source_types = sorted({article.get("source_type", "GDELT Doc API") for article in articles})
    article_texts_found = sum(1 for article in articles if article.get("article_excerpt"))
    confidence = min(0.9, 0.35 + len(articles) * 0.04 + len(domains) * 0.025)
    if article_texts_found:
        confidence = min(0.95, confidence + min(0.15, article_texts_found * 0.03))
    confidence = round(confidence, 2)
    top_titles = titles[:5]

    return {
        "source": " + ".join(source_types),
        "articles_found": len(articles),
        "article_texts_found": article_texts_found,
        "distinct_domains": len(domains),
        "top_sources": domains[:8],
        "reported_impacts_preliminary": reported_impacts,
        "summary": "Public reporting was found for this event window. Article excerpts are used when available; otherwise the system falls back to news titles and metadata.",
        "top_article_titles": top_titles,
        "confidence": confidence,
        "limitations": [
            "GDELT is a public media index and may miss local or non-indexed sources.",
            "Current extraction may use partial article text and should not be treated as verified impact assessment.",
            "Duplicate syndication and uneven media attention can bias event visibility.",
        ],
    }


def demo_news_fallback(country: str, area_name: str, event: dict[str, Any]) -> list[dict[str, Any]]:
    event_day = event["start"][:10]
    return [
        {
            "title": f"Demo fallback: wildfire reported near {area_name} with firefighting response",
            "url": "demo://gdelt-rate-limit/fallback-1",
            "domain": "demo-fallback.local",
            "language": "English",
            "source_country": country,
            "seen_date": event_day,
            "relevance": 0.75,
            "source_type": "Demo fallback",
            "is_demo_fallback": True,
        },
        {
            "title": f"Demo fallback: local authorities monitor smoke, road access, and vulnerable communities in {area_name}",
            "url": "demo://gdelt-rate-limit/fallback-2",
            "domain": "demo-fallback.local",
            "language": "English",
            "source_country": country,
            "seen_date": event_day,
            "relevance": 0.68,
            "source_type": "Demo fallback",
            "is_demo_fallback": True,
        },
        {
            "title": f"Demo fallback: preparedness lessons include early warning, evacuation communication, and resource coordination",
            "url": "demo://gdelt-rate-limit/fallback-3",
            "domain": "demo-fallback.local",
            "language": "English",
            "source_country": country,
            "seen_date": event_day,
            "relevance": 0.62,
            "source_type": "Demo fallback",
            "is_demo_fallback": True,
        },
    ]


def demo_detections(bbox: list[float], start: date) -> list[Detection]:
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
            detections.append(
                Detection(
                    lat=lat + j * 0.008,
                    lon=lon + j * 0.007,
                    acquired_at=datetime.combine(start + timedelta(days=day_offset), datetime.min.time())
                    + timedelta(hours=hour, minutes=j * 11),
                    confidence_raw="h" if idx % 2 == 0 else "n",
                    confidence_score=0.88 if idx % 2 == 0 else 0.66,
                    frp=38 + idx * 17 + j * 4,
                    sensor="VIIRS demo",
                    satellite="NPP",
                )
            )
    return detections


def cluster_events(detections: list[Detection], radius_km: float, max_gap_hours: float) -> list[dict[str, Any]]:
    clusters: list[list[Detection]] = []
    for detection in sorted(detections, key=lambda item: item.acquired_at):
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
        max_frp = max(item.frp for item in cluster)
        avg_conf = sum(item.confidence_score for item in cluster) / len(cluster)
        detection_score = min(1.0, len(cluster) / 20)
        frp_score = min(1.0, max_frp / 250)
        event_confidence = round((avg_conf * 0.55 + detection_score * 0.25 + frp_score * 0.20) * 100)
        if len(cluster) >= 8 and event_confidence >= 75:
            status = "confirmed candidate"
        elif len(cluster) >= 3 and event_confidence >= 55:
            status = "probable candidate"
        else:
            status = "possible thermal anomaly"
        events.append(
            {
                "event_id": f"WF-{idx:03d}",
                "lat": round(center_lat, 5),
                "lon": round(center_lon, 5),
                "start": min(item.acquired_at for item in cluster).isoformat(timespec="minutes"),
                "end": max(item.acquired_at for item in cluster).isoformat(timespec="minutes"),
                "detections": len(cluster),
                "max_frp": round(max_frp, 2),
                "confidence_percent": event_confidence,
                "status": status,
            }
        )
    return sorted(events, key=lambda item: (item["confidence_percent"], item["detections"]), reverse=True)


def confidence_label(percent: float | int) -> str:
    if percent >= 75:
        return "High"
    if percent >= 50:
        return "Medium"
    return "Low"


def render_list(items: list[str], empty_text: str) -> None:
    if items:
        for item in items:
            st.markdown(f"- {item}")
    else:
        st.caption(empty_text)


def render_insight_items(items: list[Any], empty_text: str) -> None:
    if not items:
        st.caption(empty_text)
        return
    for item in items:
        if isinstance(item, dict):
            label = item.get("claim") or item.get("item") or item.get("recommendation") or item.get("lesson") or str(item)
            confidence = item.get("confidence")
            source = item.get("source_url") or item.get("source_basis")
            suffix = f" Confidence: {confidence}." if confidence else ""
            st.markdown(f"- {label}{suffix}")
            if source:
                st.caption(f"Source/basis: {source}")
        else:
            st.markdown(f"- {item}")


def confidence_percent(value: str | int | float | None) -> int:
    if isinstance(value, (int, float)):
        return int(max(0, min(100, value)))
    text = str(value or "").strip().lower()
    if text == "high":
        return 85
    if text == "medium":
        return 60
    if text == "low":
        return 30
    return 45


def confidence_color(value: str | int | float | None) -> str:
    percent = confidence_percent(value)
    if percent >= 75:
        return "#16a34a"
    if percent >= 50:
        return "#d97706"
    return "#dc2626"


def insight_label(item: Any) -> str:
    if isinstance(item, dict):
        return (
            item.get("claim")
            or item.get("item")
            or item.get("recommendation")
            or item.get("lesson")
            or item.get("time")
            or str(item)
        )
    return str(item)


def render_confidence_bar(label: str, value: str | int | float | None) -> None:
    percent = confidence_percent(value)
    color = confidence_color(value)
    st.markdown(
        f"""
        <div style="margin: 0.35rem 0 0.75rem 0;">
          <div style="display:flex; justify-content:space-between; font-size:0.85rem;">
            <span>{html.escape(label)}</span><strong>{html.escape(str(value or 'unknown'))}</strong>
          </div>
          <div style="height:8px; background:#e5e7eb; border-radius:999px; overflow:hidden;">
            <div style="height:8px; width:{percent}%; background:{color};"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight_cards(title: str, items: list[Any], empty_text: str) -> None:
    st.markdown(f"#### {title}")
    if not items:
        st.caption(empty_text)
        return
    cols = st.columns(min(3, max(1, len(items))))
    for idx, item in enumerate(items):
        with cols[idx % len(cols)]:
            label = insight_label(item)
            confidence = item.get("confidence", "unknown") if isinstance(item, dict) else "unknown"
            source = ""
            if isinstance(item, dict):
                source = item.get("source_basis") or item.get("source_url") or item.get("reason") or ""
            st.markdown(
                f"""
                <div style="border:1px solid #e5e7eb; border-radius:8px; padding:0.85rem; min-height:150px; background:#ffffff;">
                  <div style="font-weight:700; margin-bottom:0.4rem;">{html.escape(label)}</div>
                  <div style="font-size:0.82rem; color:#6b7280; margin-bottom:0.55rem;">{html.escape(source[:180])}</div>
                  <div style="font-size:0.78rem; color:#374151;">Confidence</div>
                  <div style="height:7px; background:#e5e7eb; border-radius:999px; overflow:hidden;">
                    <div style="height:7px; width:{confidence_percent(confidence)}%; background:{confidence_color(confidence)};"></div>
                  </div>
                  <div style="font-size:0.78rem; color:#6b7280; margin-top:0.35rem;">{html.escape(str(confidence))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_ai_timeline(items: list[dict[str, Any]]) -> None:
    st.markdown("#### Event Timeline")
    if not items:
        st.caption("No event timeline returned.")
        return
    for item in items[:6]:
        time_text = item.get("time", "time unknown")
        claim = item.get("claim", "")
        confidence = item.get("confidence", "unknown")
        st.markdown(
            f"""
            <div style="border-left:4px solid {confidence_color(confidence)}; padding:0.4rem 0 0.65rem 0.8rem; margin-left:0.2rem;">
              <div style="font-size:0.82rem; color:#6b7280;">{html.escape(time_text)}</div>
              <div style="font-weight:600;">{html.escape(claim)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def compact_articles_for_llm(articles: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    compacted = []
    for article in articles[:limit]:
        compacted.append(
            {
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "domain": article.get("domain", ""),
                "seen_date": article.get("seen_date", ""),
                "source_type": article.get("source_type", ""),
                "is_demo_fallback": bool(article.get("is_demo_fallback", False)),
                "relevance": article.get("relevance", 0),
                "resolved_url": article.get("resolved_url", article.get("url", "")),
                "article_excerpt": article.get("article_excerpt", "")[:ARTICLE_MAX_CHARS],
                "article_fetch_status": article.get("article_fetch_status", ""),
            }
        )
    return compacted


def build_gemini_prompt(memory_record: dict[str, Any], articles: list[dict[str, Any]]) -> str:
    payload = {
        "memory_record": memory_record,
        "news_articles": compact_articles_for_llm(articles),
    }
    return (
        "You are supporting a Horizon Europe wildfire preparedness prototype called ECHOES-Wildfire. "
        "Transform the provided satellite event record and news evidence into a structured digital memory. "
        "Use only the evidence provided. Do not invent casualties, causes, damage, or official decisions. "
        "If evidence is weak or missing, say so explicitly. Keep every string under 160 characters. "
        "Use short factual phrases, not paragraphs. Return valid JSON only with this structure: "
        "{"
        '"executive_summary": string, '
        '"event_timeline": [{"time": string, "claim": string, "source_basis": string, "confidence": "low|medium|high"}], '
        '"reported_impacts": [{"claim": string, "source_url": string, "confidence": "low|medium|high", "requires_human_validation": boolean}], '
        '"response_actions": [{"claim": string, "source_url": string, "confidence": "low|medium|high", "requires_human_validation": boolean}], '
        '"affected_or_vulnerable_groups": [{"claim": string, "source_url": string, "confidence": "low|medium|high", "requires_human_validation": boolean}], '
        '"preparedness_gaps": [{"claim": string, "source_basis": string, "confidence": "low|medium|high"}], '
        '"lessons_learned": [{"lesson": string, "source_basis": string, "confidence": "low|medium|high"}], '
        '"early_action_recommendations": [{"recommendation": string, "reason": string, "confidence": "low|medium|high"}], '
        '"proposal_value": string, '
        '"validation_needs": [string], '
        '"overall_confidence": "low|medium|high"'
        "}. Evidence payload: "
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


def extract_gemini_text(response_payload: dict[str, Any]) -> str:
    candidates = response_payload.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini returned no candidates.")
    parts = candidates[0].get("content", {}).get("parts", [])
    text_parts = [part.get("text", "") for part in parts if part.get("text")]
    if not text_parts:
        raise RuntimeError("Gemini returned no text content.")
    return "\n".join(text_parts).strip()


def parse_json_response(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        json_text = text[start : end + 1]
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            return json.loads(repair_json_string_newlines(json_text))


def repair_json_string_newlines(text: str) -> str:
    repaired = []
    in_string = False
    escaped = False
    for char in text:
        if escaped:
            repaired.append(char)
            escaped = False
            continue
        if char == "\\":
            repaired.append(char)
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            repaired.append(char)
            continue
        if in_string and char in {"\n", "\r", "\t"}:
            repaired.append(" ")
            continue
        repaired.append(char)
    return "".join(repaired)


def gemini_response_schema() -> dict[str, Any]:
    claim_item = {
        "type": "object",
        "properties": {
            "claim": {"type": "string"},
            "source_url": {"type": "string"},
            "source_basis": {"type": "string"},
            "confidence": {"type": "string"},
            "requires_human_validation": {"type": "boolean"},
        },
    }
    return {
        "type": "object",
        "properties": {
            "executive_summary": {"type": "string"},
            "event_timeline": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "time": {"type": "string"},
                        "claim": {"type": "string"},
                        "source_basis": {"type": "string"},
                        "confidence": {"type": "string"},
                    },
                },
            },
            "reported_impacts": {"type": "array", "items": claim_item},
            "response_actions": {"type": "array", "items": claim_item},
            "affected_or_vulnerable_groups": {"type": "array", "items": claim_item},
            "preparedness_gaps": {"type": "array", "items": claim_item},
            "lessons_learned": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "lesson": {"type": "string"},
                        "source_basis": {"type": "string"},
                        "confidence": {"type": "string"},
                    },
                },
            },
            "early_action_recommendations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "recommendation": {"type": "string"},
                        "reason": {"type": "string"},
                        "confidence": {"type": "string"},
                    },
                },
            },
            "proposal_value": {"type": "string"},
            "validation_needs": {"type": "array", "items": {"type": "string"}},
            "overall_confidence": {"type": "string"},
        },
    }


def fallback_memory_analysis(
    memory_record: dict[str, Any],
    articles: list[dict[str, Any]],
    model: str,
    reason: str,
    response_preview: str = "",
) -> dict[str, Any]:
    event = memory_record["event"]
    satellite = memory_record["satellite_evidence"]
    narrative = memory_record["public_narrative"]
    top_titles = [article.get("title", "") for article in articles[:3] if article.get("title")]
    source_note = "Satellite detections and news metadata only"
    return {
        "executive_summary": (
            f"Candidate wildfire memory for {event['area_label']} based on "
            f"{satellite['detections']} FIRMS detections and {len(articles)} news metadata rows."
        ),
        "event_timeline": [
            {
                "time": event["start"],
                "claim": f"Clustered active-fire detections started near {event['region']}.",
                "source_basis": satellite["source"],
                "confidence": "medium",
            },
            {
                "time": event["end"],
                "claim": "Clustered detections ended within the selected event window.",
                "source_basis": satellite["source"],
                "confidence": "medium",
            },
        ],
        "reported_impacts": [
            {
                "claim": impact,
                "source_url": "",
                "source_basis": "Preliminary keyword extraction from news titles",
                "confidence": "low",
                "requires_human_validation": True,
            }
            for impact in narrative.get("reported_impacts_preliminary", [])
        ],
        "response_actions": [],
        "affected_or_vulnerable_groups": [],
        "preparedness_gaps": [
            {
                "claim": "Official incident reports, perimeter data, and verified impacts are still missing.",
                "source_basis": source_note,
                "confidence": "high",
            }
        ],
        "lessons_learned": [
            {
                "lesson": "Use satellite detections to trigger rapid evidence collection and human validation.",
                "source_basis": source_note,
                "confidence": "medium",
            }
        ],
        "early_action_recommendations": [
            {
                "recommendation": "Validate the event with civil protection or official wildfire data.",
                "reason": "FIRMS detects thermal anomalies, not confirmed wildfire perimeters.",
                "confidence": "high",
            },
            {
                "recommendation": "Collect full article text and official updates for cited impact extraction.",
                "reason": "Current news evidence is mostly metadata and titles.",
                "confidence": "high",
            },
        ],
        "proposal_value": (
            "Shows how ECHOES can convert public satellite and media signals into an auditable disaster memory."
        ),
        "validation_needs": [
            "Official incident confirmation",
            "Fire perimeter or burned-area layer",
            "Verified impacts and response actions",
            *top_titles[:2],
        ],
        "overall_confidence": "low" if not articles else "medium",
        "method": "Gemini fallback after malformed model JSON",
        "model": model,
        "important_limitation": (
            f"Gemini returned malformed JSON, so the app generated a conservative fallback. Reason: {reason}"
        ),
        "raw_gemini_preview": response_preview,
    }


def claim_from_summary(summary: str, pattern: str, label: str, source_basis: str) -> dict[str, Any] | None:
    if re.search(pattern, summary, flags=re.IGNORECASE):
        return {
            "claim": label,
            "source_url": "",
            "source_basis": source_basis,
            "confidence": "medium",
            "requires_human_validation": True,
        }
    return None


def normalize_gemini_analysis(
    analysis: dict[str, Any],
    memory_record: dict[str, Any],
    articles: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = analysis.get("executive_summary", "")
    source_basis = "Gemini executive summary derived from attached news evidence"

    if summary and not analysis.get("reported_impacts"):
        inferred_impacts = [
            claim_from_summary(summary, r"\bfatalit|dead|death|killed\b", "fatalities reported", source_basis),
            claim_from_summary(summary, r"\bevacuat|mass evacuation\b", "evacuations reported", source_basis),
            claim_from_summary(summary, r"\b(loss|losses|damage|homes?|property|economic)\b", "damage or economic losses reported", source_basis),
            claim_from_summary(summary, r"\broad|highway|closure|traffic\b", "road disruption reported", source_basis),
        ]
        analysis["reported_impacts"] = [item for item in inferred_impacts if item]

    if summary and not analysis.get("response_actions"):
        inferred_actions = [
            claim_from_summary(summary, r"\bfirefighter|firefighting|crews?|aircraft|helicopter\b", "firefighting response reported", source_basis),
            claim_from_summary(summary, r"\bevacuat|shelter\b", "evacuation or shelter response reported", source_basis),
        ]
        analysis["response_actions"] = [item for item in inferred_actions if item]

    if not analysis.get("preparedness_gaps"):
        analysis["preparedness_gaps"] = [
            {
                "claim": "Official incident reports, perimeter data, and verified impacts are still required.",
                "source_basis": "Digital memory validation checklist",
                "confidence": "high",
            }
        ]

    if not analysis.get("lessons_learned"):
        analysis["lessons_learned"] = [
            {
                "lesson": "Combine satellite detections with article-level evidence before drawing operational conclusions.",
                "source_basis": "Current evidence mix and validation gaps",
                "confidence": "medium",
            }
        ]

    if not analysis.get("early_action_recommendations"):
        analysis["early_action_recommendations"] = [
            {
                "recommendation": "Validate the candidate event with official civil protection or EFFIS/Copernicus data.",
                "reason": "FIRMS detections and media reports do not equal official incident confirmation.",
                "confidence": "high",
            },
            {
                "recommendation": "Extract cited claims from full article text and official updates.",
                "reason": "Article titles and short excerpts can miss response details and vulnerable-group impacts.",
                "confidence": "high",
            },
        ]

    if not analysis.get("affected_or_vulnerable_groups") and re.search(
        r"\bevacuated|residents|villages?|communities|elderly|children|tourists\b",
        summary,
        flags=re.IGNORECASE,
    ):
        analysis["affected_or_vulnerable_groups"] = [
            {
                "claim": "Affected residents or communities are mentioned but need source-level verification.",
                "source_url": "",
                "source_basis": source_basis,
                "confidence": "low",
                "requires_human_validation": True,
            }
        ]

    if not analysis.get("validation_needs"):
        top_titles = [article.get("title", "") for article in articles[:2] if article.get("title")]
        analysis["validation_needs"] = [
            "Official incident confirmation",
            "Fire perimeter or burned-area layer",
            "Verified casualty, evacuation, and damage figures",
            *top_titles,
        ]

    analysis["post_processed"] = True
    analysis["post_processing_note"] = (
        "Empty Gemini sections were backfilled from explicit summary claims and validation rules."
    )
    return analysis


def generate_gemini_memory_analysis(
    api_key: str,
    model: str,
    memory_record: dict[str, Any],
    articles: list[dict[str, Any]],
) -> dict[str, Any]:
    body = {
        "contents": [{"parts": [{"text": build_gemini_prompt(memory_record, articles)}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 4500,
            "responseMimeType": "application/json",
            "responseSchema": gemini_response_schema(),
        },
    }

    candidate_models = [normalize_gemini_model(model)]
    for fallback in GEMINI_MODEL_FALLBACKS:
        if fallback not in candidate_models:
            candidate_models.append(fallback)

    last_error = ""
    selected_model = candidate_models[0]
    response_payload: dict[str, Any] | None = None
    for candidate_model in candidate_models:
        url = (
            f"{GEMINI_GENERATE_URL.format(model=urllib.parse.quote(candidate_model))}?"
            f"{urllib.parse.urlencode({'key': api_key})}"
        )
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                response_payload = json.loads(response.read().decode("utf-8", errors="replace"))
            selected_model = candidate_model
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            last_error = f"Gemini model {candidate_model} returned HTTP {exc.code}. {detail}"
            if exc.code not in GEMINI_RETRYABLE_HTTP_CODES:
                raise RuntimeError(last_error) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError | socket.timeout):
                selected_model = candidate_model
                last_error = f"Gemini model {candidate_model} timed out before returning a response."
                return fallback_memory_analysis(memory_record, articles, selected_model, str(exc.reason))
            raise RuntimeError(f"Could not reach Gemini API: {exc.reason}") from exc
        except (TimeoutError, socket.timeout) as exc:
            selected_model = candidate_model
            last_error = f"Gemini model {candidate_model} timed out before returning a response."
            return fallback_memory_analysis(memory_record, articles, selected_model, str(exc))

    if response_payload is None:
        raise RuntimeError(last_error or "Gemini did not return a usable response.")

    response_text = extract_gemini_text(response_payload)
    try:
        analysis = parse_json_response(response_text)
    except json.JSONDecodeError as exc:
        preview = response_text[:220].replace("\n", " ")
        return fallback_memory_analysis(memory_record, articles, selected_model, str(exc), preview)
    analysis = normalize_gemini_analysis(analysis, memory_record, articles)
    analysis["method"] = "Gemini API structured generation over NASA FIRMS and public news metadata"
    analysis["model"] = selected_model
    analysis["important_limitation"] = (
        "The model only sees satellite metadata and news titles/metadata, not full article text or official incident reports."
    )
    return analysis


def render_memory_record(memory_record: dict[str, Any]) -> None:
    event_info = memory_record["event"]
    satellite = memory_record["satellite_evidence"]
    narrative = memory_record["public_narrative"]
    llm_analysis = memory_record.get("llm_analysis", {})

    st.subheader("Digital Wildfire Memory")
    st.markdown(
        f"**{memory_record['memory_id']}** documents a candidate wildfire event in "
        f"**{event_info['area_label']}** from **{event_info['start']}** to **{event_info['end']}**."
    )

    if event_info["status"] == "possible thermal anomaly":
        st.warning("This is a possible thermal anomaly, not a confirmed wildfire. Official or expert validation is required.")
    elif "probable" in event_info["status"]:
        st.info("This is a probable wildfire candidate based on clustered active-fire detections.")
    else:
        st.success("This is a higher-confidence wildfire candidate based on the current evidence.")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Event status", event_info["status"])
    metric_cols[1].metric("Satellite detections", satellite["detections"])
    metric_cols[2].metric("Max FRP", satellite["max_frp"])
    metric_cols[3].metric(
        "Satellite confidence",
        f"{satellite['event_confidence_percent']}%",
        confidence_label(satellite["event_confidence_percent"]),
    )

    tab_summary, tab_ai, tab_evidence, tab_gaps, tab_raw = st.tabs(
        ["Memory Summary", "Gemini Analysis", "Evidence", "Gaps & Limitations", "Raw Record"]
    )

    with tab_summary:
        st.markdown("#### What the memory currently says")
        st.write(
            f"The system detected **{satellite['detections']}** active-fire observations near "
            f"**{event_info['region']}**, clustered into event **{memory_record['memory_id']}**. "
            f"The current event status is **{event_info['status']}**."
        )
        st.markdown("#### Public narrative")
        st.write(narrative.get("summary", "No public narrative summary is available yet."))
        if narrative.get("reported_impacts_preliminary"):
            st.markdown("#### Preliminary reported impacts")
            render_list(narrative["reported_impacts_preliminary"], "No impacts extracted yet.")
        if narrative.get("top_article_titles"):
            st.markdown("#### Top article titles")
            render_list(narrative["top_article_titles"], "No article titles attached yet.")

    with tab_ai:
        if not llm_analysis:
            st.info("Gemini analysis has not been generated yet. Use the button below the news evidence table.")
        else:
            st.markdown("#### AI Memory Board")
            st.info(llm_analysis.get("executive_summary", "No Gemini summary returned."))

            confidence_cols = st.columns(3)
            with confidence_cols[0]:
                render_confidence_bar("Narrative confidence", llm_analysis.get("overall_confidence", "unknown"))
            with confidence_cols[1]:
                render_confidence_bar("Event validation confidence", satellite["event_confidence_percent"])
            with confidence_cols[2]:
                operational_confidence = "medium" if narrative.get("articles_found", 0) else "low"
                render_confidence_bar("Operational readiness", operational_confidence)

            render_ai_timeline(llm_analysis.get("event_timeline", []))

            impact_col, action_col = st.columns(2)
            with impact_col:
                render_insight_cards(
                    "Reported Impacts",
                    llm_analysis.get("reported_impacts", []),
                    "No reported impacts extracted.",
                )
            with action_col:
                render_insight_cards(
                    "Response Actions",
                    llm_analysis.get("response_actions", []),
                    "No response actions extracted.",
                )

            group_col, gap_col = st.columns(2)
            with group_col:
                render_insight_cards(
                    "Affected / Vulnerable Groups",
                    llm_analysis.get("affected_or_vulnerable_groups", []),
                    "No affected or vulnerable groups extracted.",
                )
            with gap_col:
                render_insight_cards(
                    "Preparedness Gaps",
                    llm_analysis.get("preparedness_gaps", []),
                    "No preparedness gaps extracted.",
                )

            lesson_col, early_action_col = st.columns(2)
            with lesson_col:
                render_insight_cards(
                    "Lessons Learned",
                    llm_analysis.get("lessons_learned", []),
                    "No lessons learned extracted.",
                )
            with early_action_col:
                render_insight_cards(
                    "Early-Action Recommendations",
                    llm_analysis.get("early_action_recommendations", []),
                    "No early-action recommendations extracted.",
                )

            st.markdown("#### Proposal Value")
            st.success(llm_analysis.get("proposal_value", "No proposal value returned."))
            st.markdown("#### Validation Checklist")
            validation_needs = llm_analysis.get("validation_needs", [])
            if validation_needs:
                validation_df = pd.DataFrame(
                    [{"Status": "Pending", "Validation need": item} for item in validation_needs]
                )
                st.dataframe(validation_df, use_container_width=True, hide_index=True)
            else:
                st.caption("No validation needs returned.")

            st.caption(f"Model: {llm_analysis.get('model', 'unknown')}")
            if llm_analysis.get("post_processing_note"):
                st.caption(llm_analysis["post_processing_note"])
            st.warning(llm_analysis.get("important_limitation", "AI output requires human validation."))

    with tab_evidence:
        st.markdown("#### Satellite evidence")
        st.table(
            pd.DataFrame(
                [
                    {"Field": "Source", "Value": satellite["source"]},
                    {"Field": "Evidence type", "Value": satellite["evidence_type"]},
                    {"Field": "Detections", "Value": satellite["detections"]},
                    {"Field": "Max FRP", "Value": satellite["max_frp"]},
                    {"Field": "Confidence", "Value": f"{satellite['event_confidence_percent']}%"},
                    {"Field": "Center", "Value": f"{event_info['center']['lat']}, {event_info['center']['lon']}"},
                ]
            )
        )
        st.markdown("#### News / public evidence")
        st.table(
            pd.DataFrame(
                [
                    {"Field": "Source", "Value": narrative.get("source", "not attached")},
                    {"Field": "Articles found", "Value": narrative.get("articles_found", 0)},
                    {"Field": "Article excerpts found", "Value": narrative.get("article_texts_found", 0)},
                    {"Field": "Distinct domains", "Value": narrative.get("distinct_domains", 0)},
                    {"Field": "Narrative confidence", "Value": narrative.get("confidence", 0.0)},
                    {"Field": "Status", "Value": narrative.get("status", "available")},
                ]
            )
        )
        if narrative.get("top_sources"):
            st.markdown("#### Top sources")
            render_list(narrative["top_sources"], "No sources attached yet.")

    with tab_gaps:
        st.markdown("#### Missing data")
        render_list(memory_record.get("missing_data", []), "No missing-data notes.")
        st.markdown("#### Satellite limitations")
        render_list(satellite.get("limitations", []), "No satellite limitations listed.")
        st.markdown("#### Public narrative limitations")
        render_list(narrative.get("limitations", []), "No public narrative limitations listed.")

    with tab_raw:
        st.caption("Machine-readable structured memory record for debugging, export, and future LLM/RAG modules.")
        st.json(memory_record)


st.set_page_config(page_title="ECHOES-Wildfire", layout="wide")
st.title("ECHOES-Wildfire")
st.caption("AI-ready wildfire digital memory prototype using NASA FIRMS active-fire detections.")

regions = load_regions()

with st.sidebar:
    st.header("Search")
    country = st.selectbox("Country", list(regions.keys()), index=list(regions.keys()).index("Cyprus"))
    region_names = list(regions[country]["regions"].keys())
    default_region = region_names.index("Limassol") if country == "Cyprus" and "Limassol" in region_names else 0
    region = st.selectbox("Preset region / municipality", region_names, index=default_region)
    custom_area = st.text_input(
        "Custom city/area (optional)",
        placeholder="e.g. Nicosia, Palermo, Athens, Valencia",
        help="Use this when the city is not in the preset list. The app geocodes it with OpenStreetMap Nominatim.",
    )
    start_date = st.date_input("Start date", value=date(2024, 7, 1))
    end_date = st.date_input("End date", value=date(2024, 7, 20))
    source = st.selectbox(
        "NASA FIRMS source",
        FIRMS_SOURCES,
        help="Use SP for historical dates and NRT for recent near-real-time detections.",
    )
    nasa_key = get_nasa_firms_key()
    gemini_key = get_gemini_api_key()
    gemini_model = get_gemini_model()
    if nasa_key:
        st.success("NASA FIRMS key loaded from Streamlit secrets.")
    else:
        st.info("No NASA FIRMS key found in Streamlit secrets. Demo mode is available.")
    if gemini_key:
        st.success("Gemini API key loaded from Streamlit secrets.")
    else:
        st.info("No Gemini API key found. Add GEMINI_API_KEY in Streamlit secrets to generate AI memory insights.")
    demo_mode = st.checkbox(
        "Use demo data",
        value=not bool(nasa_key),
        help="Demo data is synthetic and intentionally returns a small fixed set of sample events. Turn this off to query NASA FIRMS.",
    )
    attach_gdelt = st.checkbox("Enable GDELT/news evidence", value=False)
    gdelt_max_records = st.slider("Max GDELT articles", min_value=5, max_value=50, value=10)
    attach_google_news = st.checkbox("Enable Google News RSS evidence", value=True)
    google_news_max_records = st.slider("Max Google News RSS articles", min_value=5, max_value=30, value=10)
    use_news_fallback = st.checkbox(
        "Use demo news fallback if GDELT is rate-limited",
        value=True,
        help="Fallback rows are clearly marked as demo data and are only for presentation continuity.",
    )
    radius_km = st.slider("Cluster radius (km)", min_value=1, max_value=50, value=12)
    max_gap_hours = st.slider("Max time gap (hours)", min_value=1, max_value=120, value=36)
    run = st.button("Search Wildfire Events", type="primary")

area_name = custom_area.strip() if custom_area.strip() else region
area_label = f"{country} / {area_name}"

try:
    if custom_area.strip():
        geocoded_area = geocode_area(custom_area.strip(), country)
        bbox = geocoded_area["bbox"]
        area_label = geocoded_area["label"]
    else:
        bbox = regions[country]["regions"][region]
except Exception as exc:
    st.error(f"Could not resolve the selected area: {exc}")
    st.stop()

if "search_results" not in st.session_state:
    st.session_state.search_results = None

if not run and st.session_state.search_results is None:
    st.info("Select a country, region, and date range, then run a search.")
    st.stop()

if end_date < start_date:
    st.error("End date must be after start date.")
    st.stop()

if run:
    with st.spinner("Building wildfire event candidates..."):
        if demo_mode:
            detections = demo_detections(bbox, start_date)
            source_label = "Demo data"
            limitations = [
                "Demo mode uses synthetic detections for interface testing only.",
                "Demo mode intentionally returns a small fixed sample of event candidates for every selected area.",
                "Turn off demo mode to query real NASA FIRMS detections using the Streamlit secret key.",
            ]
            st.warning("Demo mode is on: event candidates are synthetic and will look similar across locations.")
        elif not nasa_key:
            st.error("NASA FIRMS key is missing. Add NASA_FIRMS_MAP_KEY in Streamlit app secrets or enable demo mode.")
            st.stop()
        else:
            source_to_query, source_note = effective_firms_source(source, end_date)
            if source_note:
                st.info(source_note)
            try:
                detections = fetch_firms(nasa_key, source_to_query, bbox, start_date, end_date)
            except Exception as exc:
                st.error(str(exc))
                st.info("You can enable demo mode to continue testing the interface while the NASA FIRMS request is fixed.")
                st.stop()
            source_label = f"NASA FIRMS ({source_to_query})"
            limitations = [
                "FIRMS reports active-fire and thermal-anomaly detections, not confirmed wildfire perimeters.",
                "Hotspot clusters are event candidates and require official or expert validation.",
            ]
        events = cluster_events(detections, radius_km, max_gap_hours)
        st.session_state.search_results = {
            "detections": detections,
            "events": events,
            "source_label": source_label,
            "limitations": limitations,
            "country": country,
            "area_name": area_name,
            "area_label": area_label,
            "bbox": bbox,
        }
        st.session_state.gdelt_results = {}
        st.session_state.google_news_results = {}
        st.session_state.article_text_results = {}
        st.session_state.gemini_results = {}

search_results = st.session_state.search_results
detections = search_results["detections"]
events = search_results["events"]
source_label = search_results["source_label"]
limitations = search_results["limitations"]
country = search_results["country"]
area_name = search_results["area_name"]
area_label = search_results["area_label"]
bbox = search_results["bbox"]

if not run:
    st.caption(f"Showing last search results for {area_label}. Press Search Wildfire Events to update the results.")

col1, col2, col3 = st.columns(3)
col1.metric("Detections", len(detections))
col2.metric("Event candidates", len(events))
col3.metric("Source", source_label)

if detections:
    st.subheader("Detection Map")
    st.map(pd.DataFrame([{"lat": d.lat, "lon": d.lon} for d in detections]), latitude="lat", longitude="lon")

st.subheader("Wildfire Event Candidates")
if not events:
    st.warning("No candidate events found.")
    st.stop()

events_df = pd.DataFrame(events)
st.dataframe(events_df, use_container_width=True, hide_index=True)

selected_id = st.selectbox("Select event to build digital memory", events_df["event_id"].tolist())
event = next(item for item in events if item["event_id"] == selected_id)

articles: list[dict[str, Any]] = []
gdelt_error = ""
gdelt_cache_key = (
    f"gdelt-v2|{country}|{area_name}|{selected_id}|{event['start']}|{event['end']}|"
    f"{gdelt_max_records}|fallback={use_news_fallback}"
)
google_news_cache_key = f"google-news-v1|{country}|{area_name}|{selected_id}|{google_news_max_records}"
if "gdelt_results" not in st.session_state:
    st.session_state.gdelt_results = {}
if "google_news_results" not in st.session_state:
    st.session_state.google_news_results = {}
if "article_text_results" not in st.session_state:
    st.session_state.article_text_results = {}
if "gemini_results" not in st.session_state:
    st.session_state.gemini_results = {}

if attach_gdelt or attach_google_news:
    st.subheader("News Evidence")
    st.caption("News sources are fetched only when you press a button, reducing rate-limit errors on Streamlit Cloud.")
    if st.button("Clear cached news evidence"):
        st.session_state.gdelt_results = {}
        st.session_state.google_news_results = {}
        st.session_state.article_text_results = {}
        st.info("Cleared cached news evidence for this session.")

if attach_google_news:
    if st.button("Fetch Google News RSS evidence for selected event"):
        with st.spinner("Collecting Google News RSS evidence for the selected event..."):
            try:
                google_articles = fetch_google_news_articles(country, area_name, google_news_max_records)
                st.session_state.google_news_results[google_news_cache_key] = {"articles": google_articles, "error": ""}
            except Exception as exc:
                st.session_state.google_news_results[google_news_cache_key] = {"articles": [], "error": str(exc)}

    cached_google_news = st.session_state.google_news_results.get(google_news_cache_key, {"articles": [], "error": ""})
    google_articles = cached_google_news["articles"]
    google_error = cached_google_news["error"]
    articles.extend(google_articles)
    if google_error:
        gdelt_error = f"{gdelt_error} | Google News RSS retrieval failed: {google_error}".strip(" |")

if attach_gdelt:
    if st.button("Fetch GDELT/news evidence for selected event"):
        gdelt_start, gdelt_end = event_date_window(event)
        with st.spinner("Collecting GDELT/news evidence for the selected event..."):
            try:
                fetched_articles = fetch_gdelt_articles(country, area_name, gdelt_start, gdelt_end, gdelt_max_records)
                st.session_state.gdelt_results[gdelt_cache_key] = {"articles": fetched_articles, "error": ""}
            except Exception as exc:  # Keep the memory record even if GDELT is temporarily unavailable.
                fallback_articles = demo_news_fallback(country, area_name, event) if use_news_fallback else []
                st.session_state.gdelt_results[gdelt_cache_key] = {"articles": fallback_articles, "error": str(exc)}

    cached_gdelt = st.session_state.gdelt_results.get(gdelt_cache_key, {"articles": [], "error": ""})
    articles.extend(cached_gdelt["articles"])
    if cached_gdelt["error"]:
        gdelt_error = f"{gdelt_error} | {cached_gdelt['error']}".strip(" |")
    if gdelt_error and use_news_fallback and not articles:
        articles = demo_news_fallback(country, area_name, event)
        st.session_state.gdelt_results[gdelt_cache_key] = {"articles": articles, "error": gdelt_error}

article_text_cache_key = (
    f"article-text-v1|{country}|{area_name}|{selected_id}|{event['start']}|{event['end']}|"
    f"{len(articles)}|limit={ARTICLE_FETCH_LIMIT}"
)
if attach_gdelt or attach_google_news:
    if articles:
        if st.button(f"Fetch full article excerpts for top {ARTICLE_FETCH_LIMIT} news rows"):
            with st.spinner("Fetching readable article excerpts for stronger evidence extraction..."):
                st.session_state.article_text_results[article_text_cache_key] = {
                    "articles": attach_article_excerpts(articles, ARTICLE_FETCH_LIMIT),
                    "error": "",
                }

        cached_article_text = st.session_state.article_text_results.get(
            article_text_cache_key, {"articles": [], "error": ""}
        )
        if cached_article_text["articles"]:
            articles = cached_article_text["articles"]
            excerpt_count = sum(1 for article in articles if article.get("article_excerpt"))
            st.success(f"Attached readable excerpts for {excerpt_count} article(s).")
        elif cached_article_text["error"]:
            st.warning(f"Article excerpt retrieval warning: {cached_article_text['error']}")

public_narrative = build_public_narrative(articles)
if articles and any(article.get("is_demo_fallback") for article in articles):
    public_narrative["source"] = "Demo news fallback after GDELT retrieval failure"
    public_narrative["is_demo_fallback"] = True
    public_narrative["confidence"] = min(public_narrative.get("confidence", 0.0), 0.35)
    public_narrative["limitations"] = public_narrative.get("limitations", []) + [
        "These fallback rows are not real news articles and must not be cited as external evidence.",
        "Use them only to demonstrate the Digital Memory structure when GDELT is rate-limited.",
    ]
if gdelt_error:
    if articles and any(article.get("is_demo_fallback") for article in articles):
        public_narrative["status"] = "GDELT retrieval failed; demo fallback active"
    else:
        public_narrative["status"] = "GDELT retrieval failed"
    public_narrative["retrieval_error"] = gdelt_error
    public_narrative["limitations"] = public_narrative.get("limitations", []) + [
        "GDELT could not be reached or returned an error during this run."
    ]

if attach_gdelt or attach_google_news:
    if articles:
        articles_df = pd.DataFrame(articles)
        if "is_demo_fallback" not in articles_df:
            articles_df["is_demo_fallback"] = False
        if "source_type" not in articles_df:
            articles_df["source_type"] = articles_df["domain"].apply(
                lambda domain: "Demo fallback" if domain == "demo-fallback.local" else "GDELT Doc API"
            )
        for column, default in {
            "article_fetch_status": "not requested",
            "article_excerpt": "",
            "resolved_url": "",
        }.items():
            if column not in articles_df:
                articles_df[column] = default
        st.dataframe(
            articles_df[
                [
                    "source_type",
                    "relevance",
                    "seen_date",
                    "domain",
                    "language",
                    "source_country",
                    "article_fetch_status",
                    "is_demo_fallback",
                    "title",
                    "url",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        excerpt_rows = [
            {
                "domain": article.get("domain", ""),
                "title": article.get("title", ""),
                "excerpt": article.get("article_excerpt", ""),
                "resolved_url": article.get("resolved_url", article.get("url", "")),
            }
            for article in articles
            if article.get("article_excerpt")
        ]
        if excerpt_rows:
            with st.expander("Readable article excerpts used for Gemini", expanded=False):
                st.dataframe(pd.DataFrame(excerpt_rows), use_container_width=True, hide_index=True)
        if any(article.get("is_demo_fallback") for article in articles):
            st.warning("Showing demo fallback rows because GDELT was rate-limited. These are not real news articles.")
    elif gdelt_error:
        st.warning(f"News retrieval warning: {gdelt_error}")
    else:
        st.info("No news articles attached yet. Press a fetch button above to collect news evidence.")

missing_data = [
    "Copernicus/EFFIS context is not attached yet.",
    "Preparedness gaps and lessons learned require the next LLM/RAG module.",
]
if not articles:
    missing_data.insert(0, "No GDELT/news narrative evidence is currently attached.")

memory_record = {
    "memory_id": f"{country[:2].upper()}-{area_name.replace(' ', '-').upper()}-{event['event_id']}",
    "event": {
        "type": "wildfire",
        "country": country,
        "region": area_name,
        "area_label": area_label,
        "start": event["start"],
        "end": event["end"],
        "center": {"lat": event["lat"], "lon": event["lon"]},
        "status": event["status"],
    },
    "satellite_evidence": {
        "source": source_label,
        "detections": event["detections"],
        "max_frp": event["max_frp"],
        "event_confidence_percent": event["confidence_percent"],
        "evidence_type": "active-fire detections + spatio-temporal clustering",
        "limitations": limitations,
    },
    "public_narrative": public_narrative,
    "evidence_sources": {
        "satellite": source_label,
        "public_news": {
            "gdelt": "GDELT Doc API" if attach_gdelt else "not requested",
            "google_news_rss": "Google News RSS" if attach_google_news else "not requested",
        },
        "official_context": "pending Copernicus/EFFIS connector",
        "llm_analysis": "Gemini API" if gemini_key else "pending Gemini API key",
    },
    "missing_data": missing_data,
}

gemini_cache_key = (
    f"gemini-v1|{memory_record['memory_id']}|{event['start']}|{event['end']}|"
    f"{len(articles)}|excerpts={public_narrative.get('article_texts_found', 0)}|{gemini_model}"
)
st.subheader("Gemini Memory Analysis")
st.caption(
    "Gemini turns the satellite event, news metadata, and readable article excerpts into lessons, gaps, early-action recommendations, "
    "and proposal-ready memory insights. It does not replace official validation."
)
if not gemini_key:
    st.info("Add GEMINI_API_KEY in Streamlit secrets to enable this step.")
elif not articles:
    st.warning("Fetch Google News RSS or GDELT evidence first, then generate Gemini analysis.")
else:
    col_generate, col_clear = st.columns([1, 3])
    with col_generate:
        generate_gemini = st.button("Generate Gemini memory analysis")
    with col_clear:
        if st.button("Clear cached Gemini analysis"):
            st.session_state.gemini_results.pop(gemini_cache_key, None)
            st.info("Cleared Gemini analysis for this event.")
            generate_gemini = False

    if generate_gemini:
        with st.spinner("Generating structured digital memory analysis with Gemini..."):
            try:
                st.session_state.gemini_results[gemini_cache_key] = {
                    "analysis": generate_gemini_memory_analysis(gemini_key, gemini_model, memory_record, articles),
                    "error": "",
                }
            except Exception as exc:
                st.session_state.gemini_results[gemini_cache_key] = {"analysis": {}, "error": str(exc)}

cached_gemini = st.session_state.gemini_results.get(gemini_cache_key, {"analysis": {}, "error": ""})
if cached_gemini.get("analysis"):
    memory_record["llm_analysis"] = cached_gemini["analysis"]
elif cached_gemini.get("error"):
    st.warning(f"Gemini analysis failed: {cached_gemini['error']}")

render_memory_record(memory_record)

with st.expander("Data provenance and confidence method", expanded=True):
    st.write(f"Area: {area_label}")
    st.write(f"Bounding box: {bbox}")
    st.write("Confidence formula: 55% average FIRMS confidence + 25% detection count score + 20% max FRP score.")
    for item in limitations:
        st.write(f"- {item}")
