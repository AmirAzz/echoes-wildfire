const state = {
  regions: {},
  events: [],
  provenance: null,
  methodology: null,
  selectedEvent: null,
};

const els = {
  country: document.querySelector("#country"),
  region: document.querySelector("#region"),
  startDate: document.querySelector("#startDate"),
  endDate: document.querySelector("#endDate"),
  source: document.querySelector("#source"),
  nasaKey: document.querySelector("#nasaKey"),
  demoMode: document.querySelector("#demoMode"),
  radiusKm: document.querySelector("#radiusKm"),
  maxGapHours: document.querySelector("#maxGapHours"),
  searchBtn: document.querySelector("#searchBtn"),
  runStatus: document.querySelector("#runStatus"),
  detectionsMetric: document.querySelector("#detectionsMetric"),
  eventsMetric: document.querySelector("#eventsMetric"),
  sourceMetric: document.querySelector("#sourceMetric"),
  bboxLabel: document.querySelector("#bboxLabel"),
  eventsTable: document.querySelector("#eventsTable"),
  memoryPreview: document.querySelector("#memoryPreview"),
  selectedEventLabel: document.querySelector("#selectedEventLabel"),
  provenanceBody: document.querySelector("#provenanceBody"),
  canvas: document.querySelector("#mapCanvas"),
};

function setStatus(text, tone = "ready") {
  els.runStatus.textContent = text;
  els.runStatus.style.background = tone === "error" ? "#fff0ed" : tone === "busy" ? "#fff8e8" : "#edf6f2";
  els.runStatus.style.color = tone === "error" ? "#a23321" : tone === "busy" ? "#8a520f" : "#236f63";
}

function option(label, value = label) {
  const item = document.createElement("option");
  item.value = value;
  item.textContent = label;
  return item;
}

function populateCountries() {
  els.country.innerHTML = "";
  Object.keys(state.regions).forEach((country) => els.country.append(option(country)));
  els.country.value = "Cyprus";
  populateRegions();
}

function populateRegions() {
  const country = els.country.value;
  const regionNames = Object.keys(state.regions[country]?.regions || {});
  els.region.innerHTML = "";
  regionNames.forEach((region) => els.region.append(option(region)));
  if (country === "Cyprus" && regionNames.includes("Limassol")) {
    els.region.value = "Limassol";
  }
}

function eventBadge(event) {
  if (event.event_confidence >= 75) return '<span class="badge">High</span>';
  if (event.event_confidence >= 55) return '<span class="badge warn">Medium</span>';
  return '<span class="badge low">Low</span>';
}

function renderEvents() {
  els.eventsMetric.textContent = String(state.events.length);
  if (!state.events.length) {
    els.eventsTable.className = "events-empty";
    els.eventsTable.innerHTML = "No candidate events found for this search.";
    return;
  }

  els.eventsTable.className = "";
  els.eventsTable.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Event</th>
          <th>Period</th>
          <th>Detections</th>
          <th>Max FRP</th>
          <th>Confidence</th>
          <th>Status</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        ${state.events.map((event) => `
          <tr>
            <td><strong>${event.event_id}</strong><br>${event.center.lat}, ${event.center.lon}</td>
            <td>${formatDateTime(event.start)}<br>${formatDateTime(event.end)}</td>
            <td>${event.detections}</td>
            <td>${event.max_frp}</td>
            <td>${event.event_confidence}%<br>${eventBadge(event)}</td>
            <td>${event.status}</td>
            <td><button class="mini-btn" data-event="${event.event_id}">Build Digital Memory</button></td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
  els.eventsTable.querySelectorAll("button[data-event]").forEach((button) => {
    button.addEventListener("click", () => buildMemory(button.dataset.event));
  });
}

function formatDateTime(value) {
  return value.replace("T", " ").slice(0, 16);
}

function renderProvenance() {
  if (!state.provenance) {
    els.provenanceBody.textContent = "No data loaded yet.";
    return;
  }
  const p = state.provenance;
  const m = state.methodology || {};
  els.provenanceBody.innerHTML = `
    <strong>Source:</strong> ${p.source || "-"}<br>
    <strong>Retrieved at:</strong> ${p.retrieved_at || "-"}<br>
    <strong>Area:</strong> ${p.bbox_label || "-"} ${p.bbox ? `(${p.bbox.join(", ")})` : ""}<br>
    <strong>Method:</strong> ${m.event_builder || "-"}<br>
    <strong>Confidence formula:</strong> ${m.confidence_formula || "-"}
    <ul>
      ${(p.limitations || []).map((item) => `<li>${item}</li>`).join("")}
    </ul>
  `;
}

function getBbox() {
  const country = els.country.value;
  const region = els.region.value;
  return state.regions[country]?.regions?.[region] || state.regions[country]?.bbox || [0, 0, 1, 1];
}

function drawMap() {
  const canvas = els.canvas;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const bbox = state.provenance?.bbox || getBbox();
  const [west, south, east, north] = bbox;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#e8eee9";
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = "#c9d3cc";
  ctx.lineWidth = 1;
  for (let i = 1; i < 8; i++) {
    const x = (w / 8) * i;
    const y = (h / 6) * i;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
    if (i < 6) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }
  }

  ctx.fillStyle = "#617069";
  ctx.font = "14px Segoe UI, Arial";
  ctx.fillText(`${els.country.value} / ${els.region.value}`, 18, 28);
  ctx.fillText(`bbox ${west}, ${south}, ${east}, ${north}`, 18, 50);

  const project = (lat, lon) => {
    const x = ((lon - west) / (east - west || 1)) * (w - 60) + 30;
    const y = (1 - (lat - south) / (north - south || 1)) * (h - 80) + 60;
    return [x, y];
  };

  state.events.forEach((event) => {
    (event.detections_sample || []).forEach((detection) => {
      const [x, y] = project(detection.lat, detection.lon);
      ctx.fillStyle = event.event_confidence >= 75 ? "rgba(182,66,45,0.38)" : event.event_confidence >= 55 ? "rgba(214,140,47,0.38)" : "rgba(124,135,129,0.38)";
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fill();
    });
  });

  state.events.forEach((event) => {
    const [x, y] = project(event.center.lat, event.center.lon);
    ctx.fillStyle = event.event_confidence >= 75 ? "#b6422d" : event.event_confidence >= 55 ? "#d68c2f" : "#7c8781";
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(x, y, 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#1e2923";
    ctx.font = "700 13px Segoe UI, Arial";
    ctx.fillText(event.event_id, x + 13, y + 5);
  });
}

function buildMemory(eventId) {
  const event = state.events.find((item) => item.event_id === eventId);
  if (!event) return;
  state.selectedEvent = event;
  els.selectedEventLabel.textContent = event.event_id;

  const memoryRecord = {
    memory_id: `${els.country.value.slice(0, 2).toUpperCase()}-${els.region.value.replace(/\s+/g, "-").toUpperCase()}-${event.event_id}`,
    event: {
      type: "wildfire",
      country: els.country.value,
      region: els.region.value,
      start: event.start,
      end: event.end,
      status: event.status,
      center: event.center,
    },
    satellite_evidence: {
      source: state.provenance?.source || "NASA FIRMS",
      detections: event.detections,
      max_frp: event.max_frp,
      avg_detection_confidence: event.avg_confidence,
      event_confidence_percent: event.event_confidence,
      evidence_type: event.evidence_type,
      limitations: state.provenance?.limitations || [],
    },
    current_analysis_scope: {
      included_now: ["NASA FIRMS active-fire detections", "prototype spatio-temporal clustering", "provenance and limitations"],
      not_yet_included: ["GDELT/news narratives", "Copernicus/EFFIS burned area or fire danger", "OpenStreetMap exposed assets", "LLM lessons learned"],
    },
    missing_data: [
      "No public news evidence has been attached yet.",
      "No official burned-area or evacuation data has been attached yet.",
      "Preparedness gaps and lessons learned require the next LLM/RAG module and human validation.",
    ],
    next_step: "Attach GDELT and official reports, then run LLM-based memory extraction with citations.",
  };

  els.memoryPreview.className = "";
  els.memoryPreview.innerHTML = `<pre class="memory-json">${escapeHtml(JSON.stringify(memoryRecord, null, 2))}</pre>`;
}

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  }[char]));
}

async function searchEvents() {
  setStatus("Searching", "busy");
  els.searchBtn.disabled = true;
  try {
    const params = new URLSearchParams({
      country: els.country.value,
      region: els.region.value,
      startDate: els.startDate.value,
      endDate: els.endDate.value,
      source: els.source.value,
      radiusKm: els.radiusKm.value,
      maxGapHours: els.maxGapHours.value,
    });
    if (els.demoMode.checked || !els.nasaKey.value.trim()) {
      params.set("demo", "true");
    } else {
      params.set("nasaKey", els.nasaKey.value.trim());
    }

    const response = await fetch(`/api/fires/search?${params.toString()}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Search failed");

    state.events = data.events || [];
    state.provenance = data.provenance || null;
    state.methodology = data.methodology || null;
    state.selectedEvent = null;
    els.detectionsMetric.textContent = String(data.detections_count || 0);
    els.sourceMetric.textContent = state.provenance?.source || "-";
    els.bboxLabel.textContent = state.provenance?.bbox_label || "";
    els.memoryPreview.className = "memory-empty";
    els.memoryPreview.textContent = "Select an event and build a first memory record. GDELT, Copernicus, OSM, and LLM analysis will attach here in the next module.";
    els.selectedEventLabel.textContent = "No event selected";
    renderEvents();
    renderProvenance();
    drawMap();
    setStatus("Done");
  } catch (error) {
    setStatus("Error", "error");
    els.eventsTable.className = "events-empty";
    els.eventsTable.textContent = error.message;
  } finally {
    els.searchBtn.disabled = false;
  }
}

async function init() {
  const response = await fetch("/api/regions");
  state.regions = await response.json();
  populateCountries();
  drawMap();
  els.country.addEventListener("change", () => {
    populateRegions();
    drawMap();
  });
  els.region.addEventListener("change", drawMap);
  els.searchBtn.addEventListener("click", searchEvents);
}

init().catch((error) => {
  setStatus("Error", "error");
  els.eventsTable.textContent = error.message;
});
