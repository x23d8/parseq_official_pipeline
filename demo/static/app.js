const state = {
  methods: [],
  selectedMethod: null,
  file: null,
  previewUrl: null,
  busy: false,
};

const elements = {
  systemState: document.querySelector("#system-state"),
  systemLabel: document.querySelector("#system-label"),
  deviceLabel: document.querySelector("#device-label"),
  methodCount: document.querySelector("#method-count"),
  methodSearch: document.querySelector("#method-search"),
  methodList: document.querySelector("#method-list"),
  detailRank: document.querySelector("#detail-rank"),
  detailContent: document.querySelector("#detail-content"),
  dropZone: document.querySelector("#drop-zone"),
  dropEmpty: document.querySelector("#drop-empty"),
  fileInput: document.querySelector("#file-input"),
  fileState: document.querySelector("#file-state"),
  imageStage: document.querySelector("#image-stage"),
  imagePreview: document.querySelector("#image-preview"),
  imageName: document.querySelector("#image-name"),
  imageSize: document.querySelector("#image-size"),
  autoDetectToggle: document.querySelector("#auto-detect-toggle"),
  detectorState: document.querySelector("#detector-state"),
  detectButton: document.querySelector("#detect-button"),
  compareButton: document.querySelector("#compare-button"),
  resetButton: document.querySelector("#reset-button"),
  runStatusText: document.querySelector("#run-status-text"),
  resultsSection: document.querySelector("#results-section"),
  resultsEmpty: document.querySelector("#results-empty"),
  resultsOutput: document.querySelector("#results-output"),
  resultMode: document.querySelector("#result-mode"),
  toast: document.querySelector("#toast"),
  toastText: document.querySelector("#toast-text"),
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatPercent(value, digits = 2) {
  return `${(Number(value || 0) * 100).toFixed(digits)}%`;
}

function formatConfidence(value) {
  const percent = Number(value || 0) * 100;
  if (percent > 0 && percent < 0.01) return "<0.01%";
  return `${percent.toFixed(2)}%`;
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function updateModelStatus(model) {
  const detectorReady = Boolean(model?.detector?.checkpoint_exists);
  const systemReady = Boolean(model?.checkpoint_exists && detectorReady);
  elements.systemState.classList.toggle("ready", systemReady);
  elements.systemState.classList.toggle("error", !model?.checkpoint_exists);
  elements.systemLabel.textContent = model?.checkpoint_exists
    ? model.loaded ? "OCR loaded" : "System ready"
    : "Checkpoint missing";
  elements.deviceLabel.textContent = `${model?.device || "--"} · R${model?.refine_iters ?? "-"}`;
  elements.detectorState.textContent = detectorReady
    ? model.detector.loaded ? "YOLO26 loaded" : "YOLO26 ready"
    : "Detector unavailable";
  elements.detectorState.classList.toggle("unavailable", !detectorReady);
  elements.autoDetectToggle.disabled = !detectorReady;
  if (!detectorReady) elements.autoDetectToggle.checked = false;
}

function showToast(message) {
  elements.toastText.textContent = message;
  elements.toast.classList.remove("is-hidden");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => elements.toast.classList.add("is-hidden"), 5200);
}

function renderMethodList(query = "") {
  const normalized = query.trim().toLowerCase();
  const visible = state.methods.filter((method) =>
    `${method.display_name} ${method.name} ${method.topic}`.toLowerCase().includes(normalized)
  );
  elements.methodList.innerHTML = visible.length
    ? visible.map((method) => `
        <button class="method-item ${method.name === state.selectedMethod?.name ? "active" : ""}"
                data-method="${escapeHtml(method.name)}" role="option"
                aria-selected="${method.name === state.selectedMethod?.name}">
          <span class="method-rank">${String(method.rank).padStart(2, "0")}</span>
          <span>
            <span class="method-name">${escapeHtml(method.display_name)}</span>
            <span class="method-topic">${escapeHtml(method.topic)}</span>
          </span>
          <span class="method-score">
            <strong>${method.exact_acc ? formatPercent(method.exact_acc) : "REF"}</strong>
            <small class="${method.is_baseline ? "reference-text" : ""}">${method.is_baseline ? "baseline" : `+${(method.delta_exact * 100).toFixed(2)}`}</small>
          </span>
        </button>`).join("")
    : `<div class="results-empty"><p>No matching methods.</p></div>`;

  elements.methodList.querySelectorAll("[data-method]").forEach((button) => {
    button.addEventListener("click", () => selectMethod(button.dataset.method));
  });
}

function renderMethodDetail(method) {
  elements.detailRank.textContent = `#${String(method.rank).padStart(2, "0")}`;
  const speed = method.images_per_second ? `${method.images_per_second.toFixed(0)} img/s` : "Reference";
  elements.detailContent.innerHTML = `
    <div class="method-identity">
      <h3>${escapeHtml(method.display_name)}</h3>
      <p>${escapeHtml(method.description)}</p>
    </div>
    <div class="metric-grid">
      <div class="metric-card"><span>Exact match</span><strong>${method.exact_acc ? formatPercent(method.exact_acc) : "REF"}</strong></div>
      <div class="metric-card"><span>Character acc.</span><strong>${method.char_acc ? formatPercent(method.char_acc) : "REF"}</strong></div>
      <div class="metric-card"><span>Exact delta</span><strong class="${method.delta_exact > 0 ? "positive" : ""}">${method.delta_exact > 0 ? "+" : ""}${(method.delta_exact * 100).toFixed(2)} pp</strong></div>
      <div class="metric-card"><span>Benchmark speed</span><strong>${speed}</strong></div>
    </div>
    <div class="detail-block">
      <span class="detail-label">Why it helps</span>
      <p>${escapeHtml(method.impact_reason)}</p>
    </div>
    <div class="detail-block">
      <span class="detail-label">Processing path</span>
      <ul class="pipeline-list">${method.pipeline_steps.map((step) => `<li>${escapeHtml(step)}</li>`).join("")}</ul>
    </div>
    <div class="detail-block source-row">
      <span>MEASURED SET</span><strong>${method.samples || "--"} samples</strong>
    </div>
    <div class="detail-block source-row">
      <span>CONFIG ID</span><strong>${escapeHtml(method.name)}</strong>
    </div>`;
}

function selectMethod(name) {
  const method = state.methods.find((item) => item.name === name);
  if (!method) return;
  state.selectedMethod = method;
  renderMethodList(elements.methodSearch.value);
  renderMethodDetail(method);
  elements.runStatusText.textContent = `Selected ${method.display_name}. Ready for inference.`;
}

function acceptFile(file) {
  if (!file?.type?.startsWith("image/")) {
    showToast("Please select a valid image file.");
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showToast("The image exceeds the 10 MB upload limit.");
    return;
  }
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.file = file;
  state.previewUrl = URL.createObjectURL(file);
  elements.imagePreview.src = state.previewUrl;
  elements.imageName.textContent = file.name;
  elements.imageSize.textContent = formatBytes(file.size);
  elements.fileState.textContent = "Input locked";
  elements.dropEmpty.classList.add("is-hidden");
  elements.imageStage.classList.remove("is-hidden");
  elements.detectButton.disabled = false;
  elements.compareButton.disabled = false;
  elements.runStatusText.textContent = "Image decoded locally. Select an inference action.";
}

function setBusy(mode, busy) {
  state.busy = busy;
  elements.detectButton.disabled = busy || !state.file;
  elements.compareButton.disabled = busy || !state.file;
  elements.resetButton.disabled = busy;
  elements.detectButton.classList.toggle("busy", busy && mode === "detect");
  elements.compareButton.classList.toggle("busy", busy && mode === "compare");
}

async function apiRequest(endpoint, formData) {
  const response = await fetch(endpoint, { method: "POST", body: formData });
  let payload;
  try { payload = await response.json(); } catch { payload = {}; }
  if (!response.ok) throw new Error(payload.detail || `Request failed with status ${response.status}.`);
  return payload;
}

function showResults(html, mode) {
  elements.resultsEmpty.classList.add("is-hidden");
  elements.resultsOutput.classList.remove("is-hidden");
  elements.resultsOutput.innerHTML = html;
  elements.resultMode.textContent = mode;
  elements.resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderSingle(payload) {
  const result = payload.result;
  const benchmark = result.benchmark;
  const detection = result.detection;
  const confidence = Math.max(0, Math.min(100, result.confidence * 100));
  const detectionLabel = detection.detected
    ? `${escapeHtml(detection.class_name)} · ${formatConfidence(detection.confidence)}`
    : detection.enabled ? "NO PLATE / FALLBACK" : "MANUAL CROP";
  showResults(`
    <div class="single-result">
      <div class="visual-comparison">
        <article class="visual-card">
          <header><span>Input scene</span><span>${detection.detected ? "Located" : "Original"}</span></header>
          <div class="visual-frame"><img src="${detection.annotated_image}" alt="Input image with plate detection" /></div>
        </article>
        <article class="visual-card">
          <header><span>OCR crop</span><span>${detection.crop_size.join("×")}</span></header>
          <div class="visual-frame"><img src="${detection.crop_image}" alt="Detected license plate crop" /></div>
        </article>
        <article class="visual-card">
          <header><span>Processed signal</span><span>${escapeHtml(result.runtime_method)}</span></header>
          <div class="visual-frame"><img src="${result.processed_image}" alt="Preprocessed plate" /></div>
        </article>
      </div>
      <article class="recognition-card">
        <div class="recognition-top">
          <span>PARSEQ / AUTOREGRESSIVE / REFINE ${payload.model.refine_iters}</span>
          <span>${escapeHtml(benchmark.display_name)}</span>
        </div>
        <div class="recognition-output">
          <span>Recognized sequence</span>
          <div class="plate-text">${escapeHtml(result.prediction || "NO TEXT")}</div>
          <div class="confidence-track" title="Model confidence"><i style="width:${confidence}%"></i></div>
        </div>
        <div class="recognition-stats">
          <div><span>Confidence</span><strong>${formatConfidence(result.confidence)}</strong></div>
          <div><span>Plate detector</span><strong>${detectionLabel}</strong></div>
          <div><span>Total latency</span><strong>${result.total_ms.toFixed(1)} ms</strong></div>
          <div><span>Benchmark exact</span><strong>${formatPercent(benchmark.exact_acc)}</strong></div>
        </div>
      </article>
    </div>`, "SINGLE DETECT");
}

function renderComparison(payload) {
  const detection = payload.detection;
  const bestConfidence = Math.max(...payload.results.map((item) => Number(item.confidence || 0)));
  const cards = payload.results.map((result) => {
    const benchmark = result.benchmark;
    const isBest = result.confidence === bestConfidence;
    return `
      <article class="comparison-card ${isBest ? "best-output" : ""}">
        ${isBest ? '<span class="best-tag">TOP CONF.</span>' : ""}
        <div class="comparison-thumb"><img src="${result.processed_image}" alt="${escapeHtml(benchmark.display_name)} output" /></div>
        <div class="comparison-body">
          <span class="comparison-rank">#${String(benchmark.rank).padStart(2, "0")} · ${formatPercent(benchmark.exact_acc)}</span>
          <div class="comparison-method" title="${escapeHtml(benchmark.display_name)}">${escapeHtml(benchmark.display_name)}</div>
          <div class="comparison-prediction">${escapeHtml(result.prediction || "NO TEXT")}</div>
          <div class="comparison-meta"><span>CONF ${formatConfidence(result.confidence)}</span><span>PRE ${result.preprocessing_ms.toFixed(1)} MS</span></div>
        </div>
      </article>`;
  }).join("");
  showResults(`
    <div class="compare-detection">
      <div class="detection-preview"><img src="${detection.annotated_image}" alt="Located plate in input image" /></div>
      <div class="detection-preview"><img src="${detection.crop_image}" alt="Detected plate crop used for comparison" /></div>
      <div class="detection-data">
        <span>Shared OCR input</span>
        <strong>${detection.detected ? `${escapeHtml(detection.class_name)} · ${formatConfidence(detection.confidence)}` : detection.enabled ? "No detection — original fallback" : "Manual crop mode"}</strong>
        <small>BBOX ${detection.bbox.join(", ")} · ${detection.detection_ms.toFixed(1)} MS</small>
      </div>
    </div>
    <div class="compare-summary">
      <p><strong>${payload.method_count} pipelines</strong> processed from one image and decoded in one model batch.</p>
      <span>BATCH INFERENCE ${payload.model_batch_ms.toFixed(1)} MS</span>
    </div>
    <div class="comparison-grid">${cards}</div>`, "COMPARE MATRIX");
}

async function runDetect() {
  if (!state.file || !state.selectedMethod || state.busy) return;
  const form = new FormData();
  form.append("file", state.file);
  form.append("method", state.selectedMethod.name);
  form.append("auto_detect", String(elements.autoDetectToggle.checked));
  setBusy("detect", true);
  elements.runStatusText.textContent = "Loading model and executing selected preprocessing pipeline...";
  try {
    const payload = await apiRequest("/api/detect", form);
    updateModelStatus(payload.model);
    renderSingle(payload);
    elements.runStatusText.textContent = `Detection complete: ${payload.result.prediction || "no text returned"}.`;
  } catch (error) {
    showToast(error.message);
    elements.runStatusText.textContent = "Detection failed. Inspect the error notification.";
  } finally {
    setBusy("detect", false);
  }
}

async function runCompare() {
  if (!state.file || state.busy) return;
  const form = new FormData();
  form.append("file", state.file);
  form.append("auto_detect", String(elements.autoDetectToggle.checked));
  setBusy("compare", true);
  elements.runStatusText.textContent = `Preparing ${state.methods.length} pipelines for batched comparison...`;
  try {
    const payload = await apiRequest("/api/compare", form);
    updateModelStatus(payload.model);
    renderComparison(payload);
    elements.runStatusText.textContent = `Comparison complete across ${payload.method_count} methods.`;
  } catch (error) {
    showToast(error.message);
    elements.runStatusText.textContent = "Comparison failed. Inspect the error notification.";
  } finally {
    setBusy("compare", false);
  }
}

function resetWorkspace() {
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.file = null;
  state.previewUrl = null;
  elements.fileInput.value = "";
  elements.imagePreview.removeAttribute("src");
  elements.dropEmpty.classList.remove("is-hidden");
  elements.imageStage.classList.add("is-hidden");
  elements.fileState.textContent = "No input";
  elements.detectButton.disabled = true;
  elements.compareButton.disabled = true;
  elements.resultsEmpty.classList.remove("is-hidden");
  elements.resultsOutput.classList.add("is-hidden");
  elements.resultsOutput.innerHTML = "";
  elements.resultMode.textContent = "IDLE";
  elements.runStatusText.textContent = "Waiting for an input image.";
}

async function initialize() {
  try {
    const response = await fetch("/api/methods");
    if (!response.ok) throw new Error("Could not load preprocessing catalog.");
    const payload = await response.json();
    state.methods = payload.methods;
    elements.methodCount.textContent = `${state.methods.length} cfg`;
    updateModelStatus(payload.model);
    renderMethodList();
    selectMethod(payload.default_method);
  } catch (error) {
    elements.systemState.classList.add("error");
    elements.systemLabel.textContent = "API offline";
    elements.methodList.innerHTML = '<div class="results-empty"><p>Catalog unavailable.</p></div>';
    showToast(error.message);
  }
}

elements.methodSearch.addEventListener("input", (event) => renderMethodList(event.target.value));
elements.dropZone.addEventListener("click", () => elements.fileInput.click());
elements.dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    elements.fileInput.click();
  }
});
elements.fileInput.addEventListener("change", (event) => acceptFile(event.target.files[0]));
["dragenter", "dragover"].forEach((name) => elements.dropZone.addEventListener(name, (event) => {
  event.preventDefault();
  elements.dropZone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((name) => elements.dropZone.addEventListener(name, (event) => {
  event.preventDefault();
  elements.dropZone.classList.remove("dragging");
}));
elements.dropZone.addEventListener("drop", (event) => acceptFile(event.dataTransfer.files[0]));
elements.detectButton.addEventListener("click", runDetect);
elements.compareButton.addEventListener("click", runCompare);
elements.resetButton.addEventListener("click", resetWorkspace);
elements.autoDetectToggle.addEventListener("change", () => {
  elements.runStatusText.textContent = elements.autoDetectToggle.checked
    ? "Auto plate location enabled. Full-scene and cropped inputs are supported."
    : "Manual crop mode enabled. The complete uploaded image will be sent to OCR.";
});

initialize();
