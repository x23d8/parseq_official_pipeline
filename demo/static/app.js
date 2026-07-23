const state = {
  methods: [],
  catalogExpanded: false,
  selectedMethod: null,
  pipeline: [],
  maxPipelineSteps: 5,
  draggedPipelineIndex: null,
  file: null,
  previewUrl: null,
  busy: false,
  methodFilters: new Set(["imp", "rl"]),
  theme: localStorage.getItem("parseq-theme") || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"),
};

const elements = {
  systemState: document.querySelector("#system-state"),
  systemLabel: document.querySelector("#system-label"),
  deviceLabel: document.querySelector("#device-label"),
  methodCount: document.querySelector("#method-count"),
  methodSearch: document.querySelector("#method-search"),
  methodFilters: [...document.querySelectorAll(".method-filters input")],
  themeToggle: document.querySelector("#theme-toggle"),
  methodList: document.querySelector("#method-list"),
  pipelineCanvas: document.querySelector("#pipeline-canvas"),
  pipelineEmpty: document.querySelector("#pipeline-empty"),
  pipelineCount: document.querySelector("#pipeline-count"),
  clearPipeline: document.querySelector("#clear-pipeline"),
  detailRank: document.querySelector("#detail-rank"),
  detailContent: document.querySelector("#detail-content"),
  dropZone: document.querySelector("#drop-zone"),
  dropEmpty: document.querySelector("#drop-empty"),
  fileInput: document.querySelector("#file-input"),
  browseButton: document.querySelector("#browse-button"),
  pasteButton: document.querySelector("#paste-button"),
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

const METHOD_PREVIEW_LIMIT = 10;

function canAddMethod(method) {
  if (!method?.available || state.pipeline.includes(method.name)) return false;
  if (method.exclusive) return true;
  const currentExclusive = state.pipeline.some((name) =>
    state.methods.find((item) => item.name === name)?.exclusive
  );
  return currentExclusive || state.pipeline.length < state.maxPipelineSteps;
}

function methodItemMarkup(method) {
  const catalogBadge = method.catalog_badge || (method.experimental ? "RL AGENT" : "");
  return `
    <div class="method-item ${method.name === state.selectedMethod?.name ? "active" : ""} ${!method.available ? "unavailable" : ""}"
         data-method="${escapeHtml(method.name)}" role="option" tabindex="0"
         draggable="${Boolean(method.available && method.composable)}"
         aria-selected="${method.name === state.selectedMethod?.name}"
         aria-disabled="${!method.available}">
      <span class="method-rank">${String(method.rank).padStart(2, "0")}</span>
      <span>
        <span class="method-name">${escapeHtml(method.display_name)}</span>
        <span class="method-topic">${catalogBadge ? `<b>${escapeHtml(catalogBadge)}</b> · ` : ""}${escapeHtml(method.topic)}</span>
      </span>
      <span class="method-score">
        <strong>${method.available ? (method.benchmark_available ? formatPercent(method.exact_acc) : "N/A") : "OFF"}</strong>
        <small class="${!method.benchmark_available ? "reference-text" : ""}">${method.available ? (method.benchmark_available ? `${method.delta_exact > 0 ? "+" : ""}${(method.delta_exact * 100).toFixed(2)}` : "unmeasured") : "missing"}</small>
      </span>
      <button class="method-add" type="button" data-add-method="${escapeHtml(method.name)}"
              title="Add ${escapeHtml(method.display_name)} to pipeline"
              aria-label="Add ${escapeHtml(method.display_name)} to pipeline"
              ${!canAddMethod(method) ? "disabled" : ""}>+</button>
    </div>`;
}

function renderMethodList(query = "") {
  const normalized = query.trim().toLowerCase();
  const visible = state.methods.filter((method) =>
    state.methodFilters.has(method.filter_group || "imp") &&
    `${method.display_name} ${method.name} ${method.topic}`.toLowerCase().includes(normalized)
  );
  elements.methodCount.textContent = `${visible.length} / ${state.methods.length}`;
  if (!visible.length) {
    elements.methodList.innerHTML = `<div class="results-empty"><p>No matching methods.</p></div>`;
  } else if (normalized) {
    elements.methodList.innerHTML = `
      <div class="method-section-heading search-heading">
        <strong>Search results</strong><span>${visible.length} matching blocks</span>
      </div>
      ${visible.map(methodItemMarkup).join("")}`;
  } else {
    const displayOrder = [
      ...visible.filter((method) => method.featured),
      ...visible.filter((method) => !method.featured),
    ];
    const topMethods = displayOrder.slice(0, METHOD_PREVIEW_LIMIT);
    const remainingMethods = displayOrder.slice(METHOD_PREVIEW_LIMIT);
    const moreSection = state.catalogExpanded
      ? `<section class="method-more-section" aria-label="More preprocessing methods">
          <div class="method-section-heading">
            <strong>More methods</strong><span>${remainingMethods.length} reusable blocks</span>
          </div>
          ${remainingMethods.map(methodItemMarkup).join("")}
        </section>`
      : "";
    elements.methodList.innerHTML = `
      <div class="method-section-heading top-heading">
        <strong>Featured + top methods</strong><span>Recovery shortcut, then validation rank</span>
      </div>
      ${topMethods.map(methodItemMarkup).join("")}
      ${remainingMethods.length ? `
        <button class="catalog-expand-button" type="button" data-toggle-catalog
                aria-expanded="${state.catalogExpanded}">
          <span>${state.catalogExpanded ? "Show top 10 only" : `Show more (${remainingMethods.length})`}</span>
          <b aria-hidden="true">${state.catalogExpanded ? "↑" : "↓"}</b>
        </button>` : ""}
      ${moreSection}`;
  }

  elements.methodList.querySelector("[data-toggle-catalog]")?.addEventListener("click", () => {
    state.catalogExpanded = !state.catalogExpanded;
    renderMethodList();
    if (!state.catalogExpanded) elements.methodList.scrollTo({ top: 0, behavior: "smooth" });
  });

  elements.methodList.querySelectorAll("[data-method]").forEach((button) => {
    button.addEventListener("click", () => selectMethod(button.dataset.method));
    button.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectMethod(button.dataset.method);
      }
    });
    button.addEventListener("dragstart", (event) => {
      if (button.getAttribute("aria-disabled") === "true") {
        event.preventDefault();
        return;
      }
      event.dataTransfer.effectAllowed = "copy";
      event.dataTransfer.setData("application/x-parseq-method", button.dataset.method);
      button.classList.add("drag-source");
    });
    button.addEventListener("dragend", () => button.classList.remove("drag-source"));
  });
  elements.methodList.querySelectorAll("[data-add-method]").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      addMethodToPipeline(button.dataset.addMethod);
    });
  });
}

function pipelineDisplayName() {
  return state.pipeline
    .map((name) => state.methods.find((method) => method.name === name)?.display_name || name)
    .join(" → ");
}

function renderPipeline() {
  elements.pipelineCount.textContent = `${state.pipeline.length} / ${state.maxPipelineSteps} steps`;
  elements.clearPipeline.disabled = state.pipeline.length === 0 || state.busy;
  if (!state.pipeline.length) {
    elements.pipelineCanvas.innerHTML = `
      <div class="pipeline-empty" id="pipeline-empty">
        <strong>Drag methods here</strong>
        <span>or use the + button. Execution follows the order from left to right.</span>
      </div>`;
  } else {
    elements.pipelineCanvas.innerHTML = state.pipeline.map((name, index) => {
      const method = state.methods.find((item) => item.name === name);
      return `
        ${index ? '<span class="pipeline-arrow" aria-hidden="true">→</span>' : ''}
        <article class="pipeline-chip ${method?.experimental ? "rl-chip" : ""}" draggable="true"
                 data-pipeline-index="${index}" role="listitem">
          <button class="chip-main" type="button" data-focus-method="${escapeHtml(name)}">
            <span class="chip-index">${String(index + 1).padStart(2, "0")}</span>
            <span><strong>${escapeHtml(method?.display_name || name)}</strong><small>${escapeHtml(method?.topic || "")}</small></span>
          </button>
          <div class="chip-actions">
            <button type="button" data-move-left="${index}" aria-label="Move step left" ${index === 0 ? "disabled" : ""}>←</button>
            <button type="button" data-move-right="${index}" aria-label="Move step right" ${index === state.pipeline.length - 1 ? "disabled" : ""}>→</button>
            <button type="button" data-remove-step="${index}" aria-label="Remove step">×</button>
          </div>
        </article>`;
    }).join("");
  }
  elements.detectButton.disabled = state.busy || !state.file || !state.pipeline.length;
  renderMethodList(elements.methodSearch.value);

  elements.pipelineCanvas.querySelectorAll("[data-focus-method]").forEach((button) => {
    button.addEventListener("click", () => selectMethod(button.dataset.focusMethod));
  });
  elements.pipelineCanvas.querySelectorAll("[data-remove-step]").forEach((button) => {
    button.addEventListener("click", () => removePipelineStep(Number(button.dataset.removeStep)));
  });
  elements.pipelineCanvas.querySelectorAll("[data-move-left]").forEach((button) => {
    button.addEventListener("click", () => movePipelineStep(Number(button.dataset.moveLeft), -1));
  });
  elements.pipelineCanvas.querySelectorAll("[data-move-right]").forEach((button) => {
    button.addEventListener("click", () => movePipelineStep(Number(button.dataset.moveRight), 1));
  });
  elements.pipelineCanvas.querySelectorAll("[data-pipeline-index]").forEach((chip) => {
    chip.addEventListener("dragstart", (event) => {
      state.draggedPipelineIndex = Number(chip.dataset.pipelineIndex);
      event.dataTransfer.effectAllowed = "move";
      event.dataTransfer.setData("application/x-parseq-pipeline-index", chip.dataset.pipelineIndex);
      chip.classList.add("drag-source");
    });
    chip.addEventListener("dragend", () => {
      state.draggedPipelineIndex = null;
      chip.classList.remove("drag-source");
      elements.pipelineCanvas.classList.remove("drag-over");
    });
  });
}

function addMethodToPipeline(name, insertAt = state.pipeline.length) {
  const method = state.methods.find((item) => item.name === name);
  if (!method?.available) {
    showToast(method?.unavailable_reason || "This method is unavailable.");
    return;
  }
  if (state.pipeline.includes(name)) {
    showToast("A method can appear only once in a pipeline.");
    return;
  }
  const currentExclusive = state.pipeline.some((pipelineName) =>
    state.methods.find((item) => item.name === pipelineName)?.exclusive
  );
  if (method.exclusive) {
    state.pipeline = [name];
    selectMethod(name);
    renderPipeline();
    elements.runStatusText.textContent = `${method.display_name} is a complete selector and will run alone.`;
    return;
  }
  if (currentExclusive) state.pipeline = [];
  if (state.pipeline.length >= state.maxPipelineSteps) {
    showToast(`A pipeline can contain at most ${state.maxPipelineSteps} methods.`);
    return;
  }
  const safeIndex = Math.max(0, Math.min(Number(insertAt), state.pipeline.length));
  state.pipeline.splice(safeIndex, 0, name);
  selectMethod(name);
  renderPipeline();
  elements.runStatusText.textContent = `Pipeline ready: ${pipelineDisplayName()}.`;
}

function removePipelineStep(index) {
  state.pipeline.splice(index, 1);
  renderPipeline();
  elements.runStatusText.textContent = state.pipeline.length
    ? `Pipeline updated: ${pipelineDisplayName()}.`
    : "Pipeline is empty. Add at least one processing method.";
}

function movePipelineStep(index, delta) {
  const target = index + delta;
  if (target < 0 || target >= state.pipeline.length) return;
  [state.pipeline[index], state.pipeline[target]] = [state.pipeline[target], state.pipeline[index]];
  renderPipeline();
  elements.runStatusText.textContent = `Execution order updated: ${pipelineDisplayName()}.`;
}

function renderMethodDetail(method) {
  elements.detailRank.textContent = `#${String(method.rank).padStart(2, "0")}`;
  const speed = method.benchmark_available && method.images_per_second ? `${method.images_per_second.toFixed(0)} img/s` : "N/A";
  const exact = method.benchmark_available ? formatPercent(method.exact_acc) : "N/A";
  const character = method.benchmark_available ? formatPercent(method.char_acc) : "N/A";
  const delta = method.benchmark_available
    ? `${method.delta_exact > 0 ? "+" : ""}${(method.delta_exact * 100).toFixed(2)} pp`
    : "N/A";
  const availability = method.available
    ? `<span class="availability ready">${method.experimental ? "EXPERIMENT READY" : "AVAILABLE"}</span>`
    : `<span class="availability unavailable">UNAVAILABLE</span>`;
  const experimentalLabel = method.experimental_label || (method.experimental ? "REINFORCEMENT LEARNING" : "");
  elements.detailContent.innerHTML = `
    <div class="method-identity">
      <div class="identity-flags">${availability}${experimentalLabel ? `<span class="availability experimental">${escapeHtml(experimentalLabel)}</span>` : ""}</div>
      <h3>${escapeHtml(method.display_name)}</h3>
      <p>${escapeHtml(method.description)}</p>
      ${!method.available ? `<p class="unavailable-copy">${escapeHtml(method.unavailable_reason)}</p>` : ""}
      <button class="detail-add-button" id="detail-add-button" type="button"
              ${!canAddMethod(method) ? "disabled" : ""}>
        ${state.pipeline.includes(method.name) ? "Already in pipeline" : "Add to pipeline"}
      </button>
    </div>
    <div class="metric-grid">
      <div class="metric-card"><span>Exact match</span><strong>${exact}</strong></div>
      <div class="metric-card"><span>Character acc.</span><strong>${character}</strong></div>
      <div class="metric-card"><span>Exact delta</span><strong class="${method.delta_exact > 0 ? "positive" : ""}">${delta}</strong></div>
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
      <span>MEASURED SET</span><strong>${method.benchmark_available ? `${method.samples} samples` : "Not benchmarked alone"}</strong>
    </div>
    <div class="detail-block source-row">
      <span>CONFIG ID</span><strong>${escapeHtml(method.name)}</strong>
    </div>`;
  document.querySelector("#detail-add-button")?.addEventListener("click", () => addMethodToPipeline(method.name));
}

function selectMethod(name) {
  const method = state.methods.find((item) => item.name === name);
  if (!method) return;
  state.selectedMethod = method;
  renderMethodList(elements.methodSearch.value);
  renderMethodDetail(method);
  elements.runStatusText.textContent = state.pipeline.includes(method.name)
    ? `${method.display_name} is in the pipeline. Drag its chip to change execution order.`
    : `Inspecting ${method.display_name}. Add it to the pipeline to run inference.`;
}

function clipboardFile(blob) {
  const type = blob.type || "image/png";
  const extensions = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/bmp": "bmp",
    "image/gif": "gif",
  };
  const extension = extensions[type] || "png";
  return new File([blob], `clipboard-image.${extension}`, { type, lastModified: Date.now() });
}

function acceptFile(file, source = "local") {
  if (state.busy) {
    showToast("Wait for the current inference request to finish before changing the image.");
    return;
  }
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
  elements.fileState.textContent = source === "clipboard" ? "Pasted input" : "Local input";
  elements.dropEmpty.classList.add("is-hidden");
  elements.imageStage.classList.remove("is-hidden");
  elements.detectButton.disabled = !state.pipeline.length;
  elements.compareButton.disabled = false;
  elements.runStatusText.textContent = source === "clipboard"
    ? "Clipboard image decoded locally. Select an inference action."
    : "Image decoded locally. Select an inference action.";
}

async function readClipboardImage() {
  if (state.busy) return;
  if (!navigator.clipboard?.read) {
    showToast("Direct clipboard access is unavailable here. Copy an image, then press Ctrl or Cmd + V.");
    return;
  }
  try {
    const clipboardItems = await navigator.clipboard.read();
    for (const item of clipboardItems) {
      const imageType = item.types.find((type) => type.startsWith("image/"));
      if (!imageType) continue;
      const blob = await item.getType(imageType);
      acceptFile(clipboardFile(blob), "clipboard");
      return;
    }
    showToast("The clipboard does not contain an image.");
  } catch (error) {
    const denied = error?.name === "NotAllowedError";
    showToast(denied
      ? "Clipboard permission was denied. Allow access or press Ctrl or Cmd + V."
      : "Could not read an image from the clipboard.");
  }
}

function setBusy(mode, busy) {
  state.busy = busy;
  elements.detectButton.disabled = busy || !state.file || !state.pipeline.length;
  elements.compareButton.disabled = busy || !state.file;
  elements.resetButton.disabled = busy;
  elements.fileInput.disabled = busy;
  elements.browseButton.disabled = busy;
  elements.pasteButton.disabled = busy;
  elements.dropZone.setAttribute("aria-disabled", String(busy));
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
  const benchmarkExact = benchmark.benchmark_available ? formatPercent(benchmark.exact_acc) : "Not benchmarked";
  const pipelineTrace = result.step_timings.map((step) => `
    <li><span>${String(step.position).padStart(2, "0")}</span><strong>${escapeHtml(step.runtime_method)}</strong><small>${step.milliseconds.toFixed(1)} ms</small></li>
  `).join("");
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
          <div><span>Benchmark exact</span><strong>${benchmarkExact}</strong></div>
        </div>
        <div class="executed-pipeline">
          <span>Executed left → right</span>
          <ol>${pipelineTrace}</ol>
          ${benchmark.benchmark_available ? "" : "<p>Exploratory composition — evaluate it on a locked validation set before comparing accuracy.</p>"}
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
  if (!state.file || !state.pipeline.length || state.busy) return;
  const form = new FormData();
  form.append("file", state.file);
  form.append("pipeline", JSON.stringify(state.pipeline));
  form.append("auto_detect", String(elements.autoDetectToggle.checked));
  setBusy("detect", true);
  elements.runStatusText.textContent = `Executing ${state.pipeline.length} step(s) from left to right...`;
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
  const comparisonCount = state.methods.filter((method) => method.available && method.comparison_eligible).length;
  elements.runStatusText.textContent = `Preparing ${comparisonCount} single-method pipelines for batched comparison...`;
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

function applyTheme(theme) {
  state.theme = theme === "dark" ? "dark" : "light";
  document.documentElement.dataset.theme = state.theme;
  localStorage.setItem("parseq-theme", state.theme);
  const dark = state.theme === "dark";
  elements.themeToggle.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
  elements.themeToggle.title = dark ? "Light mode" : "Dark mode";
}

async function initialize() {
  try {
    const response = await fetch("/api/methods");
    if (!response.ok) throw new Error("Could not load preprocessing catalog.");
    const payload = await response.json();
    state.methods = payload.methods;
    state.maxPipelineSteps = Number(payload.max_pipeline_steps || 5);
    updateModelStatus(payload.model);
    renderMethodList();
    selectMethod(payload.default_method);
    addMethodToPipeline(payload.default_method);
  } catch (error) {
    elements.systemState.classList.add("error");
    elements.systemLabel.textContent = "API offline";
    elements.methodList.innerHTML = '<div class="results-empty"><p>Catalog unavailable.</p></div>';
    showToast(error.message);
  }
}

function pipelineDropIndex(clientX) {
  const chips = [...elements.pipelineCanvas.querySelectorAll("[data-pipeline-index]")];
  for (let index = 0; index < chips.length; index += 1) {
    const rect = chips[index].getBoundingClientRect();
    if (clientX < rect.left + rect.width / 2) return index;
  }
  return state.pipeline.length;
}

elements.methodSearch.addEventListener("input", (event) => renderMethodList(event.target.value));
elements.methodFilters.forEach((input) => input.addEventListener("change", () => {
  if (input.checked) state.methodFilters.add(input.value);
  else state.methodFilters.delete(input.value);
  state.catalogExpanded = false;
  renderMethodList(elements.methodSearch.value);
}));
elements.themeToggle.addEventListener("click", () => applyTheme(state.theme === "dark" ? "light" : "dark"));
elements.clearPipeline.addEventListener("click", () => {
  state.pipeline = [];
  renderPipeline();
  elements.runStatusText.textContent = "Pipeline cleared. Add at least one processing method.";
});
elements.pipelineCanvas.addEventListener("dragover", (event) => {
  const types = [...event.dataTransfer.types];
  if (!types.includes("application/x-parseq-method") && !types.includes("application/x-parseq-pipeline-index")) return;
  event.preventDefault();
  event.dataTransfer.dropEffect = types.includes("application/x-parseq-pipeline-index") ? "move" : "copy";
  elements.pipelineCanvas.classList.add("drag-over");
});
elements.pipelineCanvas.addEventListener("dragleave", (event) => {
  if (!elements.pipelineCanvas.contains(event.relatedTarget)) elements.pipelineCanvas.classList.remove("drag-over");
});
elements.pipelineCanvas.addEventListener("drop", (event) => {
  event.preventDefault();
  elements.pipelineCanvas.classList.remove("drag-over");
  const insertAt = pipelineDropIndex(event.clientX);
  const sourceIndexValue = event.dataTransfer.getData("application/x-parseq-pipeline-index");
  if (sourceIndexValue !== "") {
    const sourceIndex = Number(sourceIndexValue);
    const [name] = state.pipeline.splice(sourceIndex, 1);
    const adjustedIndex = sourceIndex < insertAt ? insertAt - 1 : insertAt;
    state.pipeline.splice(Math.max(0, Math.min(adjustedIndex, state.pipeline.length)), 0, name);
    renderPipeline();
    elements.runStatusText.textContent = `Execution order updated: ${pipelineDisplayName()}.`;
    return;
  }
  const methodName = event.dataTransfer.getData("application/x-parseq-method");
  if (methodName) addMethodToPipeline(methodName, insertAt);
});
elements.browseButton.addEventListener("click", () => elements.fileInput.click());
elements.pasteButton.addEventListener("click", readClipboardImage);
elements.dropZone.addEventListener("click", () => {
  if (!state.busy) elements.fileInput.click();
});
elements.dropZone.addEventListener("keydown", (event) => {
  if (!state.busy && (event.key === "Enter" || event.key === " ")) {
    event.preventDefault();
    elements.fileInput.click();
  }
});
elements.fileInput.addEventListener("change", (event) => acceptFile(event.target.files[0], "local"));
["dragenter", "dragover"].forEach((name) => elements.dropZone.addEventListener(name, (event) => {
  event.preventDefault();
  elements.dropZone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((name) => elements.dropZone.addEventListener(name, (event) => {
  event.preventDefault();
  elements.dropZone.classList.remove("dragging");
}));
elements.dropZone.addEventListener("drop", (event) => acceptFile(event.dataTransfer.files[0], "local"));
document.addEventListener("paste", (event) => {
  if (state.busy) return;
  const imageItem = [...(event.clipboardData?.items || [])]
    .find((item) => item.kind === "file" && item.type.startsWith("image/"));
  if (!imageItem) return;
  const blob = imageItem.getAsFile();
  if (!blob) return;
  event.preventDefault();
  acceptFile(clipboardFile(blob), "clipboard");
});
elements.detectButton.addEventListener("click", runDetect);
elements.compareButton.addEventListener("click", runCompare);
elements.resetButton.addEventListener("click", resetWorkspace);
elements.autoDetectToggle.addEventListener("change", () => {
  elements.runStatusText.textContent = elements.autoDetectToggle.checked
    ? "Auto plate location enabled. Full-scene and cropped inputs are supported."
    : "Manual crop mode enabled. The complete uploaded image will be sent to OCR.";
});

applyTheme(state.theme);
initialize();
