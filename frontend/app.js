const state = {
  history: [],
};

const els = {
  modelStatus: document.querySelector("#modelStatus"),
  modelName: document.querySelector("#modelName"),
  classList: document.querySelector("#classList"),
  historyList: document.querySelector("#historyList"),
  fileInput: document.querySelector("#fileInput"),
  dropzone: document.querySelector("#dropzone"),
  previewWrap: document.querySelector("#previewWrap"),
  previewImage: document.querySelector("#previewImage"),
  decisionPill: document.querySelector("#decisionPill"),
  defectStatus: document.querySelector("#defectStatus"),
  detectedClass: document.querySelector("#detectedClass"),
  confidence: document.querySelector("#confidence"),
  qualityScore: document.querySelector("#qualityScore"),
  confidenceBar: document.querySelector("#confidenceBar"),
  recommendationText: document.querySelector("#recommendationText"),
  gradcamWrap: document.querySelector("#gradcamWrap"),
  probabilityList: document.querySelector("#probabilityList"),
  curvesArtifact: document.querySelector("#curvesArtifact"),
  matrixArtifact: document.querySelector("#matrixArtifact"),
  binaryMatrixArtifact: document.querySelector("#binaryMatrixArtifact"),
};

function pretty(value) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function recommendation(result) {
  if (result.needs_review) {
    return "Route sample for manual review; the model does not have enough confidence to approve it.";
  }
  if (result.predicted_class === "clean") {
    return "Approve sample and continue standard monitoring.";
  }
  if (result.decision_rank >= 2) {
    return "Reject or isolate batch, then trigger manual quality review.";
  }
  return "Route sample for secondary inspection before release.";
}

async function loadModel() {
  const response = await fetch("/api/model");
  const model = await response.json();
  if (!model.ready) {
    els.modelStatus.textContent = "Not ready";
    els.modelName.textContent = model.message;
    return;
  }
  els.modelStatus.textContent = "Ready";
  els.modelStatus.className = "pill ready";
  els.modelName.textContent = model.model_name;
  els.classList.innerHTML = model.classes
    .map((name) => `<span class="class-chip">${pretty(name)}</span>`)
    .join("");
}

async function loadArtifacts() {
  const response = await fetch("/api/artifacts");
  const artifacts = await response.json();
  if (artifacts.training_curves) {
    els.curvesArtifact.innerHTML = '<img src="/api/artifacts/training-curves" alt="Training curves" />';
  }
  if (artifacts.confusion_matrix) {
    els.matrixArtifact.innerHTML = '<img src="/api/artifacts/confusion-matrix" alt="Confusion matrix" />';
  }
  if (artifacts.binary_confusion_matrix) {
    els.binaryMatrixArtifact.innerHTML = '<img src="/api/artifacts/binary-confusion-matrix" alt="Defect detection confusion matrix" />';
  }
}

function renderProbabilities(probabilities) {
  const rows = Object.entries(probabilities)
    .sort((a, b) => b[1] - a[1])
    .map(
      ([name, value]) => `
        <div class="probability-row">
          <div class="probability-head">
            <span>${pretty(name)}</span>
            <strong>${(value * 100).toFixed(2)}%</strong>
          </div>
          <div class="probability-bar"><span style="width:${value * 100}%"></span></div>
        </div>
      `,
    );
  els.probabilityList.innerHTML = rows.join("");
}

function renderHistory() {
  if (!state.history.length) {
    els.historyList.innerHTML = '<p class="empty">No inspections yet.</p>';
    return;
  }
  els.historyList.innerHTML = state.history
    .slice(0, 5)
    .map(
      (item) => `
        <div class="history-item">
          <strong>${pretty(item.predicted_class)}</strong>
          <span>${(item.confidence * 100).toFixed(2)}% confidence · ${item.decision}</span>
        </div>
      `,
    )
    .join("");
}

function renderResult(result) {
  els.decisionPill.textContent = result.decision;
  els.decisionPill.className = `pill ${result.decision.toLowerCase()}`;
  els.defectStatus.textContent = result.defect_detected ? "Defect" : "Clean";
  els.detectedClass.textContent = result.display_class;
  els.confidence.textContent = `${(result.confidence * 100).toFixed(2)}%`;
  els.qualityScore.textContent = `${result.quality_score}/100`;
  els.confidenceBar.style.width = `${result.confidence * 100}%`;
  els.recommendationText.textContent = recommendation(result);
  els.gradcamWrap.innerHTML = `<img src="${result.gradcam_url}" alt="Grad-CAM overlay" />`;
  renderProbabilities(result.probabilities);
  state.history.unshift(result);
  renderHistory();
}

async function inspect(file) {
  const data = new FormData();
  data.append("file", file);
  const response = await fetch("/api/predict", {
    method: "POST",
    body: data,
  });
  if (!response.ok) {
    const error = await response.json();
    els.recommendationText.textContent = error.detail || "Inspection failed.";
    return;
  }
  renderResult(await response.json());
}

function preview(file) {
  const reader = new FileReader();
  reader.onload = () => {
    els.previewImage.src = reader.result;
    els.previewWrap.classList.remove("hidden");
  };
  reader.readAsDataURL(file);
}

els.fileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;
  preview(file);
  inspect(file);
});

["dragenter", "dragover"].forEach((eventName) => {
  els.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropzone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  els.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropzone.classList.remove("dragging");
  });
});

els.dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (!file) return;
  preview(file);
  inspect(file);
});

loadModel();
loadArtifacts();
