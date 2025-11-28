const API_BASE = window.location.origin + "/api";

const modelSelect = document.getElementById("modelSelect");
const confInput = document.getElementById("confInput");
const iouInput = document.getElementById("iouInput");
const saveBtn = document.getElementById("saveBtn");
const detectBtn = document.getElementById("detectBtn");
const statusEl = document.getElementById("status");
const resultBox = document.getElementById("resultBox");

let models = [];
let currentModel = null;

async function loadModels() {
  try {
    const res = await fetch(API_BASE + "/models");
    const data = await res.json();
    models = data;
    modelSelect.innerHTML = "";
    data.forEach(m => {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = `${m.id}`;
      modelSelect.appendChild(opt);
    });
    if (data.length) {
      currentModel = data[0];
      modelSelect.value = currentModel.id;
      confInput.value = currentModel.conf;
      iouInput.value = currentModel.iou;
    }
  } catch (e) {
    statusEl.textContent = "Error cargando modelos";
  }
}

modelSelect.addEventListener("change", () => {
  const id = modelSelect.value;
  const m = models.find(x => x.id === id);
  currentModel = m;
  if (m) {
    confInput.value = m.conf;
    iouInput.value = m.iou;
  }
});

saveBtn.addEventListener("click", async () => {
  if (!currentModel) return;
  const id = currentModel.id;
  const body = {
    conf: parseFloat(confInput.value),
    iou: parseFloat(iouInput.value),
  };
  statusEl.textContent = "Guardando...";
  try {
    const res = await fetch(API_BASE + "/models/" + id, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (data.success) {
      statusEl.textContent = "Guardado";
    } else {
      statusEl.textContent = "Error: " + data.message;
    }
  } catch (e) {
    statusEl.textContent = "Error guardando modelo";
  }
});

detectBtn.addEventListener("click", async () => {
  if (!currentModel) return;
  const id = currentModel.id;
  statusEl.textContent = "Ejecutando detección...";
  resultBox.textContent = "Esperando...";
  try {
    const res = await fetch(API_BASE + "/detect/" + id, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    const data = await res.json();
    if (data.success) {
      statusEl.textContent = "OK";
    } else {
      statusEl.textContent = "Error: " + data.message;
    }
    resultBox.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    statusEl.textContent = "Error llamando a la API";
    resultBox.textContent = "Error";
  }
});

loadModels();
