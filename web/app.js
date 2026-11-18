"use strict";

(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const SVG_NS = "http://www.w3.org/2000/svg";
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

  const store = {
    get: (key) => { try { return localStorage.getItem(key); } catch { return null; } },
    set: (key, value) => { try { localStorage.setItem(key, value); } catch { /* storage unavailable */ } },
    del: (key) => { try { localStorage.removeItem(key); } catch { /* storage unavailable */ } },
  };

  const state = {
    file: null, previewUrl: null, reference: null, busy: false,
    token: store.get("clarity.token"), username: store.get("clarity.user"),
    limitMb: 12, mode: "slider", dragging: false, raf: 0, timer: 0,
    blobs: [], historyBlobs: [],
  };

  const studio = $(".studio");
  const panels = { empty: $("#panel-empty"), ready: $("#panel-ready"), busy: $("#panel-busy"), result: $("#panel-result") };
  const el = {
    alert: $("#alert"), alertText: $("#alert-text"),
    drop: $("#drop"), file: $("#file"),
    readyImg: $("#ready-img"), busyImg: $("#busy-img"), fileName: $("#file-name"), fileMeta: $("#file-meta"),
    engineRow: $("#engine-row"), engine: $("#engine"), engineDesc: $("#engine-desc"),
    refInput: $("#ref-input"), refName: $("#ref-name"), refClear: $("#ref-clear"),
    busyTime: $("#busy-time"), busyHint: $("#busy-hint"),
    compare: $("#compare"), stage: $("#stage"), handle: $("#handle"),
    imgBefore: $("#img-before"), imgAfter: $("#img-after"), toolHint: $("#tool-hint"),
    metrics: $("#metrics"), chips: $("#chips"),
    download: $("#btn-download"), fullsize: $("#btn-fullsize"), nudge: $("#guest-nudge"),
    history: $("#history"), historyGrid: $("#history-grid"), historyEmpty: $("#history-empty"),
    auth: $("#auth"), authForm: $("#auth-form"), authUser: $("#auth-user"), authPass: $("#auth-pass"),
    authError: $("#auth-error"), authSubmit: $("#auth-submit"), authTitle: $("#auth-title"),
    authHint: $("#auth-hint"), authRule: $("#auth-rule"),
  };

  /* ---------------------------------------------------------------- helpers */

  function h(tag, attrs = {}, ...kids) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value);
    }
    node.append(...kids);
    return node;
  }

  function icon(name, cls = "i") {
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("class", cls);
    const use = document.createElementNS(SVG_NS, "use");
    use.setAttribute("href", `#i-${name}`);
    svg.append(use);
    return svg;
  }

  const fmtBytes = (n) => (n < 1048576 ? `${Math.max(1, Math.round(n / 1024))} KB` : `${(n / 1048576).toFixed(1)} MB`);
  const fmtDate = (iso) => new Date(iso).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  const signed = (v, digits) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(digits)}`;
  const revoke = (url) => { if (url) URL.revokeObjectURL(url); };

  async function api(path, { method, body, json } = {}) {
    const headers = {};
    if (state.token) headers.Authorization = `Bearer ${state.token}`;
    let payload = body;
    if (json) { headers["Content-Type"] = "application/json"; payload = JSON.stringify(json); }
    let res;
    try {
      res = await fetch(path, { method: method || (payload ? "POST" : "GET"), headers, body: payload });
    } catch {
      throw new Error("Can't reach the server. Check your connection and try again.");
    }
    if (!res.ok) {
      let message = "Something went wrong. Please try again.";
      try {
        const data = await res.json();
        if (typeof data.detail === "string") message = data.detail;
        else if (Array.isArray(data.detail)) message = "Please check what you entered.";
      } catch { /* non-JSON error body */ }
      throw Object.assign(new Error(message), { status: res.status });
    }
    return res.json();
  }

  async function fetchBlobUrl(path) {
    const res = await fetch(path, { headers: state.token ? { Authorization: `Bearer ${state.token}` } : {} });
    if (!res.ok) throw new Error("That image is no longer available.");
    return URL.createObjectURL(await res.blob());
  }

  async function fetchFile(url, name) {
    const res = await fetch(url);
    if (!res.ok) throw new Error("Couldn't load the example photo.");
    const blob = await res.blob();
    return new File([blob], name, { type: blob.type || "image/png" });
  }

  /* ------------------------------------------------------------------ stages */

  function setStage(name) {
    for (const [key, panel] of Object.entries(panels)) panel.hidden = key !== name;
    studio.dataset.stage = name;
  }

  function showAlert(message) {
    el.alertText.textContent = message;
    el.alert.hidden = false;
  }
  const hideAlert = () => { el.alert.hidden = true; };
  $("#alert-close").addEventListener("click", hideAlert);

  function keepStudioInView() {
    const rect = studio.getBoundingClientRect();
    if (rect.top < 70 || rect.bottom > innerHeight) {
      scrollTo({ top: scrollY + rect.top - 84, behavior: reduceMotion ? "auto" : "smooth" });
    }
  }

  /* ----------------------------------------------------------- choosing files */

  const isImage = (file) => file && (/^image\//.test(file.type) || /\.(jpe?g|png|webp|bmp|gif)$/i.test(file.name));

  function acceptFile(file) {
    hideAlert();
    if (!isImage(file)) return showAlert("That doesn't look like an image. Please choose a JPG, PNG or WebP photo.");
    if (file.size > state.limitMb * 1048576) {
      return showAlert(`That file is ${fmtBytes(file.size)}. Please choose one under ${state.limitMb} MB.`);
    }
    setFile(file, null);
  }

  function setFile(file, reference) {
    revoke(state.previewUrl);
    state.file = file;
    state.previewUrl = URL.createObjectURL(file);
    setReference(reference);
    el.readyImg.src = state.previewUrl;
    el.busyImg.src = state.previewUrl;
    el.fileName.textContent = file.name || "Pasted image";
    el.fileMeta.textContent = fmtBytes(file.size);
    el.readyImg.onload = () => {
      el.fileMeta.textContent = `${el.readyImg.naturalWidth} × ${el.readyImg.naturalHeight} px · ${fmtBytes(file.size)}`;
    };
    setStage("ready");
  }

  function setReference(file, label) {
    state.reference = file;
    el.refName.textContent = file ? (label || file.name) : "";
    el.refClear.hidden = !file;
  }

  function resetToEmpty() {
    hideAlert();
    revoke(state.previewUrl);
    state.file = null;
    state.previewUrl = null;
    setReference(null);
    state.blobs.splice(0).forEach(revoke);
    setStage("empty");
  }

  el.drop.addEventListener("click", () => el.file.click());
  el.drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.file.click(); }
  });
  el.file.addEventListener("change", () => { acceptFile(el.file.files[0]); el.file.value = ""; });
  $("#btn-change").addEventListener("click", resetToEmpty);
  $("#btn-again").addEventListener("click", resetToEmpty);

  const hasFiles = (e) => e.dataTransfer && [...e.dataTransfer.types].includes("Files");
  for (const type of ["dragenter", "dragover"]) {
    addEventListener(type, (e) => { if (hasFiles(e)) { e.preventDefault(); el.drop.classList.add("over"); } });
  }
  addEventListener("dragleave", (e) => { if (!e.relatedTarget) el.drop.classList.remove("over"); });
  addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    el.drop.classList.remove("over");
    acceptFile(e.dataTransfer.files[0]);
  });
  document.addEventListener("paste", (e) => {
    const file = [...(e.clipboardData?.files || [])].find(isImage);
    if (file) { e.preventDefault(); acceptFile(file); }
  });

  $("#ref-pick").addEventListener("click", () => el.refInput.click());
  el.refInput.addEventListener("change", () => {
    const file = el.refInput.files[0];
    el.refInput.value = "";
    if (!file) return;
    if (!isImage(file)) return showAlert("The clean original must be an image file.");
    setReference(file);
  });
  el.refClear.addEventListener("click", () => setReference(null));

  for (const button of $$(".sample")) {
    button.addEventListener("click", async () => {
      if (state.busy) return;
      const name = button.dataset.sample;
      try {
        const [photo, clean] = await Promise.all([
          fetchFile(`/static/samples/${name}.png`, `${name}-example.png`),
          fetchFile(`/static/samples/${name}-clean.png`, `${name}-clean.png`),
        ]);
        setFile(photo, null);
        setReference(clean, "Clean original of this example");
        restore();
      } catch (err) {
        showAlert(err.message);
      }
    });
  }

  /* ---------------------------------------------------------------- restoring */

  function startTimer() {
    const started = performance.now();
    el.busyTime.textContent = "0 s";
    el.busyHint.textContent = "This usually takes a few seconds.";
    state.timer = setInterval(() => {
      const seconds = Math.floor((performance.now() - started) / 1000);
      el.busyTime.textContent = `${seconds} s`;
      if (seconds >= 25) el.busyHint.textContent = "Still working. The server may be busy with other photos.";
      else if (seconds >= 6) el.busyHint.textContent = "Larger photos take a little longer.";
    }, 500);
  }

  async function restore() {
    if (!state.file || state.busy) return;
    state.busy = true;
    hideAlert();
    setStage("busy");
    startTimer();
    const form = new FormData();
    form.append("image", state.file, state.file.name || "photo.png");
    if (state.reference) form.append("reference", state.reference, state.reference.name || "reference.png");
    form.append("engine", el.engine.value || "unet");
    try {
      const result = await api("/api/restore", { method: "POST", body: form });
      await showResult(result, {
        before: result.input_url || state.previewUrl,
        after: result.output_url,
        download: result.download_url,
      });
      if (state.token) loadHistory();
    } catch (err) {
      setStage("ready");
      showAlert(err.message);
    } finally {
      state.busy = false;
      clearInterval(state.timer);
    }
  }
  $("#btn-restore").addEventListener("click", restore);

  /* ------------------------------------------------------------------- result */

  async function showResult(result, src) {
    el.imgBefore.src = src.before;
    el.imgAfter.src = src.after;
    await Promise.all([el.imgBefore.decode(), el.imgAfter.decode()]).catch(() => {});
    const ratio = el.imgAfter.naturalWidth && el.imgAfter.naturalHeight
      ? el.imgAfter.naturalWidth / el.imgAfter.naturalHeight : 1.5;
    el.compare.style.setProperty("--ar", ratio.toFixed(4));
    el.download.href = src.download;
    if (src.downloadName) el.download.setAttribute("download", src.downloadName);
    else el.download.setAttribute("download", "");
    el.fullsize.href = src.after;
    el.nudge.hidden = Boolean(state.token) || Boolean(result.fromHistory);
    renderMetrics(result);
    setMode("slider");
    setStage("result");
    revealSlider();
    keepStudioInView();
  }

  function metricCard(label, value, gain, was) {
    return h("div", { class: "metric" },
      h("h4", { text: label }),
      h("div", { class: "big" }, h("b", { text: value }), h("span", { class: "gain", text: gain })),
      h("p", { text: was }));
  }

  function chip(text, cls = "", iconName) {
    return h("span", { class: `chip ${cls}`.trim() }, ...(iconName ? [icon(iconName, "i i-sm")] : []), text);
  }

  function renderMetrics(result) {
    el.metrics.replaceChildren();
    el.chips.replaceChildren();
    const m = result.metrics;
    el.metrics.hidden = !m;
    if (m) {
      el.metrics.append(
        metricCard("PSNR (higher is better)", `${m.psnr.toFixed(1)} dB`, `${signed(m.psnr - m.input_psnr, 1)} dB`, `Before: ${m.input_psnr.toFixed(1)} dB`),
        metricCard("SSIM (closer to 1 is better)", m.ssim.toFixed(2), signed(m.ssim - m.input_ssim, 2), `Before: ${m.input_ssim.toFixed(2)}`),
      );
    }
    if (result.seconds != null) el.chips.append(chip(`Done in ${result.seconds} s`, "", "clock"));
    el.chips.append(chip(`${result.width} × ${result.height} px`));
    if (result.fromHistory && result.psnr != null) el.chips.append(chip(`PSNR ${result.psnr.toFixed(1)} dB vs. clean original`));
    if (result.resized) el.chips.append(chip("Large photo scaled down to keep things fast", "info"));
    if (result.note) el.chips.append(chip(result.note, "info"));
    if (!m && !result.fromHistory) el.chips.append(chip("Tip: add a clean original under Advanced options to see quality scores", "info"));
  }

  /* ------------------------------------------------------------ compare slider */

  function setPos(percent) {
    const value = Math.max(0, Math.min(100, percent));
    el.compare.style.setProperty("--pos", `${value}%`);
    el.handle.setAttribute("aria-valuenow", String(Math.round(value)));
  }

  function revealSlider() {
    cancelAnimationFrame(state.raf);
    if (reduceMotion) return setPos(50);
    const from = 96, to = 50, duration = 1000;
    let t0;
    const step = (t) => {
      t0 ??= t;
      const k = Math.min(1, (t - t0) / duration);
      setPos(from + (to - from) * (1 - Math.pow(1 - k, 3)));
      if (k < 1) state.raf = requestAnimationFrame(step);
    };
    setPos(from);
    state.raf = requestAnimationFrame(step);
  }

  const HINTS = { slider: "Drag the handle to compare", side: "Original on the left, restored on the right", after: "Showing the restored photo" };

  function setMode(mode) {
    state.mode = mode;
    el.compare.dataset.mode = mode;
    loupeHide();
    el.toolHint.textContent = HINTS[mode];
    for (const button of $$(".seg button[data-mode]")) button.setAttribute("aria-selected", String(button.dataset.mode === mode));
    if (mode === "slider") setPos(50);
  }
  for (const button of $$(".seg button[data-mode]")) button.addEventListener("click", () => setMode(button.dataset.mode));

  function moveTo(e) {
    const rect = el.stage.getBoundingClientRect();
    setPos(((e.clientX - rect.left) / rect.width) * 100);
  }
  el.stage.addEventListener("pointerdown", (e) => {
    if (state.mode !== "slider") return;
    cancelAnimationFrame(state.raf);
    state.dragging = true;
    el.stage.setPointerCapture(e.pointerId);
    moveTo(e);
  });
  el.stage.addEventListener("pointermove", (e) => { if (state.dragging) moveTo(e); });
  for (const type of ["pointerup", "pointercancel"]) el.stage.addEventListener(type, () => { state.dragging = false; });
  el.handle.addEventListener("keydown", (e) => {
    const now = Number(el.handle.getAttribute("aria-valuenow"));
    const steps = { ArrowLeft: -3, ArrowRight: 3, PageDown: -10, PageUp: 10 };
    if (e.key in steps) setPos(now + steps[e.key]);
    else if (e.key === "Home") setPos(0);
    else if (e.key === "End") setPos(100);
    else return;
    e.preventDefault();
    cancelAnimationFrame(state.raf);
  });

  /* --------------------------------------------------------------- magnifier */

  // Hold Alt over the comparison to inspect the same point before and after at
  // pixel level. Gated on Alt so it never fights with dragging the slider.
  const LOUPE_SIZE = 184; // keep in sync with .loupe in styles.css
  const loupe = { box: $("#loupe"), before: $("#loupe-before"), after: $("#loupe-after"), zoom: 2.6, shown: false };

  function loupeHide() {
    if (!loupe.shown) return;
    loupe.shown = false;
    loupe.box.hidden = true;
  }

  function loupeMove(event) {
    if (state.mode === "side") return loupeHide();
    const rect = el.stage.getBoundingClientRect();
    const fx = (event.clientX - rect.left) / rect.width;
    const fy = (event.clientY - rect.top) / rect.height;
    if (fx < 0 || fx > 1 || fy < 0 || fy > 1) return loupeHide();

    // The stage's aspect ratio matches the photo, so object-fit: cover shows it
    // whole and the cursor fraction maps straight onto the image.
    const bw = rect.width * loupe.zoom;
    const bh = rect.height * loupe.zoom;
    for (const pane of [loupe.before, loupe.after]) {
      pane.style.backgroundSize = `${bw}px ${bh}px`;
      pane.style.backgroundPosition = `${LOUPE_SIZE / 2 - fx * bw}px ${LOUPE_SIZE / 2 - fy * bh}px`;
    }
    loupe.box.style.left = `${event.clientX - rect.left}px`;
    loupe.box.style.top = `${event.clientY - rect.top}px`;

    if (!loupe.shown) {
      loupe.before.style.backgroundImage = `url("${el.imgBefore.src}")`;
      loupe.after.style.backgroundImage = `url("${el.imgAfter.src}")`;
      loupe.box.hidden = false;
      loupe.shown = true;
    }
  }

  if (loupe.box) {
    el.stage.addEventListener("pointermove", (e) => {
      if (e.altKey && !state.dragging) loupeMove(e);
      else loupeHide();
    });
    el.stage.addEventListener("pointerleave", loupeHide);
    addEventListener("keyup", (e) => { if (e.key === "Alt") loupeHide(); });
    addEventListener("blur", loupeHide);
  }

  /* ------------------------------------------------------------------ engines */

  function fillEngines(engines) {
    el.engine.replaceChildren(...engines.map((e) => h("option", { value: e.id, text: e.label })));
    el.engineRow.hidden = engines.length < 2;
    const describe = () => { el.engineDesc.textContent = (engines.find((e) => e.id === el.engine.value) || {}).description || ""; };
    el.engine.onchange = describe;
    describe();
  }

  /* ------------------------------------------------------------------ account */

  function renderAuth() {
    const signedIn = Boolean(state.token && state.username);
    $("#btn-signin").hidden = signedIn;
    $("#user-box").hidden = !signedIn;
    el.history.hidden = !signedIn;
    if (signedIn) {
      $("#user-name").textContent = state.username;
      $("#user-avatar").textContent = state.username[0];
    } else {
      el.nudge.hidden = studio.dataset.stage !== "result";
    }
  }

  function endSession() {
    state.token = state.username = null;
    store.del("clarity.token");
    store.del("clarity.user");
    state.historyBlobs.splice(0).forEach(revoke);
    el.historyGrid.replaceChildren();
    renderAuth();
  }

  let authTab = "login";
  function setAuthTab(tab) {
    authTab = tab;
    const signup = tab === "signup";
    for (const button of $$("[data-tab]", el.auth)) button.setAttribute("aria-selected", String(button.dataset.tab === tab));
    el.authTitle.textContent = signup ? "Create your account" : "Welcome back";
    el.authHint.textContent = signup ? "Save your restorations and find them again anytime." : "Sign in to see your saved restorations.";
    el.authSubmit.textContent = signup ? "Create account" : "Sign in";
    el.authPass.autocomplete = signup ? "new-password" : "current-password";
    el.authRule.hidden = !signup;
    el.authError.hidden = true;
  }

  function openAuth(tab = "login") {
    setAuthTab(tab);
    el.authForm.reset();
    el.auth.showModal();
    el.authUser.focus();
  }

  for (const button of $$("[data-tab]", el.auth)) button.addEventListener("click", () => setAuthTab(button.dataset.tab));
  $("#btn-signin").addEventListener("click", () => openAuth("login"));
  $("#nudge-signin").addEventListener("click", () => openAuth("signup"));
  $("#auth-close").addEventListener("click", () => el.auth.close());
  el.auth.addEventListener("click", (e) => { if (e.target === el.auth) el.auth.close(); });
  $("#btn-signout").addEventListener("click", () => {
    api("/api/auth/logout", { method: "POST" }).catch(() => {});
    endSession();
  });

  el.authForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    el.authError.hidden = true;
    const username = el.authUser.value.trim();
    const password = el.authPass.value;
    if (!username || !password) {
      el.authError.textContent = "Please enter a username and password.";
      el.authError.hidden = false;
      return;
    }
    el.authSubmit.disabled = true;
    try {
      const data = await api(`/api/auth/${authTab}`, { method: "POST", json: { username, password } });
      state.token = data.token;
      state.username = data.username;
      store.set("clarity.token", data.token);
      store.set("clarity.user", data.username);
      el.auth.close();
      renderAuth();
      loadHistory();
    } catch (err) {
      el.authError.textContent = err.message;
      el.authError.hidden = false;
    } finally {
      el.authSubmit.disabled = false;
    }
  });

  /* ------------------------------------------------------------------ history */

  function historyCard(item) {
    const thumb = h("img", { alt: `Restored photo from ${fmtDate(item.created_at)}`, loading: "lazy" });
    fetchBlobUrl(`/api/history/${item.id}/thumb`).then((url) => { state.historyBlobs.push(url); thumb.src = url; }).catch(() => {});

    const details = [`${item.width} × ${item.height}`];
    if (item.psnr != null) details.push(`${item.psnr.toFixed(1)} dB`);
    const open = h("button", { class: "h-open", type: "button", onclick: () => openHistory(item) },
      thumb,
      h("span", { class: "h-body" }, h("b", { text: fmtDate(item.created_at) }), h("span", { text: details.join(" · ") })));

    let armTimer = 0;
    const del = h("button", { class: "icon-btn h-del", type: "button", "aria-label": "Delete this restoration" }, icon("trash"));
    del.addEventListener("click", async () => {
      if (!del.classList.contains("armed")) {
        del.classList.add("armed");
        del.replaceChildren("Delete?");
        armTimer = setTimeout(() => { del.classList.remove("armed"); del.replaceChildren(icon("trash")); }, 3000);
        return;
      }
      clearTimeout(armTimer);
      try {
        await api(`/api/history/${item.id}`, { method: "DELETE" });
        loadHistory();
      } catch (err) {
        showAlert(err.message);
      }
    });
    return h("article", { class: "h-card" }, open, del);
  }

  async function loadHistory() {
    if (!state.token) return;
    let data;
    try {
      data = await api("/api/history");
    } catch (err) {
      if (err.status === 401) endSession();
      return;
    }
    state.historyBlobs.splice(0).forEach(revoke);
    el.historyGrid.replaceChildren(...data.items.map(historyCard));
    el.historyEmpty.hidden = data.items.length > 0;
  }

  async function openHistory(item) {
    hideAlert();
    try {
      const [before, after] = await Promise.all([
        fetchBlobUrl(`/api/history/${item.id}/input`),
        fetchBlobUrl(`/api/history/${item.id}/output`),
      ]);
      state.blobs.push(before, after);
      await showResult(
        { width: item.width, height: item.height, psnr: item.psnr, fromHistory: true },
        { before, after, download: after, downloadName: `restored-${item.id}.png` },
      );
    } catch (err) {
      showAlert(err.message);
    }
  }

  /* --------------------------------------------------------------------- init */

  async function init() {
    renderAuth();
    try {
      const status = await api("/api/status");
      state.limitMb = status.limits.max_upload_mb;
      $("#max-mb").textContent = state.limitMb;
      fillEngines(status.engines);
    } catch {
      fillEngines([{ id: "unet", label: "Standard", description: "" }]);
    }
    if (state.token) {
      try {
        state.username = (await api("/api/me")).username;
        store.set("clarity.user", state.username);
      } catch (err) {
        if (err.status === 401) endSession();
      }
      renderAuth();
      loadHistory();
    }
  }
  init();
})();
