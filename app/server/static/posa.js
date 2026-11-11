/* posa — "Prova nella tua stanza" per qualunque sito.
 *
 *   <script src="https://SERVIZIO/widget/posa.js" defer></script>
 *   <div data-posa data-chiave="pk_live_..." data-articolo="gres-grigio-60x60"
 *        data-superfici="pavimento,rivestimento"></div>
 *
 * Attributi facoltativi: data-testo (testo del pulsante), data-max-mb (default 15).
 * Sicurezza: nessun innerHTML con dati del server (solo textContent); si parla solo con
 * l'indirizzo da cui è stato caricato questo script, in https; la chiave pubblica funziona
 * solo dalle pagine dei siti registrati per il cliente. */
(function () {
  "use strict";

  var script = document.currentScript;
  var BASE = script ? new URL(script.src).origin : "";
  var POLL_MS = 3000;
  var MAX_WAIT_MS = 20 * 60 * 1000;
  var LABELS = { pavimento: "Pavimento", rivestimento: "Parete (rivestimento)" };

  var CSS =
    ".posa-w__apri{cursor:pointer;padding:.7em 1.2em;border:0;border-radius:6px;background:#1f2937;color:#fff;font:inherit}" +
    ".posa-w__finestra{border:0;border-radius:10px;padding:1.4em;max-width:min(92vw,640px);width:100%;box-shadow:0 10px 40px rgba(0,0,0,.35);font:inherit;color:#111;background:#fff}" +
    ".posa-w__finestra::backdrop{background:rgba(0,0,0,.5)}" +
    ".posa-w__finestra label{display:block;margin:.8em 0;font-size:.95em}" +
    ".posa-w__finestra input,.posa-w__finestra select{display:block;margin-top:.3em;max-width:100%;font:inherit}" +
    ".posa-w__invia{cursor:pointer;padding:.7em 1.2em;border:0;border-radius:6px;background:#1f2937;color:#fff;font:inherit}" +
    ".posa-w__invia[disabled]{opacity:.5;cursor:wait}" +
    ".posa-w__chiudi{float:right;border:0;background:none;font-size:1.5em;line-height:1;cursor:pointer}" +
    ".posa-w__stato{min-height:1.4em;margin:.8em 0}" +
    ".posa-w__risultato{display:block;max-width:100%;border-radius:6px;margin:.6em 0}" +
    ".posa-w__nota{font-size:.8em;color:#555}";

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text) e.textContent = text;
    return e;
  }

  function resultUrl(path) {
    // accetta solo i due file pubblici di una pratica del servizio, mai URL arbitrari
    if (!/^\/api\/pratiche\/[A-Za-z0-9_-]{32}\/(risultato|prima_dopo)\.jpg$/.test(path)) return null;
    return BASE + path;
  }

  function init(root) {
    if (root.dataset.posaPronto) return;
    root.dataset.posaPronto = "1";
    var key = root.dataset.chiave || "";
    var article = root.dataset.articolo || "";
    var surfaces = (root.dataset.superfici || "pavimento").split(",").map(function (s) { return s.trim(); })
      .filter(function (s) { return LABELS[s]; });
    if (!surfaces.length) surfaces = ["pavimento"];
    var maxBytes = (parseInt(root.dataset.maxMb, 10) || 15) * 1024 * 1024;

    var open = el("button", "posa-w__apri", root.dataset.testo || "Prova nella tua stanza");
    open.type = "button";
    var dialog = el("dialog", "posa-w__finestra");
    var close = el("button", "posa-w__chiudi", "×");
    close.type = "button";
    close.setAttribute("aria-label", "Chiudi");
    var form = el("form");
    var fileLabel = el("label", "", "Foto della stanza");
    var file = el("input");
    file.type = "file";
    file.accept = "image/jpeg,image/png,image/webp,image/heic,image/heif,.heic,.heif";
    file.required = true;
    fileLabel.appendChild(file);
    var surfLabel = el("label", "", "Dove vuoi vederlo");
    var surf = el("select");
    surfaces.forEach(function (s) {
      var o = el("option", "", LABELS[s]);
      o.value = s;
      surf.appendChild(o);
    });
    surfLabel.appendChild(surf);
    if (surfaces.length < 2) surfLabel.hidden = true;
    var roomLabel = el("label", "", "Misure della stanza in metri (facoltativo)");
    var room = el("input");
    room.type = "text";
    room.inputMode = "decimal";
    room.placeholder = "es. 4.2x3.6";
    room.maxLength = 11;
    roomLabel.appendChild(room);
    function syncRoom() { roomLabel.hidden = surf.value !== "pavimento"; }
    surf.addEventListener("change", syncRoom);
    syncRoom();
    var send = el("button", "posa-w__invia", "Prova questo rivestimento");
    send.type = "submit";
    var note = el("p", "posa-w__nota", "La foto serve solo a creare l'anteprima e viene cancellata automaticamente dopo pochi giorni.");
    var status = el("p", "posa-w__stato");
    status.setAttribute("role", "status");
    status.setAttribute("aria-live", "polite");
    var img = el("img", "posa-w__risultato");
    img.alt = "Il rivestimento nella tua stanza";
    img.hidden = true;
    var link = el("a", "", "Scarica l'immagine");
    link.hidden = true;
    link.download = "";
    [fileLabel, surfLabel, roomLabel, note, send].forEach(function (n) { form.appendChild(n); });
    [close, form, status, img, link].forEach(function (n) { dialog.appendChild(n); });
    root.appendChild(open);
    root.appendChild(dialog);

    function say(text) { status.textContent = text; }
    open.addEventListener("click", function () {
      if (typeof dialog.showModal === "function") dialog.showModal();
      else dialog.setAttribute("open", "");
    });
    close.addEventListener("click", function () {
      if (typeof dialog.close === "function") dialog.close();
      else dialog.removeAttribute("open");
    });

    form.addEventListener("submit", function (ev) {
      ev.preventDefault();
      var photo = file.files && file.files[0];
      if (!photo) { say("Scegli una foto."); return; }
      if (photo.size > maxBytes) { say("La foto è troppo grande."); return; }
      if (BASE.indexOf("https://") !== 0) { say("Servizio non raggiungibile in modo sicuro."); return; }
      send.disabled = true;
      img.hidden = true;
      link.hidden = true;
      say("Preparo la prova…");
      fetch(BASE + "/api/sessioni", {
        method: "POST",
        credentials: "omit",
        headers: { "Content-Type": "application/json", "X-Posa-Chiave": key },
        body: JSON.stringify({ articolo: article, superficie: surf.value })
      })
        .then(function (r) {
          return r.json().then(function (b) { if (!r.ok) throw new Error(b.errore || "sessione"); return b; });
        })
        .then(function (s) {
          var data = new FormData();
          data.append("foto", photo);
          var size = (room.value || "").trim();
          if (size && surf.value === "pavimento") data.append("stanza", size);
          return fetch(BASE + "/api/pratiche", {
            method: "POST", body: data, credentials: "omit",
            headers: { "X-Posa-Permesso": String(s.permesso || "") }
          });
        })
        .then(function (r) {
          return r.json().then(function (b) { if (!r.ok) throw new Error(b.errore || "invio"); return b; });
        })
        .then(function (b) { return poll(String(b.pratica || ""), Date.now()); })
        .catch(function (e) {
          say(e && e.message && e.message.length < 120
            ? "Non è stato possibile: " + e.message
            : "Non è stato possibile completare la prova. Riprova più tardi.");
        })
        .finally(function () { send.disabled = false; });
    });

    function poll(id, started) {
      if (!/^[A-Za-z0-9_-]{32}$/.test(id)) throw new Error("risposta non valida");
      return fetch(BASE + "/api/pratiche/" + id, { credentials: "omit" })
        .then(function (r) { return r.json(); })
        .then(function (b) {
          say(String(b.messaggio || "In lavorazione…"));
          if (b.stato === "pronto") {
            var src = resultUrl(String(b.prima_dopo || b.risultato || ""));
            var full = resultUrl(String(b.risultato || ""));
            if (src) { img.src = src; img.hidden = false; }
            if (full) { link.href = full; link.hidden = false; }
            say("Ecco il risultato.");
            root.dispatchEvent(new CustomEvent("posa:pronto", { detail: { pratica: id, risultato: full } }));
            return;
          }
          if (b.stato === "non_riuscito") {
            say("Questa foto richiede un controllo manuale. Prova con un'altra foto, più luminosa e con la superficie ben visibile.");
            return;
          }
          if (Date.now() - started > MAX_WAIT_MS) { say("Ci vuole più del previsto. Riprova più tardi."); return; }
          return new Promise(function (res) { setTimeout(res, POLL_MS); }).then(function () { return poll(id, started); });
        });
    }
  }

  function start() {
    if (!document.getElementById("posa-w-stile")) {
      var style = document.createElement("style");
      style.id = "posa-w-stile";
      style.textContent = CSS;
      document.head.appendChild(style);
    }
    document.querySelectorAll("[data-posa]").forEach(init);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
  window.posa = { avvia: start };  // per pagine che aggiungono i pulsanti dopo il caricamento
})();
