/* "Prova nella tua stanza": carica la foto, segue la pratica, mostra il risultato.
 * Sicurezza: nessun innerHTML con dati del server (solo textContent), solo URL del
 * servizio restituiti dall'App Proxy firmato, file controllati prima dell'invio. */
(function () {
  "use strict";

  var POLL_MS = 3000;
  var MAX_WAIT_MS = 20 * 60 * 1000;

  function variantId(root) {
    var fromUrl = new URLSearchParams(window.location.search).get("variant");
    if (fromUrl && /^\d{1,20}$/.test(fromUrl)) return fromUrl;
    var input = document.querySelector('form[action*="/cart/add"] [name="id"]');
    if (input && /^\d{1,20}$/.test(input.value)) return input.value;
    return root.dataset.variante || "";
  }

  function sameOrigin(base, path) {
    // accetta solo percorsi relativi del servizio, mai URL arbitrari
    if (!/^\/api\/pratiche\/[A-Za-z0-9_-]{32}\/(risultato|prima_dopo)\.jpg$/.test(path)) return null;
    return base + path;
  }

  function init(root) {
    var dialog = root.querySelector(".posa-prova__finestra");
    var form = root.querySelector(".posa-prova__modulo");
    var status = root.querySelector(".posa-prova__stato");
    var img = root.querySelector(".posa-prova__risultato");
    var link = root.querySelector(".posa-prova__scarica");
    var maxBytes = (parseInt(root.dataset.maxMb, 10) || 15) * 1024 * 1024;

    root.querySelector(".posa-prova__apri").addEventListener("click", function () {
      if (typeof dialog.showModal === "function") dialog.showModal();
      else dialog.setAttribute("open", "");
    });

    function say(text) { status.textContent = text; }

    form.addEventListener("submit", function (ev) {
      ev.preventDefault();
      var file = form.foto.files && form.foto.files[0];
      if (!file) { say("Scegli una foto."); return; }
      if (file.size > maxBytes) { say("La foto è troppo grande."); return; }
      var room = (form.stanza.value || "").trim();
      form.querySelector('[type="submit"]').disabled = true;
      img.hidden = true; link.hidden = true;
      say("Preparo la prova…");

      var session = "/apps/posa/sessione?prodotto=" + encodeURIComponent(root.dataset.prodotto) +
        "&variante=" + encodeURIComponent(variantId(root));
      fetch(session, { credentials: "same-origin" })
        .then(function (r) { if (!r.ok) throw new Error("sessione"); return r.json(); })
        .then(function (s) {
          var upload = new URL(s.carica);
          if (upload.protocol !== "https:") throw new Error("servizio non sicuro");
          var base = upload.origin;
          var data = new FormData();
          data.append("foto", file);
          if (room) data.append("stanza", room);
          return fetch(upload.href, {
            method: "POST", body: data, credentials: "omit",
            headers: { "X-Posa-Permesso": s.permesso }
          })
            .then(function (r) {
              return r.json().then(function (b) { if (!r.ok) throw new Error(b.errore || "invio"); return b; });
            })
            .then(function (b) { return poll(base, b.pratica, Date.now()); });
        })
        .catch(function (e) {
          say(e && e.message && e.message.length < 120 ? "Non è stato possibile: " + e.message : "Non è stato possibile completare la prova. Riprova più tardi.");
        })
        .finally(function () { form.querySelector('[type="submit"]').disabled = false; });
    });

    function poll(base, id, started) {
      if (!/^[A-Za-z0-9_-]{32}$/.test(id)) throw new Error("risposta non valida");
      return fetch(base + "/api/pratiche/" + id, { credentials: "omit" })
        .then(function (r) { return r.json(); })
        .then(function (b) {
          say(String(b.messaggio || "In lavorazione…"));
          if (b.stato === "pronto") {
            var src = sameOrigin(base, String(b.prima_dopo || b.risultato || ""));
            var full = sameOrigin(base, String(b.risultato || ""));
            if (src) { img.src = src; img.hidden = false; }
            if (full) { link.href = full; link.hidden = false; }
            say("Ecco il risultato.");
            return;
          }
          if (b.stato === "non_riuscito") { say("Questa foto richiede un controllo manuale: il negozio ti ricontatterà se lasci i tuoi dati."); return; }
          if (Date.now() - started > MAX_WAIT_MS) { say("Ci vuole più del previsto. Riprova più tardi."); return; }
          return new Promise(function (res) { setTimeout(res, POLL_MS); }).then(function () { return poll(base, id, started); });
        });
    }
  }

  document.querySelectorAll(".posa-prova").forEach(init);
})();
