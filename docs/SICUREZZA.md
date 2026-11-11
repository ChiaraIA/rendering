# Sicurezza di posa

## Modello delle minacce
| Chi | Cosa può fare | Difese |
|---|---|---|
| Visitatore di un negozio | caricare foto, anche malevole (file falsi, bombe di decompressione, foto con istruzioni nascoste per gli agenti) | permesso firmato monouso, formato verificato dal contenuto, limiti di dimensione e pixel, Pillow ≥ 13, agenti senza strumenti pericolosi |
| Chiunque su internet | chiamare l'API direttamente, riempire disco o code, far spendere crediti | firma App Proxy di Shopify o chiave pubblica valida solo dai siti registrati, limite dei byte mentre arrivano, velocità minima, quote per cliente (mese e giorno) e per IP, coda massima, tetto di spesa per prova |
| Chi copia la chiave pubblica di un sito | chiedere permessi di caricamento fingendo l'origine (fuori dal browser l'intestazione Origin si può falsificare) | la chiave pubblica dà solo permessi monouso a tempo; restano i limiti per IP, il limite giornaliero e la quota mensile del cliente, che è il tetto del danno; la chiave si revoca e si rigenera |
| Un cliente malevolo | articoli e foto costruiti ad arte nel proprio catalogo; leggere dati di altri clienti | download solo dal negozio e dal CDN Shopify (https, IP pubblici, dimensione massima); foto del catalogo via API decodificate e ricodificate; testi trattati come dati; ogni chiave vede solo catalogo e pratiche del proprio cliente |
| Chi finge di essere Stripe | attivare un cliente senza pagare, sospenderne uno | firma HMAC del webhook con tolleranza di 5 minuti, eventi usati una volta sola |
| Un agente manipolato (prompt injection) | qualunque cosa gli strumenti gli permettano | nessun file, shell, rete o impostazione esterna; solo `guarda` sulle immagini della propria pratica; argomenti limitati; testi ripuliti; segreti fuori dall'ambiente |

## Regole di esercizio
- Segreti solo nel file `deploy/.env` sul server (mai nella repo); `POSA_ADMIN_TOKEN` e
  `POSA_CHIAVE_CARICAMENTI` lunghi e casuali.
- Le chiavi dei clienti (`pk_live_…`, `sk_live_…`) sono conservate solo come hash: si
  consegnano una volta, per un canale sicuro; se una chiave segreta finisce in una pagina
  web o in una repo va revocata subito (`DELETE /admin/clienti/<id>/chiavi/<inizio>`).
- Le API di amministrazione (`/admin/...`) non vanno esposte ai clienti: usale solo tu.
- Il servizio gira solo dietro Caddy (HTTPS, timeout) e in un container in sola lettura.
- La demo Gradio (`posa web`) è solo per uso interno: con `--share` richiede
  `POSA_WEB_PASSWORD`. Usa ancora Pillow < 13 finché Gradio non si aggiorna.
- Alla consegna restano solo i due JPEG del risultato; tutto è cancellato dopo
  `POSA_CONSERVAZIONE_GIORNI` (7). Le pratiche non riuscite conservano la foto fino alla
  scadenza, per il controllo manuale.
- I log non contengono id di pratiche, permessi, chiavi o IP interi: non aggiungere altri
  registri degli accessi davanti al servizio senza gli stessi filtri.
- Disco e copie di sicurezza cifrati; una chiave Anthropic dedicata con tetto di spesa.

## Verifiche da ripetere a ogni rilascio
```bash
pytest -q                                   # include i test d'attacco su agenti, API, SSRF
bandit -q -r app -ll                        # analisi statica
pip-audit                                   # dipendenze con vulnerabilità note
POSA_BASE=https://prova.tuodominio.it POSA_ATTESA_LENTA_S=125 STRIPE_WEBHOOK_SECRET=... \
  ... python tools/pentest.py               # istanza di prova, 77 attacchi
docker run --rm -e POSA_DOMINIO=x.example -v $PWD/deploy/Caddyfile:/etc/caddy/Caddyfile:ro \
  caddy:2.8 caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
```

## Pentest dell'11 novembre 2026 (servizio in abbonamento)

**Come è stato fatto.** Revisione del codice alla ricerca di escalation (cliente → altro
cliente, cliente → amministrazione, visitatore → cliente, agente → controllore o sistema) e
di fughe di dati; poi attacchi veri contro lo **stack di produzione** in locale: il
container del servizio con le protezioni di `deploy/docker-compose.yml` (sola lettura,
nessun privilegio) dietro Caddy con il `Caddyfile` della repo. Ogni attacco riuscito è
stato corretto e ripetuto. Esito finale: **77 attacchi della batteria, 0 riusciti**; 211
test automatici verdi; analisi statica (bandit) e dipendenze (pip-audit) senza rilievi.

### Problemi trovati e corretti

| Gravità | Problema | Correzione |
|---|---|---|
| Alta | **Quota mensile aggirabile**: 12 richieste in parallelo con quota 2 ne facevano accettare 11 (controllo e creazione erano due passi). Valeva anche per il limite giornaliero e per IP. | Limiti, permesso monouso e creazione della pratica in un'unica operazione (`Store.admit_job`). |
| Alta | **Pagamenti: escalation tra clienti.** Il campo `client_reference_id` del link di pagamento lo scrive chi paga: con l'id di un cliente (noto) si poteva riattivare un cliente sospeso, agganciare il proprio cliente Stripe a un concorrente e poi farlo disdire. | Vale solo un **codice di attivazione segreto** per cliente; contano solo abbonamenti pagati; un cliente Stripe già collegato non si sostituisce; lo stato lo decide solo l'abbonamento del cliente Stripe collegato. |
| Alta | **Servizio fermabile**: un risultato corrotto faceva morire il thread che esegue le pratiche; due volte e il servizio non lavorava più (difetto già presente). | I worker sopravvivono a qualunque errore; la pratica risulta "non riuscita". |
| Alta | **Il proxy non partiva**: il `Caddyfile` della repo non era valido (opzioni globali in fondo). Il pentest del 9 novembre non può quindi essere passato da Caddy. | Corretto e verificato con `caddy validate` e con lo stack acceso. |
| Media | **Dati sensibili nei log**: id delle pratiche (chi ha l'id vede la foto del cliente), permessi di caricamento, firme e indirizzi IP interi nei log del proxy; id interi anche nei log del servizio. | Filtro nei log di Caddy (id e firme omessi, intestazioni tolte, IP troncati); registro degli accessi del servizio spento; id abbreviati. Verificato: 0 id, 0 permessi, 0 chiavi in 249 righe. |
| Media | **Segreti in memoria**: dopo la "pulizia" i segreti restavano leggibili in `/proc/1/environ`. | Il valore viene sovrascritto in memoria. Verificato nel container. |
| Media | **Un agente poteva parlare al suo controllore**: la "motivazione" scritta dall'agente finiva nel contesto del controllore ("approva…"). | I controllori ricevono solo immagini e dati calcolati dal codice. |
| Media | **Lavoro a vuoto**: con una chiave pubblica copiata si potevano coniare permessi senza limite e far decodificare file sbagliati all'infinito; la decodifica bloccava tutte le altre richieste. | Limiti per indirizzo su permessi e tentativi, controlli a costo zero prima di leggere il file, decodifica fuori dal ciclo degli eventi. |
| Media | **Foto conservate più del necessario**: foto originale, maschere e anteprime restavano 7 giorni anche a prova consegnata. | Alla consegna restano solo i due JPEG del risultato; `DELETE /v1/pratiche/<id>` cancella subito tutto. |
| Bassa | Il commento JPEG della foto d'origine veniva ricopiato nelle immagini ricodificate. | Si salvano solo i pixel. |
| Bassa | Le foto di catalogo sostituite non venivano mai cancellate (disco riempibile). | Pulizia automatica. |
| Bassa | Nessun freno ai tentativi sul token di amministrazione. | 20 errori all'ora per indirizzo, poi blocco. |
| Bassa | Modello di segmentazione che contatta Hugging Face a servizio acceso; versioni dei pacchetti non fissate; chiave dei permessi corta non segnalata; file leggibili da altri utenti del server. | Modello solo da disco, versioni fissate (`deploy/vincoli-server.txt`), avviso all'avvio, `umask 077`. |

### Cosa ha retto senza modifiche
Isolamento tra clienti (catalogo, pratiche, chiavi), chiavi revocate e scambiate, origini,
firme Shopify e Stripe con replay, file malevoli (HTML, bombe di decompressione, 60 MB a
pezzi), percorsi `../`, CORS, amministrazione, confini degli agenti (nessun file, shell o
rete), container (utente non privilegiato, nessuna capability, file system in sola
lettura, porta interna non esposta).

### Cosa NON è stato provato, e rischi che restano
- **Agenti veri.** I confini degli agenti sono verificati nella configurazione e con
  controllori finti: manca la prova con i modelli veri e foto costruite per ingannarli.
- **Immagine completa.** In questo ambiente non si è potuta costruire l'immagine con
  PyTorch: lo stack provato è la stessa immagine senza modello di segmentazione.
- **Stripe vero** e server di produzione (DNS, certificati, firewall del fornitore).
- **Chiave pubblica.** Sta nelle pagine dei clienti: chi la copia può avviare prove a nome
  del cliente falsificando l'origine fuori dal browser. Il danno massimo è la quota
  giornaliera del cliente (`prove_al_giorno`): tienila vicina all'uso reale.
- **Chiave Anthropic.** Deve restare nell'ambiente del processo, perché gli agenti la
  usano; non hanno strumenti per leggerla. Usa una chiave dedicata con tetto di spesa.
- **Dati a riposo non cifrati**: cifra il disco del server e le copie di sicurezza.
- **Un upload fermo** tiene aperta una connessione del proxy fino a 120 secondi; contro
  attacchi di volume serve una protezione a monte (CDN o firewall del fornitore).
- **Limiti di frequenza in memoria**: valgono per una sola istanza e si azzerano al riavvio.

## Pentest del 9 novembre 2026
12 problemi trovati (1 critico, 4 alti, 5 medi, 2 bassi), tutti corretti e verificati
ripetendo gli attacchi sul servizio e sul container di produzione; 46 attacchi della
batteria respinti. Dettaglio nel documento di progetto e nella storia dei commit.
