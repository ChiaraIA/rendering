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
- Foto e risultati sono cancellati dopo `POSA_CONSERVAZIONE_GIORNI` (7).

## Verifiche da ripetere a ogni rilascio
```bash
pytest -q                                   # include i test d'attacco su agenti, API, SSRF
bandit -q -r app -ll                        # analisi statica
pip-audit                                   # dipendenze con vulnerabilità note
POSA_BASE=https://prova.tuodominio.it ... python tools/pentest.py   # istanza di prova
```

## Novità dell'11 novembre 2026 (clienti, chiavi, piani, pagamenti, pareti)
Le superfici d'attacco nuove (chiavi API, `/api/sessioni`, `/v1/...`, `/webhooks/stripe`,
`/admin/clienti`, `/admin/piani`) sono coperte dai test automatici (`tests/server/`) e da
una nuova sezione di `tools/pentest.py` (21 prove: origini, chiavi scambiate o revocate,
isolamento tra clienti, catalogo, firme Stripe, cliente sospeso, cancellazione dei dati).
La batteria completa (67 prove) è stata rilanciata su un'istanza **locale con agenti
finti**: tutte respinte. Va ripetuta sull'istanza di prova vera (container, Caddy, agenti
reali) prima di aprire il servizio ai clienti.

## Pentest del 9 novembre 2026
12 problemi trovati (1 critico, 4 alti, 5 medi, 2 bassi), tutti corretti e verificati
ripetendo gli attacchi sul servizio e sul container di produzione; 46 attacchi della
batteria respinti. Dettaglio nel documento di progetto e nella storia dei commit.
