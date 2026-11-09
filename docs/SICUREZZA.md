# Sicurezza di posa

## Modello delle minacce
| Chi | Cosa può fare | Difese |
|---|---|---|
| Visitatore di un negozio | caricare foto, anche malevole (file falsi, bombe di decompressione, foto con istruzioni nascoste per gli agenti) | permesso firmato monouso, formato verificato dal contenuto, limiti di dimensione e pixel, Pillow ≥ 13, agenti senza strumenti pericolosi |
| Chiunque su internet | chiamare l'API direttamente, riempire disco o code, far spendere crediti | firma App Proxy di Shopify, limite dei byte mentre arrivano, velocità minima, quote per negozio e per IP, coda massima, tetto di spesa per prova |
| Un negozio malevolo | articoli e foto costruiti ad arte nel proprio catalogo | download solo dal negozio e dal CDN Shopify (https, IP pubblici, dimensione massima), testi del catalogo trattati come dati |
| Un agente manipolato (prompt injection) | qualunque cosa gli strumenti gli permettano | nessun file, shell, rete o impostazione esterna; solo `guarda` sulle immagini della propria pratica; argomenti limitati; testi ripuliti; segreti fuori dall'ambiente |

## Regole di esercizio
- Segreti solo nel file `deploy/.env` sul server (mai nella repo); `POSA_ADMIN_TOKEN` e
  `POSA_CHIAVE_CARICAMENTI` lunghi e casuali.
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

## Pentest del 9 novembre 2026
12 problemi trovati (1 critico, 4 alti, 5 medi, 2 bassi), tutti corretti e verificati
ripetendo gli attacchi sul servizio e sul container di produzione; 46 attacchi della
batteria respinti. Dettaglio nel documento di progetto e nella storia dei commit.
