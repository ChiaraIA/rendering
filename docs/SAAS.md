# posa come servizio in abbonamento

`posa server` è un servizio unico per tutti i clienti che pagano il canone. Un **cliente**
è un sito che integra la prova "nella tua stanza":

| Canale | Come si collega | Catalogo |
|---|---|---|
| `sito` | widget JavaScript (chiave pubblica) e/o API dal proprio server (chiave segreta) | caricato via API, articolo per articolo |
| `shopify` | app Shopify con il blocco sulla pagina prodotto (vedi `shopify/README.md`) | letto dal negozio e aggiornato dai webhook |

Ogni cliente ha i suoi articoli, le sue pratiche e le sue chiavi: nessuno vede i dati di un
altro (verificato dai test in `tests/server/test_saas.py`).

## Piani e quote

Un piano stabilisce **quante prove consegnate al mese** sono comprese, **su quali
superfici** (pavimento, rivestimento a parete) e che cosa succede oltre la quota (ci si
ferma, oppure si continua "a consumo" e le prove extra vengono contate).

Piani di partenza, modificabili in qualunque momento:

| Piano | Prove al mese | Superfici | Oltre la quota |
|---|---|---|---|
| `prova` | 30 | pavimento, rivestimento | stop |
| `base` | 150 | pavimento | stop |
| `professionale` | 600 | pavimento, rivestimento | a consumo |
| `azienda` | 3000 | pavimento, rivestimento | a consumo |

Il **prezzo non è scritto nel codice**: `canone_eur` e `prova_extra_eur` sono campi del
piano che decidi tu. Per ragionare sul prezzo: ogni prova costa circa 0,5-1,5 USD di API
(tutti gli agenti sul modello più capace; meno con `POSA_MODEL`), più il server. Il costo
reale di ogni cliente, mese per mese, è in `/admin/clienti/<id>/uso`
(`costo_agenti_usd`): è il numero da confrontare con il canone.

Cosa conta e cosa no:
- conta una prova **consegnata** (il cliente finale ha visto il risultato);
- una prova non riuscita non consuma la quota, ma costa: oltre il doppio della quota in
  tentativi il cliente viene fermato (foto inadatte in massa, abusi);
- valgono sempre anche il limite giornaliero del cliente (`prove_al_giorno`) e quello per
  indirizzo IP: sono il tetto di spesa anche per i piani a consumo.

```bash
curl -X PUT https://SERVER/admin/piani/professionale -H "Authorization: Bearer $POSA_ADMIN_TOKEN" -H "Content-Type: application/json" -d '{"nome":"Professionale","prove_al_mese":600,"superfici":["pavimento","rivestimento"],"oltre_quota_a_consumo":true,"canone_eur":299,"prova_extra_eur":0.9}'
```

## Attivare un nuovo cliente (sito)

1. Crea il cliente con il suo sito (da lì, e solo da lì, funzionerà il widget):
   ```bash
   curl -X POST https://SERVER/admin/clienti -H "Authorization: Bearer $POSA_ADMIN_TOKEN" -H "Content-Type: application/json" -d '{"id":"ceramiche-bianchi","nome":"Ceramiche Bianchi","sito":"https://www.ceramiche-bianchi.it","origini":["https://staging.ceramiche-bianchi.it"],"piano":"professionale"}'
   ```
2. Crea le due chiavi (si vedono **una volta sola**: sul server resta solo l'hash):
   ```bash
   curl -X POST https://SERVER/admin/clienti/ceramiche-bianchi/chiavi -H "Authorization: Bearer $POSA_ADMIN_TOKEN" -H "Content-Type: application/json" -d '{"tipo":"pubblica"}'
   ```
   ```bash
   curl -X POST https://SERVER/admin/clienti/ceramiche-bianchi/chiavi -H "Authorization: Bearer $POSA_ADMIN_TOKEN" -H "Content-Type: application/json" -d '{"tipo":"segreta"}'
   ```
   - `pk_live_…` (pubblica) va nella pagina web: da sola permette solo di chiedere un
     permesso di caricamento, e solo dalle pagine dei siti registrati;
   - `sk_live_…` (segreta) resta sul server del cliente: catalogo, pratiche, uso.
3. Consegna al cliente le chiavi e la sezione "Integrazione" qui sotto.

Altre operazioni: `GET /admin/clienti` (elenco con uso del mese), `POST
/admin/clienti/<id>/stato` (`attivo`, `sospeso`, `disdetto`), `DELETE
/admin/clienti/<id>/chiavi/<inizio>` (revoca), `DELETE /admin/clienti/<id>` (fine del
rapporto: cancella chiavi, catalogo, foto e risultati).

## Integrazione per il sito cliente

### 1. Catalogo (dal server del cliente, chiave segreta)

Per ogni articolo: la foto frontale di **una piastrella** (a tutto campo o su sfondo
uniforme; più foto = varianti alternate nella posa) e le regole di posa.

```bash
curl -X PUT https://SERVER/v1/catalogo/metro-bianco-10x20 -H "Authorization: Bearer sk_live_..." -F nome="Metro bianco lucido 10x20" -F formato=20x10 -F fuga=2 -F posa=running_bond -F finitura=lucida -F uso=rivestimento -F foto=@piastrella.jpg
```

| Campo | Valori | Note |
|---|---|---|
| `nome`, `formato` | testo, `60x60` (cm) | obbligatori; esagoni: solo la larghezza punta-punta |
| `fuga`, `colore_fuga` | mm, `#rrggbb` | fuga 0 = a incastro; colore in tinta se omesso |
| `posa` | `grid`, `running_bond`, `hex` | `sfalsamento` (0-1) per la posa a correre |
| `rotazione` | `0`, `45`, `90` | 45 = diagonale; 90 = piastrelle rettangolari in verticale |
| `finitura` | `opaca`, `satinata`, `lucida` | decide i riflessi |
| `uso` | `pavimento`, `rivestimento` o entrambi | dove l'articolo si può provare |

La risposta dice, foto per foto, se è stata usata o scartata e perché. `GET /v1/catalogo`
elenca gli articoli, `DELETE /v1/catalogo/<codice>` ne toglie uno.

### 2. Il pulsante nella pagina prodotto (chiave pubblica)

```html
<script src="https://SERVER/widget/posa.js" defer></script>
<div data-posa
     data-chiave="pk_live_..."
     data-articolo="metro-bianco-10x20"
     data-superfici="pavimento,rivestimento"
     data-testo="Prova nella tua stanza"></div>
```

Il cliente finale carica la foto, sceglie pavimento o parete (se l'articolo li consente
entrambi) e in 2-5 minuti vede il prima/dopo. A risultato pronto il `div` emette l'evento
`posa:pronto` (`detail.pratica`, `detail.risultato`), utile per le statistiche del sito.
Se il sito ha una Content-Security-Policy, deve consentire `script-src` e `connect-src`
verso `https://SERVER`, e `img-src https://SERVER`.

### 3. In alternativa: tutto dal server del cliente

```bash
curl -X POST "https://SERVER/v1/pratiche?articolo=metro-bianco-10x20&superficie=rivestimento&zona=paraschizzi" -H "Authorization: Bearer sk_live_..." -F foto=@cucina.jpg
```
```bash
curl https://SERVER/v1/pratiche/<pratica> -H "Authorization: Bearer sk_live_..."
```
```bash
curl https://SERVER/v1/account -H "Authorization: Bearer sk_live_..."
```

`superficie`: `pavimento` o `rivestimento`. `zona` (solo rivestimento): `paraschizzi` (tra
piano di lavoro e pensili o finestre), `parete` (parete intera) o `automatica`. `stanza`
(campo del modulo, facoltativo): misure in metri, `4.2x3.6` per il pavimento, larghezza x
altezza della zona per una parete. Stati: `in_coda`, `in_lavorazione`, `pronto`,
`non_riuscito`. Risposte d'errore: `401` chiave, `402` quota del mese finita, `403`
abbonamento o piano, `409` articolo non adatto alla superficie, `429` troppe richieste.

## Pagamenti (Stripe)

Il servizio non incassa: legge lo stato dell'abbonamento da Stripe e attiva o sospende il
cliente di conseguenza.

1. In Stripe crea un prodotto ricorrente per ogni piano e un Payment Link (o Checkout) che
   passi l'id del cliente in `client_reference_id`; nei metadati dell'abbonamento metti
   `piano` (es. `professionale`) e, se vuoi, `cliente`.
2. Crea l'endpoint webhook `https://SERVER/webhooks/stripe` con gli eventi
   `checkout.session.completed` e `customer.subscription.*`, e metti il suo segreto in
   `STRIPE_WEBHOOK_SECRET`.
3. Da quel momento: pagamento riuscito → cliente `attivo` e collegato al suo `cus_…`;
   pagamento scaduto (`past_due`, `unpaid`) → `sospeso` (il widget risponde che
   l'abbonamento non è attivo, i dati restano); disdetta → `disdetto`.

Senza Stripe funziona tutto lo stesso: lo stato si cambia a mano con
`POST /admin/clienti/<id>/stato`. **Prima di incassare davvero**, prova il giro completo in
modalità test di Stripe: la verifica della firma e gli eventi sono coperti dai test
automatici, ma non sono stati provati contro un account Stripe reale.

## Prima di vendere: cose da decidere o verificare

- **Prova con gli agenti veri.** I test automatici usano controllori finti: la procedura
  completa su foto reali (soprattutto cucine e stanze piene di oggetti) va provata con la
  chiave API prima di mostrarla a un cliente. Le pratiche che non superano i controlli
  finiscono in `non_riuscito` con `ESCALATION.txt`: guardale, dicono dove migliorare.
- **Licenza del modello di segmentazione.** I pesi UperNet-ConvNeXt sono rilasciati con
  licenza MIT, ma sono addestrati su ADE20K, le cui immagini sono distribuite per ricerca
  non commerciale. È una situazione comune tra i prodotti commerciali, ma per un servizio
  a pagamento conviene un parere legale (o un modello addestrato su dati con licenza
  commerciale: il modello si cambia con `POSA_SEG_MODEL`).
- **Privacy.** Le foto delle case dei clienti finali sono dati personali: servono
  informativa, nomina a responsabile del trattamento nei contratti con i clienti e il
  periodo di conservazione (`POSA_CONSERVAZIONE_GIORNI`, 7 giorni) scritto nei contratti.
- **Capacità.** Un server da 4 CPU esegue 2 pratiche insieme (`POSA_PRATICHE_IN_PARALLELO`)
  in 2-5 minuti l'una: circa 500-1000 prove al giorno. Oltre, servono più istanze e una
  coda condivisa (oggi coda e archivio sono locali: SQLite su un solo server).
- **Copie di sicurezza.** L'archivio (`/dati/servizio.db`: clienti, chiavi, uso) e i
  cataloghi (`/dati/clienti/`) vanno salvati ogni giorno; le pratiche no.
