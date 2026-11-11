# Manuale: demo e vendita di posa

Questo manuale è per te. Dice come far funzionare posa per una demo, cosa mostrare e cosa
dire, e poi come attivare un cliente che ha deciso di comprare. I dettagli tecnici delle
API sono in [SAAS.md](SAAS.md), la sicurezza in [SICUREZZA.md](SICUREZZA.md).

**Una cosa da sapere prima di tutto.** Il sistema è stato provato in ogni sua parte con test
automatici e con foto vere, ma **la procedura completa con gli agenti veri non è mai stata
eseguita dopo le ultime modifiche** (contorni, pareti): serve la tua chiave API. Per questo
la prova generale del capitolo 1 non è facoltativa: va fatta prima di sederti davanti a un
cliente, con le stesse foto che userai.

---

## 1. Prima della demo: preparazione (1-2 ore, il giorno prima)

### 1.1 La chiave API

Gli agenti sono modelli Claude: serve una chiave Anthropic con credito.

1. Vai su <https://console.anthropic.com>, crea una chiave (`sk-ant-...`).
2. Nella console imposta un **limite di spesa mensile** (es. 50 USD per le demo).
3. Ogni prova costa indicativamente 0,5-1,5 USD e ha comunque un tetto di 4 USD
   (`POSA_BUDGET_PRATICA_USD`). Il costo vero lo leggi alla fine di ogni prova, nel
   registro ("costo stimato X USD") e accanto a "Consegnato" nella pagina web. Segnatelo durante la prova generale: ti serve per i
   prezzi.

### 1.2 Installazione

**Strada A — GitHub Codespaces (niente da installare, consigliata se non hai Python).**

1. Nella repo: **Code → Codespaces → ⋯ → New with options**, scegli il branch e crea.
   Quando GitHub lo chiede, inserisci la tua `ANTHROPIC_API_KEY`: resta un segreto del
   Codespace, non finisce nel codice. Se la aggiungi dopo, il Codespace va ricreato.
2. Aspetta la fine dell'installazione automatica (5-10 minuti: scarica anche il modello che
   riconosce pavimenti e pareti, circa 470 MB).

**Strada B — sul tuo computer (Python 3.11 o più recente).**

```bash
git clone https://github.com/ChiaraIA/rendering.git && cd rendering
```
```bash
python -m venv .venv && source .venv/bin/activate
```
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
```bash
pip install -e '.[demo,segmentazione]'
```
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 1.3 Controllo dell'installazione

```bash
posa verifica --modelli
```

Deve finire con tutte spunte verdi. Poi una chiamata di prova agli agenti (pochi centesimi):

```bash
posa verifica --online
```

Se "Segmentazione automatica" è in avviso, **fermati e risolvilo**: senza quel modello i
contorni li disegnano gli agenti a mano e il risultato è peggiore. Il comando ti dice cosa
installare.

### 1.4 Prova generale (non saltarla)

Fai tre prove complete, nell'ordine. Per ognuna segna: tempo, costo, esito.

```bash
posa demo
```
Stanza sintetica, pavimento. Serve a vedere che tutta la catena funziona (3-5 minuti).

```bash
posa demo --parete
```
Cucina sintetica, rivestimento tra piano di lavoro e pensili.

```bash
posa web
```
Si apre <http://localhost:7860> (in Codespaces: scheda **Ports → 7860 → Open in Browser**).
Carica **le foto vere che userai in demo** e provale una per una.

Come leggere l'esito:
- **"Consegnato"** con il prima/dopo: la foto è buona per la demo. Guarda il risultato da
  vicino (zoom sui bordi di sedie, tavoli, prese): se vedi un difetto, scarta la foto.
- **"Non consegnato: passato a un operatore"**: i controllori hanno bocciato tre volte lo
  stesso passo. Non è un guasto, è il sistema che rifiuta di consegnare un lavoro brutto.
  Per la demo, però, quella foto non usarla. Il motivo è scritto nel registro.
- **Errore tecnico** (chiave, rete, credito): `posa verifica --online` dice cosa manca.

Tieni **da parte i risultati riusciti**: ogni prova è una cartella in `pratiche/<data-ora>/`
e dentro `consegna/` ci sono `prima_dopo.jpg` e `final.png`. Sono il tuo piano B.

### 1.5 Quali foto scegliere

Le foto fanno la differenza tra una demo che convince e una che zoppica.

Per il **pavimento**:
- stanza illuminata, foto nitida, scattata in orizzontale ad altezza d'uomo;
- si vede una buona porzione di pavimento e almeno due linee muro-pavimento;
- mobili sì (fanno scena: il pavimento passa dietro le gambe), ma evita il caos totale,
  tappeti che coprono quasi tutto, specchi a terra, pavimenti molto lucidi con riflessi forti;
- se conosci le misure della stanza, inseriscile (es. `4.2x3.6`): la scala delle piastrelle
  sarà più credibile.

Per la **parete** (paraschizzi della cucina):
- cucina fotografata abbastanza di fronte, con piano di lavoro e pensili entrambi visibili;
- qualche oggetto sul piano va bene; evita pareti quasi del tutto nascoste da oggetti;
- funziona anche con cucine ad angolo (due pareti).

Prepara 3-4 foto buone di ambienti diversi e sai già quali riescono.

### 1.6 Il catalogo del cliente (l'effetto più forte)

Vedere **le proprie piastrelle** vale più di qualunque spiegazione.

Se il cliente ha un negozio Shopify, bastano il suo indirizzo e un minuto (nessuna password:
si leggono le pagine pubbliche):

```bash
posa shopify --negozio www.sito-del-cliente.it
```

Il comando elenca gli articoli importati e quelli scartati con il motivo. Riavvia `posa web`
e li trovi nel catalogo, con la ricerca.

Se non ha Shopify, fatti mandare 2-3 foto frontali di singole piastrelle e aggiungile:

```bash
posa aggiungi-materiale piastrella.jpg --codice metro-bianco-10x20 --nome "Metro bianco 10x20" --formato 20x10 --fuga 2 --colore-fuga "#e8e6e1" --posa running_bond --finitura lucida
```

Controlla con `posa catalogo` e fai una prova con quell'articolo prima della demo.

### 1.7 Lista della sera prima

- [ ] `posa verifica --modelli` tutto verde, `posa verifica --online` riuscito
- [ ] tre prove riuscite, con tempi e costi annotati
- [ ] 3-4 foto scelte e già provate
- [ ] catalogo del cliente caricato e provato con almeno un articolo
- [ ] risultati già pronti aperti in una cartella (piano B)
- [ ] credito sulla chiave API, computer in carica, connessione di riserva (telefono)
- [ ] `posa web` avviato **prima** che arrivi il cliente (il primo avvio carica il modello)

---

## 2. La demo (15 minuti)

### 2.1 Scaletta

1. **Il problema (1 minuto).** "Chi compra un pavimento o un rivestimento non riesce a
   immaginarlo a casa sua. Qui carica una foto e lo vede."
2. **Prima prova, pavimento (5 minuti).** In `posa web`: carica la foto, scegli un articolo
   del *loro* catalogo, lascia "Pavimento", premi **Posa il rivestimento**.
3. **Mentre lavora, racconta** (ci vogliono 2-5 minuti: non restare in silenzio). A destra
   scorre il registro degli agenti. Fai notare:
   - un agente delimita il pavimento, un altro trova la prospettiva, un altro regola luci e
     ombre: ognuno fa una cosa sola;
   - dopo ogni passo un **controllore indipendente** approva o boccia, e se boccia il lavoro
     si rifà: "non vi arriva un risultato che nessuno ha guardato";
   - la piastrella non è "disegnata" dall'intelligenza artificiale: è **la foto del vostro
     articolo**, nel formato e con la fuga del catalogo.
4. **Il risultato.** Muovi il cursore prima/dopo. Ingrandisci su una gamba di sedia o su un
   mobile: il pavimento ci passa dietro, le ombre sono rimaste. Apri la galleria "Lavoro
   degli agenti" e mostra gli ingrandimenti che hanno visto i controllori.
5. **La garanzia.** "Fuori dal pavimento la foto è identica all'originale, pixel per pixel:
   lo verifica il programma, non una persona."
6. **Seconda prova, parete (4 minuti).** Foto di una cucina, scegli "Parete". Mentre lavora
   passa al punto 7.
7. **Come arriva sul vostro sito.** "Sul vostro sito è un pulsante nella pagina prodotto."
   Mostra le due righe da incollare (capitolo 3.4) e spiega: il cliente carica la foto dal
   telefono, sceglie pavimento o parete, dopo qualche minuto vede il risultato e lo scarica.
8. **Piani.** Canone mensile con un numero di prove comprese (capitolo 4).

### 2.2 Cosa dire con onestà (ti evita problemi dopo la firma)

- **Tempi:** 2-5 minuti a prova, non è istantaneo. Sul sito il cliente vede l'avanzamento.
- **Non tutte le foto riescono.** Foto buie, mosse o con la superficie quasi tutta coperta
  vengono rifiutate invece di produrre un risultato brutto: il cliente finale riceve
  l'invito a riprovare con un'altra foto. Le prove non riuscite non consumano la quota.
- **Una superficie per prova:** pavimento *oppure* parete.
- **È un'anteprima,** non un progetto di posa: le misure della stanza, se non indicate,
  sono stimate.

### 2.3 Domande che ti faranno

| Domanda | Risposta |
|---|---|
| Le foto dei miei clienti dove finiscono? | Sul nostro server. A prova consegnata resta solo il risultato, cancellato dopo 7 giorni; l'originale viene eliminato subito. Non raccogliamo nomi né email. Le immagini vengono elaborate da Anthropic (il fornitore dei modelli). |
| Funziona con il mio sito? | Con qualunque sito: due righe di HTML. Con Shopify c'è un'app che tiene anche il catalogo allineato. |
| Come carico i miei articoli? | Una foto frontale della singola piastrella più formato e fuga. Da Shopify in automatico. |
| E se il risultato è sbagliato? | Ogni passo è controllato; quello che non passa i controlli non viene consegnato. |
| Posso provarlo? | Sì: piano di prova con 30 prove al mese. |
| Vale per il bagno, per le pareti intere? | Sì: la zona può essere il paraschizzi o la parete intera. |
| Quanto costa a me ogni prova? | È compresa nel canone fino alla quota del piano. |

### 2.4 Se in demo qualcosa va storto

- **Ci mette più di 6-7 minuti o finisce "non consegnato":** non insistere. "Questa foto il
  sistema la manda al controllo manuale: è la stessa prudenza che avrete sul vostro sito."
  Passa a una foto già provata o apri un risultato pronto dalla cartella `pratiche/`.
- **Errore di rete o di chiave:** mostra i risultati pronti e riprova a fine incontro.
- **Porta occupata:** `posa web` ne sceglie un'altra da solo e te la scrive.

Per mostrarlo a distanza, a un cliente che non è nella stanza:

```bash
export POSA_WEB_PASSWORD='una-password-per-questa-demo'
```
```bash
posa web --share
```

Ottieni un link pubblico temporaneo protetto dalla password. Spegnilo a demo finita.

---

## 3. Il cliente ha detto sì: come glielo fai usare

`posa web` è solo la tua demo. I clienti usano il **servizio** (`posa server`): un server
tuo, sempre acceso, a cui i loro siti si collegano.

### 3.1 Una volta sola: accendere il servizio

Ti servono un server Linux con Docker (indicativamente 4 CPU e 8 GB di RAM), un dominio che
punta al server (es. `posa.tuodominio.it`) e la chiave API.

```bash
git clone https://github.com/ChiaraIA/rendering.git && cd rendering
```
```bash
cp deploy/.env.example deploy/.env
```

Compila `deploy/.env`: dominio, indirizzo pubblico in `https://`, chiave Anthropic, e i
segreti. Per generare ogni segreto (token di amministrazione, chiave dei permessi):

```bash
python -c "import secrets;print(secrets.token_urlsafe(32))"
```

Poi:

```bash
docker compose -f deploy/docker-compose.yml up -d --build
```
```bash
curl https://posa.tuodominio.it/salute
```

L'ultima riga deve rispondere `{"stato":"ok"}`. Il certificato HTTPS lo ottiene Caddy da
solo. Salva il token di amministrazione in un gestore di password: è la chiave di tutto.

Negli esempi qui sotto `SERVER` è il tuo dominio e il token è in una variabile:

```bash
export POSA_ADMIN_TOKEN='il-tuo-token'
```

### 3.2 Per ogni cliente: creazione (2 minuti)

```bash
curl -X POST https://SERVER/admin/clienti -H "Authorization: Bearer $POSA_ADMIN_TOKEN" -H "Content-Type: application/json" -d '{"id":"ceramiche-bianchi","nome":"Ceramiche Bianchi","sito":"https://www.ceramiche-bianchi.it","piano":"prova"}'
```

- `id`: nome breve, minuscole e trattini. `sito`: l'indirizzo esatto in `https://` da cui
  il pulsante funzionerà (se ha anche un sito di prova, aggiungilo in `"origini":[...]`).
- `piano`: parti da `prova` (30 prove al mese) e passa al piano pagato alla firma.

Crea le due chiavi. **Si vedono una volta sola**: copiale subito.

```bash
curl -X POST https://SERVER/admin/clienti/ceramiche-bianchi/chiavi -H "Authorization: Bearer $POSA_ADMIN_TOKEN" -H "Content-Type: application/json" -d '{"tipo":"pubblica"}'
```
```bash
curl -X POST https://SERVER/admin/clienti/ceramiche-bianchi/chiavi -H "Authorization: Bearer $POSA_ADMIN_TOKEN" -H "Content-Type: application/json" -d '{"tipo":"segreta"}'
```

- `pk_live_...` (pubblica): va nella pagina web del cliente.
- `sk_live_...` (segreta): serve a caricare il catalogo. Mandala per un canale sicuro, mai
  in chiaro in una email insieme alle istruzioni.

### 3.3 Il catalogo del cliente

**Cliente con Shopify:** segui `shopify/README.md` (registri il negozio, lui installa l'app;
il catalogo si aggiorna da solo).

**Qualunque altro sito:** un comando per articolo, con la chiave segreta. Puoi farlo tu per
lui nei primi tempi: è un servizio che si vende bene ("vi carichiamo noi il catalogo").

```bash
curl -X PUT https://SERVER/v1/catalogo/metro-bianco-10x20 -H "Authorization: Bearer sk_live_..." -F nome="Metro bianco lucido 10x20" -F formato=20x10 -F fuga=2 -F posa=running_bond -F finitura=lucida -F uso=rivestimento -F foto=@piastrella.jpg
```

La risposta dice se la foto è stata accettata. Regole per una foto buona: **una sola
piastrella, frontale**, a tutto campo o su sfondo uniforme; niente ambientazioni. Campi e
valori sono nella tabella di [SAAS.md](SAAS.md). `uso` decide dove l'articolo si può
provare: `pavimento`, `rivestimento` o tutti e due.

### 3.4 Il pulsante sul sito del cliente

Questo è tutto quello che il suo webmaster deve incollare nella pagina prodotto:

```html
<script src="https://SERVER/widget/posa.js" defer></script>
<div data-posa
     data-chiave="pk_live_..."
     data-articolo="metro-bianco-10x20"
     data-superfici="pavimento,rivestimento"
     data-testo="Prova nella tua stanza"></div>
```

- `data-articolo`: il codice dell'articolo di quella pagina (lo stesso usato nel catalogo).
- `data-superfici`: `pavimento`, `rivestimento` o entrambi; con entrambi sceglie il cliente.
- Il pulsante funziona solo dalle pagine del sito registrato al punto 3.2, e solo in https.

Fate insieme una prova dal sito vero prima di annunciarlo.

### 3.5 Il messaggio da mandare al cliente (da copiare)

> Buongiorno, il servizio "Prova nella tua stanza" è attivo per il vostro sito.
>
> 1. Nella pagina di ogni prodotto incollate le due righe che trovate qui sotto, cambiando
>    solo il codice dell'articolo.
> 2. I vostri clienti vedranno il pulsante "Prova nella tua stanza": caricano una foto,
>    scelgono pavimento o parete e in pochi minuti vedono il risultato, che possono scaricare.
> 3. Consigli da dare ai clienti: foto luminosa e nitida, con la superficie ben visibile.
> 4. Il vostro piano comprende N prove consegnate al mese; le prove non riuscite non contano.
> 5. Le foto servono solo a creare l'anteprima: l'originale viene cancellato alla consegna,
>    il risultato dopo 7 giorni. Per cancellare subito una prova, scriveteci.
>
> Per aggiungere o cambiare articoli mandateci la foto frontale della piastrella con formato
> e fuga, oppure usate la chiave segreta che vi abbiamo inviato a parte.

### 3.6 Il pagamento

Finché non colleghi Stripe decidi tu chi è attivo:

```bash
curl -X POST https://SERVER/admin/clienti/ceramiche-bianchi/stato -H "Authorization: Bearer $POSA_ADMIN_TOKEN" -H "Content-Type: application/json" -d '{"stato":"sospeso"}'
```

(`attivo`, `sospeso`, `disdetto`.) Per cambiare piano alla firma:

```bash
curl -X POST https://SERVER/admin/clienti/ceramiche-bianchi/fatturazione -H "Authorization: Bearer $POSA_ADMIN_TOKEN" -H "Content-Type: application/json" -d '{"piano":"professionale"}'
```

Con Stripe il servizio attiva e sospende da solo in base all'abbonamento: i passi sono in
[SAAS.md](SAAS.md), capitolo "Pagamenti". Ricorda che nel link di pagamento va il **codice
di pagamento** del cliente (campo `codice_pagamento` di `GET /admin/clienti`), non il suo
id. Prova tutto il giro in modalità test di Stripe prima di incassare davvero.

### 3.7 La gestione di tutti i giorni

| Cosa vuoi sapere o fare | Comando |
|---|---|
| Tutti i clienti, con stato e uso del mese | `curl https://SERVER/admin/clienti -H "Authorization: Bearer $POSA_ADMIN_TOKEN"` |
| Uso e costo di un cliente, mese per mese | `curl https://SERVER/admin/clienti/ID/uso -H "Authorization: Bearer $POSA_ADMIN_TOKEN"` |
| I piani | `curl https://SERVER/admin/piani -H "Authorization: Bearer $POSA_ADMIN_TOKEN"` |
| Chiave finita nelle mani sbagliate | `curl -X DELETE https://SERVER/admin/clienti/ID/chiavi/INIZIO -H "Authorization: Bearer $POSA_ADMIN_TOKEN"` poi creane una nuova (`INIZIO` = i primi 14 caratteri) |
| Un cliente finale chiede di cancellare la sua foto | `curl -X DELETE https://SERVER/v1/pratiche/ID_PRATICA -H "Authorization: Bearer sk_live_..."` |
| Fine del rapporto: cancellare tutto | `curl -X DELETE https://SERVER/admin/clienti/ID -H "Authorization: Bearer $POSA_ADMIN_TOKEN"` |

Guarda una volta al mese `costo_agenti_usd` di ogni cliente: è quello che hai speso per lui
e va confrontato con il canone. Fai una copia giornaliera della cartella dei dati del
server (clienti, chiavi, cataloghi): senza, un guasto ti fa perdere tutti i clienti.

---

## 4. I prezzi: come ragionare

Il programma non contiene prezzi: li decidi tu e li scrivi nei piani. Il conto è questo.

- **Costo di una prova**: quello che hai misurato nella prova generale (indicativamente
  0,5-1,5 USD). Anche le prove non riuscite costano, ma non le fatturi.
- **Costo fisso**: il server.
- **Canone minimo sensato** = prove comprese × costo di una prova × 1,2 (per le non
  riuscite) + la tua quota di server, più il tuo margine.

Esempio di calcolo, **solo per capire il meccanismo** (i numeri veri sono i tuoi): se una
prova ti costa 1 USD, un piano da 150 prove ti costa fino a circa 180 USD al mese quando
viene usato tutto. Un canone sotto quella cifra è in perdita con i clienti che lo usano
davvero.

Piani già pronti: `prova` (30), `base` (150, solo pavimento), `professionale` (600,
pavimento e parete, a consumo oltre la quota), `azienda` (3000). Per cambiare quote e
scrivere i prezzi:

```bash
curl -X PUT https://SERVER/admin/piani/professionale -H "Authorization: Bearer $POSA_ADMIN_TOKEN" -H "Content-Type: application/json" -d '{"nome":"Professionale","prove_al_mese":600,"superfici":["pavimento","rivestimento"],"oltre_quota_a_consumo":true,"canone_eur":0,"prova_extra_eur":0}'
```

(metti i tuoi valori al posto degli zeri). La parete come opzione dei piani più alti è un
modo naturale di differenziare l'offerta. Per abbassare il costo per prova si può usare un
modello più economico (`POSA_MODEL`), ma va riprovata la qualità.

---

## 5. Prima di incassare dal primo cliente

Queste cose non le può verificare il programma al posto tuo:

- [ ] procedura completa provata con gli agenti veri su almeno 20 foto reali, pavimenti e
      pareti, con la percentuale di prove riuscite annotata (è il numero che i clienti
      vorranno sapere);
- [ ] giro completo dei pagamenti in modalità test di Stripe;
- [ ] batteria d'attacco rilanciata sul server vero (comandi in [SICUREZZA.md](SICUREZZA.md));
- [ ] disco del server e copie di sicurezza cifrati; chiave API dedicata con tetto di spesa;
- [ ] contratto e informativa: foto come dati personali, conservazione di 7 giorni,
      Anthropic come fornitore che elabora le immagini;
- [ ] parere sulla licenza del modello di segmentazione (pesi MIT, addestrati su un archivio
      di immagini distribuito per ricerca): vedi [SAAS.md](SAAS.md).
