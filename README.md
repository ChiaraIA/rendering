# posa — pavimenti e rivestimenti nuovi, nella foto del cliente

**posa** è un sistema **multiagente** venduto come servizio: il cliente carica la foto della
sua stanza, sceglie un articolo dal catalogo e riceve la stessa foto con

- il nuovo **pavimento** posato in prospettiva, oppure
- il nuovo **rivestimento a parete**: per esempio le piastrelle della cucina tra il piano
  di lavoro e i pensili (o le finestre), anche su due pareti ad angolo.

**La foto originale non viene mai toccata fuori dalla superficie rivestita**, e il materiale
inserito è esattamente l'articolo del catalogo.

```
                      ┌──────────────────── ORCHESTRATORE (Claude) ─────────────────────┐
 foto + richiesta ──► │   pianifica · delega · ripete con il feedback · escalation       │
                      └───┬────────────┬────────────┬────────────┬────────────┬─────────┘
                          ▼            ▼            ▼            ▼            ▼
                     «pavimento»  «geometra»   «materiali»    «luci»      esegui_posa
                     o «parete»   4 angoli +   codice o      ombre,       render con texture
                     maschera     misure (per  campione      riflessi,    del catalogo e
                     precisa      ogni parete)               luce calda   luci approvate
                          │            │            │            │            │
                          ▼            ▼            ▼            ▼            ▼
                     controllo-   controllo-   controllo-   controllo-   controllo-integrità
                     maschera     geometria    materiale    luce         (codice) + revisore-
                     (ingrandi-   (+ focale                              finale con i bordi
                     menti veri)   EXIF)                                 ingranditi (Claude)
                                                                              │
                                     CANCELLI NEL CODICE ─────────────►  consegna
```

Per venderlo come abbonamento ai siti che lo integrano (clienti, chiavi API, piani, quote,
widget, pagamenti): **[docs/SAAS.md](docs/SAAS.md)**.

## Perché ci si può fidare del risultato

| Garanzia | Come è ottenuta |
|---|---|
| L'originale non cambia | Viene letto in sola lettura, copiato a `0444` e identificato con SHA-256. L'immagine finale nasce come copia esatta e viene modificata **solo dentro la maschera**. Dopo il render il codice verifica `np.array_equal(final[alpha == 0], original[alpha == 0])`; se fallisce, non c'è consegna. |
| Il rivestimento è quello giusto | Texture, formato, fughe e schema di posa arrivano **solo dal catalogo**, con l'hash di ogni foto verificato. Nessun agente "disegna" il materiale. |
| I contorni li decide la foto | Gli agenti dicono *cosa* escludere (un riquadro attorno a un vaso, a una presa); il contorno esatto lo calcola il codice sulla foto a piena risoluzione. I controllori giudicano su ingrandimenti presi dall'originale, scelti dove un errore è più probabile. |
| Nessun agente si approva da solo | I controllori sono sessioni Claude **separate, avviate dal codice** dentro gli strumenti: vedono solo le immagini da giudicare e possono rispondere solo con un verdetto. Le approvazioni le scrive il codice. |
| Niente scorciatoie | `esegui_posa` e `consegna` rifiutano di procedere se manca un'approvazione o se l'integrità fallisce. Dopo 3 bocciature dello stesso passo la pratica passa a un operatore. |
| Tracciabilità | La cartella della pratica contiene ogni proposta, overlay, verdetto (con motivo), gli input grezzi degli agenti (`lavoro/tracce.jsonl`) e `report.json`. |

## Provarlo su GitHub Codespaces (niente da installare)

1. Su GitHub: **Code → Codespaces → ⋯ → New with options**, scegli il branch e crea il
   Codespace. Quando lo chiede, inserisci la tua `ANTHROPIC_API_KEY`: viene salvata come
   segreto di Codespaces e non finisce nel codice.
2. Attendi l'installazione automatica (`pip install -e '.[demo]'`), poi nel terminale scrivi:
   ```bash
   posa web
   ```
3. Si apre la scheda della porta 7860. Se non si apre, vai su **Ports → 7860 → Open in
   Browser**. Carica la tua foto e premi **"Posa il rivestimento"**.

**Se qualcosa non parte**
- Nel terminale esegui `posa verifica`: controlla l'installazione e per ogni problema dice
  come risolverlo. Con `posa verifica --online` fa anche una chiamata di prova agli agenti
  (pochi centesimi).
- Se il comando `posa` non esiste, l'installazione automatica non è finita o è fallita:
  rilanciala con `pip install -e '.[demo]'`. Il dettaglio è nel log di creazione: Command Palette
  (F1) → **"Codespaces: View Creation Log"**.
- Se `posa web` dice che la porta 7860 è in uso, un'altra copia è già accesa: aprila dalla
  scheda **PORTS**, oppure fermala con `pkill -f "posa web"`. In ogni caso `posa web` si
  avvia da solo sulla prima porta libera e te la indica.
- Se la chiave è stata aggiunta dopo aver creato il Codespace, il Codespace non la vede:
  ricrealo oppure, solo per la sessione, `export ANTHROPIC_API_KEY=sk-ant-...`.

## Avvio rapido (demo)

Requisiti: Python 3.11+ e una chiave API Anthropic.

```bash
git clone https://github.com/ChiaraIA/rendering.git && cd rendering
```
```bash
python -m venv .venv && source .venv/bin/activate
```
```bash
pip install -e '.[demo]'
```
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```
```bash
posa web
```

Apri <http://localhost:7860>, premi **"Usa una stanza di esempio"** (oppure carica la foto di
una stanza vera), scegli il rivestimento, dove posarlo (pavimento o parete) e premi **"Posa il rivestimento"**. A destra vedi in
diretta cosa fanno gli agenti, il prima/dopo con il cursore e i file da scaricare.

Per mandare un link temporaneo al cliente:

```bash
posa web --share
```

### Da terminale

```bash
posa run foto_stanza.jpg --materiale gres-grigio-30x30 --stanza 4.2x3.6
```
```bash
posa run foto_stanza.heic --campione piastrella_negozio.jpg
```
```bash
posa run cucina.jpg --materiale ceramica-blu-20x20 --superficie rivestimento --zona paraschizzi
```
```bash
posa demo
```
```bash
posa demo --parete
```
```bash
posa catalogo
```

La pratica finisce in `pratiche/<data-ora>/`, e il risultato in `consegna/`:
`final.png` (piena risoluzione, profilo colore originale), `mask.png`, `preview.jpg`,
`prima_dopo.jpg` e `report.json`.

## Contorni precisi, anche con molti oggetti

Il difetto che un cliente nota subito è il contorno: il nuovo pavimento disegnato sopra un
oggetto, oppure una striscia del pavimento vecchio rimasta accanto a una sedia. La maschera
nasce così:

1. **Dove.** Un modello di segmentazione (**UperNet-ConvNeXt base**, pesi con licenza MIT,
   addestrato su ADE20K) riconosce pavimento, pareti e 130 tipi di oggetti. Dove ha un
   dubbio fondato che ci sia un oggetto, l'oggetto vince.
2. **Il bordo.** Il bordo del modello viene portato sui contorni veri della foto con un
   filtro guidato: decide zona per zona, quindi l'ombra sotto un tavolo resta pavimento e il
   bordo di un tappeto viene seguito. (La versione precedente usava GrabCut con un unico
   modello di colore per tutta la foto: nelle stanze piene di oggetti *peggiorava* la
   maschera del modello.) Il bordo finale è calcolato alla risoluzione piena della foto.
3. **Le correzioni.** L'agente «pavimento» (o «parete») guarda il risultato e corregge solo
   dove serve. Lo strumento principale è `escludi_oggetti`: un riquadro largo attorno a ciò
   che è rimasto dentro (una ciotola, un cavo, una presa); il contorno preciso lo trova il
   codice. I poligoni disegnati dall'agente vengono anch'essi portati sui contorni veri.
4. **I controlli.** Il controllo-maschera riceve ingrandimenti presi dalla **foto
   originale**, scelti dove il bordo è più fitto e dove una mappa di rischio segnala un
   probabile errore (una zona di colore diverso dentro la maschera, un bordo che non
   coincide con nessun contorno). Il revisore finale vede il PRIMA/DOPO ingrandito negli
   stessi punti: se trova un oggetto coperto o una chiazza di pavimento vecchio, la
   maschera si rifà e la posa precedente non è più consegnabile.

**Quanto migliora, misurato.** Stesse 109 foto vere di stanze piene di oggetti (ADE20K
validation, verità disegnata a mano), stessa misura, solo la maschera automatica cioè
*prima* delle correzioni degli agenti. Gli errori sono in % dell'area della superficie.

| Maschera automatica | Versione precedente | Ora |
|---|---|---|
| Pavimento finito sopra oggetti — foto tipica (mediana) | 0,44% | **0,31%** |
| Pavimento finito sopra oggetti — foto difficili (90° percentile) | 7,3% | **5,1%** |
| Chiazze visibili di pavimento sopra oggetti, per foto | 1,6 | **1,0** |
| Pavimento vecchio rimasto scoperto — mediana / 90° percentile | 1,8% / 12,6% | 2,1% / 12,6% |
| Parete: rivestimento sopra oggetti — mediana / 90° percentile | 4,9% / 29,7% | **4,4% / 19,2%** |
| Parete: muro rimasto scoperto — mediana / 90° percentile | 4,3% / 19,9% | **3,1% / 13,1%** |

Cosa dicono questi numeri: la maschera automatica copre meno oggetti (un terzo di chiazze
in meno) e sulle pareti migliora da entrambi i lati; il pavimento vecchio lasciato scoperto
dalla sola maschera automatica **non** è migliorato, perché nel dubbio ora vince l'oggetto.
Quello che resta (tappeti, oggetti che il modello non conosce, pavimento di un'altra
stanza) è il lavoro degli agenti e dei controllori, che ora hanno strumenti e ingrandimenti
adatti. La misura non vede gli scarti sotto i 3-4 pixel (la verità di ADE20K è a poligoni):
la pulizia dei bordi si giudica a occhio sugli ingrandimenti. **Non è ancora stata fatta una
prova completa con gli agenti veri** (serve la chiave API): va fatta prima della vendita.

Se PyTorch non è installato la maschera la disegnano gli agenti (meno precisa). In locale:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
```bash
pip install -e ".[segmentazione]"
```
```bash
posa verifica --modelli
```

L'ultimo comando scarica il modello (circa 470 MB, una volta sola) da Hugging Face. Per
disattivare la segmentazione: `POSA_SEGMENTAZIONE=0`. Per un altro modello UperNet:
`POSA_SEG_MODEL` e `POSA_SEG_REVISIONE` (la versione dei pesi è fissata apposta).

Per usare un modello più economico per gli agenti: `posa web --modello sonnet`. Costa circa
la metà, ma nelle prove è meno affidabile sulla geometria della stanza.

## Rivestimenti a parete

Con `--superficie rivestimento` (o scegliendo "Parete" nel widget) gli stessi cancelli
valgono per una zona di muro:

- **zona**: `paraschizzi` (la fascia tra piano di lavoro e pensili, cappa o finestre),
  `parete` (parete intera) o `automatica`. L'agente «parete» racchiude la zona; il modello
  riconosce il muro, quindi piano, pensili, finestre e oggetti davanti al muro restano
  fuori. Prese, interruttori e ciò che il modello non conosce si escludono con un riquadro;
- **prospettiva**: per ogni parete (fino a 3: cucine a L o a U) un rettangolo sul muro con
  larghezza e altezza reali, stimate da indizi (50-60 cm tra piano e pensili, moduli da
  60 cm) o date dal cliente. Il controllore giudica una griglia di quadrati da 10 cm
  disegnata sul muro;
- **posa**: la prima fila poggia **intera sul piano di lavoro** e i tagli stanno in alto,
  come fa un posatore; le file proseguono alla stessa altezza oltre lo spigolo;
- **luce**: l'ombra dei pensili sul muro resta, l'ombra di contatto è più leggera che a
  terra;
- **catalogo**: ogni articolo dichiara dove si può posare (`uso`: pavimento, rivestimento
  o entrambi); un articolo solo da pavimento non viene proposto a parete.

## Luci e ombre (agente «luci»)

Un pavimento giusto ma illuminato "a caso" sembra incollato. Per questo luci e ombre sono un
passo a sé, con un proprio agente e un proprio controllore. Il codice ricava la luce dal
**vecchio pavimento della foto**, dopo averne cancellato il disegno (fughe, venature):

- **luce d'ambiente**: zone più illuminate e più buie della stanza;
- **ombre**: sotto il tavolo, le sedie, lungo i mobili, come nella foto;
- **riflessi** di lampade, tende e mobili, dosati secondo la **finitura** del materiale
  (opaca, satinata, lucida);
- **ombra di contatto** a ridosso di ciò che poggia a terra e dei muri;
- **colore della luce** (stimato dalle superfici bianche della foto: lampade = luce calda)
  e **grana** della fotocamera.

L'agente «luci» decide quanto pesa ciascuna parte (`ombre`, `riflessi`, `contatto`,
`luce_calda`, `grana`); il **controllo-luce** confronta PRIMA e DOPO, anche con ingrandimenti
dove il pavimento tocca gli oggetti, e boccia ombre mancanti o esagerate, "fantasmi" del
vecchio pavimento, riflessi incoerenti con la finitura o un pavimento più acceso della
stanza. `esegui_posa` parte solo con luci approvate.

## Catalogo dal negozio Shopify

Gli articoli in vendita sul vostro negozio Shopify diventano rivestimenti da provare, con
le foto che caricate sulle pagine prodotto. Non serve nessuna chiave: il programma legge
l'elenco pubblico degli articoli (`https://<negozio>/products.json`).

```bash
posa shopify --negozio mio-negozio.myshopify.com
```

Va bene anche il dominio del sito (`www.miosito.it`). Il negozio viene ricordato: da quel
momento **ogni avvio di `posa web` riscarica il catalogo**, e nella pagina c'è il pulsante
**"Aggiorna dal negozio Shopify"** per farlo senza riavviare. Gli articoli non cambiati non
vengono riscaricati; quelli tolti dal negozio spariscono dal catalogo. Per limitarsi a una
collezione: `--collezione pavimenti` (l'handle della collezione).

**Quale foto viene usata.** Il modo più sicuro: su Shopify, nel **testo alternativo** della
foto della singola piastrella, scrivete `texture`. Senza questa indicazione il programma
sceglie da solo:
- la foto in cui riconosce **una piastrella su sfondo uniforme** (anche un po' storta: viene
  raddrizzata);
- oppure la **prima foto**, se è la superficie della piastrella a tutto campo;
- le **ambientazioni** vengono scartate.

Ogni **variante con una foto propria** (es. i colori) diventa un articolo a sé. Più foto
`texture` sullo stesso articolo (o il tag `misto`) = piastrelle diverse mescolate nella posa,
come le cementine.

**Regole di posa.** Il formato si legge dal titolo (`Gres 60x60`, `Rovere 20x120`,
`Esagona 25 cm`) o dall'opzione "Formato" della variante. Il resto ha valori ragionevoli
per tipo: legno e laminato sfalsati e senza fuga, esagoni per i titoli con "esagona",
fuga di 2-3 mm per le piastrelle, finitura lucida se nel testo c'è "lucido", "lappato" o
"levigato". Per cambiarli si usano i **tag** dell'articolo su Shopify:

| Tag | Esempio | Cosa fa |
|---|---|---|
| `formato:` | `formato:60x60` | misura in cm (per gli esagoni: punta-punta) |
| `fuga:` | `fuga:2` | larghezza della fuga in mm (`fuga:0` = senza fuga) |
| `colore-fuga:` | `colore-fuga:#cfccc5` | colore della fuga (altrimenti in tinta con la piastrella) |
| `posa:` | `posa:running_bond` | `grid`, `running_bond` (sfalsata) o `esagoni` |
| `sfalsamento:` | `sfalsamento:0.33` | sfalsamento della posa a correre |
| `rotazione:` | `rotazione:45` | posa in diagonale |
| `finitura:` | `finitura:satinata` | `opaca`, `satinata` o `lucida` (riflessi) |
| `render:no` | | esclude l'articolo (colle, accessori…) |

Sono riconosciuti anche i tag nello stile `Chiave_Valore` (`Formato_60x120`,
`Formato_Esagonale`, `Superficie_Lucida`, `Utilizzo_Pavimento`, `Utilizzo_Rivestimento`).
Gli articoli con il tag `Fuori Produzione` e quelli che non sono rivestimenti (box doccia,
accessori…) vengono saltati; la posa a spina di pesce non è ancora supportata.

Quando la foto è la superficie della piastrella va bene in qualunque posizione (anche
ruotata); più foto della stessa piastrella con colori simili diventano **varianti** alternate
nella posa, come le diverse facce di un gres. Le foto con più piastrelle e fughe
(composizioni) e le ambientazioni vengono scartate.

Con migliaia di articoli, nella pagina web c'è la **ricerca** (es. `grigio cemento 60x60`):
la galleria mostra i risultati e un clic su una miniatura sceglie l'articolo. Anche
l'agente «materiali» cerca per parole invece di scorrere tutto il catalogo.

`posa shopify` stampa per ogni articolo cosa ha fatto: foto usate e scartate, valori presi
per buoni e cosa aggiungere quando un articolo non si può posare.

## Il servizio in abbonamento

`posa server` è il prodotto vero: un solo server per tutti i clienti che pagano il canone.

- **Qualunque sito**: una riga di HTML (`/widget/posa.js` + chiave pubblica) aggiunge il
  pulsante "Prova nella tua stanza"; il catalogo si carica via API con la chiave segreta.
- **Negozi Shopify**: l'app nella cartella `shopify/` aggiunge il blocco alla pagina
  prodotto e tiene il catalogo allineato al negozio.
- **Piani**: prove al mese, superfici comprese (pavimento, rivestimento), stop o consumo
  oltre la quota; uso e costo degli agenti per cliente, mese per mese.
- **Pagamenti**: l'abbonamento Stripe attiva, sospende o chiude il cliente da solo.

Tutto, passo per passo, in [docs/SAAS.md](docs/SAAS.md).

```bash
cp deploy/.env.example deploy/.env
```
```bash
docker compose -f deploy/docker-compose.yml up -d --build
```

Poi si attivano i clienti: i siti con `docs/SAAS.md`, i negozi Shopify con
`shopify/README.md`. Il servizio si installa con `pip install ".[server,segmentazione]"`:
il gruppo `server` porta Pillow ≥ 13 e non va installato insieme a `demo`.

Sicurezza: modello delle minacce, regole di esercizio e verifiche da ripetere a ogni
rilascio in [docs/SICUREZZA.md](docs/SICUREZZA.md).

## Mettere i vostri rivestimenti nel catalogo

Per ogni articolo servono le foto di **una singola piastrella o doga**: frontale, ritagliata
sul bordo, senza fuga. Più foto diventano **varianti**, che vengono alternate nella posa.

```bash
posa aggiungi-materiale foto1.jpg foto2.jpg foto3.jpg --codice gres-antracite-60x60 --nome "Gres antracite 60x60" --formato 60x60 --fuga 2 --colore-fuga "#3a3a3c"
```
```bash
posa aggiungi-materiale doga1.jpg doga2.jpg --codice rovere-naturale-120x20 --nome "Rovere naturale" --formato 120x20 --posa running_bond --sfalsamento 0.33
```

Fughe, schema (`grid` o `running_bond`), rotazione (`--rotazione 45` per la diagonale) e
finitura (`--finitura opaca|satinata|lucida`, decide i riflessi) sono **fissati dal
materiale**: il cliente sceglie solo l'articolo. I materiali di esempio nella
repo sono texture sintetiche, generate via codice.

## Cosa aspettarsi

- **Tempo:** 2-5 minuti a pratica. **Costo:** circa 0,5-1,5 USD di API, con tutti gli agenti su `claude-opus-5-5`. Si può cambiare modello con `POSA_MODEL`.
- **Foto supportate:** JPEG, HEIC (iPhone, anche Display P3), PNG, TIFF e WebP a 8 bit. L'orientamento EXIF viene applicato.
- **Limiti:** una superficie per prova (il pavimento *oppure* le pareti: per entrambe servono due prove); solo superfici piane; fino a 3 pareti per rivestimento. Oggetti che il modello non conosce (prese, cavi, giochi) dipendono dall'occhio degli agenti e dei controllori. Se il cliente non indica le misure, le stimano gli agenti e il report lo segnala.

## Struttura

```
app/
  agents/      orchestrator.py (Agent SDK), tools.py (strumenti + cancelli), checkers.py
               (controllori indipendenti), state.py (stato e cancelli), prompts.py
               (pavimento e parete)
  adapters/    semantic.py (modello di segmentazione: pavimento, parete, oggetti)
  core/        ingest, colorspace/icc, masking (bordi sulla foto, oggetti nei riquadri,
               mappa di rischio), geometry (pavimento e pareti), material (posa in
               prospettiva), lighting, compositor (unico creatore del risultato),
               validator, overlays
  catalog/     catalogo (schema 2.0, varianti con hash, uso pavimento/rivestimento)
  server/      api.py (widget, API clienti, Shopify, Stripe, amministrazione), store.py
               (clienti, chiavi, piani, uso), jobs.py (coda), static/posa.js (widget)
  storage/     cartella della pratica, exporter (unico che scrive il master), catalogo
               dei clienti (Shopify e API)
  devtools/    stanze e texture sintetiche per test e demo
  cli.py  web.py
tests/         unit, agents (strumenti e cancelli con controllori finti, pavimento e parete),
               server (servizio, clienti, quote, pagamenti), architecture
```

## Sviluppo

```bash
pip install -e ".[dev]"
```
```bash
pytest -q
```
```bash
ruff check . && mypy app
```

I test non chiamano i modelli: gli strumenti e i cancelli vengono verificati con controllori
finti, comprese bocciature, limite di tentativi, escalation e integrità violata (1 bit
alterato fuori dal pavimento → nessuna consegna).
