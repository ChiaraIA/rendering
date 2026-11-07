# posa — il pavimento nuovo, nella foto della stanza del cliente

**posa** è un sistema **multiagente**: il cliente carica la foto della sua stanza, sceglie un
rivestimento dal catalogo (o fotografa un campione) e riceve la stessa foto con il nuovo
pavimento posato in prospettiva. **La foto originale non viene mai toccata fuori dal
pavimento**, e il rivestimento inserito è esattamente l'articolo del catalogo.

```
                         ┌──────────────── ORCHESTRATORE (Claude) ────────────────┐
 foto + richiesta ──►    │ pianifica · delega · ripete con il feedback · escalation│
                         └───┬───────────────┬───────────────┬──────────────┬─────┘
                             ▼               ▼               ▼              ▼
                        «pavimento»      «geometra»      «materiali»    esegui_posa
                        maschera del     4 angoli della  codice o       render deterministico
                        pavimento        stanza + misure campione       con la texture vera
                             │               │               │              │
                             ▼               ▼               ▼              ▼
                     controllo-maschera controllo-geometria controllo-  controllo-integrità (codice)
                     (Claude, sessione  (Claude + verifica  materiale   + revisore-finale (Claude)
                      separata)          focale EXIF)                       │
                                                                            ▼
                                     CANCELLI NEL CODICE ───────────► consegna
```

## Perché ci si può fidare del risultato

| Garanzia | Come è ottenuta |
|---|---|
| L'originale non cambia | Viene letto in sola lettura, copiato a `0444` e identificato con SHA-256. L'immagine finale nasce come copia esatta e viene modificata **solo dentro la maschera**. Dopo il render il codice verifica `np.array_equal(final[alpha == 0], original[alpha == 0])`; se fallisce, non c'è consegna. |
| Il rivestimento è quello giusto | Texture, formato, fughe e schema di posa arrivano **solo dal catalogo**, con l'hash di ogni foto verificato. Nessun agente "disegna" il pavimento. |
| Nessun agente si approva da solo | I controllori sono sessioni Claude **separate, avviate dal codice** dentro gli strumenti: vedono solo le immagini da giudicare e possono rispondere solo con un verdetto. Le approvazioni le scrive il codice. |
| Niente scorciatoie | `esegui_posa` e `consegna` rifiutano di procedere se manca un'approvazione o se l'integrità fallisce. Dopo 3 bocciature dello stesso passo la pratica passa a un operatore. |
| Tracciabilità | La cartella della pratica contiene ogni proposta, overlay, verdetto (con motivo), gli input grezzi degli agenti (`lavoro/tracce.jsonl`) e `report.json`. |

## Provarlo su GitHub Codespaces (niente da installare)

1. Su GitHub: **Code → Codespaces → ⋯ → New with options**, scegli il branch e crea il
   Codespace. Quando lo chiede, inserisci la tua `ANTHROPIC_API_KEY`: viene salvata come
   segreto di Codespaces e non finisce nel codice.
2. Attendi l'installazione automatica (`pip install -e .`), poi nel terminale scrivi:
   ```bash
   posa web
   ```
3. Si apre la scheda della porta 7860. Se non si apre, vai su **Ports → 7860 → Open in
   Browser**. Carica la tua foto e premi **"Posa il pavimento"**.

**Se qualcosa non parte**
- Nel terminale esegui `posa verifica`: controlla l'installazione e per ogni problema dice
  come risolverlo. Con `posa verifica --online` fa anche una chiamata di prova agli agenti
  (pochi centesimi).
- Se il comando `posa` non esiste, l'installazione automatica non è finita o è fallita:
  rilanciala con `pip install -e .`. Il dettaglio è nel log di creazione: Command Palette
  (F1) → **"Codespaces: View Creation Log"**.
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
pip install -e .
```
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```
```bash
posa web
```

Apri <http://localhost:7860>, premi **"Usa una stanza di esempio"** (oppure carica la foto di
una stanza vera), scegli il rivestimento e premi **"Posa il pavimento"**. A destra vedi in
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
posa demo
```
```bash
posa catalogo
```

La pratica finisce in `pratiche/<data-ora>/`, e il risultato in `consegna/`:
`final.png` (piena risoluzione, profilo colore originale), `mask.png`, `preview.jpg`,
`prima_dopo.jpg` e `report.json`.

## Mettere i vostri rivestimenti nel catalogo

Per ogni articolo servono le foto di **una singola piastrella o doga**: frontale, ritagliata
sul bordo, senza fuga. Più foto diventano **varianti**, che vengono alternate nella posa.

```bash
posa aggiungi-materiale foto1.jpg foto2.jpg foto3.jpg --codice gres-antracite-60x60 --nome "Gres antracite 60x60" --formato 60x60 --fuga 2 --colore-fuga "#3a3a3c"
```
```bash
posa aggiungi-materiale doga1.jpg doga2.jpg --codice rovere-naturale-120x20 --nome "Rovere naturale" --formato 120x20 --posa running_bond --sfalsamento 0.33
```

Fughe, schema (`grid` o `running_bond`) e rotazione (`--rotazione 45` per la diagonale) sono
**fissati dal materiale**: il cliente sceglie solo l'articolo. I materiali di esempio nella
repo sono texture sintetiche, generate via codice.

## Cosa aspettarsi

- **Tempo:** 2-5 minuti a pratica. **Costo:** circa 0,5-1,5 USD di API, con tutti gli agenti su `claude-opus-5-5`. Si può cambiare modello con `POSA_MODEL`.
- **Foto supportate:** JPEG, HEIC (iPhone, anche Display P3), PNG, TIFF e WebP a 8 bit. L'orientamento EXIF viene applicato.
- **Limiti della prima versione:** una sola superficie piana (il pavimento); niente riflessi o superfici curve. Se il cliente non indica le misure della stanza, le stimano gli agenti e il report lo segnala.

## Struttura

```
app/
  agents/      orchestrator.py (Agent SDK), tools.py (strumenti + cancelli), checkers.py
               (controllori indipendenti), state.py (stato e cancelli), prompts.py
  core/        ingest, colorspace/icc, masking, geometry, material (posa in prospettiva),
               lighting, compositor (unico creatore del risultato), validator, overlays
  catalog/     catalogo (schema 2.0, varianti con hash)
  storage/     cartella della pratica, exporter (unico che scrive il master)
  devtools/    stanze e texture sintetiche per test e demo
  cli.py  web.py
tests/         unit, agents (strumenti e cancelli con controllori finti), architecture
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
