# posa

Posa un materiale da catalogo su una superficie di una foto, **senza modificare nessun pixel fuori dalla maschera**.

Input: una foto, una maschera confermata della superficie e un materiale a catalogo.
Output: la foto con il materiale applicato (master PNG/TIFF con il profilo ICC originale), la maschera usata, un'anteprima e un `report.json` con parametri ed esito dei controlli.

> Stato: **M1**. Struttura, contratti dati, CLI, compositor e test di integrità.
> In M1 il materiale viene reso come **riempimento piatto** con il colore medio della texture.
> Tiling, prospettiva e scala reale arrivano in M2, la luce in M3.

## Garanzie

| Regola | Come è garantita |
|---|---|
| Originale in sola lettura, identificato da SHA-256 | si legge solo in `rb`; copia nel job con permessi `0444`; array con `writeable=False`; ri-hash a fine render |
| Stessa risoluzione | controllo bloccante `resolution_match` + `pixel_format_match` |
| Master PNG/TIFF con ICC preservato | il master viene riletto dopo la scrittura: pixel e byte ICC devono coincidere |
| Solo `compositor.py` crea l'immagine finale | `CompositeResult` è sigillato; test AST sui confini |
| Nessun output generativo come risultato finale | gli adapter restituiscono solo `MaskProposal` o `LightHint` |
| Texture e misure solo dal catalogo | `RenderRequest` contiene un `material_id`; l'hash della texture viene verificato |
| Render solo dopo la conferma della maschera | conferma legata a sha della maschera + sha dell'immagine |
| `np.array_equal(final[alpha == 0], original[alpha == 0])` | eseguito dopo ogni render (e garantito per costruzione: `final = original.copy()`) |
| Un controllo bloccante fallito → niente master | viene scritto solo `report.json` (`--keep-invalid` salva in `quarantine/`) |

## Installazione e test

```bash
python -m venv .venv && source .venv/bin/activate
```
```bash
pip install -e ".[dev]"
```
```bash
pytest -q --cov=app --cov-report=term-missing
```
```bash
ruff check . && mypy app
```

## Uso

```bash
posa demo --out runs/demo
```
```bash
posa ingest foto.jpg --job runs/cucina
```
```bash
posa mask import runs/cucina maschera.png
```
```bash
posa mask confirm runs/cucina --sha <primi 12 caratteri dello sha>
```
```bash
posa render runs/cucina --material gres-grigio-30x30
```
```bash
posa catalog list
```

La maschera è un PNG in scala di grigi a 8 bit, della stessa risoluzione della foto **già orientata** (dopo l'EXIF): `0` = fuori, `255` = dentro, valori intermedi = bordo morbido.

Exit code: `0` ok · `2` input non valido · `3` maschera assente o non confermata · `4` geometria (M2) · `5` risultato non valido.

## Struttura

```
app/
  cli.py  config.py
  core/        models, errors, colorspace, ingest, segmentation, material, compositor, validator
  catalog/     loader + data/ (catalogo di esempio, rigenerabile con `python -m app.devtools.synthetic`)
  storage/     jobstore (layout dei job), exporter (unico punto di scrittura del master)
  pipeline/    job (macchina a stati), render (orchestrazione)
  adapters/    contratti per SAM / depth / Nano Banana (M4, M6)
  devtools/    generatori di dati sintetici per test e demo
tests/         unit, integration, architecture
```

Layout di un job:

```
runs/<job>/source/<file originale> (0444)   normalized.png   source.icc   job.json
           masks/<sha>.png|.json
           renders/render-NNN/final.png  mask.png  preview.jpg  report.json
           quarantine/render-NNN.INVALID.png   (solo con --keep-invalid)
```
