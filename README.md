# Sax AI — prepoznavanje nota saksofona

Projekat za prepoznavanje muzičkih nota (tonova) odsviranih na saksofonu, na
osnovu audio snimka. Zvuk se konvertuje u mel-spektrogram (slika), nad kojim
se trenira konvolutivna neuronska mreža (CNN) koja klasifikuje notu
(A, A#, B, C, C#, C2, D, D#, E, F, F#, G, G#).

Projekat je nivo 3 — uključuje kompletan proces treninga modela (priprema
podataka, trening, evaluacija) kao i mogućnost evaluacije modela nad novim,
neviđenim podacima (snimak uživo preko mikrofona ili preko web/Flask API-ja).

## Struktura projekta

```
code/
  image/
    create-spectograms.py   # .wav -> mel-spektrogram (.npy)
    create-X-Y.py           # spaja .npy spektrograme u X.npy / Y.npy / label_map.npy
    model/
      create.py             # trening CNN modela (čuva sax_model.h5 i training_krive.png)
      evaluate.py            # evaluacija modela (classification report + confusion matrix)
      api.py                 # Flask API za predikciju note iz audio snimka
      terminal.py             # CLI demo: snimanje sa mikrofona i predikcija u realnom vremenu
  audio/                    # alternativni audio pipeline (DAG, augmentacija podataka)
doc/                         # primeri mel-spektrograma i feature vektori
sausau/                      # generisane melodije i test snimci
ui/
  index.html                 # jednostavan web frontend koji poziva Flask API
main.py
pyproject.toml
```

## Instalacija (uv)

Projekat koristi [uv](https://docs.astral.sh/uv/) za upravljanje Python
okruženjem i zavisnostima.

```bash
# instalacija uv (ako nije instaliran)
# https://docs.astral.sh/uv/getting-started/installation/

# instalacija svih zavisnosti definisanih u pyproject.toml / uv.lock
uv sync
```

Sve dalje komande pokreću se preko `uv run`, čime se automatski koristi
virtuelno okruženje projekta.

## Priprema podataka

Podaci su već pripremljeni i podeljeni po notama u
`code/audio/dag/test-data/` (po jedan podfolder za svaku notu, npr. `A (1)`,
`C# (1)`...), a iz njih su već generisani gotovi feature fajlovi
`X.npy` / `Y.npy` / `label_map.npy` u `code/image/model/`, korišćeni za
trening modela koji je već istreniran (`sax_model.h5`).

Skripte `create-spectograms.py` i `create-X-Y.py` u `code/image/` ne treba
ponovo pokretati — one postoje kao dokumentacija pipeline-a (kako se
od sirovih `.wav` fajlova dolazi do `X.npy`/`Y.npy`) i koriste se samo ako
se dataset proširi novim snimcima i model treba ponovo istrenirati od nule.

## Trening modela

Model je već istreniran (`sax_model.h5`), ali za potrebe odbrane (Nivo 3,
prikaz procesa treninga) trening se može ponoviti istom komandom:

```bash
cd code/image/model

uv run python create.py
```

Skripta učitava `X.npy`/`Y.npy`, deli podatke na trening/validaciju/test
(70/15/15), trenira CNN kroz 20 epoha, ispisuje finalnu tačnost na test
skupu, čuva model (`sax_model.h5`) i grafik krivih treninga
(`training_krive.png`).

## Evaluacija modela

```bash
cd code/image/model

uv run python evaluate.py
```

Generiše classification report (precision/recall/F1 po noti) i matricu
konfuzije (`confusion_matrix_cnn.png`).

## Evaluacija nad novim podacima (demonstracija za odbranu)

Dva načina da se model isproba na novim, neviđenim snimcima:

**1. Preko mikrofona (CLI):**

```bash
cd code/image/model

uv run python terminal.py
```

Pritiskom na Enter snima se 3 sekunde audio sa mikrofona, generiše se
mel-spektrogram istim postupkom kao u treningu, i model ispisuje top 3
predikcije sa pripadajućim procentima.

**2. Preko Flask API-ja i web interfejsa:**

```bash
cd code/image/model
uv run python api.py
```

Zatim otvoriti `ui/index.html` u browseru — dugme "Start"/"Stop" snima
zvuk sa mikrofona i šalje ga na `/predict` endpoint, koji vraća
predviđenu notu.

## Zavisnosti

Sve zavisnosti su definisane u `pyproject.toml` i zaključane u `uv.lock`
(TensorFlow, librosa, scikit-learn, matplotlib, seaborn, Flask, sounddevice,
soundfile, scipy, pandas, numpy...). `uv sync` instalira tačne verzije
korišćene tokom razvoja.
