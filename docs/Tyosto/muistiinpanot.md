## Muistiinpanot

**Huomoita**
- mites noi sähköautot pitäisi huomioida moottorin suhteen, nyt aika paljon polttomoottorin ominaisuuksia
- normalisoidut hevosvoimat: df['norm_hp'] = df['horsepower'] / df.groupby('year')['horsepower'].transform('median')

- LabelEncoder on sallittu vain, jos kategoria on oikeasti järjestysasteikollinen (ordinal). Fuel type ei ole. Ja silloin malli luulee että isompi parempi.
- eli fueltypelle pitää tehdä onehot enkoodaus
- paitsi jos: CatBoost ja LightGBM osaavat käsitellä kategorioita ilman one-hot‑enkoodausta - voi syöttää fuel_typen sellaisenaan
- Polttomoottorien featuret voi pitää ja täyttää luonnollisilla nollilla sähköautoille.
    - eli niissä sarakkeissa laiteen N/A arvoille 0 jos sähköauto. Muuten pois?
- Yhdistetty teho‑indeksi toimii hyvin koko aineistolle - 
df['engine_perf'] = (
      0.6 * df['horsepower'].fillna(0)
    + 0.2 * (df['engine_displacement'].fillna(0) / 1000)
    + 0.2 * df['cylinders'].fillna(0)
)
- malli suosittaa käyttämään kaikkia kolmea: engine_perf, normnalisointi ja fuel_type onehot
