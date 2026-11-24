## Phase 6 (Deployment Plan)

**Sovellus:** AIDA Pricer  
**URL:** https://aida-project.onrender.com/  
---

## 1. Johdanto

Tässä vaiheessa projektin tulokset — erityisesti hintaennustemalli — viedään tuotantoon käyttäjien saataville. Deployment kattaa tuotantoympäristön valinnan, mallin julkaisemisen, käyttöliittymän rakentamisen sekä ylläpidon ja seurannan suunnittelun.

Sovelluksen pääasiallinen käyttäjä on autokauppias tai muu autoalan ammattilainen, joka tarvitsee nopeasti realistisen markkinahinta-arvion käytetylle autolle.

---

## 2. Tuotantoympäristö

### 2.1 Valittu alusta: Render.com

Tuotantoalustaksi valittiin **Render.com**, joka tarjoaa:

- Ilmaisen Web Service -tason prototyypeille ja koulutusdemoille
- Automaattisen CI/CD-integraation GitHub-repositorioon — jokainen `main`-branchin push käynnistää uuden deployn automaattisesti
- HTTPS-päätepisteet oletuksena ilman erillistä konfiguraatiota
- Ympäristömuuttujien (Environment Variables) turvallisen hallinnan suoraan hallintapaneelista
- Gunicorn-pohjaisen Python WSGI -palvelimen tuen

### 2.2 Repositoriorakenne

```
aida-pricer/
├── app.py                        # Flask-backend, API-reitit, feature engineering
├── requirements.txt              # Python-riippuvuudet
├── templates/
│   └── index.html                # Käyttöliittymä (Jinja2-template)
├── static/
│   └── style.css                 # Tyylitiedosto
└── data/
    ├── models/
    │   └── catboost_price.cbm    # Koulutettu CatBoost-malli (~76 MB)
    ├── counties_lookup.json      # Osavaltiot ja pirikunnat koordinaatteineen
    ├── makes_models.json         # Automerkit ja -mallit (generoitu train-datasta)
    └── brand_model_stats.json    # Aggregaatit ennustetta varten (generoitu train-datasta)
```

### 2.3 Suurten tiedostojen hallinta

GitHub estää normaalissa repossa **yli 100 Mt** tiedostot, joten CatBoost-mallia ei versionhallita suoraan Gitissä, vaan se jaetaan **GitHub Releases** ‑assetina.  
Tässä projektissa mallitiedosto **`catboost_price.cbm` on n. 196 Mt**.
Releases-assetin enimmäiskoko on **alle 2 Gt**

`train.parquet` (124 MB) **ei** ole Renderissä — se on korvattu kahdella kevyellä JSON-tiedostolla (ks. luku 3.2).

---

## 3. Mallin vieminen tuotantoon

### 3.1 Malli

Hintaennuste perustuu **CatBoost-regressiomalliin**, joka on koulutettu vaiheessa 4:

| Ominaisuus | Arvo |
|---|---|
| Malli | CatBoostRegressor |
| Kohdemuuttuja | `price` (USD) |
| MAE | $2 495 |
| R² | 0.91 |
| MAPE | 9.2% |
| Koulutusaineisto | 2,6 M ajoneuvoa (USA, ~2020) |
| Tallennusformaatti | `.cbm` (CatBoost natiivi) |

Malli ladataan ensimmäisen ennustepyynnön yhteydessä (lazy loading) ja pidetään sen jälkeen muistissa, jolloin myöhemmät ennusteet toimivat viiveettä.

### 3.2 Aggregaattitiedot: train-datasta johdetut JSON-tiedostot

Ennuste hyödyntää useita aggregaattifeatureja, joista tärkein on `brand_model_avg_price` (feature importance 28,6 %). Nämä on esikäsitelty kahdeksi JSON-tiedostoksi, jotka generoidaan paikallisesti omalla koneella ennen deployta:

**`generate_makes_models.py`** — tuottaa `makes_models.json`  
Rakenne: `{"Toyota": ["Camry", "Corolla", ...], "Ford": [...], ...}`  
Käytetään käyttöliittymän merkki/malli-dropdownien täyttämiseen.

**`generate_brand_model_stats.py`** — tuottaa `brand_model_stats.json`  
Sisältää neljä osaa:
- `brand_model_stats` — merkki+malli -tason hintatilastot (`bm_avg_price`, `bm_avg_dom`, `bm_avg_km`)
- `geoid_stats` — pirikuntatason hintatilastot (`county_avg_price`, `county_avg_dom`)
- `brand_stats` — merkkitason fallback-tilastot
- `global_fallbacks` — viimeisenä hätävara jos merkki/sijainti ei löydy

Tämä rakenne korvaa `train.parquet`:n kokonaan tuotantoympäristössä. JSON-tiedostojen koko on murto-osa parquetista.

### 3.3 Enkoodaukset

CatBoost-malli tallentaa kategoristen muuttujien enkoodauksen sisäisesti `.cbm`-tiedostoon. Tämän vuoksi kategorisia featureja **ei enkoodata numeerisesti** backendissä — ne välitetään alkuperäisinä merkkijonoina täsmälleen kuten koulutusaineistossa:

| Feature | Välitetään backendistä |
|---|---|
| `make_name` | `"Toyota"` |
| `model_name` | `"Camry"` |
| `body_type` | `"SUV / Crossover"` |
| `fuel_type` | `"Gasoline"` |
| `wheel_system` | `"AWD"` |
| `engine_type` | `"V6"` |
| `GEOID` | `"06037"` (string, 5 merkkiä) |
| `STATEFP` | `"06"` (string, 2 merkkiä) |

Numeeriset johdetut featuret (`trans_group`, `age_bin`, `mileage_bin`, `car_age`, `km_per_year`, `hp_per_liter`, `power_density`, `luxury_flag`, `is_ev` jne.) lasketaan backendissä käyttäjän syötteen perusteella ennen ennustetta.

---

## 4. Käyttöliittymä

### 4.1 Tekninen toteutus

Käyttöliittymä on toteutettu yksisivuisena web-sovelluksena:

- **Backend:** Python / Flask — hoitaa API-reitit, feature engineeringin ja mallin kutsumisen
- **Frontend:** HTML5 + CSS3 + vanilla JavaScript — ei erillisiä frameworkeja
- **Template engine:** Jinja2 (Flaskin sisäänrakennettu)
- **Fontit:** Syne (otsikot), DM Mono (numerot), Inter (leipäteksti)

### 4.2 API-reitit

| Reitti | Metodi | Kuvaus |
|---|---|---|
| `/` | GET | Pääsivu (index.html) |
| `/api/makes` | GET | Automerkit listana |
| `/api/models/<make>` | GET | Valitun merkin mallit |
| `/api/counties/states` | GET | Osavaltiot FIPS-koodeineen |
| `/api/counties/<statefp>` | GET | Osavaltion pirikunnat |
| `/api/gas_price` | GET | Tuorein bensahinta EIA:lta |
| `/api/predict` | POST | Hintaennuste JSON-syötteellä |
| `/api/health` | GET | Palvelimen tilatarkistus |

### 4.3 Lomakkeen kentät

Käyttöliittymä kerää seuraavat tiedot viidessä lohkossa:

**01 Merkki & malli** — kaksitasoinen dropdown: merkki ensin, malli suodattuu sen mukaan dynaamisesti (`/api/models/<make>`)

**02 Perustiedot** — vuosimalli, kilometrit (mailia), hevosvoimat, moottorin tilavuus (litraa)

**03 Tekniset tiedot** — korimuoto, käyttövoima, vetotapa, moottorin tyyppi, vaihteisto, listauksen päivämäärä

**04 Myyntisijainti** — osavaltio + piirikunta (2 217 vaihtoehtoa), GEOID haetaan valinnasta automaattisesti

**05 Varusteet** — 11 merkittävintä varustelipukkea aina näkyvissä, 11 lisävarusteen laajennettu osio erikseen. Varusteiden tärkeysjärjestys perustuu feature importance -arvoihin.

### 4.4 Bensahinnan automaattinen haku

Bensahinta (`gas_price`) haetaan automaattisesti **EIA:n (U.S. Energy Information Administration)** julkisesta API:sta käynnistyksen yhteydessä. Hinta välimuistitetaan tunnin ajaksi palvelimelle — jokainen ennuste ei siis tee erillistä API-kutsua. Sähköautoille bensahinnaksi asetetaan automaattisesti 0,0.

API-avain tallennetaan Render.com:n Environment Variables -asetuksiin muuttujana `EIA_API_KEY` — se ei näy koodissa eikä repositoriossa.

### 4.5 Tuloskortti

Ennuste näytetään sivun oikean laidan pysyvässä tuloskortissa:

- Arvioitu myyntihinta (USD)
- Vaihteluväli: ennuste ± MAE ($2 495)
- Mallin selitysaste (R² = 91 %)
- Yhteenveto syötetyistä tiedoista
- Huomio yli $75 000 autojen heikommasta tarkkuudesta

---

## 5. Uuden datan lisääminen palvelimelle

Kun train-data päivittyy tai malli koulutetaan uudelleen, päivitysprosessi on:

1. **Paikallisesti omalla koneella:**
   - Kouluta uusi malli (`catboost_price.cbm`)
   - Aja `generate_makes_models.py` → uusi `makes_models.json`
   - Aja `generate_brand_model_stats.py` → uusi `brand_model_stats.json`

2. **Commitoi GitHubiin:**
   ```bash
   git add data/models/catboost_price.cbm
   git add data/makes_models.json
   git add data/brand_model_stats.json
   git commit -m "Päivitetty malli ja aggregaatit"
   git push origin main
   ```

3. **Render deployaa automaattisesti** — uusi versio on tuotannossa muutamassa minuutissa.

`counties_lookup.json` on pysyvä ja päivittyy vain jos USA:n piirikuntatiedot muuttuvat (harvinaista).

---

## 6. Seuranta

### 6.1 Render.com:n sisäinen seuranta

Render tarjoaa suoraan hallintapaneelista:

- **Lokit reaaliajassa** — kaikki Flask/Gunicorn-lokit näkyvissä
- **Deploy-historia** — jokainen deploy kirjataan aikaleimalla
- **Palvelimen tila** — käynnissä / pysäytetty / virhe

### 6.2 Health-endpoint

`/api/health` palauttaa JSON-statuksen joka kertoo onko malli ja kaikki tarvittavat datatiedostot löydettävissä:

```json
{
  "status": "ok",
  "model_exists": true,
  "counties_exists": true,
  "makes_exists": true,
  "stats_exists": true
}
```

### 6.3 Tunnetut rajoitteet

| Rajoite | Kuvaus |
|---|---|
| Alle $10 000 ja yli $75 000 autot | Malli ei ole näissä hintasegmenteissä luotettava — ennuste on vain suuntaa-antava (malli optimoitu tyypillisimpiin tapauksiin). |
| Aineiston ajallinen rajaus | Opetettu ~2020 datalla — ei välttämättä heijasta nykyisiä markkinahintoja. |
| Varusteiden vaikutus voi olla epäintuitiivinen | Varusteiden lisääminen voi joissain merkki–malli–yhdistelmissä laskea ennustetta. Varusteet ovat mallille “absoluuttisia” (ei suhteessa make+model -normiin), ja tietyt varusteyhdistelmät voivat korreloida halvempien segmenttien kanssa tai aiheuttaa epälineaarisia vuorovaikutuksia. |
| `brand_model_avg_price` dominoi yksilötasolla | Aggregaattifeature parantaa keskimääräistä tarkkuutta, mutta yksilötasolla voi yliohjata iän ja ajomäärän signaaleja ja tuottaa poikkeustapauksissa epärealistisia arvioita. (Ablation: poisto heikensi MAE:ta +$379 → aggregaattitasolla hyödyllinen, yksilötasolla voi olla riskialtis.) |
| Myynnissäoloaikamalli | Ei ole käytössä tässä versiossa. |
| Tuntemattomat merkit/mallit | Jos merkki/malli puuttuu train-datasta (ei löydy dropdown-listasta), ennuste vaatii fallback-logiikan (esim. merkkitaso / globaali) ja tulos tulee merkitä matalamman luottamuksen ennusteeksi. |

---

## 7. Ylläpitosuunnitelma

| Toimenpide | Aikataulu | Kuvaus |
|---|---|---|
| Mallin uudelleenkoulutus | Tarvittaessa | Kun uudempaa dataa saatavilla |
| Aggregaattien päivitys | Mallin päivityksen yhteydessä | `brand_model_stats.json` ja `makes_models.json` |
| EIA API -avaimen tarkistus | Vuosittain | Avain ei vanhene automaattisesti mutta kannattaa tarkistaa |
| Health-endpoint -tarkistus | Deployn jälkeen | Varmistaa että kaikki tiedostot ladattu oikein |
| Render ilmaistason rajoitukset | Jatkuva | Ilmainen tier sammuttaa palvelimen n. 15 min käyttämättömyyden jälkeen — ensimmäinen pyyntö herättää sen (kylmäkäynnistys ~30 s) |

---

## 8. Projektin arviointi

### 8.1 Tavoitteiden toteutuminen

| Tavoite | Tulos |
|---|---|
| Hintaennuste pirikunnan tarkkuudella | ✅ Toteutunut — 2 217 pirikuntaa |
| CatBoost-malli tuotannossa | ✅ Toteutunut — R²=0.91, MAE $2 495 |
| Autokauppiaan työkalu | ✅ Toteutunut — AIDA Pricer käytössä |
| Bensahinnan integraatio | ✅ Toteutunut — EIA API reaaliajassa |
| Myynnissäoloajan ennuste | ⏳ Ei tässä versiossa — jatkokehitykseen |

### 8.2 Tekniset oppimispisteet

Projekti osoitti useita tärkeitä havaintoja deployment-vaiheessa:

**Enkoodausyhteensopivuus on kriittistä.** CatBoost tallentaa kategoriset enkoodaukset `.cbm`-tiedostoon — backendin ei tule enkoodata kategorioita numeerisesti ennen mallin kutsumista. Tämä eroaa XGBoost- ja LightGBM-malleista, jotka vaativat erillisen OrdinalEncoder-tallenteen.

**Tiedostokokorajoitukset on huomioitava etukäteen.** GitHub:n 100 MB raja ja `train_slim.parquet`:n 39 MB koko edellyttivät strategista päätöstä: aggregaatit esikäsitellään JSON-tiedostoiksi paikallisesti eikä parquettia viedä palvelimelle. Git LFS hoitaa mallin `.cbm`-tiedoston.

**Aggregaattifeaturet ovat mallin kannalta elintärkeitä.** `brand_model_avg_price` on ylivoimaisesti tärkein feature (importance 28,6 %). Jos tätä ei olisi saatavilla ennusteen yhteydessä, mallin tarkkuus romahtaisi merkittävästi.

### 8.3 Jatkokehityssuositukset

1. **Soveltamisalueen “guardrail” + epävarmuusviestintä (UI + backend)**
   - Malli toimii luotettavimmin hintaluokassa **$10k–$75k**; tämän ulkopuolella ennuste on vain suuntaa-antava.
   - Toteutus: jos ennuste < $10k tai > $75k (tai syöte on muuten selvästi poikkeava), näytä tuloskortissa varoitus + laajempi epävarmuusväli (ei pelkkä ±MAE).

2. **Varusteiden käsittely uusiksi: “absoluuttisista checkboxeista” suhteelliseksi varustetasoksi**
   - Havainto: varusteiden lisääminen voi laskea ennustetta (esim. navi ↑ mutta navi+sunroof ↓, “kaikki varusteet” ↓), mikä tuntuu käyttäjälle epäloogiselta.
   - Selitys: varusteet voivat korreloida datassa tiettyihin (halvempiin) trim/segmentteihin ja varusteyhdistelmät voivat ohjata mallin eri epälineaarisiin vuorovaikutuksiin.
   - Toteutusvaihtoehdot:
     - (A) **Varustetaso Low / Medium / High** (deployment-tasolle “pika-arvio”): päätellään taso option_countin tai painotetun option_score:n perusteella.
     - (B) **Merkki+malli-kohtainen normalisointi**: “kuinka hyvin varusteltu tämä yksilö on verrattuna saman make+model -ryhmän autoihin datassa?” (percentile/z-score).
     - (C) Pidä checkboxit, mutta lisää UI-huomio: “varusteiden vaikutus ei ole aina additiivinen”.

3. **`brand_model_avg_price`-dominanssin hillitseminen (yksilötapauksien realismi)**
   - Teillä `brand_model_avg_price` on erittäin dominoiva featurenä (feature importance ~28,6 %), mikä parantaa kokonaisosumatarkkuutta mutta voi yksilötasolla “yliohjata” erityisesti vanhoja / paljon ajettuja tapauksia.
   - Toteutusvaihtoehdot:
     - (A) Lisää eksplisiittisiä interaktioita: `brand_model_avg_price × age_bin` ja `brand_model_avg_price × mileage_bin`
     - (B) Laske aggregaatit ikä-/mileage-segmenteittäin (make+model+age_bin -keskiarvot), jolloin “uudet yksilöt” eivät vedä “vanhoja yksilöitä” ylös.
     - (C) Two-stage / residual-malli: ensin baseline aggregaateista, sitten toinen malli ennustaa korjauksen (delta) yksilöfeatureista.

4. **OOD-tunnistus (out-of-distribution) + “luottamusindikaattori”**
   - Havainto: harvinaiset yhdistelmät (esim. malli, jossa sunroof on datassa poikkeus + vanha vuosimalli + isot mailit) voivat tuottaa “mitä sattuu” -ennusteita.
   - Toteutus: laske syötteelle yksinkertainen OOD-score (percentile-rajat age/mileage make+model -tasolla, tai etäisyys tyypillisistä arvoista). Jos OOD korkea → varoitus, leveämpi vaihteluväli ja lokitus mallin jatkokehitystä varten.

5. **Tuntemattomat merkit/mallit: eksplisiittinen fallback ja UI-polku**
   - Koska make/model -valinnat tulevat train-datasta generoituina, täysin uusi merkki/malli ei ole suoraan tuettu.
   - Toteutus: “Other / Manual entry” -polku → fallback-ennuste merkkitasosta (brand_stats) tai globaalista fallbackista (global_fallbacks) + selkeä “alhaisempi luottamus” -merkintä.

6. **Segmentoidut mallit vaikeisiin alueisiin (premium + extreme old/high-mileage)**
   - Premium-segmentille (> $75k) erillinen malli (tai vähintään oma kalibrointi) vähentää suuria virheitä.
   - Lisäksi “vanha / high-mileage” -segmentti voi hyötyä omasta mallista tai erillisestä kalibrointikerroksesta (price floor/ceiling + delta-malli).

7. **Selitettävyys ja QA: SHAP/ablation + automaattiset sanity-check -testit**
   - Tee toistettavat testitapaukset: “lisää varusteita” ei saisi systemaattisesti romahduttaa hintaa tietyissä perus-caseissa ilman selitystä.
   - Käytä SHAP/feature contribution -analyyseja make+model -tasolla erityisesti varusteille (navigation, sunroof), jotta nähdään missä kontekstissa vaikutus kääntyy negatiiviseksi.
   - Laadi regression-testipaketti (10–50 käsin valittua autoa) ja aja se jokaisen uudelleenkoulutuksen/deployn yhteydessä.

8. **Datan päivitys**
   - Uudempi data parantaa markkinatilanteen vastaavuutta ja vähentää “2020-vääristymää”.

---

## 9. Yhteenveto

Manuaalisessa testauksessa havaittiin kaksi käytännön rajoitetta: (1) varusteiden lisääminen voi joissain merkki–malli–yhdistelmissä laskea ennustetta epäintuitiivisesti, mikä viittaa epälineaarisiin vuorovaikutuksiin ja segmenttikorrelatioihin datassa, sekä (2) `brand_model_avg_price` on erittäin dominoiva signaali, joka parantaa keskimääräistä tarkkuutta mutta voi yksilötasolla yliohjata iän ja ajomäärän vaikutuksia poikkeavissa tapauksissa.

Hintamalli soveltuu parhaiten hintaluokkaan **$10 000–$75 000** hinnoittelun tukityökaluksi (ei automaattiseksi päätöksentekijäksi). Jatkokehityksessä tärkeimmät parannukset ovat varusteiden abstraktointi varustetasoksi (Low/Medium/High tai make+model-normalisointi), aggregaattifeaturen dominanssin hillitseminen (ikä/ajomäärä-sidonta) sekä poikkeavien syötteiden (OOD) tunnistaminen ja selkeä epävarmuusviestintä käyttöliittymässä.