# Nettside for prediksjon av huspris

Her lager vi en enkel nettside for å predikere huspris. Se [https://blasern.github.io/data-science-forelesningsnotater/implementation.html](forelesningsnotater) for mer forklaring. 

Først må vi analysere data, tilpasse, velge ut og evaluere en modell. For å gjøre det har vi filen [`modeling.py`](modeling.py). Til slutt lagrer vi den beste modellen i filen `model.pkl`. 

Så må vi lage en nettside som inneholder et skjema der brukere kan skrive inn forskjellige data om huset. Den ligger under [`templates/index.html`](templates/index.html). 

Til slutt kan vi se på hvordan vi kommuniserer med nettsiden fra python. Her har vi to versjoner, [`simpleapp.py`](simpleapp.py), som inneholder kun de minimale elementene for å kunne kjøre nettsiden og [`app.py`](app.py) som faktisk bruker modellen for å predikere huspris. 

Vi kjøre nettsiden ved å kjøre

```
python app.py
```

i en shell og så går vi til [http://localhost:8080/](http://localhost:8080/) i en nettleser. Hvis vi har skrevet ut noe med print, så dukker dette opp i shell.
