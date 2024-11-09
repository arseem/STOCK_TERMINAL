#### Instalacja zależności
```bash
pip install -r requirements.txt
```

#### Uruchomienie aplikacji
```bash
textual run main.py
```

#### Korzystanie z aplikacji
- a - dodaj nowy symbol
- R - usuń symbol (rząd, w którym znajduje się kursor)
- F5 - odśwież dane
- q - zapisz dane i wyjdź
- ctrl+d - tryb debugowania (konsola z logami)

Aby uruchomić wykres, należy skierować kursor na pole 'Active graph' wybranego rzędu w tabeli i nacisnąć klawisz 'Enter'. Taką samą operację należy wykonać, aby zamknąć wykres.

Aby zmienić okres lub interwał wykresu, należy skierować kursor na pole 'Period' lub 'Interval' wybranego rzędu w tabeli i nacisnąć klawisz 'Enter'. Następnie należy wybrać strząłkami góra/dół odpowiednią wartość i zatwierdzić ją klawiszem 'Enter'.

Aby nałożyć na wykres wybrane wskaźniki (MAVG - średnia ruchoma, EMA - wykładnicza średnia ruchoma, BOLLINGER - wstęgi Bollingera), należy skierować kursor na odpowiednie pole w tabeli i nacisnąć klawisz 'Enter'. Aby usunąć wskaźniki, należy ponownie nacisnąć klawisz 'Enter'.