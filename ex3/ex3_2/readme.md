# Aufgabe 2

Patrick Plewka (3761940)
Matthias Riegler (3761312)

## Idee
Annahme:
- Der Einfachheit halber nehmen wir an, dass es keine signifikant große Anzahl an 0 in den Zufallswerten vorkommt. Bei Gleichverteilung und z.B. `int` als Datentyp, sollte es die Lösung nicht wirklich beeinflussen.

Die Idee ist es, dass wir `ceil(N  / 2)` Threads erzeugen und sich jeder Thread immer um zwei Zeilen kümmern muss.
Thread `i` kümmert sich um Zeile `i` und `(N - i - 1)`.
Das führt dazu, dass jeder Thread in Zeile `i` `i` Nullwerte und `N - i` Zufallswerte hat; und in Zeile `(N - i - 1)` `(N - i - 1)` Nullwerte und `N - (N - i - 1) = i + 1` Zufallswerte.
Das sind zusammen `i + N - i - 1 = N - 1` Nullwerte und `N - i + i + 1 = N + 1` Zufallswerte.
D.h. jeder Thread hat exakt gleich viele Zufallswerte und Nullwerte, da diese nicht mehr abhängig von der Threadnummer `i` sind.

Eine Ausnahme gibt es dabei:
Ist `N` ungerade, so haben wird der Thread `N / 2` sich nur um eine Zeile in der Mitte kümmern müssen.
Das das aber maximal einmal passiert und `N` groß gewählt wird, bewerten wir es ähnlich, wie wenn man 5 Threads auf 4 CPU-Kernen aufteilen will.
Es geht halt rechnerisch kaum und es kommt zu einer kleinen Unausgewogenheit gegen Ende des Programms.

## Messdaten

Für die Messungen wurde ein Ryzen 7 3800X (8 Kerne/16 Threads) mit 16GB Arbeitsspeicher verwendet.
N wurde auf 36000 gesetzt, was bei `double` Datentype ungefähr 10GB für die Matrix entspricht.
Es wurden immer 5 Testläufe gemacht und gemessen wurde mit `double omp_get_wtime(void);`.
 
| Typ         | Avg   | #1    | #2    | #3    | #4    | #5    |
|-------------|-------|-------|-------|-------|-------|-------|
| parallel    | 0.164 | 0.167 | 0.155 | 0.156 | 0.187 | 0.156 | 
| sequentiell | 1.627 | 1.660 | 1.613 | 1.630 | 1.614 | 1.616 |

Der durchschnittliche Speedup ist demnach 9.92, was im Bereich des Erwarteten liegt.

Eine weitere Messreihe wurde in Aufgabe 3 noch einmal durchgeführt um GPU und CPU Programme miteinander zu vergleichen.