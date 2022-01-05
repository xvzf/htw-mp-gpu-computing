# Aufgabe 1

Patrick Plewka (3761940)
Matthias Riegler (3761312)


## Messdaten

Für die Messungen wurde eine GTX 970 verwendet.

### Version A

Die Messwerte der 5 Testläufe #1 bis #5 mit dem gegebenen k und dem Durchschnitt.

| k   | avg    | #1   | #2   | #3   | #4   | #5   |
|-----|--------|------|------|------|------|------|
| 0   | 38.6ms | 39ms | 38ms | 40ms | 37ms | 39ms |
| 1   | 53.6ms | 54ms | 54ms | 54ms | 52ms | 54ms |
| 2   | 49.4ms | 48ms | 48ms | 51ms | 49ms | 51ms |

k >= 3 stürzt ab. 32 * 5 ^ 3 = 4000. Zu viele Threads.
### Version B

Die Messwerte der 5 Testläufe #1 bis #5 mit dem gegebenen k und dem Durchschnitt. 

| k   | avg | #1  | #2  | #3  | #4  | #5   |
|-----|-----|-----|-----|-----|-----|------|
| 0   | 2ms | 2ms | 2ms | 2ms | 2ms | 2ms  |
| 1   | 2ms | 2ms | 2ms | 2ms | 2ms | 2ms  |
| 2   | 2ms | 2ms | 2ms | 2ms | 2ms | 2ms  |

k >= 3 stürzt ab. 32 * 5 ^ 3 = 4000. Zu viele Threads.

## Bewertung der Messdaten

Version A ist durchweg um Faktor ~20-25 langsamer als Version 2.
Eine Änderung von k führt zwar zu anderen Messwerten in Version A, kann das grundlegende Problem jedoch nicht eliminieren.
Eine Änderung von k führt zu keinen ersichtlichen Messwerten in Version B, das hat jedoch nichts zu bedeuten.
Die Laufzeiten von Version B sind so gering, dass tatsächlich auftretende Änderungen durch Änderung von k einfach übersehen werden.
Die größte Schwankung in Version A ist zwischen 37ms und 54ms. Das wäre eine Änderung von +46%. Diese Änderung wäre in Version B wahrscheinlich nicht sichtbar.

Eine Erklärung des Verhaltens ist eine bessere Ausnutzung der Speicheranbindung.
Bei Variante A ist jeder Thread i alle nebeneinanderliegenden Speicherzellen ab 5000 * i zuständig.
Das heißt, dass zu einem gegebenen Zeitpunkt t beispielsweise die Speicherzelle 5000 * i + t gelesen werden muss.
Da Threads parallel ausgeführt werden, wird die Grafikkarte zum Zeitpunkt t alle Speicherzellen 5000 * i + t gleichzeitig laden müssen.
Also beispielsweise t, 5000 + t, 10000 + t, 150000 + t, etc.
Die zu ladenden Speicherzellen sind dabei immer exakt 5000 Zellen voneinander entfernt.
Die ist eine schlechte Idee, da heutige Speicher extrem effizient sind, nebeneinanderliegende Speicherzellen zu laden.

Variante B nutzt die Speicheranbindung viel geschickter aus.
Zum Zeitpunkt t werden die Speicherzellen i + t * 20000 ausgelesen.
Beispielsweise t * 20000, 1 + t * 20000, 2 + t * 20000, etc.
Wie man sieht, liegen die Speicherzellen zum Zeitpunkt t alle nebeneinander, was viel vorteilhafter in Hinsicht auf die Speicheranbindung ist.

Die Änderungen der Messdaten in Variante A, wenn man k erhöht, lassen sich vielleicht auch dadurch erklären, dass mit höherem k sich mehr Threads gegenseitig im Weg stehen.
Je mehr Threads gleichzeitig diese schlechte Ladetechnik verwenden, desto länger müssen die einzelnen Threads auch warten.
Das wäre auch eine mögliche Erklärung, wieso man keine Änderung der Zeiten in Variante B sieht.
Die Threads stehen sich nicht wirklich gegenseitig im Weg.