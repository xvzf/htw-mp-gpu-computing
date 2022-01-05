# Aufgabe 3

Patrick Plewka (3761940)
Matthias Riegler (3761312)

## Idee

Die grundsätzliche Idee ist dieselbe, wie in Aufgabe 2:
Immer eine Zeile oben und eine Zeile unten im selben Thread berechnen, sodass alle Threads die gleiche Arbeit haben.

Einziger Unterschied ist, dass wir im Vergleich zu Aufgabe 2 ein eindimensionales Array statt einem zweidimensionalen verwendet haben.

## Messdaten

Gemessen wurde mit einer GTX970, einem I5-6600K (4Kerne/4Threads) und 16GB Arbeitsspeicher.
N wurde auf 20000 gelegt, was knapp über 3GB Speicherverbrauch bedeutet.
Die GTX970 besitzt zwar 4GB Speicher, jedoch hat sie (wie alle GTX970) einen Designfehler, der dazu führt, dass nur die ersten 3.5GB performant angesprochen werden können.
Deswegen sind wir unter den theoretisch möglichen 4GB geblieben, um die Messung nicht durch diesen Designfehler zu verfälschen.

Da die Grafikkarte in einem anderen PC, als der Test-PC in Aufgabe 2 eingebaut ist, haben wir auch die Tests der Aufgabe 2 auf diesem PC wiederholt.
Außerdem musste, wegen dem kleinen Grafikspeicher, N sowieso kleiner gewählt werden, was auch zum erneuten Durchführen der Tests geführt hat.

In der nachfolgenden Tabelle stehen für alle 4 möglichen Messarten der Speedup im Vergleich zum sequentiellen Programm, die Durchschnittsdauer in Sekunden und die tatsächlichen Messwerte der 5 Testläufe.

Da wir CPU und GPU Programme miteinander vergleichen, haben wir das GPU Programm auf 2 Arten gemessen.
Einmal wurde nur die notwendige Rechenzeit im Kernel, ein anderes Mal die Rechenzeit im Kernel plus Allokieren des Grafikspeichers und Kopieren zum/vom Grafikspeicher.
Die erste Messart macht in der Realität Sinn, wenn z.B.:
- die Daten sowieso schon im Grafikspeicher sind
- die Daten dort bleiben werden
- die Berechnung öfters aufgerufen werden muss.

Die zweite Messart macht Sinn, wenn dem nicht so ist.
Beispielsweise unser Programm, welches die Berechnung einmal durchführt, das Ergebnis entgegennimmt und sich dann beendet.
Den Extraaufwand die Grafikkarte anzusprechen, ist dann real existierender Aufwand, den man in der Überlegung, ob man eine Grafikkarte benutzt, einbeziehen muss.
Beide Messarten haben demnach ihre Berechtigung und es kommt auf das restliche Programm an, welche der Messarten eher der Realität entsprechen.

| Typ         | Speedup | Avg   | #1    | #2    | #3    | #4    | #5    |
|-------------|---------|-------|-------|-------|-------|-------|-------|
| parallel    | 3.994   | 0.166 | 0.166 | 0.166 | 0.166 | 0.166 | 0.166 | 
| sequentiell | 1.000   | 0.663 | 0.663 | 0.663 | 0.663 | 0.663 | 0.663 |
| cuda        | 8.610   | 0.077 | 0.077 | 0.077 | 0.077 | 0.077 | 0.077 |
| cuda + mem  | 0.994   | 0.667 | 0.671 | 0.672 | 0.658 | 0.662 | 0.672 |

## Deutung der Messdaten

Zuallererst ist erfreulich, dass der Speedup des parallelen OpenMP Programms sich nahezu perfekt an den zweiten Test-PC angepasst hat:
Ein Vierkerner mit 4 Threads, der einen Speedup um 3.994 durch Multicoreprogrammierung erhält.

Außerdem interessant ist es, dass die Werte vom parallelen und vom sequentiellen Programm so nah aneinander sind.
Sie haben sich oft ab der 4. Nachkommastelle unterschieden, aber so genau braucht man es hierfür nicht.

Betrachtet man die Messart `cuda` so sieht man einen Speedup von 8.61.
Mehr als doppelt so viel, wie `parallel`.
Das wäre schon beachtlich, wenn wir in Aufgabe 2 nicht einen Speedup von 9.92 mit einer aktuelleren CPU erreicht hätten.
Schließt man dann noch die Speicherallozierung und das Kopieren mit ein, hat man einen Performancenachteil sogar gegenüber dem sequentiellen Programm.
Fairerweise muss man aber auch dazu sagen, dass die CPU in Aufgabe 2 um einiges neuer ist als alle Hardware in dieser Messreihe.

Trotzdem würden wir die Performance als positiv bewerten, da:
- das CUDA Programm noch optimiert werden kann. Die Technik, welche in Aufgabe 1 einen Performancebonus von Faktor 20-25 gebracht hat, kann noch verwendet werden. 
   Außerdem gibt es noch viele andere Techniken den CUDA Code zu beschleunigen. Die Blockgröße könnte man noch tunen, und und und...
- Verglichen mit der CPU mit ähnlichem Baujahr mehr als eine doppelte Geschwindigkeit in der Berechnung rausgeholt werden konnte. Eine neuere Grafikkarte könnte bestimmt einen besseren Speedup rausholen, aber dafür fehlt uns die Hardware.
