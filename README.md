# Optical-Character-Recognition
The program is based on the machine learning idea. It goes through a stage of learning and then can recognize different characters. The project was developed with a teammate.

*(Code and Readme are written in **Romanian language**)*

Algoritmul utilizat:
- Majoritatea explicatiilor pentru a intelege functiile implementate se
gasesc in comentariile codului.
- Programul incepe cu formarea arborilor de decizie (etapa de invatare). Pe
baza informatiilor de la input, se apeleaza functia train care va forma un nod
de decizie si se va apela recursiv pe subarborii stang si drept sau va forma o
frunza, depinde de caz. Pentru a gasit cel mai bun split, train apeleaza
find_best_split.
- Functia find_best_split determina combinatia split_index si split_value
pentru care Information Gain al nodului curent ia cea mai mare valoare dintre
toate posibilitatile.
- In urma formarii tuturor arborilor, incepe etapa de prezicere in care
pentru fiecare imagine (vector) primita de la input, se cauta in fiecare arbore
raspunsul dat de acesta. Dupa ce au fost calculate toate prezicerile din
arbori, se alege cea care apare de cele mai multe ori. De aceste lucruri se
ocupa functiile predict din randomForest.cpp si predict din decizionTree.cpp.

Complexitatea programului:
- Pana sa incepem implementarea codului, tema ni s-a parut grea, enuntul
era ambiguu, nu intelegeam clar ce trebuie facut, insa incet-incet am inteles
fiecare functie ce trebuie sa faca si am reusit sa le implementam pe toate.
- In urma acestor neclaritati si a problemelor pe care le-am intampinat,
tema ni s-a parut de o dificultate medie spre grea.
- Pe de alta parte, tema a fost accesibila deoarece oferea un punctaj
destul de mare numai prin implementarea celor 5 functii care aduceau 50p.

Alte precizari:
- Intrucat pe parcursul implementarii si testarii temei am intampinat doua
mari greseli in urma carora programul se oprea cu eroarea Segmentation Fault,
am decis sa folosim programarea defensiva si sa oprim executia programului
inainte de a accesa zone din memorie neinitializate sau invalide.
- Am incercat sa obtinem o precizie cat mai buna in cadrul temei si un timp
de executie cat mai scazut, ajungand la precizia de 92% in 20-25 de secunde.
