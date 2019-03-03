#include "./decisionTree.h"  // NOLINT(build/include)
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <random>

using std::string;
using std::pair;
using std::vector;
using std::unordered_map;
using std::make_shared;

// structura unui nod din decision tree
// splitIndex = dimensiunea in functie de care se imparte
// split_value = valoarea in functie de care se imparte
// is_leaf si result sunt pentru cazul in care avem un nod frunza
Node::Node() {
    is_leaf = false;
    left = nullptr;
    right = nullptr;
}

void Node::make_decision_node(const int index, const int val) {
    split_index = index;
    split_value = val;
}

void Node::make_leaf(const vector<vector<int>> &samples,
                     const bool is_single_class) {
    // Seteaza nodul ca fiind de tip frunza (modificati is_leaf si result)
    // is_single_class = true -> toate testele au aceeasi clasa (acela e result)
    // is_single_class = false -> se alege clasa care apare cel mai des
    is_leaf = true;
    if (is_single_class) {
        result = samples[0][0];
    } else {
        // res_freq = frecventa numerelor (numarul lor de aparitii)
        // max = cel mai mare numar de aparitii dintre cifre
        // final_res = raspunsul cel mai des intalnit
        // size = numarul de linii(imagini) din samples
        vector<int> res_freq(10, 0);
        int max = -1, final_res = -1, size = samples.size();

        for (int i = 0; i < size; i++) {
            // res = cifra reprezentata de sample-ul de la linia i
            int res = samples[i][0];
            // Numarul de aparitii al cifrei respective creste.
            res_freq[res]++;

            if (res_freq[res] > max) {
                max = res_freq[res];
                final_res = res;
            }
        }

        result = final_res;
    }
}

pair<int, int> find_best_split(const vector<vector<int>> &samples,
                               const vector<int> &dimensions) {
    // Intoarce cea mai buna dimensiune si valoare de split dintre testele
    // primite. Prin cel mai bun split (dimensiune si valoare)
    // ne referim la split-ul care maximizeaza IG
    // pair-ul intors este format din (split_index, split_value)
    int splitIndex = -1, splitValue = -1;
    // size = numarul de linii(imagini) din samples
    // nr_col = numarul de elemente din dimensions (coloane din samples)
    // maxIG = cel mai mare Information Gain calculat
    int size = samples.size();
    int nr_col = dimensions.size();
    float maxIG = std::numeric_limits<float>::min();

    float node_entropy = get_entropy(samples);

    for (int i = 0; i < nr_col; i++) {
        vector<int> test_values = compute_unique(samples, dimensions[i]);
        // nr_values = numarul de valori pentru care se va calcula IG
        int nr_values = test_values.size();

        for (int j = 0; j < nr_values; j++) {
            pair<vector<int>, vector<int>> split_indexes = get_split_as_indexes
                                    (samples, dimensions[i], test_values[j]);

            // Daca split-ul nu formeaza copil stang sau drept (nu este split
            // bun) se renunta la acest split.
            if (split_indexes.first.size() == 0 ||
                                split_indexes.second.size() == 0) {
                continue;
            }

            // Se calculeaza Information Gain pentru split-ul curent.
            float IG = node_entropy - (split_indexes.first.size() *
                        get_entropy_by_indexes(samples, split_indexes.first) +
                        split_indexes.second.size() *
                        get_entropy_by_indexes(samples, split_indexes.second)) /
                        size;

            if (IG > maxIG) {
                maxIG = IG;
                splitIndex = dimensions[i];
                splitValue = test_values[j];
            }
        }
    }

    return pair<int, int>(splitIndex, splitValue);
}

void Node::train(const vector<vector<int>> &samples) {
    // Antreneaza nodul curent si copii sai, daca e nevoie
    // 1) verifica daca toate testele primite au aceeasi clasa (raspuns)
    // Daca da, acest nod devine frunza, altfel continua algoritmul.
    // 2) Daca nu exista niciun split valid, acest nod devine frunza. Altfel,
    // ia cel mai bun split si continua recursiv
    // size = numarul de linii(imagini) din samples
    int size = samples.size();

    if (same_class(samples)) {
        make_leaf(samples, true);
    } else {
        // best_split = indexul si valoarea celui mai bun split. Vectorul de
        // dimensiuni format pentru calcularea lui best_split contine elemente
        // random intre 0 si numarul de coloane din samples (785 pentru noi).
        pair<int, int> best_split = find_best_split(samples,
                                        random_dimensions(samples[0].size()));

        // Daca nu exista niciun split valid, nodul devine frunza.
        if (best_split.first == -1 || best_split.second == -1) {
            make_leaf(samples, false);
        } else {
            // Daca exista un split valid, nodul curent devine nod de decizie.
            make_decision_node(best_split.first, best_split.second);

            // Se calculeaza cele doua subseturi obtinute in urma split-ului.
            pair<vector<vector<int>>, vector<vector<int>>> split_sets =
                                    split(samples, split_index, split_value);
            // Formarea arborelui continua recursiv cu cei doi subarbori.
            left = make_shared<Node> (Node());
            assert(left && "Not enough memory!");
            left->train(split_sets.first);
            right = make_shared<Node> (Node());
            assert(right && "Not enough memory!");
            right->train(split_sets.second);
        }
    }
}

int Node::predict(const vector<int> &image) const {
    // Intoarce rezultatul prezis de catre decision tree
    // Daca s-a ajuns la o frunza, se returneaza valoarea stocata in ea.
    if (is_leaf) {
        return result;
    }

    // Se opeste executia programului daca valoarea split_index este incorecta.
    assert((split_index <= 784 || split_index >= 1) && "Wrong split_index!");
    // Daca nu s-a ajuns la o frunza, se continua cautarea in subarborele
    // corespunzator.
    if (image[split_index - 1] <= split_value) {
        return left->predict(image);
    }
    return right->predict(image);
}

bool same_class(const vector<vector<int>> &samples) {
    // Verifica daca testele primite ca argument au toate aceeasi
    // clasa(rezultat). Este folosit in train pentru a determina daca
    // mai are rost sa caute split-uri
    // ref = valoare de referinta folosita la compararea numerelor
    // size = numarul de linii(imagini) din samples
    int ref = samples[0][0];
    int size = samples.size();

    // Se verifica pentru fiecare imagine(linie) din samples daca este aceeasi
    // cu cea stocata in ref, iar in caz contrar se returneaza false.
    for (int i = 1; i < size; i++) {
        if (samples[i][0] != ref) {
            return false;
        }
    }

    return true;
}

float get_entropy(const vector<vector<int>> &samples) {
    // Intoarce entropia testelor primite
    assert(!samples.empty());
    vector<int> indexes;

    int size = samples.size();
    for (int i = 0; i < size; i++) indexes.push_back(i);

    return get_entropy_by_indexes(samples, indexes);
}

float get_entropy_by_indexes(const vector<vector<int>> &samples,
                             const vector<int> &index) {
    // Intoarce entropia subsetului din setul de teste total(samples)
    // Cu conditia ca subsetul sa contina testele ale caror indecsi se gasesc in
    // vectorul index (Se considera doar liniile din vectorul index)
    // freq = frecventa numerelor (numarul lor de aparitii)
    // size = numarul de linii din vectorul index
    // entropy = entropia calculata
    vector<float> freq(10, 0.0f);
    int size = index.size();
    float entropy = 0.0f;

    // Se calculeaza numarul de aparitii al fiecarei cifre, apoi se calculeaza
    // entropia.
    for (int i = 0; i < size; i++) {
        freq[samples[index[i]][0]]++;
    }

    for (int i = 0; i < 10; i++) {
        freq[i] /= size;

        if (freq[i] > 0.0f)
            entropy += freq[i] * log2(freq[i]);
    }

    return -entropy;
}

vector<int> compute_unique(const vector<vector<int>> &samples, const int col) {
    // Intoarce toate valorile (se elimina duplicatele)
    // care apar in setul de teste, pe coloana col
    vector<int> uniqueValues;
    // size = numarul de linii(imagini) din samples
    int size = samples.size();

    for (int i = 0; i < size; i++) {
        uniqueValues.push_back(samples[i][col]);
    }

    // Valorile se sorteaza, apoi se elimina duplicatele.
    sort(uniqueValues.begin(), uniqueValues.end());
    // new_end = pointer catre elementul din vector in urma caruia valorile
    //           stocate pot fi sterse
    auto new_end = unique(uniqueValues.begin(), uniqueValues.end());
    uniqueValues.resize(distance(uniqueValues.begin(), new_end));

    return uniqueValues;
}

pair<vector<vector<int>>, vector<vector<int>>> split(
    const vector<vector<int>> &samples, const int split_index,
    const int split_value) {
    // Intoarce cele 2 subseturi de teste obtinute in urma separarii
    // In functie de split_index si split_value
    vector<vector<int>> left, right;

    auto p = get_split_as_indexes(samples, split_index, split_value);
    for (const auto &i : p.first) left.push_back(samples[i]);
    for (const auto &i : p.second) right.push_back(samples[i]);

    return pair<vector<vector<int>>, vector<vector<int>>>(left, right);
}

pair<vector<int>, vector<int>> get_split_as_indexes(
    const vector<vector<int>> &samples, const int split_index,
    const int split_value) {
    // Intoarce indecsii sample-urilor din cele 2 subseturi obtinute in urma
    // separarii in functie de split_index si split_value
    vector<int> left, right;
    // size = numarul de linii(imagini) din samples
    int size = samples.size();

    for (int i = 0; i < size; i++) {
        if (samples[i][split_index] <= split_value) {
            left.push_back(i);
        } else {
            right.push_back(i);
        }
    }

    return make_pair(left, right);
}

vector<int> random_dimensions(const int size) {
    // Intoarce sqrt(size) dimensiuni diferite pe care sa caute splitul maxim
    // Precizare: Dimensiunile gasite sunt > 0 si < size
    vector<int> rez;
    // taken = monitorizeaza numerele deja considerate
    // random = genereaza un numar aleator
    // rez_size = numarul de elemente pe care trebuie sa il aiba la final rez
    unordered_map<int, int> taken;
    std::random_device random;
    int rez_size = sqrt(size), i = 0;

    while (i < rez_size) {
        // dim = dimensiune generata aleator (0 <= dim < size)
        int dim = random() % size;

        // Daca dimensiunea este 0 sau daca a fost deja adaugata, se ignora.
        if (dim == 0 || taken[dim]) {
            continue;
        } else {
            taken[dim] = 1;
            rez.push_back(dim);
        }

        i++;
    }

    return rez;
}
