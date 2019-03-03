#include "randomForest.h"
#include <iostream>
#include <random>
#include <vector>
#include <string>
#include "decisionTree.h"

using std::vector;
using std::pair;
using std::string;
using std::mt19937;

vector<vector<int>> get_random_samples(const vector<vector<int>> &samples,
                                       int num_to_return) {
    // Intoarce un vector de marime num_to_return cu elemente random,
    // diferite din samples
    vector<vector<int>> ret;
    // random = genereaza un numar aleator
    // taken = retine liniile din samples deja considerate
    // size = numarul de linii(imagini) din samples
    std::random_device random;
    vector<int> taken;
    int size = samples.size();

    // Daca numarul de linii cerut depaseste dimensiunea matricei.
    if (num_to_return >= size) {
        return samples;
    }

    int i = 0;
    while (i < num_to_return) {
        // line = indicele unei linii aleatoare din samples
        int line = random() % size;
        // Daca linia a fost deja adaugata, nu o adaug a doua oara.
        if (find(taken.begin(), taken.end(), line) != taken.end()) {
            continue;
        }
        ret.push_back(samples[line]);
        taken.push_back(line);
        i++;
    }

    return ret;
}

RandomForest::RandomForest(int num_trees, const vector<vector<int>> &samples)
    : num_trees(num_trees), images(samples) {}

void RandomForest::build() {
    // Aloca pentru fiecare Tree cate n / num_trees
    // Unde n e numarul total de teste de training
    // Apoi antreneaza fiecare tree cu testele alese
    assert(!images.empty());
    vector<vector<int>> random_samples;

    int data_size = images.size() / num_trees;

    for (int i = 0; i < num_trees; i++) {
        // cout << "Creating Tree nr: " << i << endl;
        random_samples = get_random_samples(images, data_size);

        // Construieste un Tree nou si il antreneaza
        trees.push_back(Node());
        trees[trees.size() - 1].train(random_samples);
    }
}

int RandomForest::predict(const vector<int> &image) {
    // Va intoarce cea mai probabila prezicere pentru testul din argument
    // se va interoga fiecare Tree si se va considera raspunsul final ca
    // fiind cel majoritar
    // res_freq = frecventa numerelor (numarul lor de aparitii)
    // max = cel mai mare numar de aparitii dintre cifre
    // final_res = raspunsul ce mai majoritar (raspunsul final)
    vector<int> res_freq(10, 0);
    int max = 0, final_res = 0;

    for (int i = 0; i < trees.size(); i++) {
        // res = cifra prezisa din arborele i
        int res = trees[i].predict(image);
        res_freq[res]++;

        // Daca rezultatul gasit apare de cele mai multe ori, se actualizeaza
        // raspunsul cel mai majoritar.
        if (res_freq[res] > max) {
            max = res_freq[res];
            final_res = res;
        }
    }

    return final_res;
}
