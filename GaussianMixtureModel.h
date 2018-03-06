#ifndef GAUSSIAN_MIXTURE_MODEL
#define GAUSSIAN_MIXTURE_MODEL

#include <vector>
#include <tr1/random>
#include <cassert>
#include <iostream>
#include <math.h>
#include <random>

class GaussianMixtureModel {

public:
    GaussianMixtureModel();
    ~GaussianMixtureModel() {}
    GaussianMixtureModel(int seed);
    GaussianMixtureModel(std::vector<double> mixing_coefficients, std::vector<double> means, std::vector<double> variances, int seed);
    void set_mixing_coefficients(std::vector<double> mixing_coefficients);
    void set_means(std::vector<double> means);
    void set_variances(std::vector<double> variances);
    void set_models(std::vector<double> mixing_coefficients, std::vector<double> means, std::vector<double> variances);
    std::vector<double> get_mixing_coefficients() {return this->mixing_coefficients;}
    std::vector<double> get_means() {return this->means;}
    std::vector<double> get_variances() {return this->variances;}
    std::vector<double> get_standard_deviations() {return this->standard_deviations;}
    double get_sample();
    std::vector<double> get_samples(int size);

private:
    std::vector<double> mixing_coefficients, means, variances, standard_deviations;
    int num_distributions;

    std::vector<std::normal_distribution<double>> normal_distributions;
    std::uniform_real_distribution<double> distribution_sample{0.0, 1.0};
    std::default_random_engine generator_sample;
    std::default_random_engine generator_mix;

    void load_distributions();
    double get_random_mixing_probability();
    void assert_mixing_coefficients();
    void load_std_dev();

    int seed;
};


#endif
