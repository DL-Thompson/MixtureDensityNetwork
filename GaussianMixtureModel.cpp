#include "GaussianMixtureModel.h"

GaussianMixtureModel::GaussianMixtureModel() {
}

GaussianMixtureModel::GaussianMixtureModel(int seed)
        : generator_sample(seed), generator_mix(seed) {
    this->seed = seed;
}

GaussianMixtureModel::GaussianMixtureModel(std::vector<double> mixing_coefficients, std::vector<double> means, std::vector<double> variances, int seed)
        : generator_sample(seed), generator_mix(seed) {
    assert(mixing_coefficients.size() == means.size() && mixing_coefficients.size() == variances.size());
    this->seed = seed;
    this->mixing_coefficients = mixing_coefficients;
    this->assert_mixing_coefficients();
    this->means = means;
    this->variances = variances;
    this->load_std_dev();
    this->num_distributions = this->mixing_coefficients.size();
    this->load_distributions();
}

void GaussianMixtureModel::set_mixing_coefficients(std::vector<double> mixing_coefficients) {
    assert(mixing_coefficients.size() == this->means.size() && mixing_coefficients.size() == this->variances.size());
    assert(mixing_coefficients.size() > 0);
    this->num_distributions = this->mixing_coefficients.size();
    this->mixing_coefficients = mixing_coefficients;
    this->assert_mixing_coefficients();
    this->load_distributions();
}

void GaussianMixtureModel::set_means(std::vector<double> means) {
    assert(means.size() == this->mixing_coefficients.size() && means.size() == this->variances.size());
    assert(means.size() > 0);
    this->means = means;
    this->load_distributions();
}

void GaussianMixtureModel::set_variances(std::vector<double> variances) {
    assert(variances.size() == this->mixing_coefficients.size() && variances.size() == this->means.size());
    assert(variances.size() > 0);
    this->variances = variances;
    this->load_std_dev();
    this->load_distributions();
}

void GaussianMixtureModel::set_models(std::vector<double> mixing_coefficients, std::vector<double> means,
                     std::vector<double> variances) {
    assert(mixing_coefficients.size() == means.size() && mixing_coefficients.size() == variances.size());
    assert(mixing_coefficients.size() > 0 && means.size() > 0 && variances.size() > 0);
    this->num_distributions = this->mixing_coefficients.size();
    this->mixing_coefficients = mixing_coefficients;
    this->assert_mixing_coefficients();
    this->means = means;
    this->variances = variances;
    this->load_std_dev();
    this->num_distributions = mixing_coefficients.size();
    this->load_distributions();
}

void GaussianMixtureModel::load_distributions() {
    this->normal_distributions.clear();
    for (int i = 0; i < this->num_distributions; i++) {
        this->normal_distributions.push_back(std::normal_distribution<double>(this->means[i], this->variances[i]));
    }
}

double GaussianMixtureModel::get_sample() {
    double random_mix_number = this->get_random_mixing_probability();
    double sample = 0;
    double cumalative_prob = 0;
    for (int i = 0; i < this->num_distributions; i++) {
        cumalative_prob += this->mixing_coefficients[i];
        if (random_mix_number < cumalative_prob) {
            sample = this->normal_distributions[i](this->generator_sample);
            break;
        }
    }
    return sample;
}

std::vector<double> GaussianMixtureModel::get_samples(int size) {
    std::vector<double> samples;
    for (int i = 0; i < size; i++) {
        samples.push_back(this->get_sample());
    }
    return samples;
}

double GaussianMixtureModel::get_random_mixing_probability() {
    return this->distribution_sample(this->generator_sample);
}

void GaussianMixtureModel::assert_mixing_coefficients() {
    double total = 0;
    for (int i = 0; i < this->mixing_coefficients.size(); i++) {
        total += this->mixing_coefficients[i];
    }
    assert(total <= 1.0000001 && total >= 0.9999999);
}

void GaussianMixtureModel::load_std_dev() {
    this->standard_deviations.clear();
    for (int i = 0; i < this->variances.size(); i++) {
        this->standard_deviations.push_back(this->variances[i]);
    }
}
