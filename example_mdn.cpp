#include <cstdio>
#include <gnuplot-iostream.h>
#include "MixtureDensityNetwork.h"

void plot_original_sin(int num_gaussian);
void plot_inverted_sin(int num_gaussian);
void plot_sin_training();
void plot_sin_inverted_training();
void plot_graph(std::vector<double> in, std::vector<double> real_out, std::string title_1,
                std::vector<double> estimated_in, std::vector<double> estimated_out, std::string title_2,
                std::string main_title);

int main(int argc, char** argv) {
    printf("Beginning NN Sin Example...\n");

    plot_sin_training();
    plot_sin_inverted_training();
    plot_original_sin(5);
    plot_inverted_sin(5);

    printf("End of program.\n");
    return 0;
}

void plot_inverted_sin(int num_gaussian) {
    int mdn_seed = 0;
    int mdn_num_input = 1;
    int mdn_num_hidden = 10;
    int mdn_num_output = 1;
    int mdn_num_gaussian = num_gaussian;
    std::string save_dir = "./mdn_saves";
    int mdn_num_epochs = 10000;
    double mdn_min_error = -900;
    double mdn_learn_rate = 0.0005;
    double mdn_learn_decay = 0.995;
    double mdn_min_learn = 0.00005;
    int num_plot_samples = 5;

    printf("Running Inverted Sin:\n");
    printf(" Seed: %i\n", mdn_seed);
    printf(" Input: %i\n", mdn_num_input);
    printf(" Hidden: %i\n", mdn_num_hidden);
    printf(" Output: %i\n", mdn_num_output);
    printf(" Gaussian: %i\n", mdn_num_gaussian);
    printf(" Save: %s\n", save_dir.c_str());
    printf(" Epochs: %i\n", mdn_num_epochs);
    printf(" Min Error: %f\n", mdn_min_error);
    printf(" Learn Rate: %f\n", mdn_learn_rate);
    printf(" Learn Decay: %f\n", mdn_learn_decay);
    printf(" Min Learn: %f\n", mdn_min_learn);
    printf(" Plot Samples: %i\n", num_plot_samples);

    printf("Loading Original Sin Data...\n");
    std::vector<double> sin_input;
    std::vector<double> sin_output;
    std::ifstream sin_file;
    sin_file.open("../data_sin_inv.txt");
    if (!sin_file) {
        printf("Error loading data_sin_inv.txt\n");
        exit(1);
    }
    double x, y;
    while (sin_file >> x >> y) {
        sin_input.push_back(x);
        sin_output.push_back(y);
    }
    sin_file.close();

    printf("Creating MDN Original Sin Data...\n");
    std::vector<std::vector<double>> sin_mdn_input;
    std::vector<std::vector<double>> sin_mdn_output;
    for (int i = 0; i < sin_input.size(); i++) {
        sin_mdn_input.push_back(std::vector<double>(1, sin_input[i]));
        sin_mdn_output.push_back(std::vector<double>(1, sin_output[i]));
    }
    printf("MDN data created.\n");

    printf("Creating and Training FANN network...");
    MixtureDensityNetwork mdn(mdn_seed, mdn_num_input, mdn_num_hidden, mdn_num_output, mdn_num_gaussian, save_dir);
    mdn.train_network(sin_mdn_input, sin_mdn_output, mdn_num_epochs, mdn_min_error, mdn_learn_rate, mdn_learn_decay, mdn_min_learn);
    printf("Training complete.\n");

    printf("Plotting Sin Data...\n");
    std::vector<double> estimated_in;
    for (double i = -0.5; i <= 0.5; i += 0.005) {
        estimated_in.push_back(i);
    }
    std::vector<double> plot_in;
    std::vector<double> plot_out;
    for (int i = 0; i < estimated_in.size(); i++) {
        for (int j = 0; j < num_plot_samples; j++) {
            std::vector<double> nn_input;
            nn_input.push_back(estimated_in[i]);
            std::vector<double> result = mdn.run_network(nn_input);
            plot_in.push_back(estimated_in[i]);
            plot_out.push_back(result[0]);
        }
    }
    std::string main_title = std::string("Rotated Sin - " + std::to_string(num_gaussian) + " Gaussian");
    plot_graph(sin_input, sin_output, "Real Sin Data", plot_in, plot_out, "Estimated Sin Data", main_title);
}

void plot_original_sin(int num_gaussian) {
    int mdn_seed = 0;
    int mdn_num_input = 1;
    int mdn_num_hidden = 20;
    int mdn_num_output = 1;
    int mdn_num_gaussian = num_gaussian;
    std::string save_dir = "./mdn_saves";
    int mdn_num_epochs = 1000;
    double mdn_min_error = -900;
    double mdn_learn_rate = 0.0025;
    double mdn_learn_decay = 0.999;
    double mdn_min_learn = 0.0005;
    int num_plot_samples = 10;

    printf("Running Inverted Sin:\n");
    printf(" Seed: %i\n", mdn_seed);
    printf(" Input: %i\n", mdn_num_input);
    printf(" Hidden: %i\n", mdn_num_hidden);
    printf(" Output: %i\n", mdn_num_output);
    printf(" Gaussian: %i\n", mdn_num_gaussian);
    printf(" Save: %s\n", save_dir.c_str());
    printf(" Epochs: %i\n", mdn_num_epochs);
    printf(" Min Error: %f\n", mdn_min_error);
    printf(" Learn Rate: %f\n", mdn_learn_rate);
    printf(" Learn Decay: %f\n", mdn_learn_decay);
    printf(" Min Learn: %f\n", mdn_min_learn);
    printf(" Plot Samples: %i\n", num_plot_samples);

    printf("Loading Original Sin Data...\n");
    std::vector<double> sin_input;
    std::vector<double> sin_output;
    std::ifstream sin_file;
    sin_file.open("../data_sin.txt");
    if (!sin_file) {
        printf("Error loading data_sin.txt\n");
        exit(1);
    }
    double x, y;
    while (sin_file >> x >> y) {
        sin_input.push_back(x);
        sin_output.push_back(y);
    }
    sin_file.close();

    printf("Creating MDN Original Sin Data...\n");
    std::vector<std::vector<double>> sin_mdn_input;
    std::vector<std::vector<double>> sin_mdn_output;
    for (int i = 0; i < sin_input.size(); i++) {
        sin_mdn_input.push_back(std::vector<double>(1, sin_input[i]));
        sin_mdn_output.push_back(std::vector<double>(1, sin_output[i]));
    }
    printf("MDN data created.\n");

    printf("Creating and Training FANN network...");
    MixtureDensityNetwork mdn(mdn_seed, mdn_num_input, mdn_num_hidden, mdn_num_output, mdn_num_gaussian, save_dir);
    mdn.train_network(sin_mdn_input, sin_mdn_output, mdn_num_epochs, mdn_min_error, mdn_learn_rate, mdn_learn_decay, mdn_min_learn);
    printf("Training complete.\n");

    printf("Plotting Sin Data...\n");
    std::vector<double> estimated_in;
    for (double i = -0.5; i <= 0.5; i += 0.005) {
        estimated_in.push_back(i);
    }
    std::vector<double> plot_in;
    std::vector<double> plot_out;
    for (int i = 0; i < estimated_in.size(); i++) {
        for (int j = 0; j < num_plot_samples; j++) {
            std::vector<double> nn_input;
            nn_input.push_back(estimated_in[i]);
            std::vector<double> result = mdn.run_network(nn_input);
            plot_in.push_back(estimated_in[i]);
            plot_out.push_back(result[0]);
            printf("In: %f Out: %f\n", nn_input[0], result[0]);
        }
    }
    std::string main_title = std::string("Normal Sin - " + std::to_string(num_gaussian) + " Gaussian");
    plot_graph(sin_input, sin_output, "Real Sin Data", plot_in, plot_out, "Estimated Sin Data", main_title);


}

void plot_sin_training() {
    printf("Loading Original Sin Data...\n");
    std::vector<double> sin_input;
    std::vector<double> sin_output;
    std::ifstream sin_file;
    sin_file.open("../data_sin.txt");
    if (!sin_file) {
        printf("Error loading data_sin.txt\n");
        exit(1);
    }
    double x, y;
    while (sin_file >> x >> y) {
        sin_input.push_back(x);
        sin_output.push_back(y);
    }
    sin_file.close();
    plot_graph(sin_input, sin_output, "0.3x*sin(4*pi*x) + noise", std::vector<double>(), std::vector<double>(), "", "Sin Training Data");
}

void plot_sin_inverted_training() {
    printf("Loading Original Sin Data...\n");
    std::vector<double> sin_input;
    std::vector<double> sin_output;
    std::ifstream sin_file;
    sin_file.open("../data_sin_inv.txt");
    if (!sin_file) {
        printf("Error loading data_sin.txt\n");
        exit(1);
    }
    double x, y;
    while (sin_file >> x >> y) {
        sin_input.push_back(x);
        sin_output.push_back(y);
    }
    sin_file.close();
    plot_graph(sin_input, sin_output, "0.3x*sin(4*pi*x) + noise (x,y flipped)", std::vector<double>(), std::vector<double>(), "", "Rotated Sin Training Data");
}

void plot_graph(std::vector<double> in, std::vector<double> real_out, std::string title_1,
                std::vector<double> estimated_in, std::vector<double> estimated_out, std::string title_2,
                std::string main_title) {
    std::vector<std::pair<float, float>> plot_one;
    std::vector<std::pair<float, float>> plot_two;
    for (int i = 0; i < in.size(); i++) {
        plot_one.push_back(std::make_pair(in[i], real_out[i]));
    }
    for (int i = 0; i < estimated_in.size(); i++) {
        plot_two.push_back(std::make_pair(estimated_in[i], estimated_out[i]));
    }
    Gnuplot gp;
    gp << "set yrange [-1:1]\n";
    gp << "set title \"" << main_title << "\"\n";
    gp << "plot '-' with points lt rgb \"blue\" title '";
    gp << title_1;
    gp << "', '-' with points lt rgb \"red\" title '";
    gp << title_2;
    gp << "'\n";
    gp.send1d(plot_one);
    gp.send1d(plot_two);
}