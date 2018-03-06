#ifndef MIXTURE_DENSITY_NETWORK
#define MIXTURE_DENSITY_NETWORK

#include <vector>
#include <random>
#include <fstream>
#include <math.h>
#include "GaussianMixtureModel.h"
#include "DirTools.h"

#define DEFAULT_LEARN_RATE 0.0001
#define DEFAULT_DECAY_LEARN -1.0
#define DEFAULT_RMSPROP_REGC 1e-8
#define DEFAULT_RMSPROP_CLIPVAL 5.0
#define DEFAULT_RMSPROP_DECAY 0.999
#define DEFAULT_RMSPROP_SMOOTH 1e-8
#define DEFAULT_MIN_LEARN 0.0001

class MixtureDensityNetwork {

public:
    MixtureDensityNetwork(int seed, int num_input_dimensions, int num_hidden, int num_output_dimensions, int num_gaussians, std::string saving_dir);
    ~MixtureDensityNetwork() {delete this->gmm;}

    void train_network(std::vector<std::vector<double>> input, std::vector<std::vector<double>> output,
                       int max_iter, double min_error,
                       double learn_rate = DEFAULT_LEARN_RATE, double training_decay_rate = DEFAULT_DECAY_LEARN,
                       double min_learn = DEFAULT_MIN_LEARN, double regc = DEFAULT_RMSPROP_REGC, double clipval = DEFAULT_RMSPROP_CLIPVAL,
                       double rmsprop_decay = DEFAULT_RMSPROP_DECAY, double rmsprop_smooth = DEFAULT_RMSPROP_SMOOTH);


    std::vector<double> get_negative_log_errors() {return this->errors_nl_list;}
    std::vector<double> get_train_decay_values() {return this->train_decay_list;}

    std::vector<double> run_network(const std::vector<double> &input);

    void load_weights(std::string dir, std::string file);
    void save_weights(std::string dir, std::string file);
    void save_error(std::string dir, std::string file);
    void load_error(std::string dir, std::string file);
    void save_learning_decay(std::string dir, std::string file);
    void load_learning_decay(std::string dir, std::string file);

    void test();


private:
    int seed;
    int num_input_dimensions, num_hidden, num_output_dimensions, num_gaussians;
    int num_gaussian_output, num_components;
    std::string saving_dir;

    //output indexes
    int pi_start, pi_end;
    int sigma_start, sigma_end;
    int mu_start, mu_end;

    //NNet parameters
    double nn_learning_rate, nn_min_learn, nn_max_epochs, nn_max_error, nn_training_decay;
    //RMSProp parameters
    double rmsprop_decay, rmsprop_smooth, rmsprop_clipval, rmsprop_regc;

    //network weights
    std::vector<std::vector<double>> theta_in, theta_out;
    std::vector<std::vector<double>> theta_in_gradients, theta_out_gradients;
    std::vector<std::vector<double>> theta_in_updates, theta_out_updates;

    //network nodes
    std::vector<double> nn_input, nn_hidden, nn_output, nn_delta_output, nn_delta_hidden;

    //training data
    std::vector<std::vector<double>> training_input, training_output;

    //Holds a list of the past errors and training rates for plotting
    std::vector<double> errors_nl_list, train_decay_list;

    void init_network();
    std::vector<double> forward(const std::vector<double>& input);
    void backward_update(const std::vector<double> &training_input, const std::vector<double> &training_output);

    GaussianMixtureModel *gmm;

    void activation_hidden();
    void activation_output();

    std::vector<double> get_normals(const std::vector<double>& training_output);
    std::vector<double> get_gamma(const std::vector<double>& training_output);
    void set_output_deltas(const std::vector<double>& training_output);

    double get_cost_negative_log();

    //random number generation
    double get_real_random();
    std::uniform_real_distribution<double> real_uniform{-1.0, 1.0};
    std::default_random_engine generator;

    //helper functions
    void resize_vectors(std::vector<std::vector<double>>& vec, int rows, int cols);
    void resize_vectors(std::vector<double>& vec, int cols);
    void randomize_vectors(std::vector<std::vector<double>>& vec);
    void set_learning_decay(int timestep);

    //save the best network
    void save_local_weights();
    void load_local_weights();
    std::vector<std::vector<double>> best_theta_in, best_theta_out;
    std::vector<std::vector<double>> last_theta_in, last_theta_out;
    double best_error;
    int best_iter;
    bool learning_rate_loaded;


    //temp debug stuff
    int curr_iter, curr_ex;
    void print_output(bool before_activation);
    void print_theta_in();
    void print_theta_out();
    void print_hidden(bool before_activation);
    void print_input();
    void print_theta_in_update();
    void print_theta_out_update();
    void print_theta_in_grad();
    void print_theta_out_grad();
    void print_output_deltas();
    void print_hidden_deltas();
    void print_all();
    void print_training_ex(int ex);
    double prev_error, curr_error;
    double original_learning_rate;
    double max_update, min_update;

};



#endif
