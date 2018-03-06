#include <iomanip>
#include "MixtureDensityNetwork.h"


MixtureDensityNetwork::MixtureDensityNetwork(int seed, int num_input_dimensions, int num_hidden, int num_output_dimensions,
                           int num_gaussians, std::string saving_dir) {
    this->seed = seed;
    this->num_input_dimensions = num_input_dimensions;
    this->num_hidden = num_hidden;
    this->num_output_dimensions = num_output_dimensions;
    this->num_gaussians = num_gaussians;
    this->gmm = new GaussianMixtureModel(seed);
    this->init_network();
    this->prev_error = INFINITY;
    this->curr_error = INFINITY;
    this->original_learning_rate = DEFAULT_LEARN_RATE;
    this->best_error = INFINITY;
    this->saving_dir = saving_dir;
    this->best_iter = 0;
    this->curr_ex = 0;
    this->curr_iter = 0;
    this->learning_rate_loaded = false;
    this->max_update = -INFINITY;
    this->min_update = INFINITY;
}


void MixtureDensityNetwork::train_network(std::vector<std::vector<double>> input, std::vector<std::vector<double>> output,
                                 int max_iter, double min_error, double learn_rate, double training_decay_rate, double min_learn,
                                 double regc, double clipval, double rmsprop_decay, double rmsprop_smooth) {
    assert(input[0].size() == this->num_input_dimensions && output[0].size() == this->num_output_dimensions);
    std::cout << "Training on " << input.size() << " training examples." << std::endl;
    //save parameters
    this->nn_min_learn = min_learn;
    this->training_input = input;
    this->training_output = output;
    this->rmsprop_decay = rmsprop_decay;
    this->rmsprop_smooth = rmsprop_smooth;
    this->rmsprop_clipval = clipval;
    this->rmsprop_regc = regc;
    if (!this->learning_rate_loaded) {
        this->nn_learning_rate = learn_rate;
    }
    this->original_learning_rate = learn_rate;
    this->nn_max_epochs = max_iter;
    this->nn_max_error = min_error;
    this->nn_training_decay = training_decay_rate;

    //begin training iterations
    for (int iter = 0; iter < this->nn_max_epochs; iter++) {

        this->curr_iter = iter;
        this->last_theta_in = this->theta_in;
        this->last_theta_out = this->theta_out;
        std::cout << "Iteration: " << iter;
        if (this->nn_training_decay > 0) {
            this->set_learning_decay(iter);
        }
        std::cout << " Learn Rate: " << this->nn_learning_rate;
        for (int ex = 0; ex < this->training_input.size(); ex++) {
            this->curr_ex = ex;
            this->forward(this->training_input[ex]);
            this->backward_update(this->training_input[ex], this->training_output[ex]);
        }

        //check error and max epoch exit condition
        double error_nl = this->get_cost_negative_log();
        std::cout << " NL Error: " << error_nl << std::endl;
        this->errors_nl_list.push_back(error_nl);
        this->train_decay_list.push_back(this->nn_learning_rate);
        this->prev_error = this->curr_error;
        this->curr_error = error_nl;
        if (error_nl < this->best_error) {
            this->best_iter = iter;
            this->best_error = error_nl;
            this->save_local_weights();
            this->save_weights(this->saving_dir, "weights_best.txt");
        }
        if (error_nl < this->nn_max_error || iter == this->nn_max_epochs-1) {
            std::cout << "Loading weights with best error. Iteration: " << this->best_iter << std::endl;
            this->load_local_weights();
            break;
        }

    }
}

void MixtureDensityNetwork::init_network() {
    this->num_components = this->num_output_dimensions + 2;
    this->num_gaussian_output = this->num_components * this->num_gaussians;
    //set the pi, sigma, mu output indexes
    this->pi_start = 0;
    this->pi_end = (this->num_gaussian_output / this->num_components) - 1;
    this->sigma_start = this->pi_end + 1;
    this->sigma_end = this->sigma_start + (this->num_gaussian_output / this->num_components) - 1;
    this->mu_start = this->sigma_end + 1;
    this->mu_end = this->num_gaussian_output - 1;
    //be sure all default network values are set
    //initialize matrices
    this->resize_vectors(this->theta_in, this->num_input_dimensions + 1, this->num_hidden);
    this->resize_vectors(this->theta_in_gradients, this->num_input_dimensions + 1, this->num_hidden);
    this->resize_vectors(this->theta_in_updates, this->num_input_dimensions + 1, this->num_hidden);
    this->randomize_vectors(this->theta_in);
    this->resize_vectors(this->theta_out, this->num_hidden + 1, this->num_gaussian_output);
    this->resize_vectors(this->theta_out_updates, this->num_hidden + 1, this->num_gaussian_output);
    this->resize_vectors(this->theta_out_gradients, this->num_hidden + 1, this->num_gaussian_output);
    this->randomize_vectors(this->theta_out);
    //vectors used in propagation, iniatilize size
    this->resize_vectors(this->nn_input, this->num_input_dimensions);
    this->resize_vectors(this->nn_hidden, this->num_hidden + 1);
    this->resize_vectors(this->nn_output, this->num_gaussian_output);
    this->resize_vectors(this->nn_delta_hidden, this->num_hidden);
    this->resize_vectors(this->nn_delta_output, this->num_gaussian_output);
}

double MixtureDensityNetwork::get_real_random() {
    return this->real_uniform(this->generator);
}

void MixtureDensityNetwork::resize_vectors(std::vector<std::vector<double>> &vec, int rows, int cols) {
    vec.clear();
    vec.resize(rows);
    for (int r = 0; r < rows; r++) {
        vec[r].resize(cols);
    }
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            vec[r][c] = 0;
        }
    }
}

void MixtureDensityNetwork::resize_vectors(std::vector<double> &vec, int cols) {
    vec.clear();
    vec.resize(cols);
}

void MixtureDensityNetwork::randomize_vectors(std::vector<std::vector<double>> &vec) {
    for (int r = 0; r < vec.size(); r++) {
        for (int c = 0; c < vec[r].size(); c++) {
            vec[r][c] = this->get_real_random();
        }
    }
}

std::vector<double> MixtureDensityNetwork::forward(const std::vector<double> &input) {
    //create input vector plus one for bias
    this->nn_input.resize(input.size() + 1);
    this->nn_input[0] = 1.0;
    for(int i = 0; i < input.size(); i++) {
        this->nn_input[i+1] = input[i];
    }
    //propagate the input vector to the hidden vector
    //zero the non bias parts of the hidden node vector
    this->nn_hidden[0] = 1.0;
    std::fill(this->nn_hidden.begin()+1, this->nn_hidden.end(), 0);
    for (int i = 0; i < this->nn_input.size(); i++) {
        for (int h = 0; h < this->theta_in[i].size(); h++) {
            this->nn_hidden[h+1] += this->nn_input[i] * this->theta_in[i][h];
        }
    }
    //pass hidden vector through hidden activation function
    this->activation_hidden();
    //propagate the hidden layer to the output layer
    std::fill(this->nn_output.begin(), this->nn_output.end(), 0);
    for (int h = 0; h < this->nn_hidden.size(); h++) {
        for (int o = 0; o < this->theta_out[h].size(); o++) {
            this->nn_output[o] += this->nn_hidden[h] * this->theta_out[h][o];
        }
    }
    //pass the output vector through output activation functions
    this->activation_output();
    return this->nn_output;
}

void MixtureDensityNetwork::activation_hidden() {
    //apply tanh to hidden nodes, start at 1 to skip bias
    for (int h = 1; h < this->nn_hidden.size(); h++) {
        this->nn_hidden[h] = tanh(this->nn_hidden[h]);
    }
}

void MixtureDensityNetwork::activation_output() {
    //the mixing coefficients use a softmax output to between 0 and 1 and sum to 1
    //sum the exp() of each coefficient and normalize
    double pi_sum = 0;
    for (int p = this->pi_start; p <= this->pi_end; p++) {
        this->nn_output[p] = exp(this->nn_output[p]);
        pi_sum += this->nn_output[p];
    }
    for (int p = this->pi_start; p <= this->pi_end; p++) {
        this->nn_output[p] /= pi_sum;
    }
    //the sigma values must be greater than zero, we take the exp() values
    for(int s = this->sigma_start; s <= this->sigma_end; s++) {
        this->nn_output[s] = exp(this->nn_output[s]);
    }
    //the mu values use the unactivated output values, so no changes
}

void MixtureDensityNetwork::test() {
    this->rmsprop_decay = 0.999;
    this->rmsprop_smooth = 1e-8;
    this->rmsprop_clipval = 5.0;
    this->rmsprop_regc = 1e-8;
    this->nn_learning_rate = 0.0001;

    double x = -0.41310404748988916;
    std::vector<double> input;
    input.push_back(x);

    double y = -0.48;
    std::vector<double> output;
    output.push_back(y);

    //bias to hidden
    this->theta_in[0][0] = 0.13026377684977217;
    this->theta_in[0][1] = -0.05460019822804883;

    //input to hidden
    this->theta_in[1][0] = 0.1501047967523503;
    this->theta_in[1][1] = 0.018649899326648792;

    //bias to out
    this->theta_out[0][0] = -0.09285849206870445;
    this->theta_out[0][1] = 0.044786479818621154;
    this->theta_out[0][2] = -0.100333640084536;
    this->theta_out[0][3] = -0.07901363087425885;
    this->theta_out[0][4] = 0.028698961448273892;
    this->theta_out[0][5] = -0.043661150692133846;

    //hidden 1 to out
    this->theta_out[1][0] = -0.046456407568532795;
    this->theta_out[1][1] = 0.0009427038824426906;
    this->theta_out[1][2] = 0.1330239021756948;
    this->theta_out[1][3] = -0.12957073863057852;
    this->theta_out[1][4] = -0.05175823985495203;
    this->theta_out[1][5] = -0.1145592315239981;

    //hidden 2 to out
    this->theta_out[2][0] = 0.14076261763817066;
    this->theta_out[2][1] = -0.022861658695397833;
    this->theta_out[2][2] = -0.05877340458806982;
    this->theta_out[2][3] = -0.040381312848643804;
    this->theta_out[2][4] = 0.08038873832506978;
    this->theta_out[2][5] = -0.02261080733654924;

    //previous update values for theta updates
    //theta in bias update
    this->theta_in_updates[0][0] = 0.14731295391793797;
    this->theta_in_updates[0][1] = 0.022700082233958405;

    //theta in input update
    this->theta_in_updates[1][0] = 0.030704597087497648;
    this->theta_in_updates[1][1] = 0.0047312114609080235;

    //theta out bias update
    this->theta_out_updates[0][0] = 0.003756112398781859;
    this->theta_out_updates[0][1] = 0.0037561123987819973;
    this->theta_out_updates[0][2] = 4.666369382949723;
    this->theta_out_updates[0][3] = 8.34663517306497;
    this->theta_out_updates[0][4] = 3.646795805560068;
    this->theta_out_updates[0][5] = 3.8406498174046275;

    //hidden 1 to out update
    this->theta_out_updates[1][0] = 0.000013094020115657277;
    this->theta_out_updates[1][1] = 0.00001309402011565775;
    this->theta_out_updates[1][2] = 0.01626608459270234;
    this->theta_out_updates[1][3] = 0.02909476073178411;
    this->theta_out_updates[1][4] = 0.012711561352667326;
    this->theta_out_updates[1][5] = 0.013387360381632492;

    //hidden 2 to out update
    this->theta_out_updates[2][0] = 0.000017181764352933618;
    this->theta_out_updates[2][1] = 0.000017181764352933808;
    this->theta_out_updates[2][2] = 0.021345801544155513;
    this->theta_out_updates[2][3] = 0.03818077949043361;
    this->theta_out_updates[2][4] = 0.016681954069541313;
    this->theta_out_updates[2][5] = 0.017568705779312143;

    std::cout << "Input: " << x << std::endl;
    std::cout<<"----------------"<<std::endl;
    std::vector<double> test_output = this->forward(input);
    for (int o = 0; o < test_output.size(); o++) {
        std::cout << "Output[" << o << "]: " << test_output[o] << std::endl;
    }
    std::cout<<"----------------"<<std::endl;
    std::vector<double> training_ex(1, y);
    std::vector<double> normals = this->get_normals(training_ex);
    for (int n = 0; n < normals.size(); n++) {
        std::cout << "Normal[" << n << "]: " << normals[n] << std::endl;
    }
    std::cout<<"----------------"<<std::endl;
    this->set_output_deltas(training_ex);
    for (int o = 0; o < this->nn_delta_output.size(); o++) {
        std::cout << "Delta_Out[" << o << "]: " << this->nn_delta_output[o] << std::endl;
    }
    std::cout<<"----------------"<<std::endl;
    std::cout << "Theta In Before: " << std::endl;
    for (int i = 0; i < this->num_input_dimensions + 1; i++) {
        for (int h = 0; h < this->num_hidden; h++) {
            std::cout << "(" << i << "->" << h << "): " << this->theta_in[i][h] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << "Theta Out Before: " << std::endl;
    for (int h = 0; h < this->num_hidden + 1; h++) {
        for (int o = 0; o < this->num_gaussian_output; o++) {
            std::cout << "(" << h << "->" << o << "): " << this->theta_out[h][o] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    this->forward(input);
    this->backward_update(input, output);

    std::cout << "Theta In After: " << std::endl;
    for (int i = 0; i < this->num_input_dimensions + 1; i++) {
        for (int h = 0; h < this->num_hidden; h++) {
            std::cout << "(" << i << "->" << h << "): " << this->theta_in[i][h] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << "Theta Out After: " << std::endl;
    for (int h = 0; h < this->num_hidden + 1; h++) {
        for (int o = 0; o < this->num_gaussian_output; o++) {
            std::cout << "(" << h << "->" << o << "): " << this->theta_out[h][o] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;


}

std::vector<double> MixtureDensityNetwork::get_normals(const std::vector<double>& training_output) {
    std::vector<double> normals(this->num_gaussians, 0);
    //sum the mu values minus the training example squared values (y_i - mu_i)^2
    int n_i = 0;
    for (int m = this->mu_start; m <= this->mu_end; m += this->num_output_dimensions) {
        for (int m_i = 0; m_i < this->num_output_dimensions; m_i++) {
            normals[n_i] += pow(training_output[m_i] - this->nn_output[m + m_i], 2);
        }
        n_i++;
    }
    //divide by 2 * sigma and take exp()
    //multiply by 2pi^(num_outputs_dimenions/2)*sigma^(num_output_dimenions)
    n_i = 0;
    for (int s = this->sigma_start; s <= this->sigma_end; s++) {
        normals[n_i] /= (2 * pow(this->nn_output[s], 2));
        normals[n_i] = exp(-1 * normals[n_i]);
        normals[n_i] *= 1 / (pow(2 * M_PI, this->num_output_dimensions / 2.0) * pow(this->nn_output[s], this->num_output_dimensions));
        n_i++;
    }
    return normals;
}

std::vector<double> MixtureDensityNetwork::get_gamma(const std::vector<double>& training_output) {

    std::vector<double> gammas(this->num_gaussians, 0);
    std::vector<double> normals = this->get_normals(training_output);

    double sum = 0;
    int p_i = this->pi_start;
    for (int g = 0; g < this->num_gaussians; g++) {
        sum += this->nn_output[p_i] * normals[g];
        p_i++;
    }
    p_i = this->pi_start;
    for (int g = 0; g < this->num_gaussians; g++) {
        if (sum == 0) {
            gammas[g] = 0;
        }
        else {
            gammas[g] = this->nn_output[p_i] * normals[g] / sum;
        }
        p_i++;
    }
    return gammas;
}

void MixtureDensityNetwork::set_output_deltas(const std::vector<double>& training_output) {

    std::vector<double> gammas = this->get_gamma(training_output);
    int g_i = 0;

    //assign the partial derivatives for mixing coefficients pi
    for (int p = this->pi_start; p <= this->pi_end; p++) {
        this->nn_delta_output[p] = this->nn_output[p] - gammas[g_i];
        g_i++;
    }

    //assign partial derivatives for sigma values
    g_i = 0;
    int s_i = this->sigma_start;
    for (int m = this->mu_start; m <= this->mu_end; m += this->num_output_dimensions) {
        this->nn_delta_output[s_i] = 0;
        for (int m_i = 0; m_i < this->num_output_dimensions; m_i++) {
            this->nn_delta_output[s_i] += pow(training_output[m_i] - this->nn_output[m + m_i], 2);
        }
        double calc_sigma = this->nn_delta_output[s_i] / pow(this->nn_output[s_i], 2);
        this->nn_delta_output[s_i] = -gammas[g_i] * (calc_sigma - 1);
        s_i++;
        g_i++;
    }

    //assign partial derivatives for mu values
    g_i = 0;
    s_i = this->sigma_start;
    for (int m = this->mu_start; m <= this->mu_end; m += this->num_output_dimensions) {
        for (int y = 0; y < this->num_output_dimensions; y++) {
            double calc_mu = ((this->nn_output[m+y] - training_output[y]) / pow(this->nn_output[s_i], 2));
            this->nn_delta_output[m+y] = gammas[g_i] * calc_mu;
        }
        g_i++;
        s_i++;
    }

}


void MixtureDensityNetwork::backward_update(const std::vector<double> &training_input, const std::vector<double> &training_output) {
    //calculate gradients for weights from hidden to output
    this->set_output_deltas(training_output);

    for (int h = 0; h < this->num_hidden + 1; h++) {
        for (int o = 0; o < this->num_gaussian_output; o++) {
            this->theta_out_gradients[h][o] = this->nn_delta_output[o] * this->nn_hidden[h];
        }
    }

    for (int h = 0; h < this->num_hidden; h++) {
        this->nn_delta_hidden[h] = 0;
        for (int o = 0; o < this->num_gaussian_output; o++) {
            this->nn_delta_hidden[h] += this->theta_out[h+1][o] * this->nn_delta_output[o];
        }
        this->nn_delta_hidden[h] *= (1 - pow(this->nn_hidden[h+1], 2));

    }

    for (int i = 0; i < this->num_input_dimensions + 1; i++) {
        for (int h = 0; h < this->num_hidden; h++) {
            this->theta_in_gradients[i][h] = this->nn_delta_hidden[h] * this->nn_input[i];
        }
    }

    std::vector<std::vector<double>> prev_theta_in = this->theta_in;

    for (int i = 0; i < this->num_input_dimensions + 1; i++) {
        for (int h = 0; h < this->num_hidden; h++) {
            double mdwi = this->theta_in_gradients[i][h];
            this->theta_in_updates[i][h] = this->theta_in_updates[i][h] * this->rmsprop_decay + (1.0 - this->rmsprop_decay) * mdwi * mdwi;
            if (mdwi > this->rmsprop_clipval) {
                mdwi = this->rmsprop_clipval;
            } else if (mdwi < -this->rmsprop_clipval) {
                mdwi = -this->rmsprop_clipval;
            }
            this->theta_in[i][h] += -this->nn_learning_rate * mdwi / sqrt(this->theta_in_updates[i][h] + this->rmsprop_smooth) - this->rmsprop_regc * this->theta_in[i][h];
            this->theta_in_gradients[i][h] = 0;
        }
    }

    std::vector<std::vector<double>> prev_theta_out = this->theta_out;
    for (int h = 0; h < this->num_hidden + 1; h++) {
        for (int o = 0; o < this->num_gaussian_output; o++) {
            double mdwi = this->theta_out_gradients[h][o];
            this->theta_out_updates[h][o] = this->theta_out_updates[h][o] * this->rmsprop_decay + (1.0 - this->rmsprop_decay) * mdwi * mdwi;
            if (mdwi > this->rmsprop_clipval) {
                mdwi = this->rmsprop_clipval;
            } else if (mdwi < -this->rmsprop_clipval) {
                mdwi = -this->rmsprop_clipval;
            }
            this->theta_out[h][o] += -this->nn_learning_rate * mdwi / sqrt(this->theta_out_updates[h][o] + this->rmsprop_smooth) - this->rmsprop_regc * this->theta_out[h][o];
            this->theta_out_gradients[h][o] = 0;
        }
    }


}

void MixtureDensityNetwork::set_learning_decay(int timestep) {
    //this->nn_learning_rate = 1.0 / (1 + this->nn_training_decay * timestep);
    if (this->nn_learning_rate * this->nn_training_decay >= this->nn_min_learn ) {
        this->nn_learning_rate *= this->nn_training_decay;
    }
}

double MixtureDensityNetwork::get_cost_negative_log() {
    double log_sum = 0;
    for (int ex = 0; ex < this->training_input.size(); ex++) {
        this->forward(this->training_input[ex]);
        std::vector<double> normals = this->get_normals(this->training_output[ex]);
        double normal_sum = 0;
        int p_i = this->pi_start;
        for (int g = 0; g < this->num_gaussians; g++) {
            normal_sum += this->nn_output[p_i] * normals[g];
            p_i++;
        }
        double log_normal_sum = log(normal_sum);
        log_sum += log_normal_sum;
    }
    return -log_sum;
}

std::vector<double> MixtureDensityNetwork::run_network(const std::vector<double> &input) {
    assert(input.size() == this->num_input_dimensions);
    this->forward(input);
    std::vector<double> pi(this->nn_output.begin() + this->pi_start, this->nn_output.begin() + this->pi_end + 1);
    std::vector<double> sigma(this->nn_output.begin() + this->sigma_start, this->nn_output.begin() + this->sigma_end + 1);
    std::vector<double> samples;
    for (int o = 0; o < this->num_output_dimensions; o++) {
        std::vector<double> mu;
        for (int m = this->mu_start; m < this->num_gaussian_output; m += this->num_output_dimensions) {
            mu.push_back(this->nn_output[m + o]);
        }
        this->gmm->set_models(pi, mu, sigma);
        double sample = gmm->get_sample();
        samples.push_back(sample);
    }
    return samples;
}

void MixtureDensityNetwork::print_output(bool before_activation) {
    if (before_activation) {
        std::cout << "Output Non-Activated:" << std::endl;
    }
    else {
        std::cout << "Output Activated:" << std::endl;
    }
    for (int o = 0; o < this->nn_output.size(); o++) {
        std::cout << " out[" << o << "]: " << this->nn_output[o] << std::endl;
    }
    std::cout << std::endl;
}

void MixtureDensityNetwork::print_theta_in() {
    std::cout << "Theta_In:" << std::endl;
    for (int i = 0; i < this->theta_in.size(); i++) {
        for (int h = 0; h < this->theta_in[i].size(); h++) {
            std::cout << "(" << i << "," << h <<")=" << this->theta_in[i][h] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void MixtureDensityNetwork::print_theta_out() {
    std::cout << "Theta_Out:" << std::endl;
    for (int h = 0; h < this->theta_out.size(); h++) {
        for (int o = 0; o < this->theta_out[h].size(); o++) {
            std::cout << "(" << h << "," << o <<")=" << this->theta_out[h][o] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void MixtureDensityNetwork::print_hidden(bool before_activation) {
    if (before_activation) {
        std::cout << "Hidden Non-Activated:" << std::endl;
    }
    else {
        std::cout << "Hidden Activated:" << std::endl;
    }
    for (int h = 0; h < this->nn_hidden.size(); h++) {
        std::cout << "hid[" << h << "]: " << this->nn_hidden[h] << std::endl;
    }
    std::cout << std::endl;
}

void MixtureDensityNetwork::print_input() {
    std::cout << "Input:" << std::endl;
    for (int i = 0; i < this->nn_input.size(); i++) {
        std::cout << "In[" << i << "]: " << this->nn_input[i] << std::endl;
    }
    std::cout << std::endl;
}

void MixtureDensityNetwork::load_local_weights() {
    this->theta_in = this->best_theta_in;
    this->theta_out = this->best_theta_out;
}

void MixtureDensityNetwork::save_local_weights() {
    this->best_theta_in = this->theta_in;
    this->best_theta_out = this->theta_out;
}

void MixtureDensityNetwork::load_weights(std::string dir, std::string file) {
    std::string file_path = DirTools::get_file_from_pwd(dir, file);
    std::cout << "Loading weights from: " << file_path << std::endl;
    int theta_in_rows = 0;
    int theta_in_cols = 0;
    int theta_out_rows = 0;
    int theta_out_cols = 0;

    std::ifstream in;
    in.open(file_path);
    if (!in.is_open()) {
        printf("Error loading file: %s\n", file_path.c_str());
    }

    in >> theta_in_rows;
    in >> theta_in_cols;
    in >> theta_out_rows;
    in >> theta_out_cols;

    this->theta_in.resize(theta_in_rows);
    for (int r = 0; r < theta_in_rows; r++) {
        this->theta_in[r].resize(theta_in_cols);
    }
    this->theta_out.resize(theta_out_rows);
    for (int r = 0; r < theta_out_rows; r++) {
        this->theta_out[r].resize(theta_out_cols);
    }

    double data;
    for (int r = 0; r < theta_in_rows; r++) {
        for (int c = 0; c < theta_in_cols; c++) {
            in >> data;
            this->theta_in[r][c] = data;
        }
    }
    for (int r = 0; r < theta_in_rows; r++) {
        for (int c = 0; c < theta_in_cols; c++) {
            in >> data;
            this->theta_in_updates[r][c] = data;
        }
    }
    for (int r = 0; r < theta_out_rows; r++) {
        for (int c = 0; c < theta_out_cols; c++) {
            in >> data;
            this->theta_out[r][c] = data;
        }
    }
    for (int r = 0; r < theta_out_rows; r++) {
        for (int c = 0; c < theta_out_cols; c++) {
            in >> data;
            this->theta_out_updates[r][c] = data;
        }
    }
    std::cout << "Weights loaded." << std::endl;
}

void MixtureDensityNetwork::save_weights(std::string dir, std::string file) {
    if (!DirTools::check_dir_exists(DirTools::get_path_from_pwd(dir))) {
        std::string dir_path = DirTools::get_path_from_pwd(dir);
        printf("Creating directory: %s\n", dir_path.c_str());
        bool dir_created = DirTools::create_dir(dir_path);
        if (!dir_created) {
            printf("Error creating directory: %s\n", dir_path.c_str());
            return;
        }
    }
    std::string file_path = DirTools::get_file_from_pwd(dir, file);
    std::ofstream out;
    if (!DirTools::check_file_exists(file_path)) {
        printf("Creating file: %s\n", file_path.c_str());
    }
    out.open(file_path);
    out << std::setprecision(std::numeric_limits<long double>::digits10);
    if (!out.is_open()) {
        printf("Error creating file: %s\n", file_path.c_str());
        return;
    }
    //write data
    out << this->theta_in.size() << " " << this->theta_in[0].size() << " " << this->theta_out.size() << " " << this->theta_out[0].size() << std::endl;

    for (int i = 0; i < this->theta_in.size(); i++) {
        for (int h = 0;h < this->theta_in[i].size(); h++) {
            out << this->theta_in[i][h] << " ";
        }
        out << std::endl;
    }
    for (int i = 0; i < this->theta_in_updates.size(); i++) {
        for (int h = 0; h < this->theta_in_updates[i].size(); h++) {
            out << this->theta_in_updates[i][h] << " ";
        }
        out << std::endl;
    }

    for (int h = 0; h < this->theta_out.size(); h++) {
        for (int o = 0; o < this->theta_out[h].size(); o++) {
            out << this->theta_out[h][o] << " ";
        }
        out << std::endl;
    }
    for (int h = 0; h < this->theta_out_updates.size(); h++) {
        for (int o = 0; o < this->theta_out_updates[h].size(); o++) {
            out << this->theta_out_updates[h][o] << " ";
        }
        out << std::endl;
    }
    out.close();
}

void MixtureDensityNetwork::print_theta_in_update() {
    std::cout << "Theta_In Update:" << std::endl;
    for (int i = 0; i < this->theta_in_updates.size(); i++) {
        for (int h = 0; h < this->theta_in_updates[i].size(); h++) {
            std::cout << "(" << i << "," << h <<")=" << this->theta_in_updates[i][h] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void MixtureDensityNetwork::print_theta_out_update() {
    std::cout << "Theta_Out Update:" << std::endl;
    for (int h = 0; h < this->theta_out_updates.size(); h++) {
        for (int o = 0; o < this->theta_out_updates[h].size(); o++) {
            std::cout << "(" << h << "," << o <<")=" << this->theta_out_updates[h][o] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void MixtureDensityNetwork::print_theta_in_grad() {
    std::cout << "Theta_In Gradient:" << std::endl;
    for (int i = 0; i < this->theta_in_gradients.size(); i++) {
        for (int h = 0; h < this->theta_in_gradients[i].size(); h++) {
            std::cout << "(" << i << "," << h <<")=" << this->theta_in_gradients[i][h] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void MixtureDensityNetwork::print_theta_out_grad() {
    std::cout << "Theta_Out Gradient:" << std::endl;
    for (int h = 0; h < this->theta_out_gradients.size(); h++) {
        for (int o = 0; o < this->theta_out_gradients[h].size(); o++) {
            std::cout << "(" << h << "," << o <<")=" << this->theta_out_gradients[h][o] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void MixtureDensityNetwork::print_output_deltas() {
    std::cout << "Output Deltas:" << std::endl;
    for (int o = 0; o < this->nn_delta_output.size(); o++) {
        std::cout << "delta_out[" << o << "]: " << this->nn_delta_output[o] << std::endl;
    }
}

void MixtureDensityNetwork::print_hidden_deltas() {
    std::cout << "Hidden Deltas:" << std::endl;
    for (int o = 0; o < this->nn_delta_hidden.size(); o++) {
        std::cout << "delta_hid[" << o << "]: " << this->nn_delta_hidden[o] << std::endl;
    }
}

void MixtureDensityNetwork::save_error(std::string dir, std::string file) {
    if (!DirTools::check_dir_exists(DirTools::get_path_from_pwd(dir))) {
        std::string dir_path = DirTools::get_path_from_pwd(dir);
        printf("Creating directory: %s\n", dir_path.c_str());
        bool dir_created = DirTools::create_dir(dir_path);
        if (!dir_created) {
            printf("Error creating directory: %s\n", dir_path.c_str());
            return;
        }
    }
    std::string file_path = DirTools::get_file_from_pwd(dir, file);
    if (!DirTools::check_file_exists(file_path)) {
        printf("Creating file: %s\n", file_path.c_str());
    }
    std::ofstream out;
    out.open(file_path);
    if (!out.is_open()) {
        printf("Error creating file: %s\n", file_path.c_str());
        return;
    }
    //write data
    for (int i = 0; i < this->errors_nl_list.size(); i++) {
        out << this->errors_nl_list[i] << std::endl;
    }
    out.close();
}

void MixtureDensityNetwork::save_learning_decay(std::string dir, std::string file) {
    if (!DirTools::check_dir_exists(DirTools::get_path_from_pwd(dir))) {
        std::string dir_path = DirTools::get_path_from_pwd(dir);
        printf("Creating directory: %s\n", dir_path.c_str());
        bool dir_created = DirTools::create_dir(dir_path);
        if (!dir_created) {
            printf("Error creating directory: %s\n", dir_path.c_str());
            return;
        }
    }
    std::string file_path = DirTools::get_file_from_pwd(dir, file);
    if (!DirTools::check_file_exists(file_path)) {
        printf("Creating file: %s\n", file_path.c_str());
    }
    std::ofstream out;
    out.open(file_path);
    if (!out.is_open()) {
        printf("Error creating file: %s\n", file_path.c_str());
        return;
    }
    //write data
    for (int i = 0; i < this->train_decay_list.size(); i++) {
        out << this->train_decay_list[i] << std::endl;
    }
    out.close();
}

void MixtureDensityNetwork::print_all() {
    this->print_input();
    this->print_theta_in();
    this->print_theta_in_grad();
    this->print_theta_in_update();
    this->print_hidden(false);
    this->print_theta_out();
    this->print_theta_out_grad();
    this->print_theta_out_update();
    this->print_output(false);
    this->print_output_deltas();
    this->print_hidden_deltas();
}

void MixtureDensityNetwork::print_training_ex(int ex) {
    std::cout << "Training Example: " << ex << std::endl;
    std::cout << " Input: ";
    for (int i = 0; i < this->training_input[ex].size(); i++) {
        std::cout << this->training_input[ex][i] << " ";
    }
    std::cout << std::endl;
    std::cout << " Output: ";
    for (int o = 0; o < this->training_output[ex].size(); o++) {
        std::cout << this->training_output[ex][o] << " ";
    }
    std::cout << std::endl;
}

void MixtureDensityNetwork::load_error(std::string dir, std::string file) {
    std::string file_path = DirTools::get_file_from_pwd(dir, file);
    std::cout << "Loading errors from: " << file_path << std::endl;

    std::ifstream in;
    in.open(file_path);
    if (!in.is_open()) {
        printf("Error loading file: %s\n", file_path.c_str());
    }

    double data;
    while (in >> data) {
        this->errors_nl_list.push_back(data);
    }

    in.close();
}

void MixtureDensityNetwork::load_learning_decay(std::string dir, std::string file) {
    std::string file_path = DirTools::get_file_from_pwd(dir, file);
    std::cout << "Loading errors from: " << file_path << std::endl;

    std::ifstream in;
    in.open(file_path);
    if (!in.is_open()) {
        printf("Error loading file: %s\n", file_path.c_str());
    }

    double data;
    while (in >> data) {
        this->train_decay_list.push_back(data);
    }

    this->nn_learning_rate = this->train_decay_list.back();
    this->learning_rate_loaded = true;

    in.close();
}



