#include <cstdio>
#include <floatfann.h>
#include <fann_cpp.h>
#include <gnuplot-iostream.h>

void plot_original_sin();
void plot_inverted_sin();
void plot_graph(std::vector<double> in, std::vector<double> real_out, std::string title_1,
                std::vector<double> estimated_in, std::vector<double> estimated_out, std::string title_2,
                std::string main_title);

//NN parameters
int num_input = 1;
int num_output = 1;
int num_hidden = 10;
int num_layers = 3;
int num_epochs = 5000;
int num_report = 100;
double max_error = 0.0001;
double learning_rate = 0.1;

//parameters for plotting graph
double plot_start = -0.5;
double plot_end = 0.5;
double plot_increment = 0.005;

int main(int argc, char** argv) {
    printf("Beginning NN Sin Example...\n");

    plot_original_sin();
    plot_inverted_sin();

    printf("End of program.\n");
    return 0;
}

void plot_inverted_sin() {
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

    printf("Creating FANN Original Sin Data...\n");
    fann_type** sin_fann_input = new fann_type*[sin_input.size()];
    fann_type** sin_fann_output = new fann_type*[sin_input.size()];
    for (int i = 0; i < sin_input.size(); i++) {
        sin_fann_input[i] = new fann_type[1];
        sin_fann_input[i][0] = sin_input[i];
        sin_fann_output[i] = new fann_type[1];
        sin_fann_output[i][0] = sin_output[i];
    }
    FANN::training_data train_data;
    train_data.set_train_data(sin_input.size(), 1, sin_fann_input, 1, sin_fann_output);
    printf("FANN data created.\n");

    printf("Creating and Training FANN network...");
    FANN::neural_net nn;
    nn.create_standard(num_layers, num_input, num_hidden, num_output); //layers, input, hidden, output
    nn.set_activation_function_hidden(FANN::SIGMOID_SYMMETRIC);
    nn.set_activation_function_output(FANN::LINEAR);
    nn.init_weights(train_data);
    nn.set_learning_rate(learning_rate);
    nn.train_on_data(train_data, num_epochs, num_report, max_error); //max epochs, epoch report, min error
    printf("Training complete.\n");

    printf("Plotting Sin Data...\n");
    std::vector<double> estimated_in;
    for (double i = plot_start; i <= plot_end; i += plot_increment) {
        estimated_in.push_back(i);
    }
    std::vector<double> estimated_out;
    for (int i = 0; i < estimated_in.size(); i++) {
        fann_type nn_input[1];
        nn_input[0] = estimated_in[i];
        fann_type* result = nn.run(nn_input);
        estimated_out.push_back(result[0]);
    }
    plot_graph(sin_input, sin_output, "Real Sin Data", estimated_in, estimated_out, "Estimated Sin Data", "Rotated Sin");


    printf("Deleting FANN Sin Data...\n");
    for (int i = 0; i < sin_input.size(); i++) {
        delete sin_fann_input[i];
        delete sin_fann_output[i];
    }
    delete sin_fann_input;
    delete sin_fann_output;
    printf("Data deleted.\n");
}

void plot_original_sin() {
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

    printf("Creating FANN Original Sin Data...\n");
    fann_type** sin_fann_input = new fann_type*[sin_input.size()];
    fann_type** sin_fann_output = new fann_type*[sin_input.size()];
    for (int i = 0; i < sin_input.size(); i++) {
        sin_fann_input[i] = new fann_type[1];
        sin_fann_input[i][0] = sin_input[i];
        sin_fann_output[i] = new fann_type[1];
        sin_fann_output[i][0] = sin_output[i];
    }
    FANN::training_data train_data;
    train_data.set_train_data(sin_input.size(), 1, sin_fann_input, 1, sin_fann_output);
    printf("FANN data created.\n");

    printf("Creating and Training FANN network...");
    FANN::neural_net nn;
    nn.create_standard(num_layers, num_input, num_hidden, num_output);
    nn.set_activation_function_hidden(FANN::SIGMOID_SYMMETRIC);
    nn.set_activation_function_output(FANN::LINEAR);
    nn.init_weights(train_data);
    nn.set_learning_rate(learning_rate);
    nn.train_on_data(train_data, num_epochs, num_report, max_error);
    printf("Training complete.\n");

    printf("Plotting Sin Data...\n");
    std::vector<double> estimated_in;
    for (double i = -0.5; i <= 0.5; i += 0.005) {
        estimated_in.push_back(i);
    }
    std::vector<double> estimated_out;
    for (int i = 0; i < estimated_in.size(); i++) {
        fann_type nn_input[1];
        nn_input[0] = estimated_in[i];
        fann_type* result = nn.run(nn_input);
        estimated_out.push_back(result[0]);
    }
    plot_graph(sin_input, sin_output, "Real Sin Data", estimated_in, estimated_out, "Estimated Sin Data", "Normal Sin");


    printf("Deleting FANN Sin Data...\n");
    for (int i = 0; i < sin_input.size(); i++) {
        delete sin_fann_input[i];
        delete sin_fann_output[i];
    }
    delete sin_fann_input;
    delete sin_fann_output;
    printf("Data deleted.\n");
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

//void plot_graph(std::vector<double> in, std::vector<double> real_out, std::string title_1,
//                std::vector<double> estimated_in, std::vector<double> estimated_out, std::string title_2) {
//    std::vector<std::pair<float, float>> plot_one;
//    std::vector<std::pair<float, float>> plot_two;
//    for (int i = 0; i < in.size(); i++) {
//        plot_one.push_back(std::make_pair(in[i], real_out[i]));
//    }
//    for (int i = 0; i < estimated_in.size(); i++) {
//        plot_two.push_back(std::make_pair(estimated_in[i], estimated_out[i]));
//    }
//    Gnuplot gp;
//    gp << "set yrange [-1:1]\n";
//    gp << "plot '-' with points lt rgb \"blue\" title '";
//    gp << title_1;
//    gp << "', '-' with lines lt rgb \"red\" lw 3 title '";
//    gp << title_2;
//    gp << "'\n";
//    gp.send1d(plot_one);
//    gp.send1d(plot_two);
//}