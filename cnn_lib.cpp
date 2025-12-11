#include "cnn_lib.h"

//=====================================================================
//================== Basix Matrix Operation Function ==================
//=====================================================================

bool matrix_add_scalar(std::vector<float> *matrix, float scalar){
    if(matrix == nullptr){
        std::cerr << "Error: Matrix pointer not provided" << std::endl;
        return false;
    }
    for(int i = 0; i < matrix->size(); i++){
        (*matrix)[i] += scalar;
    }
    return true;
}

bool matrix_add_matrix(std::vector<float> matrix_a, std::vector<float> matrix_b, int a_height, int a_width, int b_height, int b_width, std::vector<float> *result){
    if(result == nullptr){
        std::cerr << "Error: Result pointer not provided" << std::endl;
        return false;
    }
    else if(a_height != b_height || a_width != b_width){
        std::cerr << "Error: Matrix dimensions are incompatible for addition" << std::endl;
        return false;
    }

    result->clear();
    result->resize(a_height * a_width); //since in matrix addition 2 matrices have the same size so it will output that same size

    for(int i = 0; i < result->size(); i++){
        (*result)[i] = matrix_a[i] + matrix_b[i];
    }
    return true;
}

bool matrix_multiply(std::vector<float> matrix_a, std::vector<float> matrix_b, int a_height, int a_width, int b_height, int b_width, std::vector<float> *result){
    if(result == nullptr){
        std::cerr << "Error: Result pointer not provided" << std::endl;
        return false;
    }
    else if(a_width != b_height){
        std::cerr << "Error: Matrix dimensions are incompatible for multiplication" << std::endl;
        return false;
    }

    int result_height = a_height;
    int result_width = b_width;
    int common_dim = a_width;

    result->clear();
    result->resize(result_height * result_width);

    for (int i = 0; i < result_height; ++i) { 
        for (int j = 0; j < result_width; ++j) {  
            float sum = 0.0f;
            for (int k = 0; k < common_dim; ++k) {
                float a_val = matrix_a[i * a_width + k];
                float b_val = matrix_b[k * b_width + j];
                
                sum += a_val * b_val;
            }
            (*result)[i * result_width + j] = sum;
        }
    }
    return true;
}

bool matrix_transpose(const std::vector<float>& matrix_in, int height, int width, std::vector<float> *result){
    if(result == nullptr){
        std::cerr << "Error: Result pointer not provided" << std::endl;
        return false;
    }
    
    int expected_size = height * width;
    if(matrix_in.size() != expected_size){
        std::cerr << "Error: Input matrix size does not match provided dimensions" << std::endl;
        return false;
    }

    result->clear();
    result->resize(expected_size); 

    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            (*result)[j * height + i] = matrix_in[i * width + j];
        }
    }
    return true;
}

//=========================================================
//================== Base Layer Function ==================
//=========================================================

void Layer::set_activation_type(ActivationType input_type){
    act_type = input_type;
}

float Layer::apply_activation_function(float input){
    float result;
    switch(act_type) {
        case ActivationType::RELU:
            // ReLU: a = max(0, z)
            result = std::max(0.0f, input);
            break;
            
        case ActivationType::SIGMOID:
            // Sigmoid: a = 1 / (1 + e^(-z))
            result = 1.0f / (1.0f + std::exp(input));
            break;
            
        case ActivationType::NONE:
            // Linear Activation (for regression output layer): a = z
            result = input;
            break;
        default:
            break;
    }
    return result;
}

//========================================================
//================== Dense Layer Class ===================
//========================================================

DenseLayer::DenseLayer(int num_node_curr, int num_node_prev_in)
    : num_node(num_node_curr),
      num_node_prev(num_node_prev_in)
{
    weight_matrix.resize(num_node * num_node_prev);
    bias.resize(num_node, 0.0f);
    output.resize(num_node);
    weighted_sum.resize(num_node);

    // Xavier initialization
    float limit = std::sqrt(6.0f / (num_node_prev + num_node));
    std::uniform_real_distribution<float> dist(-limit, limit);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    for (int i = 0; i < weight_matrix.size(); ++i) {
        weight_matrix[i] = dist(generator); // Now this will be a unique random number
    }
}

bool DenseLayer::forward_propagation(const std::vector<float>& input){
    input_cache = input;

    if(!matrix_multiply(weight_matrix, input, num_node, num_node_prev, num_node_prev, 1, &weighted_sum)){ //multiply input with weight matrix to calculate z for this layer
        return false;
    }

    if(!matrix_add_matrix(weighted_sum, bias, num_node, 1, num_node, 1, &weighted_sum)){ //add the bias vector for output
        return false;
    }

    output.resize(num_node);
    for(int i = 0; i < num_node; i++) {
        output[i] = apply_activation_function(weighted_sum[i]); //apply activation function
    }

    return true;
}

std::vector<float> DenseLayer::backward_propagation(const std::vector<float>& dA, float learning_rate)
{
    std::vector<float> dZ(num_node); //dZ is the derivative of the error (loss) with respect to pre activation value Z
    switch (act_type){
    case ActivationType::RELU:
        for(int j = 0; j < num_node; j++){
            dZ[j] = (weighted_sum[j] > 0 ? 1.0f : 0.0f) * dA[j]; //dZ = dA * ReLU'(weight)
        }
        break;
    
    case ActivationType::SIGMOID:
        for(int j = 0; j < num_node; j++){
            float A = output[j];
            dZ[j] = A * (1.0f - A) * dA[j]; //dZ = dA * sigmoid'(weight)
        }
        break;

    case ActivationType::NONE:
        for(int j = 0; j < num_node; j++){
            dZ[j] = dA[j]; //dZ = dA * 1
        }
        break;
    default:
        break;
    }

    //derivative of error with respect to previous activation is derivative of error with respect to Z times derivative of Z with respect to previous activation, which is dZ * weight (why is weight matrix need to be transpose? idk)
    std::vector<float> weight_transpose;
    if (!matrix_transpose(weight_matrix, num_node, num_node_prev, &weight_transpose)) {
        std::cerr << "Error: Failed to transpose weight matrix." << std::endl;
        return std::vector<float>{}; 
    }
    std::vector<float> dA_prev;
    if (!matrix_multiply(weight_transpose, dZ, num_node_prev, num_node, num_node, 1, &dA_prev)){
        std::cerr << "Error: Failed to multiply W^T * dZ." << std::endl;
        return std::vector<float>{}; 
    }

    for(int j = 0; j < num_node; j++){
        bias[j] -= learning_rate * dZ[j]; //derivative of error with respect to b is derivative of error with respect to Z times derivative of Z with respect to b, which is dZ times 1
        for(int i = 0; i < num_node_prev; i++){
            float dW = dZ[j] * input_cache[i]; //derivative of error with respect to weight is just equal to the previous layer activation value times derivative of error with respect to Z (chain rule)
            weight_matrix[j * num_node_prev + i] -= learning_rate * dW; //update weight matrix based on calculated gradient
        }
    }

    //return dA (error for previous layer) for further back propagation
    return dA_prev;
}

//=============================================================
//================== Convolution Layer Class ==================
//=============================================================

ConvLayer::ConvLayer(int in_h_, int in_w_, int in_ch_,
                     int f_h, int f_w, int out_ch_)
        : input_height(in_h_), input_width(in_w_), input_channel(in_ch_),
        filter_height(f_h), filter_width(f_w), output_channel(out_ch_){
    
    output_height = input_height - filter_height + 1;
    output_width = input_width - filter_width + 1;

    int filter_size = filter_height * filter_width * input_channel * output_channel; //size of filter frame size, times the input channel (each input channel has to have it's own filter layer) and output channel (number of different filter for this layer)

    filters.resize(filter_size);
    bias.resize(output_channel, 0.0f);
    weighted_sum.resize(output_height * output_width * output_channel, 0.0f);
    output.resize(output_height * output_width * output_channel, 0.0f);

    // Xavier
    float limit = std::sqrt(6.0f / (filter_height * filter_width * input_channel + output_channel));
    std::uniform_real_distribution<float> dist(-limit, limit);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    for (int i = 0; i < filters.size(); ++i) { 
        filters[i] = dist(generator); // Use the filters vector
    }
}

bool ConvLayer::set_filter(std::vector<float> input_filter, int input_filter_height, int input_filter_width, int channel_num, int filter_index){
    if(input_filter_height != filter_height || input_filter_width != filter_width || channel_num != input_channel){
        std::cerr << "Error: Filter dimensions (H, W, In_C) mismatch layer's dimensions." << std::endl;
        return false;
    }
    else if(filter_index < 0 || filter_index >= output_channel){
        std::cerr << "Error: Filter index (filter_index=" << filter_index << ") is out of bounds." << std::endl;
        return false;
    }
    
    int expected_block_size = filter_height * filter_width * input_channel;
    if (input_filter.size() != expected_block_size) {
        std::cerr << "Error: Input filter vector size mismatch. Expected: " << expected_block_size 
                  << ", Got: " << input_filter.size() << std::endl;
        return false;
    }

    int start_index = filter_index * expected_block_size;
    for (size_t i = 0; i < input_filter.size(); ++i) {
        filters[start_index + i] = input_filter[i];
    }

    return true;
}
bool ConvLayer::set_all_filter(std::vector<float> input_all_filter){
    int expected_size = filter_height * filter_width * input_channel * output_channel;
    if (input_all_filter.size() != expected_size) {
        std::cerr << "Error: Input vector size mismatch." << std::endl;
        return false;
    }
    filters = input_all_filter; 
    return true;
}
bool ConvLayer::add_filter(std::vector<float> input_filter, int input_filter_height, int input_filter_width, int channel_num){
    if (input_filter_height != filter_height || input_filter_width != filter_width || channel_num != input_channel) {
        std::cerr << "Error: Filter dimensions (H, W, In_C) mismatch layer's dimensions. Cannot add filter." << std::endl;
        return false;
    }

    int expected_block_size = filter_height * filter_width * input_channel;
    if (input_filter.size() != expected_block_size) {
        std::cerr << "Error: Input filter vector size mismatch. Expected: " << expected_block_size 
                  << ", Got: " << input_filter.size() << std::endl;
        return false;
    }

    output_channel++;
    filters.insert(filters.end(), input_filter.begin(), input_filter.end());
    bias.resize(output_channel, 0.0f);
    int new_output_size = output_height * output_width * output_channel;
    weighted_sum.resize(new_output_size, 0.0f);
    output.resize(new_output_size, 0.0f);

    return true;
}

bool ConvLayer::forward_propagation(const std::vector<float>& input){
    if (input.size() != input_channel * input_height * input_width) {
        std::cerr << "Error in ConvLayer::forward_propagation: Input size mismatch." << std::endl;
        return false; 
    }
    input_cache = input;
    for(int oc = 0; oc < output_channel; oc++){ //for every output of each kernel filter
        for(int oy = 0; oy < output_height; oy++){
            for(int ox = 0; ox < output_width; ox++){ //compute every pixel of the output picture array

                float sum = bias[oc]; //init with assigned bias for that kernel filter

                for(int ic = 0; ic < input_channel; ic++){ //loop through each input channel
                    for(int ky = 0; ky < filter_height; ky++){
                        for(int kx = 0; kx < filter_width; kx++){ //and each pixel of the current kernel filter

                            //calculate input coordinate from current output and kernel coordinate
                            int iy = oy + ky;
                            int ix = ox + kx;

                            //get pixel value from that coordinate
                            float pixel = input[(ic * input_height + iy) * input_width + ix];

                            //calculate the coresponding filter weight index based on which filter is being used (oc), which filter channel is being used (ic), and pixel coordinate within that filter
                            int w_idx =
                                (((oc * input_channel + ic) * filter_height + ky) * filter_width) + kx;

                            //compute the dot product
                            sum += pixel * filters[w_idx];
                        }
                    }
                }

                //assign the final output value for that output pixel
                weighted_sum[(oc * output_height + oy) * output_width + ox] = sum;
                output[(oc * output_height + oy) * output_width + ox] = apply_activation_function(sum);
            }
        }
    }
    return true;
}

std::vector<float> ConvLayer::backward_propagation(const std::vector<float>& dA, float learning_rate){
    if (dA.size() != output_channel * output_height * output_width) {
        std::cerr << "Error in ConvLayer::backward_propagation: Incoming gradient (dA) size mismatch." << std::endl;
        return std::vector<float>{}; 
    }

    std::vector<float> dZ(weighted_sum.size());
    switch (act_type){
    case ActivationType::RELU:
        for(int j = 0; j < weighted_sum.size(); j++){
            dZ[j] = (weighted_sum[j] > 0 ? 1.0f : 0.0f) * dA[j]; //dZ = dA * ReLU'(weight)
        }
        break;
    
    case ActivationType::SIGMOID:
        for(int j = 0; j < output.size(); j++){
            float A = output[j];
            dZ[j] = A * (1.0f - A) * dA[j]; //dZ = dA * sigmoid'(weight)
        }
        break;

    case ActivationType::NONE:
        for(int j = 0; j < weighted_sum.size(); j++){
            dZ[j] = dA[j]; //dZ = dA * 1
        }
        break;
    default:
        break;
    }

    //calculate dA_prev
    std::vector<float> dA_prev(input_channel * input_height * input_width, 0.0f);

    for(int oc = 0; oc < output_channel; oc++){
        for(int oy = 0; oy < output_height; oy++){
            for(int ox = 0; ox < output_width; ox++){

                int output_idx = (oc * output_height + oy) * output_width + ox;

                for(int ic = 0; ic < input_channel; ic++){
                    for(int ky = 0; ky < filter_height; ky++){
                        for(int kx = 0; kx < filter_width; kx++){
                            
                            //calculate input coordinate just like forward prop
                            int iy = oy + ky;
                            int ix = ox + kx;
                            int input_idx = (ic * input_height + iy) * input_width + ix;
                            
                             //calculate the coresponding filter weight index again like forward prop
                            int w_idx = (((oc * input_channel + ic) * filter_height + ky) * filter_width) + kx;

                            //just like above dA/dA_Prev is derivative of dA/dZ times dZ/dA_prev, which is dZ * weight
                            dA_prev[input_idx] += dZ[output_idx] * filters[w_idx];
                        }
                    }
                }

            }
        }
    }

    std::vector<float> dFilters(filters.size(), 0.0f);
    std::vector<float> dBias(output_channel, 0.0f);

    //update weight and bias
    for(int oc = 0; oc < output_channel; oc++){
        for(int oy = 0; oy < output_height; oy++){
            for(int ox = 0; ox < output_width; ox++){
                
                //Accumulate bias gradient: dl/db = sum(dl/dz * dz/db) = sum(dz) * 1
                float d_val = dZ[(oc * output_height + oy) * output_width + ox];
                dBias[oc] += d_val; 

                for(int ic = 0; ic < input_channel; ic++){
                    for(int ky = 0; ky < filter_height; ky++){
                        for(int kx = 0; kx < filter_width; kx++){

                            // Reconstruct indices
                            int iy = oy + ky;
                            int ix = ox + kx;
                            float input_pixel = input_cache[(ic * input_height + iy) * input_width + ix];
                            
                            int w_idx = (((oc * input_channel + ic) * filter_height + ky) * filter_width) + kx;

                            // Calculate dW (Gradient w.r.t filter weight)
                            // dL/dW = dL/dZ * dZ/dW = dZ * Input
                            dFilters[w_idx] += d_val * input_pixel;

                        }
                    }
                }
            }
        }
    }

    // 2. Apply Updates
    for(size_t i = 0; i < filters.size(); i++) {
        filters[i] -= learning_rate * dFilters[i];
    }
    for(size_t i = 0; i < bias.size(); i++) {
        bias[i] -= learning_rate * dBias[i];
    }

    return dA_prev;
}

//========================================================
//================== Pool Layer Class ====================
//========================================================

PoolLayer::PoolLayer(int in_h_, int in_w_, int in_ch_,
                     int ph, int pw)
    : input_height(in_h_), input_width(in_w_), input_channel(in_ch_),
      pool_height(ph), pool_width(pw)
{
    output_height = input_height / pool_height;
    output_width = input_width / pool_width;

    output.resize(output_height * output_width * input_channel);
}

bool PoolLayer::forward_propagation(const std::vector<float>& input){
    max_indices_cache.resize(output_height * output_width * input_channel);
    for (int c = 0; c < input_channel; c++) {
        for (int oy = 0; oy < output_height; oy++) {
            for (int ox = 0; ox < output_width; ox++) {

                float max_val = -1e9;
                int max_input_index = -1;

                for (int ky = 0; ky < pool_height; ky++) {
                    for (int kx = 0; kx < pool_width; kx++) {

                        int iy = oy * pool_height + ky;
                        int ix = ox * pool_width + kx;

                        float v = input[(c * input_height + iy) * input_width + ix];
                        if(v > max_val){
                            max_val = v;
                            max_input_index = (c * input_height + iy) * input_width + ix;; //store the index of the max
                        }
                    }
                }

                output[(c * output_height + oy) * output_width + ox] = max_val;
                max_indices_cache[(c * output_height + oy) * output_width + ox] = max_input_index;
            }
        }
    }
    return true;
}

std::vector<float> PoolLayer::backward_propagation(const std::vector<float>& dA, float learning_rate){
    std::vector<float> dA_prev(input_channel * input_height * input_width, 0.0f); 
    int output_size = output_height * output_width * input_channel;

    for(int i = 0; i < output_size; ++i){
        float incoming_gradient = dA[i];
        int input_index_to_route = max_indices_cache[i];
        dA_prev[input_index_to_route] += incoming_gradient;
    }
    
    // Return the gradient for the previous layer
    return dA_prev;
}

//========================================================
//==================== Model Class =======================
//========================================================

std::vector<float> Model::softmax(const std::vector<float>& logits) {
    std::vector<float> result(logits.size());
    float max_val = -1e9;
    
    // Find max to prevent overflow during exp()
    for (float v : logits) if (v > max_val) max_val = v;

    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); i++) {
        result[i] = std::exp(logits[i] - max_val);
        sum += result[i];
    }
    for (size_t i = 0; i < result.size(); i++) {
        result[i] /= sum;
    }
    return result;
}

int Model::argmax(const std::vector<float>& vec) {
    int max_idx = 0;
    float max_val = vec[0];
    for (size_t i = 1; i < vec.size(); i++) {
        if (vec[i] > max_val) {
            max_val = vec[i];
            max_idx = i;
        }
    }
    return max_idx;
}

Model::Model() {}

Model::~Model() {
    for (Layer* layer : layers) {
        delete layer;
    }
    layers.clear();
}

void Model::add(Layer* layer) {
    layers.push_back(layer);
}

std::vector<float> Model::predict(const std::vector<float>& input) {
    std::vector<float> current_data = input;

    for (Layer* layer : layers) {
        if (!layer->forward_propagation(current_data)) {
            std::cerr << "Error: Forward propagation failed." << std::endl;
            return {};
        }
        current_data = layer->get_output();
    }

    return softmax(current_data);
}

void Model::fit(const std::vector<std::vector<float>>& x_train, 
                const std::vector<std::vector<float>>& y_train, 
                int epochs, float learning_rate) {
    
    if (x_train.size() != y_train.size()) {
        std::cerr << "Error: X and Y training data size mismatch." << std::endl;
        return;
    }

    std::cout << "Starting training on " << x_train.size() << " samples..." << std::endl;
    const size_t LOG_FREQUENCY = 100;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        int correct = 0;

        for (size_t i = 0; i < x_train.size(); i++) {
            // --- 1. Forward Pass ---
            std::vector<float> current_data = x_train[i];
            for (Layer* layer : layers) {
                layer->forward_propagation(current_data);
                current_data = layer->get_output();
            }

            // Apply Softmax
            std::vector<float> probs = softmax(current_data);
            const std::vector<float>& target = y_train[i];

            // --- 2. Calculate Loss & Accuracy ---
            float sample_loss = 0.0f;
            for (size_t j = 0; j < probs.size(); j++) {
                if (target[j] > 0.5f) { 
                    sample_loss -= std::log(probs[j] + 1e-9f);
                }
            }
            total_loss += sample_loss;

            if (argmax(probs) == argmax(target)) {
                correct++;
            }

            // --- 3. Backward Pass ---
            std::vector<float> d_error(probs.size());
            for (size_t j = 0; j < probs.size(); j++) {
                d_error[j] = probs[j] - target[j];
            }

            std::vector<float> current_gradient = d_error;
            for (int k = layers.size() - 1; k >= 0; k--) {
                current_gradient = layers[k]->backward_propagation(current_gradient, learning_rate);
                if (current_gradient.empty()) {
                    std::cerr << "Backprop failed at layer " << k << std::endl;
                    return;
                }
            }
            
            // =======================================================
            // === NEW: INTERMEDIATE LOGGING BLOCK ===================
            // =======================================================
            if ((i + 1) % LOG_FREQUENCY == 0) {
                // Calculate average metrics over the last LOG_FREQUENCY samples
                float batch_accuracy = (float)correct / (i + 1);
                float batch_loss = total_loss / (i + 1);
                
                // Use carriage return '\r' to update the line in the terminal
                std::cout << "\rEpoch " << epoch + 1 << " | Sample " << (i + 1) 
                          << " / " << x_train.size()
                          << " | Avg Loss: " << std::fixed << std::setprecision(4) << batch_loss 
                          << " | Avg Acc: " << batch_accuracy * 100.0f << "%"
                          << std::flush;
            }
            // =======================================================
        } // End of training set loop

        // Print Epoch Stats
        float acc = (float)correct / x_train.size();
        float avg_loss = total_loss / x_train.size();
        std::cout << std::endl << "Epoch " << epoch + 1 << "/" << epochs 
                  << " | Loss: " << avg_loss 
                  << " | Accuracy: " << acc * 100.0f << "%" << std::endl;
    }
}

float Model::evaluate(const std::vector<std::vector<float>>& x_test, 
                      const std::vector<std::vector<float>>& y_test) {
    int correct = 0;
    for (size_t i = 0; i < x_test.size(); i++) {
        std::vector<float> pred = predict(x_test[i]);
        if (argmax(pred) == argmax(y_test[i])) correct++;
    }
    return (float)correct / x_test.size();
}