#ifndef HEADER_H
#define HEADER_H

#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>


//================== Base Layer Class ==================
enum ActivationType {RELU, SIGMOID, NONE};
class Layer {
protected:
    ActivationType act_type;
public:
    virtual ~Layer() {}

    void set_activation_type(ActivationType input_type);
    float apply_activation_function(float input);
    virtual bool forward_propagation(const std::vector<float>& input) = 0;
    virtual std::vector<float> backward_propagation(const std::vector<float>& dA, float lr) = 0;

    virtual const std::vector<float>& get_output() const = 0;
};

class DenseLayer : public Layer {
private:
    int num_node;
    int num_node_prev;

    std::vector<float> output;
    std::vector<float> bias;
    std::vector<float> weight_matrix;
    std::vector<float> weighted_sum;
    std::vector<float> input_cache;

public:
    DenseLayer(int num_node_curr, int num_node_prev_in);

    bool forward_propagation(const std::vector<float>& input) override;
    std::vector<float> backward_propagation(const std::vector<float>& dA, float lr) override;

    const std::vector<float>& get_output() const override {
        return output;
    }
};

//================== Convolution Layer Class ==================

class ConvLayer : public Layer {
private:
    int input_height, input_width, input_channel;
    int output_height, output_width, output_channel;
    int filter_height, filter_width;

    std::vector<float> input_cache;
    std::vector<float> filters;    // flattened 4D tensor
    std::vector<float> bias;
    std::vector<float> weighted_sum;
    std::vector<float> output;     // flattened 3D tensor

public:
    ConvLayer(int in_h, int in_w, int in_ch,
              int f_h, int f_w, int out_ch);

    bool set_filter(std::vector<float> input_filter, int input_filter_height, int input_filter_width, int channel_num, int filter_index);
    bool set_all_filter(std::vector<float> input_all_filter);
    bool add_filter(std::vector<float> input_filter, int input_filter_height, int input_filter_width, int channel_num);
    bool forward_propagation(const std::vector<float>& input) override;
    std::vector<float> backward_propagation(const std::vector<float>& dA, float lr) override;

    int get_output_height(){
        return output_height;
    }
    int get_output_width(){
        return output_width;
    }
    int get_output_channel(){
        return output_channel;
    }

    const std::vector<float>& get_output() const override {
        return output;
    }
};

//================== Pooling Layer Class ==================

class PoolLayer : public Layer {
private:
    int input_height, input_width, input_channel;
    int pool_height, pool_width;
    int output_height, output_width;

    std::vector<int> max_indices_cache; //stores the 1D index of the max element in the input
    std::vector<float> output;

public:
    PoolLayer(int in_h, int in_w, int in_ch, int ph, int pw);

    bool forward_propagation(const std::vector<float>& input) override;
    std::vector<float> backward_propagation(const std::vector<float>& dA, float lr) override;

    int get_output_height(){
        return output_height;
    }
    int get_output_width(){
        return output_width;
    }
    int get_channel_num(){
        return input_channel;
    }

    const std::vector<float>& get_output() const override {
        return output;
    }
};

//================== Model Class ==================

class Model {
private:
    std::vector<Layer* > layers;
    std::vector<float> softmax(const std::vector<float>& logits);
    int argmax(const std::vector<float>& vec);
public:
    Model();
    ~Model();

    void add(Layer* layer);
    std::vector<float> predict(const std::vector<float>& input);
    void fit(const std::vector<std::vector<float>>& x_train, const std::vector<std::vector<float>>& y_train, int epochs, float learning_rate);
    float evaluate(const std::vector<std::vector<float>>& x_test, const std::vector<std::vector<float>>& y_test);
};


#endif