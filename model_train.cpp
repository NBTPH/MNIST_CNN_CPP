#include "cnn_lib.h"
#include <algorithm>
#include <cmath>

#define NUM_SAMPLE_TRAIN 5000 //must be from 1 to 60000
#define NUM_SAMPLE_TEST 100 //must be from 1 to 10000
#define EPOCHS 4 
#define LEARNING_RATE 0.01f

void read_dataset(const std::string& path, std::vector<std::vector<float>>& x_out, std::vector<std::vector<float>>& y_out, size_t limit){
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file" << std::endl;
        return;
    }

    std::string line;
    x_out.clear();
    y_out.clear();
    int sample_count = 0;

    while(std::getline(file, line)){
        if(line.empty()) continue;
        if(limit > 0 && sample_count >= limit){
            break;
        }
        std::stringstream ss(line);
        std::string cell;
        
        //read label (first column)
        if(!std::getline(ss, cell, ',')) continue; //read the label cell
        try{
            int label = std::stoi(cell);
            
            //encode one hot
            std::vector<float> one_hot_label(10, 0.0f);
            if(label >= 0 && label < 10){
                one_hot_label[label] = 1.0f; //since one hot lable has only 10 elements, just label the corresponding index as 1
                y_out.push_back(one_hot_label); //put the read sample into the vector
            } 
            else{
                std::cerr << "Invalid label " << label << " skipped." << std::endl;
                continue;
            }

        } catch (const std::exception& e) {
            std::cerr << "Failed to parse label: " << e.what() << ". Skipping line." << std::endl;
            continue;
        }

        //read pixel (the rest of the collumn)
        std::vector<float> image_data;
        int pixel_count = 0;
        while (std::getline(ss, cell, ',')) {
            try {
                //read pixel value (from 0 to 255)
                float pixel_val = std::stof(cell); 
                
                //normalization: divide by 255.0f
                image_data.push_back(pixel_val / 255.0f);
                pixel_count++;
            } catch (const std::exception& e) {
                break; 
            }
        }
        
        //check if we got the expected number of pixels
        if (pixel_count == (28 * 28)) {
            x_out.push_back(image_data);
            sample_count++;
        } else {
            //remove the label if we couldn't get the full image data
            if (!y_out.empty()) y_out.pop_back(); 
            std::cerr << "Line skipped due to incomplete pixel data. Expected 784, got " << pixel_count << "." << std::endl;
        }
    }

    std::cout << "Successfully loaded " << sample_count << " samples from " << path << std::endl;
}


int main(){
    std::vector<std::vector<float>> X_train, Y_train;
    std::vector<std::vector<float>> X_test, Y_test;

    std::cout << std::endl;
    read_dataset("dataset/mnist_train.csv", X_train, Y_train, NUM_SAMPLE_TRAIN);
    read_dataset("dataset/mnist_test.csv", X_test, Y_test, NUM_SAMPLE_TEST); 

    if (X_train.empty() || X_test.empty()) {
        std::cerr << "Data loading failed" << std::endl;
        return 1;
    }

    // Build model
    Model model;

    ConvLayer* c1 = new ConvLayer(28, 28, 1, 3, 3, 8); 
    c1->set_activation_type(ActivationType::RELU);
    model.add(c1);

    PoolLayer* p1 = new PoolLayer(c1->get_output_height(), c1->get_output_width(), c1->get_output_channel(), 2, 2); 
    model.add(p1);

    const int dense_input_size = p1->get_output_height() * p1->get_output_width() * p1->get_channel_num();
    DenseLayer* d1 = new DenseLayer(256, dense_input_size); 
    d1->set_activation_type(ActivationType::RELU);
    model.add(d1);

    DenseLayer* d2 = new DenseLayer(10, 256);
    d2->set_activation_type(ActivationType::NONE); 
    model.add(d2);

    //Train model
    std::cout << "\nTraining Model...\n";
    model.fit(X_train, Y_train, EPOCHS, LEARNING_RATE);
    
    //Test model
    std::cout << "\nEvaluating Model...\n";
    float accuracy = model.evaluate(X_test, Y_test);
    std::cout << "Test Set Accuracy: " << accuracy * 100.0f << "%\n";

    //Save model to a txt file
    model.output_model();

    return 0;
}