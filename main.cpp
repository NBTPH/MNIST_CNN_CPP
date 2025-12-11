#include "cnn_lib.h"
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>

// Define constants for clarity
// MNIST images are 28x28 = 784 pixels
const int PIXEL_COUNT = 784;
const int NUM_CLASSES = 10; 

/**
 * Reads data from a CSV file, parses the labels and pixel data, 
 * normalizes the pixels, and converts the labels to one-hot encoding.
 * * The CSV format is: Label, Pixel1, Pixel2, ..., Pixel784
 */
void read_dataset(const std::string& path, 
                  std::vector<std::vector<float>>& x_out, 
                  std::vector<std::vector<float>>& y_out,
                    size_t limit) {

    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file at " << path << std::endl;
        return;
    }

    std::string line;
    // Skip the header row if your CSV has one (e.g., "label,pixel1,pixel2,...")
    // If your CSV starts directly with data, comment out the line below:
    // std::getline(file, line); 

    x_out.clear();
    y_out.clear();
    int sample_count = 0;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (limit > 0 && sample_count >= limit) {
            std::cout << "Reached sample limit of " << limit << "." << std::endl;
            break; // Exit the while loop
        }
        std::stringstream ss(line);
        std::string cell;
        
        // --- 1. Get Label (First column) ---
        if (!std::getline(ss, cell, ',')) continue; // Read the label cell
        try {
            int label = std::stoi(cell);
            
            // One-Hot Encoding
            std::vector<float> one_hot_label(NUM_CLASSES, 0.0f);
            if (label >= 0 && label < NUM_CLASSES) {
                one_hot_label[label] = 1.0f;
                y_out.push_back(one_hot_label);
            } else {
                std::cerr << "Warning: Invalid label " << label << " skipped." << std::endl;
                continue;
            }

        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to parse label: " << e.what() << ". Skipping line." << std::endl;
            continue;
        }

        // --- 2. Get Pixels (Remaining columns) ---
        std::vector<float> image_data;
        int pixel_count = 0;
        while (std::getline(ss, cell, ',')) {
            try {
                // Read pixel value (0-255)
                float pixel_val = std::stof(cell); 
                
                // Normalization: Divide by 255.0f
                image_data.push_back(pixel_val / 255.0f);
                pixel_count++;
            } catch (const std::exception& e) {
                // Handle incomplete line or bad data mid-line
                break; 
            }
        }
        
        // Check if we got the expected number of pixels
        if (pixel_count == PIXEL_COUNT) {
            x_out.push_back(image_data);
            sample_count++;
        } else {
            // Remove the label if we couldn't get the full image data
            if (!y_out.empty()) y_out.pop_back(); 
            std::cerr << "Warning: Line skipped due to incomplete pixel data. Expected 784, got " << pixel_count << "." << std::endl;
        }
    }

    std::cout << "Successfully loaded " << sample_count << " samples from " << path << std::endl;
}


int main(){
    // --- Data Variables ---
    std::vector<std::vector<float>> X_train, Y_train;
    std::vector<std::vector<float>> X_test, Y_test;

    // --- 1. Load Data ---
    // NOTE: Change these file paths to match your actual files!
    read_dataset("mnist_train.csv", X_train, Y_train, 2000);
    read_dataset("mnist_test.csv", X_test, Y_test, 20); 

    if (X_train.empty() || X_test.empty()) {
        std::cerr << "Fatal Error: Data loading failed. Cannot proceed." << std::endl;
        return 1;
    }

    // --- 2. Build Model (LeNet-5 inspired architecture) ---
    Model model;

    // Input image is 28x28x1

    // Layer 1: Conv -> 8 filters, 3x3 kernel
    // Output size: (28-3+1) = 26x26x8
    ConvLayer* c1 = new ConvLayer(28, 28, 1, 3, 3, 8); 
    c1->set_activation_type(ActivationType::RELU);
    model.add(c1);

    // Layer 2: Pool -> 2x2 max pooling
    // Output size: 13x13x8
    PoolLayer* p1 = new PoolLayer(26, 26, 8, 2, 2); 
    model.add(p1);

    // Layer 3: Dense Hidden Layer (Flattening)
    // Input Size: 13 * 13 * 8 = 1352
    const int DENSE_INPUT_SIZE = 13 * 13 * 8;
    const int DENSE_NODES = 128; // Hidden nodes
    DenseLayer* d1 = new DenseLayer(DENSE_NODES, DENSE_INPUT_SIZE); 
    d1->set_activation_type(ActivationType::RELU);
    model.add(d1);

    // Layer 4: Output Layer
    // Input Size: 128, Output Size: 10 (classes)
    DenseLayer* d2 = new DenseLayer(NUM_CLASSES, DENSE_NODES);
    // CRITICAL: Must be NONE, Model::fit applies Softmax
    d2->set_activation_type(ActivationType::NONE); 
    model.add(d2);

    // --- 3. Train Model ---
    const int EPOCHS = 40;
    const float LEARNING_RATE = 0.05f;

    std::cout << "\nTraining Model...\n";
    model.fit(X_train, Y_train, EPOCHS, LEARNING_RATE);
    
    // --- 4. Evaluate Model ---
    std::cout << "\nEvaluating Model...\n";
    float accuracy = model.evaluate(X_test, Y_test);
    
    std::cout << "Test Set Accuracy: " << accuracy * 100.0f << "%\n";

    return 0;
}