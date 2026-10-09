#include <torch/torch.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <map>
#include <algorithm>
#include <iostream>
#include <regex>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace fs = std::filesystem;

/**
 * Converts downloaded color JPEG images to PyTorch tensors using stb_image.h.
 * Creates:
 * - arclas/tensors/x_{i}.tensor (image tensors as [3, 256, 256])
 * - arclas/labels (artist IDs, one per line)
 * - arclas/artist_mapping (label_id -> artist_name mapping)
 */

int main() {
    const std::string images_dir = "arclas/images";
    const std::string output_dir = "arclas/tensors";
    const std::string labels_file = "arclas/labels";
    const std::string mapping_file = "arclas/artist_mapping";
    
    fs::create_directories(output_dir);
    
    // Step 1: Extract artist names from filenames and create mapping
    std::map<std::string, int> artist_to_id;
    std::vector<std::pair<std::string, std::string>> image_files;  // {filename, artist}
    int next_artist_id = 0;
    
    std::regex filename_regex(R"(^([a-zA-Z0-9_]+)_\d{3}\.jpg$)");
    
    // Collect all JPEG files
    std::vector<fs::path> all_files;
    for (const auto& entry : fs::directory_iterator(images_dir)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            if (filename.find(".jpg") != std::string::npos || 
                filename.find(".jpeg") != std::string::npos) {
                all_files.push_back(entry.path());
            }
        }
    }
    
    // Sort for consistent ordering
    std::sort(all_files.begin(), all_files.end());
    
    for (const auto& filepath : all_files) {
        std::string filename = filepath.filename().string();
        std::smatch match;
        
        if (std::regex_match(filename, match, filename_regex)) {
            std::string artist = match[1].str();
            
            // Assign ID if new artist
            if (artist_to_id.find(artist) == artist_to_id.end()) {
                artist_to_id[artist] = next_artist_id++;
            }
            
            image_files.push_back({filepath.string(), artist});
        }
    }
    
    std::cout << "Found " << artist_to_id.size() << " unique artists" << std::endl;
    for (const auto& [artist, id] : artist_to_id) {
        std::cout << "  Artist " << id << ": " << artist << std::endl;
    }
    
    // Step 2: Convert images to tensors
    std::ofstream labels_out(labels_file);
    if (!labels_out.is_open()) {
        std::cerr << "Error: Could not open " << labels_file << std::endl;
        return 1;
    }
    
    int tensor_count = 0;
    
    for (const auto& [img_path, artist] : image_files) {
        try {
            int width, height, channels;
            
            // Load image as RGB (force 3 channels)
            unsigned char* img_data = stbi_load(img_path.c_str(), &width, &height, &channels, 3);
            
            if (!img_data) {
                std::cerr << "Failed to read: " << img_path << " (" 
                          << stbi_failure_reason() << ")" << std::endl;
                continue;
            }
            
            // Verify dimensions
            if (width != 256 || height != 256) {
                std::cerr << "Warning: " << img_path << " is not 256x256 (got " 
                          << width << "x" << height << ")" << std::endl;
            }
            
            // Create tensor from image data (stb_image returns RGB, which is what we want)
            // Data layout: [H, W, C] = [256, 256, 3]
            torch::Tensor tensor = torch::from_blob(
                img_data,
                {height, width, 3},
                torch::kUInt8
            ).clone();  // Clone to own the data (before stbi_image_free)
            
            // Free stb_image memory
            stbi_image_free(img_data);
            
            // Transpose to [C, H, W] format
            tensor = tensor.permute({2, 0, 1});
            
            // Save tensor
            std::string tensor_path = (fs::path(output_dir) / 
                                      std::format("x_{}.tensor", tensor_count)).string();
            torch::save(tensor, tensor_path);
            
            // Write label (artist ID)
            int artist_id = artist_to_id[artist];
            labels_out << artist_id << "\n";
            
            std::cout << "Tensor " << tensor_count << ": " << img_path 
                      << " -> artist_id=" << artist_id << " (shape: " 
                      << tensor.sizes() << ")" << std::endl;
            
            tensor_count++;
            
        } catch (const std::exception& e) {
            std::cerr << "Error processing " << img_path << ": " << e.what() << std::endl;
        }
    }
    
    labels_out.close();
    
    // Step 3: Write artist mapping file
    std::ofstream mapping_out(mapping_file);
    if (!mapping_out.is_open()) {
        std::cerr << "Error: Could not open " << mapping_file << std::endl;
        return 1;
    }
    
    for (const auto& [artist, id] : artist_to_id) {
        mapping_out << id << " " << artist << "\n";
    }
    
    mapping_out.close();
    
    std::cout << "\n=== Conversion Complete ===" << std::endl;
    std::cout << "Tensors created: " << tensor_count << std::endl;
    std::cout << "Unique artists: " << artist_to_id.size() << std::endl;
    std::cout << "Tensors saved to: " << output_dir << std::endl;
    std::cout << "Labels saved to: " << labels_file << std::endl;
    std::cout << "Artist mapping saved to: " << mapping_file << std::endl;
    
    return 0;
}
