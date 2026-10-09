#include "common.hpp"

// Calculate accuracy: (correct predictions) / (total predictions)
torch::Tensor accuracy(torch::Tensor predictions, torch::Tensor targets) {
    // Ensure tensors are on the same device and dtype
    predictions = predictions.to(targets.device());
    targets = targets.to(torch::kLong);
    
    // Get predicted classes (argmax for multi-class, threshold for binary)
    torch::Tensor predicted = torch::argmax(predictions, /*dim=*/ 1);
    
    // Count correct predictions
    torch::Tensor correct = torch::eq(predicted, targets).to(torch::kFloat);
    
    return torch::mean(correct);
}

// Calculate recall per class, returns tensor of shape [num_classes]
torch::Tensor recall(torch::Tensor predictions, torch::Tensor targets, int64_t num_classes) {
    predictions = predictions.to(targets.device());
    targets = targets.to(torch::kLong);
    
    torch::Tensor predicted = torch::argmax(predictions, /*dim=*/ 1);
    
    torch::Tensor recalls = torch::zeros(num_classes, targets.device());
    
    for (int64_t c = 0; c < num_classes; ++c) {
        // True positives: predicted == c AND targets == c
        torch::Tensor tp = torch::logical_and(
            torch::eq(predicted, c),
            torch::eq(targets, c)
        ).to(torch::kFloat).sum();
        
        // False negatives: predicted != c BUT targets == c
        torch::Tensor fn = torch::logical_and(
            torch::ne(predicted, c),
            torch::eq(targets, c)
        ).to(torch::kFloat).sum();
        
        // Recall = TP / (TP + FN), handle division by zero
        recalls[c] = torch::where(
            (tp + fn) > 0,
            tp / (tp + fn),
            torch::tensor(0.0, targets.device())
        );
    }
    
    return recalls;
}

// Macro recall (average across all classes)
torch::Tensor recall_macro(torch::Tensor predictions, torch::Tensor targets, int64_t num_classes) {
    return torch::mean(recall(predictions, targets, num_classes));
}

torch::Tensor f_score(torch::Tensor predictions, torch::Tensor targets, int64_t num_classes) {
    return 2*(accuracy(predictions, targets) * recall_macro(predictions, targets, num_classes))/(accuracy(predictions, targets) + recall_macro(predictions, targets, num_classes));
}
