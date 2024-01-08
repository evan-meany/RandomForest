#ifdef __cplusplus
extern "C" {
#endif

#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include "Data.h"

// Categorical and Regression Decision Tree Nodes
DLL_EXPORT struct Node
{
   bool leaf;

   // For non-leaf nodes
   size_t feature;
   double threshold; 
   double informationGain;
   struct Node* left; // left <= threshold
   struct Node* right; // right > threshold

   // For leaf nodes
   size_t sizeOfValues;
   size_t modeClass;
};

DLL_EXPORT struct DecisionTree
{
   struct Node* head;
   size_t* featureIndices;
   size_t numberOfFeatures;
};

DLL_EXPORT struct DecisionTree BuildTree(const struct Dataset* train,
                                         const size_t numberOfFeatures);
DLL_EXPORT double Predict(const struct Node* head, const struct Dataset* test);
DLL_EXPORT size_t PredictSingleRecursive(const struct Node* head,
                                         const struct Observation* observation);
DLL_EXPORT void DestroyTree(struct DecisionTree* tree);
DLL_EXPORT void PrintTree(const struct DecisionTree* tree);


#endif // End DECISION_TREE_H
#ifdef __cplusplus
}
#endif