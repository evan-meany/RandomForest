#ifdef __cplusplus
extern "C" {
#endif
#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include "Core.h"
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

DLL_EXPORT struct Node* BuildTree(struct Dataset* train);
DLL_EXPORT void PrintTree(struct Node* head, size_t tab);


#endif
#ifdef __cplusplus
}
#endif