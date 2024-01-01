#ifdef __cplusplus
extern "C" {
#endif
#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include "Core.h"

// Categorical and Regression Decision Tree Nodes
DLL_EXPORT struct CatNode
{
   int dimension;
   double threshold; 
   struct CatNode* left; // left <= threshold
   struct CatNode* right; // right > threshold
   double informationGain;

   // For leaf nodes
   double* values;
   size_t sizeOfValues;
   double modeValue;
};

// Regression Node
DLL_EXPORT struct RegNode
{
   int dimension;
   double threshold; 
   struct RegNode* left; // left <= threshold
   struct RegNode* right; // right > threshold
   double varianceReduction;

   // For leaf nodes
   double* values;
   size_t sizeOfValues;
   double averageValue;
};

#endif
#ifdef __cplusplus
}
#endif