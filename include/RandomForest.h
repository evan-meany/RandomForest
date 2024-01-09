#ifdef __cplusplus
extern "C" {
#endif
#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include "DecisionTree.h"

DLL_EXPORT struct RandomForest
{
   struct DecisionTree* trees;
   size_t numberOfTrees;
};

DLL_EXPORT struct RandomForest BuildForest(const struct Dataset* train, 
                                           const size_t numberOfTrees);
DLL_EXPORT size_t Predict(const struct RandomForest* forest,
                          const struct Dataset* test);
DLL_EXPORT void DestroyForest(struct RandomForest* randomForest);

#endif
#ifdef __cplusplus
}
#endif