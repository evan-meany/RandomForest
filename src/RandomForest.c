#include "RandomForest.h"
#include "DecisionTree.h"

static const size_t NUMBER_OF_FEATURES = 2;

DLL_EXPORT struct RandomForest BuildForest(const struct Dataset* train, 
                                           const size_t numberOfTrees)
{
   // Setup RandomForest structure
   struct RandomForest randomForest;
   randomForest.trees = malloc(numberOfTrees * sizeof(struct DecisionTree));
   randomForest.numberOfTrees = numberOfTrees;

   // Build each decision tree
   for (size_t i = 0; i < numberOfTrees; i++)
   {
      randomForest.trees[i] = BuildTree(train, NUMBER_OF_FEATURES);
      PrintTree(&randomForest.trees[i]);
   }

   return randomForest;
}

DLL_EXPORT void DestroyForest(struct RandomForest* randomForest)
{
   for (size_t i = 0; i < randomForest->numberOfTrees; i++)
   {
      DestroyTree(&randomForest->trees[i]);
   }
   free(randomForest->trees);
}