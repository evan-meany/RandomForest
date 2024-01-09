#include "RandomForest.h"
#include "DecisionTree.h"

static const size_t NUMBER_OF_FEATURES = 3;

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
      struct Dataset randomDataset = BuildRandomDataset(train);
      randomForest.trees[i] = BuildTree(&randomDataset, NUMBER_OF_FEATURES);
      // PrintTree(&randomForest.trees[i]);
      DestroyDataset(&randomDataset);
   }

   return randomForest;
}

DLL_EXPORT size_t Predict(const struct RandomForest* forest,
                          const struct Dataset* test)
{
   size_t* classPredicitons = malloc(forest->numberOfTrees * sizeof(size_t));

   // Predict observations on each decision tree
   size_t totalCorrect = 0;
   for (size_t i = 0; i < test->numberOfObservations; i++)
   {
      for (size_t j = 0; j < forest->numberOfTrees; j++) { classPredicitons[j] = 0; }

      for (size_t j = 0; j < forest->numberOfTrees; j++)
      {
         size_t prediction = PredictSingleRecursive(forest->trees[j].head, 
                                                    test->observations[i]);
         classPredicitons[prediction]++;
      }

      size_t highestTotal = 0, classPrediction = 0;
      for (size_t j = 0; j < forest->numberOfTrees; j++)
      {
         if (classPredicitons[j] > highestTotal) 
         {
            highestTotal = classPredicitons[j];
            classPrediction = j;
         }
      }

      if (classPrediction == test->observations[i]->classification) { totalCorrect++; }
   }

   return totalCorrect;
}

DLL_EXPORT void DestroyForest(struct RandomForest* randomForest)
{
   for (size_t i = 0; i < randomForest->numberOfTrees; i++)
   {
      DestroyTree(&randomForest->trees[i]);
   }
   free(randomForest->trees);
}