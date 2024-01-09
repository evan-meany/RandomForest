#include "DecisionTree.h"

static const size_t MAX_DEPTH = 2;
static const size_t MIN_SAMPLES = 2;
#define UNIQUE_CLASSIFICATIONS 3

// Returns a double representing information gain
double InformationGain(const struct Dataset* dataset,
                       const size_t* datasetIndices, 
                       const size_t indicesSize,
                       const double threshold,
                       const size_t featureIndex,
                       const double parentEntropy)
{
   // Find the class totals after a potential split at threshold
   size_t leftClasses[UNIQUE_CLASSIFICATIONS] = {0};
   size_t rightClasses[UNIQUE_CLASSIFICATIONS] = {0};
   size_t leftTotal = 0, rightTotal = 0;
   for(size_t i = 0; i < indicesSize; i++)
   {
      const size_t index = datasetIndices[i];
      if (dataset->observations[index]->features[featureIndex] <= threshold)
      {
         leftClasses[dataset->observations[index]->classification]++;
         leftTotal++;
      }
      else
      {
         rightClasses[dataset->observations[index]->classification]++;
         rightTotal++;
      }
   }

   // Calcuate entropies and weight based on fraction of observations
   double leftWeight = (double)leftTotal / (double)indicesSize;
   double rightWeight = (double)rightTotal / (double)indicesSize;
   double leftEntropy = 0, rightEntropy = 0;
   for (size_t i = 0; i < UNIQUE_CLASSIFICATIONS; i++)
   {
      double leftClassProb = (double)leftClasses[i] / (double)indicesSize;
      double rightClassProb = (double)rightClasses[i] / (double)indicesSize;
      if (leftClassProb > 0)
      {
         leftEntropy -= leftClassProb * log2(leftClassProb);
      }
      if (rightClassProb > 0)
      {
         rightEntropy -= rightClassProb * log2(rightClassProb);
      }
   }

   return parentEntropy - (leftWeight * leftEntropy) - (rightWeight * rightEntropy);
}

// Returns head of tree
struct Node* BTRecursive(const struct Dataset* dataset,
                         const size_t* datasetIndices, 
                         const size_t datasetIndicesSize,
                         const size_t* featureIndices,
                         const size_t featureIndicesSize,
                         const size_t depth)
{
   // Calculate class totals
   size_t classTotals[UNIQUE_CLASSIFICATIONS] = {0};
   for(size_t i = 0; i < datasetIndicesSize; i++)
   {
      const size_t index = datasetIndices[i];
      classTotals[dataset->observations[index]->classification]++;
   }

   // Check size and depth
   if (datasetIndicesSize < MIN_SAMPLES || depth > MAX_DEPTH)
   {
      // Get mode class
      size_t highestTotal = 0, modeClass = 0;
      for (size_t i = 0; i < UNIQUE_CLASSIFICATIONS; i++)
      {
         if (classTotals[i] > highestTotal)
         {
            highestTotal = classTotals[i];
            modeClass = i;
         }
      }

      // Create leaf node and return it
      struct Node* node = malloc(sizeof(struct Node));
      node->leaf = true;
      node->modeClass = modeClass;
      node->sizeOfValues = datasetIndicesSize;
      node->left = NULL;
      node->right = NULL;
      return node;
   }

   // Calculate Parent Entropy
   double parentEntropy = 0;
   for (size_t i = 0; i < UNIQUE_CLASSIFICATIONS; i++)
   {
      double classProb = (double)classTotals[i] / (double)datasetIndicesSize;
      if (classProb > 0)
      {
         parentEntropy -= classProb * log2(classProb);
      }
   }

   // Find threshold value that maximizes information gain
   size_t bestFeature = featureIndices[0];
   double bestThreshold = dataset->observations[0]->features[bestFeature];
   double bestInformationGain = 0;
   for (size_t i = 0; i < datasetIndicesSize; i++)
   {
      const size_t datasetIndex = datasetIndices[i];
      for (size_t j = 0; j < featureIndicesSize; j++)
      {
         size_t featureIndex = featureIndices[j];
         double threshold = dataset->observations[datasetIndex]->features[featureIndex];
         double informationGain = InformationGain(dataset, datasetIndices, datasetIndicesSize,
                                                  threshold, featureIndex, parentEntropy);
         if (informationGain > bestInformationGain)
         {
            bestInformationGain = informationGain;
            bestFeature = featureIndex;
            bestThreshold = threshold;
         }
      }
   }

   // If there is no information gain then don't split
   if (bestInformationGain <= 0)
   {
      // Get mode class
      size_t highestTotal = 0, modeClass = 0;
      for (size_t i = 0; i < UNIQUE_CLASSIFICATIONS; i++)
      {
         if (classTotals[i] > highestTotal)
         {
            highestTotal = classTotals[i];
            modeClass = i;
         }
      }

      // Create leaf node and return it
      struct Node* node = malloc(sizeof(struct Node));
      node->leaf = true;
      node->modeClass = modeClass;
      node->sizeOfValues = datasetIndicesSize;
      node->left = NULL;
      node->right = NULL;
      return node;
   }

   // Now split data along the best threshold
   size_t leftDatasetSize = 0, rightDatasetSize = 0;
   for (size_t i = 0; i < datasetIndicesSize; i++)
   {
      const size_t datasetIndex = datasetIndices[i];
      if (dataset->observations[datasetIndex]->features[bestFeature] <= bestThreshold) 
      { 
         leftDatasetSize++;
      }
      else { rightDatasetSize++; }
   }

   size_t* leftDatasetIndices = malloc(leftDatasetSize * sizeof(size_t));
   size_t* rightDatasetIndices = malloc(rightDatasetSize * sizeof(size_t));
   size_t leftDatasetIndex = 0, rightDatasetIndex = 0;
   for (size_t i = 0; i < datasetIndicesSize; i++)
   {
      const size_t datasetIndex = datasetIndices[i];
      if (dataset->observations[datasetIndex]->features[bestFeature] <= bestThreshold) 
      { 
         leftDatasetIndices[leftDatasetIndex++] = datasetIndex;
      }
      else { rightDatasetIndices[rightDatasetIndex++] = datasetIndex; }
   }

   // Create new node and connect to parent
   struct Node* node = malloc(sizeof(struct Node));
   node->leaf = false;
   node->feature = bestFeature;
   node->threshold = bestThreshold;
   node->informationGain = bestInformationGain;
   node->left = BTRecursive(dataset, leftDatasetIndices, leftDatasetSize, 
                            featureIndices, featureIndicesSize, depth + 1);
   node->right = BTRecursive(dataset, rightDatasetIndices, rightDatasetSize, 
                             featureIndices, featureIndicesSize, depth + 1);
   free(leftDatasetIndices);
   free(rightDatasetIndices);
   return node;
}

// Returns DecisionTree
DLL_EXPORT struct DecisionTree BuildTree(const struct Dataset* dataset, 
                                         const size_t numberOfFeatures)
{
   // Setup indices array
   size_t* allIndices = malloc(dataset->numberOfObservations * sizeof(size_t));
   for (size_t i = 0; i < dataset->numberOfObservations; i++) { allIndices[i] = i; }

   // Setup featureIndices
   size_t* featureIndices = GetRandomFeatureIndices(dataset, numberOfFeatures);
   // printf("Feature indices: ");
   // for (size_t i = 0; i < numberOfFeatures; i++)
   // {
   //    printf("%d, ", featureIndices[i]);
   // }
   // printf("\n");

   // Recursively create tree
   struct Node* head  = BTRecursive(dataset, allIndices, dataset->numberOfObservations, 
                                    featureIndices, numberOfFeatures, 0);

   // Remove indices array
   free(allIndices);

   // Create and return DecisionTree struct
   struct DecisionTree tree;
   tree.head = head;
   tree.featureIndices = featureIndices;
   tree.numberOfFeatures = numberOfFeatures;
   return tree;
}

// Returns the classification prediction of an observation
DLL_EXPORT size_t PredictSingleRecursive(const struct Node* head,
                                         const struct Observation* observation)
{
   // If at leaf return the leaf's classification
   if (head == NULL) { return 0; }
   if (head->leaf == true)
   {
      return head->modeClass;
   }

   // Continue recursing
   if (observation->features[head->feature] <= head->threshold)
   {
      return PredictSingleRecursive(head->left, observation);
   }
   else { return PredictSingleRecursive(head->right, observation); }
}

// Returns raw number of correct choices
size_t PredictRecursive(const struct Node* head,
                        const struct Dataset* test, 
                        const size_t* indices, 
                        const size_t indicesSize)
{
   // Return number of correct predictions
   if (head == NULL) { return 0; }
   if (head->leaf == true)
   {  
      size_t totalCorrect = 0;
      for (size_t i = 0; i < indicesSize; i++)
      {
         const size_t index = indices[i];
         if (test->observations[index]->classification == head->modeClass)
         {
            totalCorrect++;
         }
      }

      return totalCorrect;
   }

   // Split data and recurse further
   size_t leftSize = 0, rightSize = 0;
   for (size_t i = 0; i < indicesSize; i++)
   {
      const size_t index = indices[i];
      const size_t feature = head->feature;
      if (test->observations[index]->features[feature] <= head->threshold) { leftSize++; }
      else { rightSize++; }
   }

   size_t* leftIndices = malloc(leftSize * sizeof(size_t));
   size_t* rightIndices = malloc(rightSize * sizeof(size_t));
   size_t leftIndex = 0, rightIndex = 0;
   for (size_t i = 0; i < indicesSize; i++)
   {
      const size_t index = indices[i];
      const size_t feature = head->feature;
      if (test->observations[index]->features[feature] <= head->threshold) 
      { 
         leftIndices[leftIndex++] = index;
      }
      else { rightIndices[rightIndex++] = index; }
   }

   size_t totalCorrect = 0;
   totalCorrect += PredictRecursive(head->left, test, leftIndices, leftSize);
   totalCorrect += PredictRecursive(head->right, test, rightIndices, rightSize);
   free(leftIndices);
   free(rightIndices);
   return totalCorrect;
}

// Returns the percentage of correct predictions
// DLL_EXPORT double Predict(const struct Node* head, 
//                           const struct Dataset* test)
// {
//    size_t* allIndices = malloc(test->numberOfFeatures * sizeof(size_t));
//    for (size_t i = 0; i < test->numberOfFeatures; i++) { allIndices[i] = i; }
//    size_t totalCorrect = PredictRecursive(head, test, 
//                                           allIndices,
//                                           test->numberOfFeatures);
//    free(allIndices);
//    return (double)totalCorrect / (double)test->numberOfObservations;
// }

void DestroyTreeRecursive(struct Node* head)
{
   if (head == NULL) { return; }
   if (head->leaf == true) 
   {
      free(head);
      return;
   }

   DestroyTreeRecursive(head->left);
   DestroyTreeRecursive(head->right);
   free(head);
}

DLL_EXPORT void DestroyTree(struct DecisionTree* tree)
{
   DestroyTreeRecursive(tree->head);
   tree->head = NULL;

   // Free feature indices
   free(tree->featureIndices);
   tree->featureIndices = NULL;
}

void PrintTreeRecursive(const struct Node* head, const size_t tab)
{
   for (size_t i = 0; i < tab; i++) { printf("   "); }
   if (head->leaf == true)
   {
      printf("size: %d, mode: %d\n", head->sizeOfValues, head->modeClass);
   }
   else
   {
      PrintTreeRecursive(head->left, tab + 1);
      printf("feature: %d, threshold: %f, IG: %f\n", head->feature,
                                                     head->threshold,
                                                     head->informationGain);
      PrintTreeRecursive(head->right, tab + 1);
   }
}

DLL_EXPORT void PrintTree(const struct DecisionTree* tree)
{
   printf("\n===========\n");
   PrintTreeRecursive(tree->head, 0);
   printf("===========\n");
}
