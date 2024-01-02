#include "DecisionTree.h"

static const size_t MAX_DEPTH = 2;
static const size_t MIN_SAMPLES = 2;
#define UNIQUE_CLASSIFICATIONS 3

double InformationGain(struct Dataset* train,
                       const size_t* indices, 
                       const size_t indicesSize,
                       const double threshold,
                       const size_t featureIndex,
                       const double parentEntropy)
{
   size_t leftClasses[UNIQUE_CLASSIFICATIONS] = {0};
   size_t rightClasses[UNIQUE_CLASSIFICATIONS] = {0};
   size_t leftTotal = 0, rightTotal = 0;
   for(size_t i = 0; i < indicesSize; i++)
   {
      const size_t index = indices[i];
      if (train->observations[index].features[featureIndex] <= threshold)
      {
         leftClasses[train->observations[index].classification]++;
         leftTotal++;
      }
      else
      {
         rightClasses[train->observations[index].classification]++;
         rightTotal++;
      }
   }

   double leftWeight = (double)leftTotal / (double)indicesSize;
   double rightWeight = (double)rightTotal / (double)indicesSize;
   double leftEntropy = 0, rightEntropy = 0;
   for (size_t i = 0; i < UNIQUE_CLASSIFICATIONS; i++)
   {
      double leftClassProb = (double)leftClasses[i] / (double)indicesSize;
      double rightClassProb = (double)rightClasses[i] / (double)indicesSize;
      leftEntropy -= leftClassProb * log2(leftClassProb);
      rightEntropy -= rightClassProb * log2(rightClassProb);
   }

   return parentEntropy - (leftWeight * leftEntropy) - (rightWeight * rightEntropy);
}

// returns head of tree
struct Node* BTRecursive(struct Dataset* train,
                         const size_t* indices, 
                         const size_t indicesSize,
                         const size_t depth)
{
   // Calculate class totals
   size_t classTotals[UNIQUE_CLASSIFICATIONS] = {0};
   for(size_t i = 0; i < indicesSize; i++)
   {
      const size_t index = indices[i];
      classTotals[train->observations[index].classification]++;
   }

   // Check size and depth
   if (indicesSize < MIN_SAMPLES || depth > MAX_DEPTH)
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

      struct Node* node = malloc(sizeof(struct Node));
      node->leaf = true;
      node->modeClass = modeClass;
      node->sizeOfValues = indicesSize;
      node->left = NULL;
      node->right = NULL;
      return node;
   }

   // Calculate Parent Entropy
   double parentEntropy = 0;
   for (size_t i = 0; i < UNIQUE_CLASSIFICATIONS; i++)
   {
      double classProb = (double)classTotals[i] / (double)indicesSize;
      parentEntropy -= classProb * log2(classProb);
   }

   // Find threshold value that maximizes information gain
   size_t bestFeature = 0;
   double bestThreshold = train->observations[0].features[bestFeature];
   double bestInformationGain = 0;
   for (size_t i = 0; i < indicesSize; i++)
   {
      const size_t index = indices[i];
      for (size_t j = 0; j < train->numberOfFeatures; j++)
      {
         double threshold = train->observations[index].features[j];
         double informationGain = InformationGain(train, indices, indicesSize,
                                                  threshold, j, parentEntropy);
         if (informationGain > bestInformationGain)
         {
            bestInformationGain = informationGain;
            bestFeature = j;
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

      struct Node* node = malloc(sizeof(struct Node));
      node->leaf = true;
      node->modeClass = modeClass;
      node->sizeOfValues = indicesSize;
      node->left = NULL;
      node->right = NULL;
      return node;
   }

   // Now split data along the best threshold
   size_t leftSize = 0, rightSize = 0;
   for (size_t i = 0; i < indicesSize; i++)
   {
      const size_t index = indices[i];
      if (train->observations[index].features[bestFeature] <= bestThreshold) { leftSize++; }
      else { rightSize++; }
   }

   size_t* leftIndices = malloc(leftSize * sizeof(size_t));
   size_t* rightIndices = malloc(rightSize * sizeof(size_t));
   size_t leftIndex = 0, rightIndex = 0;
   for (size_t i = 0; i < indicesSize; i++)
   {
      const size_t index = indices[i];
      if (train->observations[index].features[bestFeature] <= bestThreshold) 
      { 
         leftIndices[leftIndex++] = index;
      }
      else { rightIndices[rightIndex++] = index; }
   }

   // Create new node and connect to parent
   struct Node* node = malloc(sizeof(struct Node));
   node->leaf = false;
   node->feature = bestFeature;
   node->threshold = bestThreshold;
   node->informationGain = bestInformationGain;
   node->left = BTRecursive(train, leftIndices, leftSize, depth + 1);
   node->right = BTRecursive(train, rightIndices, rightSize, depth + 1);
   free(leftIndices);
   free(rightIndices);
   return node;
}

// returns head of tree
DLL_EXPORT struct Node* BuildTree(struct Dataset* train)
{
   size_t* allIndices = malloc(train->numberOfRecords * sizeof(size_t));
   for (size_t i = 0; i < train->numberOfRecords; i++) { allIndices[i] = i; }
   struct Node* head  = BTRecursive(train, allIndices, train->numberOfRecords, 0);
   free(allIndices);
   return head;
}

// returns the classification prediction of an observation
DLL_EXPORT size_t PredictSingleRecursive(struct Node* head,
                                         const struct Observation* observation)
{
   if (head->leaf == true || head == NULL)
   {
      return head->modeClass;
   }

   if (observation->features[head->feature] <= head->threshold)
   {
      return PredictSingleRecursive(head->left, observation);
   }
   else { return PredictSingleRecursive(head->right, observation); }
}

// returns raw number of correct choices
size_t PredictRecursive(struct Node* head,
                        struct Dataset* test, 
                        const size_t* indices, 
                        const size_t indicesSize)
{
   // Return number of correct predictions
   if (head->leaf == true || head == NULL)
   {  
      size_t totalCorrect = 0;
      for (size_t i = 0; i < indicesSize; i++)
      {
         const size_t index = indices[i];
         if (test->observations[index].classification == head->modeClass)
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
      if (test->observations[index].features[feature] <= head->threshold) { leftSize++; }
      else { rightSize++; }
   }

   size_t* leftIndices = malloc(leftSize * sizeof(size_t));
   size_t* rightIndices = malloc(rightSize * sizeof(size_t));
   size_t leftIndex = 0, rightIndex = 0;
   for (size_t i = 0; i < indicesSize; i++)
   {
      const size_t index = indices[i];
      const size_t feature = head->feature;
      if (test->observations[index].features[feature] <= head->threshold) 
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

// returns the percentage of correct predictions
DLL_EXPORT double Predict(struct Node* head, 
                          struct Dataset* test)
{
   size_t* allIndices = malloc(test->numberOfRecords * sizeof(size_t));
   for (size_t i = 0; i < test->numberOfRecords; i++) { allIndices[i] = i; }
   size_t totalCorrect = PredictRecursive(head, test, 
                                          allIndices,
                                          test->numberOfRecords);
   free(allIndices);
   return (double)totalCorrect / (double)test->numberOfRecords;
}

DLL_EXPORT void DestroyTree(struct Node* head)
{
   if (head->leaf == true || head == NULL)
   {
      free(head);
      return;
   }
   DestroyTree(head->left);
   DestroyTree(head->right);
   free(head);
}

DLL_EXPORT void PrintTree(struct Node* head, size_t tab)
{
   for (size_t i = 0; i < tab; i++) { printf(" "); }
   if (head->leaf == true)
   {
      printf("size: %d, mode: %d\n", head->sizeOfValues, head->modeClass);
   }
   else
   {
      PrintTree(head->left, tab + 1);
      printf("feature: %d, threshold: %f, IG: %f\n", head->feature,
                                                     head->threshold,
                                                     head->informationGain);
      PrintTree(head->right, tab + 1);
   }
}
