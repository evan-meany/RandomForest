#include "RandomForest.h"

int main()
{
   srand(time(NULL));

   // Gather data
   struct ObservationPool iris;
   if (ImportIrisDataset(&iris)) { return 1; }

   // Split observations into training and test datasets
   struct Dataset train, test;
   SplitPool(&iris, &train, &test, 0.90);

   // Build forest
   struct RandomForest randomForest = BuildForest(&train, 100);

   // Test forest
   size_t totalCorrect = Predict(&randomForest, &test);
   printf("total correct: %zu / %zu", totalCorrect, test.numberOfObservations);

   // Destroy created structures
   DestroyForest(&randomForest);
   DestroyObservationPool(&iris);
   DestroyDataset(&train);
   DestroyDataset(&test);

   return 0;
}