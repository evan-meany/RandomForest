#include "RandomForest.h"

int main()
{
   srand(time(NULL));

   // Gather data
   struct ObservationPool iris;
   if (ImportIrisDataset(&iris)) { return 1; }

   // Split observations into training and test datasets
   struct Dataset train, test;
   SplitPool(&iris, &train, &test, 0.95);

   // Build forest
   struct RandomForest randomForest = BuildForest(&train, 4);

   // Test forest
   // PrintObservationPool(&iris);
   // PrintDataset(&train);
   // PrintDataset(&test);

   // Destroy created structures
   DestroyForest(&randomForest);
   DestroyObservationPool(&iris);
   DestroyDataset(&train);
   DestroyDataset(&test);

   return 0;
}