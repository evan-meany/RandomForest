// #include "DecisionTree.h"
#include "Data.h"

int main()
{
   struct Dataset irisDataset, irisTrain, irisTest;
   if (ImportIrisDataset(&irisDataset))
   {
      // Issue with import
      perror("Import error");
      return 1;
   }

   SplitDataset(&irisDataset, &irisTrain, &irisTest);
   PrintDataset(&irisTrain);
   DestroyDataset(&irisDataset);
   DestroyDataset(&irisTrain);
   DestroyDataset(&irisTest);
   return 0;
}