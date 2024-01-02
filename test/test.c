#include "Data.h"
#include "DecisionTree.h"

int main()
{
   struct Dataset irisDataset, irisTrain, irisTest;
   if (ImportIrisDataset(&irisDataset))
   {
      perror("Import error");
      return 1;
   }

   SplitDataset(&irisDataset, &irisTrain, &irisTest);

   struct Node* head = BuildTree(&irisTrain);
   PrintTree(head, 0);

   DestroyDataset(&irisDataset);
   DestroyDataset(&irisTrain);
   DestroyDataset(&irisTest);
   return 0;
}