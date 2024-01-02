#ifndef DATA_H
#define DATA_H

#include "Core.h"

DLL_EXPORT struct Observation
{
   double* features;
   size_t classification;
};

DLL_EXPORT struct Dataset
{
   struct Observation* observations;
   size_t numberOfRecords;
   size_t numberOfFeatures;
};

DLL_EXPORT void DestroyDataset(struct Dataset* dataset);
DLL_EXPORT size_t IrisPetalToClassification(const char* petal);
DLL_EXPORT int ImportIrisDataset(struct Dataset* dataset);
DLL_EXPORT void SplitDataset(struct Dataset* dataset, 
                             struct Dataset* train,
                             struct Dataset* test);
DLL_EXPORT void PrintDataset(struct Dataset* dataset);

#endif // End DATA_H
