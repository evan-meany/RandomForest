#ifdef __cplusplus
extern "C" {
#endif
#ifndef DATA_H
#define DATA_H

#include "Core.h"

DLL_EXPORT struct Observation
{
   double* features;
   size_t classification;
};

DLL_EXPORT struct ObservationPool
{
   struct Observation* observations;
   size_t numberOfObservations;
   size_t numberOfFeatures;
};

DLL_EXPORT struct Dataset
{
   const struct Observation** observations;
   size_t numberOfObservations;
   size_t numberOfFeatures;
};

// ObservationPool functions
DLL_EXPORT void DestroyObservationPool(struct ObservationPool* pool);
DLL_EXPORT void SplitPool(const struct ObservationPool* pool, 
                          struct Dataset* train,
                          struct Dataset* test,
                          const double fractionTraining);
DLL_EXPORT void PrintObservationPool(const struct ObservationPool* pool);

// Dataset functions
DLL_EXPORT void DestroyDataset(struct Dataset* dataset);
DLL_EXPORT void PrintDataset(struct Dataset* dataset);

// Import Functions
DLL_EXPORT size_t IrisPetalToClassification(const char* petal);
DLL_EXPORT int ImportIrisDataset(struct ObservationPool* pool);

// Misc. functions
size_t* GetRandomFeatureIndices(const struct Dataset* dataset,
                                const size_t numberOfFeatures); 

#endif // End DATA_H
#ifdef __cplusplus
}
#endif