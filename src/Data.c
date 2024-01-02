#include "Data.h"

DLL_EXPORT void DestroyDataset(struct Dataset* dataset)
{
   free(dataset->observations);
}

DLL_EXPORT size_t IrisPetalToClassification(const char* petal)
{
   if (strcmp(petal, "Iris-setosa\n") == 0) { return 0; }
   if (strcmp(petal, "Iris-versicolor\n") == 0) { return 1; }
   if (strcmp(petal, "Iris-virginica\n") == 0) { return 2; }
   return 0;
}

// Fisher-Yates (or Knuth) shuffle algorithm
void ShuffleDataset(struct Dataset* dataset) 
{
   if (dataset == NULL || dataset->numberOfRecords <= 1) { return; }

   // Initialize random number generator
   srand((unsigned int)time(NULL));

   for (size_t i = dataset->numberOfRecords - 1; i > 0; i--) 
   {
      // Generate a random index between 0 and i
      size_t j = rand() % (i + 1);

      // Swap features[i] with features[j]
      struct Observation tempObservation = dataset->observations[i];
      dataset->observations[i] = dataset->observations[j];
      dataset->observations[j] = tempObservation; 
   }
}

// returns 0 on success
DLL_EXPORT int ImportIrisDataset(struct Dataset* dataset)
{
   const char* filename = "data/iris-data.csv";
   const int numberOfColumns = 5;
   const int classColumn = 4;

   FILE *file = fopen(filename, "r");
   if (file == NULL) 
   { 
      perror("Unable to open file"); 
      return 1; 
   }

   char line[1024];
   int rowCount = -1;
   while (fgets(line, sizeof(line), file)) { rowCount++; }
   rewind(file); 
   dataset->observations = malloc(rowCount * sizeof(struct Observation));
   if (!dataset->observations) 
   {
      perror("Memory allocation failed");
      fclose(file);
      return 1;
   }
   dataset->numberOfRecords = rowCount;
   dataset->numberOfFeatures = numberOfColumns - 1;

   size_t i = 0;
   char *token;
   fgets(line, sizeof(line), file);
   while (fgets(line, sizeof(line), file))
   {
      dataset->observations[i].features = malloc(dataset->numberOfFeatures * sizeof(double));
      if (!dataset->observations[i].features) 
      {
         perror("Memory allocation failed");
         DestroyDataset(dataset);
         fclose(file);
         return 1;
      }
      
      token = strtok(line, ",");
      for (size_t column = 0; column < numberOfColumns; column++)
      {
         if (!token) 
         {
            perror("Error parsing file");
            DestroyDataset(dataset);
            fclose(file);
            return 1;
         }

         if (column == classColumn)
         {
            dataset->observations[i].classification = IrisPetalToClassification(token);
         }
         else
         {
            dataset->observations[i].features[column] = strtod(token, NULL);
         }
         token = strtok(NULL, ","); // Get next token
      }
      i++;
   }

   // Close file
   fclose(file);

   // Randomize dataset
   ShuffleDataset(dataset);

   return 0;
}

// Splits dataset into training data and testing data
DLL_EXPORT void SplitDataset(struct Dataset* dataset, 
                             struct Dataset* train,
                             struct Dataset* test)
{
   const size_t trainingSize = 130;
   const size_t testSize = dataset->numberOfRecords - trainingSize;

   // Setup new datasets
   train->numberOfRecords = trainingSize; 
   test->numberOfRecords = testSize;
   train->numberOfFeatures = dataset->numberOfFeatures;
   test->numberOfFeatures = dataset->numberOfFeatures;
   train->observations = malloc(trainingSize * sizeof(struct Observation));
   test->observations = malloc(testSize * sizeof(struct Observation));

   // Copy data from full dataset to train and test datasets
   for (size_t i = 0; i < trainingSize; i++)
   {
      train->observations[i].features = malloc(train->numberOfFeatures * sizeof(double));
      for (size_t j = 0; j < train->numberOfFeatures; j++)
      {
         train->observations[i].features[j] = dataset->observations[i].features[j];
      }

      train->observations[i].classification = dataset->observations[i].classification;
   }

   size_t datasetIndex = trainingSize;
   for (size_t i = 0; i < testSize; i++)
   {
      test->observations[i].features = malloc(test->numberOfFeatures * sizeof(double));
      for (size_t j = 0; j < test->numberOfFeatures; j++)
      {
         test->observations[i].features[j] = dataset->observations[datasetIndex].features[j];
      }

      test->observations[i].classification = dataset->observations[datasetIndex].classification;
   }
}

DLL_EXPORT void PrintDataset(struct Dataset* dataset)
{
   for (size_t i = 0; i < dataset->numberOfRecords; i++)
   {
      printf("\nrow %d:", i);
      for (size_t j = 0; j < dataset->numberOfFeatures; j++)
      {
         printf(" %f,", dataset->observations[i].features[j]);
      }
      printf(" %zu", dataset->observations[i].classification);
   }
}
