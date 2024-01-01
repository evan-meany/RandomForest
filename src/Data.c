#include "Data.h"

DLL_EXPORT void DestroyDataset(struct Dataset* dataset)
{
   for (size_t i = 0; i < dataset->numberOfRecords; i++)
   {
      free(dataset->features[i]); // Free each features array
   }
   free(dataset->features);       // Free the array of feature pointers
   free(dataset->classification); // Free classification array
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
      double* tempFeatures = dataset->features[i];
      dataset->features[i] = dataset->features[j];
      dataset->features[j] = tempFeatures;

      // Swap classification[i] with classification[j]
      size_t tempClassification = dataset->classification[i];
      dataset->classification[i] = dataset->classification[j];
      dataset->classification[j] = tempClassification;
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
   dataset->features = malloc(rowCount * sizeof(double*));
   dataset->classification = malloc(rowCount * sizeof(size_t));
   if (!dataset->features || !dataset->classification) 
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
      dataset->features[i] = malloc(dataset->numberOfFeatures * sizeof(double));
      if (!dataset->features[i]) 
      {
         perror("Memory allocation failed");
         // Free already allocated memory
         while (i > 0) 
         {
            free(dataset->features[--i]);
         }
         free(dataset->features);
         free(dataset->classification);
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
            dataset->classification[i] = IrisPetalToClassification(token);
         }
         else
         {
            dataset->features[i][column] = strtod(token, NULL);
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
   const size_t trainingSize = 100;
   const size_t testSize = dataset->numberOfRecords - trainingSize;

   // Setup new datasets
   train->numberOfRecords = trainingSize; 
   test->numberOfRecords = testSize;
   train->numberOfFeatures = dataset->numberOfFeatures;
   test->numberOfFeatures = dataset->numberOfFeatures;
   train->features = malloc(trainingSize * sizeof(double*));
   test->features = malloc(testSize * sizeof(double*));
   train->classification = malloc(trainingSize * sizeof(size_t));
   test->classification = malloc(testSize * sizeof(size_t));

   // Copy data from full dataset to train and test datasets
   for (size_t i = 0; i < trainingSize; i++)
   {
      train->features[i] = malloc(train->numberOfFeatures * sizeof(double));
      for (size_t j = 0; j < train->numberOfFeatures; j++)
      {
         train->features[i][j] = dataset->features[i][j];
      }

      train->classification[i] = dataset->classification[i];
   }

   size_t datasetIndex = trainingSize;
   for (size_t i = 0; i < testSize; i++)
   {
      test->features[i] = malloc(test->numberOfFeatures * sizeof(double));
      for (size_t j = 0; j < test->numberOfFeatures; j++)
      {
         test->features[i][j] = dataset->features[datasetIndex][j];
      }

      test->classification[i] = dataset->classification[datasetIndex++];
   }
}

DLL_EXPORT void PrintDataset(struct Dataset* dataset)
{
   for (size_t i = 0; i < dataset->numberOfRecords; i++)
   {
      printf("\nrow %d:", i);
      for (size_t j = 0; j < dataset->numberOfFeatures; j++)
      {
         printf(" %f,", dataset->features[i][j]);
      }
      printf(" %zu", dataset->classification[i]);
   }
}
