#include "RandomForest.h"
#include <stdio.h>

__declspec(dllexport) void IncrementEntity(struct Entity* entity)
{
   entity->x++;
   entity->y += 1.0;
}

__declspec(dllexport) void PrintEntity(struct Entity* entity)
{
   printf("Entity: {x : %d}, {y : %lf}\n", entity->x, entity->y);
}
