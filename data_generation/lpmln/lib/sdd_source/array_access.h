#include "sddapi.h"


SddNode* sdd_array_element(SddNode** arr, int element) {
    return arr[element];
}

int sdd_array_int_element(int* arr, int element) {
    return arr[element];
}

SddLiteral* new_array_of_sdd_literals(int size){
   SddLiteral * arr = (SddLiteral *)malloc(sizeof(SddLiteral)*size); 
   return arr;
}

void set_element_of_sdd_lit_array(SddLiteral* a, int i, SddLiteral v){
   a[i]=v;
}
