// Function: sub_15B1000
// Address: 0x15b1000
//
unsigned __int8 *__fastcall sub_15B1000(unsigned __int8 *a1)
{
  unsigned __int8 *result; // rax

  for ( result = a1;
        (unsigned int)*result - 18 <= 1;
        result = *(unsigned __int8 **)&result[8 * (1LL - *((unsigned int *)result + 2))] )
  {
    ;
  }
  return result;
}
