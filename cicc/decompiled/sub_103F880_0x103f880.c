// Function: sub_103F880
// Address: 0x103f880
//
_QWORD *__fastcall sub_103F880(__int64 a1)
{
  __int64 v1; // rcx
  _QWORD *result; // rax
  _QWORD *i; // rdx

  v1 = *(unsigned int *)(a1 + 24);
  result = *(_QWORD **)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  for ( i = &result[7 * v1]; result != i; result += 7 )
  {
    if ( result )
    {
      *result = -4096;
      result[1] = -4096;
      result[2] = -3;
      result[3] = 0;
      result[4] = 0;
      result[5] = 0;
      result[6] = 0;
    }
  }
  return result;
}
