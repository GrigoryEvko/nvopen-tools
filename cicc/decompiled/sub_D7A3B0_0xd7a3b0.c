// Function: sub_D7A3B0
// Address: 0xd7a3b0
//
_QWORD *__fastcall sub_D7A3B0(__int64 a1)
{
  __int64 v1; // rdx
  _QWORD *result; // rax
  _QWORD *i; // rdx

  v1 = *(unsigned int *)(a1 + 24);
  result = *(_QWORD **)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  for ( i = &result[5 * v1]; result != i; result += 5 )
  {
    if ( result )
    {
      *result = 0;
      result[1] = -1;
      result[2] = 0;
      result[3] = 0;
      result[4] = 0;
    }
  }
  return result;
}
