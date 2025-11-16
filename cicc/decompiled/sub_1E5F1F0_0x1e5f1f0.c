// Function: sub_1E5F1F0
// Address: 0x1e5f1f0
//
_QWORD *__fastcall sub_1E5F1F0(__int64 a1)
{
  __int64 v1; // rdx
  _QWORD *result; // rax
  _QWORD *i; // rdx

  v1 = *(unsigned int *)(a1 + 24);
  result = *(_QWORD **)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  for ( i = &result[9 * v1]; i != result; result += 9 )
  {
    if ( result )
      *result = -8;
  }
  return result;
}
