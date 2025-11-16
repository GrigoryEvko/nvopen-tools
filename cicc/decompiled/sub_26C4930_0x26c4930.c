// Function: sub_26C4930
// Address: 0x26c4930
//
_QWORD *__fastcall sub_26C4930(__int64 a1)
{
  __int64 v1; // rdx
  _QWORD *result; // rax
  _QWORD *i; // rdx

  v1 = *(unsigned int *)(a1 + 24);
  result = *(_QWORD **)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  for ( i = &result[3 * v1]; i != result; result += 3 )
  {
    if ( result )
      *result = -1;
  }
  return result;
}
