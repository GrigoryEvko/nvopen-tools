// Function: sub_18A8C30
// Address: 0x18a8c30
//
_QWORD *__fastcall sub_18A8C30(__int64 a1)
{
  __int64 v1; // rdx
  _QWORD *result; // rax
  _QWORD *i; // rdx

  v1 = *(unsigned int *)(a1 + 24);
  result = *(_QWORD **)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  for ( i = &result[3 * v1]; result != i; result += 3 )
  {
    if ( result )
    {
      *result = -8;
      result[1] = -8;
    }
  }
  return result;
}
