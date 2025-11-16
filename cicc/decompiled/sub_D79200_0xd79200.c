// Function: sub_D79200
// Address: 0xd79200
//
_QWORD *__fastcall sub_D79200(__int64 a1)
{
  __int64 v1; // rdx
  _QWORD *result; // rax
  _QWORD *i; // rdx

  v1 = *(unsigned int *)(a1 + 24);
  result = *(_QWORD **)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  for ( i = &result[2 * v1]; result != i; result += 2 )
  {
    if ( result )
      *result = -8;
  }
  return result;
}
