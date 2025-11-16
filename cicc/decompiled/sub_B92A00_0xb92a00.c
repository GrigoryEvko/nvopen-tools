// Function: sub_B92A00
// Address: 0xb92a00
//
_QWORD *__fastcall sub_B92A00(__int64 a1)
{
  bool v1; // zf
  _QWORD *result; // rax
  __int64 v3; // rdx
  _QWORD *i; // rdx

  v1 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v1 )
  {
    result = *(_QWORD **)(a1 + 16);
    v3 = 3LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    result = (_QWORD *)(a1 + 16);
    v3 = 12;
  }
  for ( i = &result[v3]; result != i; result += 3 )
  {
    if ( result )
      *result = -4096;
  }
  return result;
}
