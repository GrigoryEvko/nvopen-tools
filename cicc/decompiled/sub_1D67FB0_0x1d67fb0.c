// Function: sub_1D67FB0
// Address: 0x1d67fb0
//
_QWORD *__fastcall sub_1D67FB0(__int64 a1)
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
    v3 = 2LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    result = (_QWORD *)(a1 + 16);
    v3 = 16;
  }
  for ( i = &result[v3]; result != i; result += 2 )
  {
    if ( result )
    {
      *result = -8;
      result[1] = -8;
    }
  }
  return result;
}
