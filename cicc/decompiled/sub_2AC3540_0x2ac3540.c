// Function: sub_2AC3540
// Address: 0x2ac3540
//
_DWORD *__fastcall sub_2AC3540(__int64 a1)
{
  bool v1; // zf
  _DWORD *result; // rax
  __int64 v3; // rdx
  _DWORD *i; // rdx

  v1 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v1 )
  {
    result = *(_DWORD **)(a1 + 16);
    v3 = 2LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    result = (_DWORD *)(a1 + 16);
    v3 = 8;
  }
  for ( i = &result[v3]; result != i; result += 2 )
  {
    if ( result )
      *result = -1;
  }
  return result;
}
