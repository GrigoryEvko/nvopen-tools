// Function: sub_37BEA60
// Address: 0x37bea60
//
_DWORD *__fastcall sub_37BEA60(__int64 a1)
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
    v3 = 4LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    result = (_DWORD *)(a1 + 16);
    v3 = 32;
  }
  for ( i = &result[v3]; result != i; result += 4 )
  {
    if ( result )
      *result = -1;
  }
  return result;
}
