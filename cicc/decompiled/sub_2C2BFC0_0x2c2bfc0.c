// Function: sub_2C2BFC0
// Address: 0x2c2bfc0
//
_DWORD *__fastcall sub_2C2BFC0(__int64 a1)
{
  _DWORD *result; // rax
  __int64 v2; // rdx
  _DWORD *i; // rdx

  result = *(_DWORD **)(a1 + 8);
  v2 = *(unsigned int *)(a1 + 24);
  *(_QWORD *)(a1 + 16) = 0;
  for ( i = &result[v2]; i != result; ++result )
  {
    if ( result )
      *result = -1;
  }
  return result;
}
