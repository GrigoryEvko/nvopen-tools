// Function: sub_2E86750
// Address: 0x2e86750
//
__int64 __fastcall sub_2E86750(__int64 a1, __int64 a2)
{
  _BYTE *v2; // rbx
  __int64 result; // rax
  _BYTE *i; // r12
  _BYTE *v6; // rsi

  v2 = *(_BYTE **)(a1 + 32);
  result = 5LL * (*(_DWORD *)(a1 + 40) & 0xFFFFFF);
  for ( i = &v2[40 * (*(_DWORD *)(a1 + 40) & 0xFFFFFF)]; i != v2; result = sub_2EBEAE0(a2, v6) )
  {
    while ( *v2 )
    {
      v2 += 40;
      if ( i == v2 )
        return result;
    }
    v6 = v2;
    v2 += 40;
  }
  return result;
}
