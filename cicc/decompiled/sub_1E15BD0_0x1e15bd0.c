// Function: sub_1E15BD0
// Address: 0x1e15bd0
//
__int64 __fastcall sub_1E15BD0(__int64 a1, __int64 a2)
{
  _BYTE *v2; // rbx
  __int64 result; // rax
  _BYTE *i; // r12
  _BYTE *v6; // rsi

  v2 = *(_BYTE **)(a1 + 32);
  result = 5LL * *(unsigned int *)(a1 + 40);
  for ( i = &v2[40 * *(unsigned int *)(a1 + 40)]; i != v2; result = sub_1E69A50(a2, v6) )
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
