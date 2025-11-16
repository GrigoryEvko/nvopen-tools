// Function: sub_37BB410
// Address: 0x37bb410
//
_DWORD *__fastcall sub_37BB410(__int64 a1, unsigned int *a2)
{
  _DWORD *v4; // r8
  __int64 v5; // rdx
  __int64 v6; // rax
  unsigned int v7; // esi
  __int64 v8; // rdx
  unsigned int *v9; // rcx

  v4 = *(_DWORD **)a1;
  v5 = 5LL * *(unsigned int *)(a1 + 8);
  v6 = *(unsigned int *)(a1 + 8);
  if ( v5 )
  {
    v7 = *a2;
    do
    {
      while ( 1 )
      {
        v8 = v6 >> 1;
        v9 = &v4[5 * (v6 >> 1)];
        if ( *v9 < v7 || *v9 == v7 && v9[1] < a2[1] )
          break;
        v6 >>= 1;
        if ( v8 <= 0 )
          return v4;
      }
      v4 = v9 + 5;
      v6 = v6 - v8 - 1;
    }
    while ( v6 > 0 );
  }
  return v4;
}
