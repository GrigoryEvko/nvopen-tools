// Function: sub_30F3CB0
// Address: 0x30f3cb0
//
__int64 __fastcall sub_30F3CB0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r8
  __int64 i; // rsi
  int v5; // edi
  __int64 v6; // rax
  bool v7; // cc

  v3 = a1;
  for ( i = 0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 3); i > 0; i >>= 1 )
  {
    while ( 1 )
    {
      v5 = *(_DWORD *)(a3 + 16);
      v6 = v3 + 8 * ((i >> 1) + (i & 0xFFFFFFFFFFFFFFFELL));
      v7 = *(_DWORD *)(v6 + 16) < v5;
      if ( *(_DWORD *)(v6 + 16) == v5 )
        v7 = *(_QWORD *)(v6 + 8) < *(_QWORD *)(a3 + 8);
      if ( v7 )
        break;
      v3 = v6 + 24;
      i = i - (i >> 1) - 1;
      if ( i <= 0 )
        return v3;
    }
  }
  return v3;
}
