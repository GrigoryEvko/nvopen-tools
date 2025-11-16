// Function: sub_30F3D20
// Address: 0x30f3d20
//
__int64 __fastcall sub_30F3D20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r8
  __int64 i; // rsi
  __int64 v5; // rax
  int v6; // edi
  bool v7; // cc

  v3 = a1;
  for ( i = 0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 3); i > 0; i >>= 1 )
  {
    while ( 1 )
    {
      v5 = v3 + 8 * ((i >> 1) + (i & 0xFFFFFFFFFFFFFFFELL));
      v6 = *(_DWORD *)(v5 + 16);
      v7 = *(_DWORD *)(a3 + 16) < v6;
      if ( *(_DWORD *)(a3 + 16) == v6 )
        v7 = *(_QWORD *)(a3 + 8) < *(_QWORD *)(v5 + 8);
      if ( !v7 )
        break;
      v3 = v5 + 24;
      i = i - (i >> 1) - 1;
      if ( i <= 0 )
        return v3;
    }
  }
  return v3;
}
