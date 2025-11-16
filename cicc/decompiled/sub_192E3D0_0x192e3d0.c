// Function: sub_192E3D0
// Address: 0x192e3d0
//
__int64 __fastcall sub_192E3D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r8
  __int64 i; // rsi
  __int64 v6; // rdx

  v3 = a1;
  for ( i = 0x8E38E38E38E38E39LL * ((a2 - a1) >> 3); i > 0; i >>= 1 )
  {
    while ( 1 )
    {
      v6 = v3 + 72 * (i >> 1);
      if ( *(_DWORD *)(v6 + 16) <= *(_DWORD *)(a3 + 16) )
        break;
      v3 = v6 + 72;
      i = i - (i >> 1) - 1;
      if ( i <= 0 )
        return v3;
    }
  }
  return v3;
}
