// Function: sub_16FF680
// Address: 0x16ff680
//
__int64 __fastcall sub_16FF680(__int64 a1, __int64 a2, _DWORD *a3)
{
  __int64 v3; // r8
  __int64 i; // rsi
  __int64 v6; // rdx

  v3 = a1;
  for ( i = (a2 - a1) >> 3; i > 0; i >>= 1 )
  {
    while ( 1 )
    {
      v6 = v3 + 8 * (i >> 1);
      if ( *a3 <= *(_DWORD *)(v6 + 4) )
        break;
      v3 = v6 + 8;
      i = i - (i >> 1) - 1;
      if ( i <= 0 )
        return v3;
    }
  }
  return v3;
}
