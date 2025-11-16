// Function: sub_1EF8050
// Address: 0x1ef8050
//
__int64 __fastcall sub_1EF8050(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r8
  __int64 i; // rsi
  __int64 v6; // rdx

  v3 = a1;
  for ( i = 0xCCCCCCCCCCCCCCCDLL * ((a2 - a1) >> 3); i > 0; i >>= 1 )
  {
    while ( 1 )
    {
      v6 = v3 + 40 * (i >> 1);
      if ( *(_DWORD *)(a3 + 8) > *(_DWORD *)(v6 + 8) )
        break;
      v3 = v6 + 40;
      i = i - (i >> 1) - 1;
      if ( i <= 0 )
        return v3;
    }
  }
  return v3;
}
