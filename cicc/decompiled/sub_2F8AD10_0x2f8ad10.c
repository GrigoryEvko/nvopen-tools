// Function: sub_2F8AD10
// Address: 0x2f8ad10
//
__int64 __fastcall sub_2F8AD10(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r8
  __int64 i; // rsi
  __int64 v6; // rdx

  v3 = a1;
  for ( i = 0x2E8BA2E8BA2E8BA3LL * ((a2 - a1) >> 3); i > 0; i >>= 1 )
  {
    while ( 1 )
    {
      v6 = v3 + 88 * (i >> 1);
      if ( *(_DWORD *)(a3 + 8) > *(_DWORD *)(v6 + 8) )
        break;
      v3 = v6 + 88;
      i = i - (i >> 1) - 1;
      if ( i <= 0 )
        return v3;
    }
  }
  return v3;
}
