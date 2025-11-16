// Function: sub_2912520
// Address: 0x2912520
//
__int64 __fastcall sub_2912520(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rcx
  unsigned int v3; // r8d
  unsigned __int64 v4; // rsi

  _BitScanReverse64(&v2, 1LL << (*(_WORD *)(a1 + 2) >> 1));
  v3 = -1;
  v4 = -(__int64)(a2 | (0x8000000000000000LL >> ((unsigned __int8)v2 ^ 0x3Fu)))
     & (a2 | (0x8000000000000000LL >> ((unsigned __int8)v2 ^ 0x3Fu)));
  if ( v4 )
  {
    _BitScanReverse64(&v4, v4);
    return 63 - ((unsigned int)v4 ^ 0x3F);
  }
  return v3;
}
