// Function: sub_2AAE0E0
// Address: 0x2aae0e0
//
__int64 __fastcall sub_2AAE0E0(__int64 a1)
{
  unsigned __int64 v1; // rdx

  _BitScanReverse64(&v1, 1LL << (*(_WORD *)(a1 + 2) >> 1));
  return 63 - ((unsigned int)v1 ^ 0x3F);
}
