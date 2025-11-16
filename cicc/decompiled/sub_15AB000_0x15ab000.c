// Function: sub_15AB000
// Address: 0x15ab000
//
__int64 __fastcall sub_15AB000(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned int v3; // r8d

  v2 = sub_15AAE60(a1, a2);
  v3 = -1;
  if ( v2 )
  {
    _BitScanReverse(&v2, v2);
    return 31 - (v2 ^ 0x1F);
  }
  return v3;
}
