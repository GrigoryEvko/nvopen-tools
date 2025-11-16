// Function: sub_12FBC80
// Address: 0x12fbc80
//
__int64 __fastcall sub_12FBC80(unsigned int a1)
{
  unsigned int v1; // ecx

  if ( !a1 )
    return -23;
  _BitScanReverse(&v1, a1);
  return 1 - (char)((v1 ^ 0x1F) - 8);
}
