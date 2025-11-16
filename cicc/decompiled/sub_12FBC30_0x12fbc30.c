// Function: sub_12FBC30
// Address: 0x12fbc30
//
__int64 __fastcall sub_12FBC30(unsigned __int16 a1)
{
  unsigned int v1; // ecx

  if ( !a1 )
    return 246;
  _BitScanReverse(&v1, a1);
  return (unsigned __int8)(1 - ((v1 ^ 0x1F) - 21));
}
