// Function: sub_12FBCB0
// Address: 0x12fbcb0
//
__int64 __fastcall sub_12FBCB0(unsigned __int64 a1)
{
  unsigned __int64 v1; // rcx

  if ( !a1 )
    return -52;
  _BitScanReverse64(&v1, a1);
  return 1 - (char)((v1 ^ 0x3F) - 11);
}
