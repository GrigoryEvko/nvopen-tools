// Function: sub_12FA470
// Address: 0x12fa470
//
unsigned __int64 __fastcall sub_12FA470(unsigned __int64 a1)
{
  __int64 v1; // r8
  unsigned __int64 v2; // rcx
  unsigned __int8 v3; // al

  v1 = 0;
  if ( a1 )
  {
    _BitScanReverse64(&v2, a1);
    v3 = (v2 ^ 0x3F) + 49;
    if ( v3 <= 0x3Fu )
      return a1 << v3;
  }
  return v1;
}
