// Function: sub_12FA4C0
// Address: 0x12fa4c0
//
unsigned __int64 __fastcall sub_12FA4C0(__int64 a1)
{
  unsigned __int64 v1; // rdi
  unsigned __int64 v2; // rcx
  unsigned __int8 v3; // al

  if ( !a1 )
    return 0;
  v1 = abs64(a1);
  _BitScanReverse64(&v2, v1);
  v3 = (v2 ^ 0x3F) + 49;
  if ( v3 > 0x3Fu )
    return 0;
  else
    return v1 << v3;
}
