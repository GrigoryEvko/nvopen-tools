// Function: sub_A71F30
// Address: 0xa71f30
//
__int16 __fastcall sub_A71F30(__int64 *a1)
{
  char v1; // bl
  unsigned __int64 v2; // rax
  char v3; // dl
  __int16 result; // ax

  v2 = sub_A71B70(*a1);
  v3 = 0;
  if ( v2 )
  {
    _BitScanReverse64(&v2, v2);
    v3 = 1;
    v1 = 63 - (v2 ^ 0x3F);
  }
  LOBYTE(result) = v1;
  HIBYTE(result) = v3;
  return result;
}
