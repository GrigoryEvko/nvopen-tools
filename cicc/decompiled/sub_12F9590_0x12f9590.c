// Function: sub_12F9590
// Address: 0x12f9590
//
__int64 __fastcall sub_12F9590(unsigned int a1)
{
  unsigned int v2; // ecx
  char v3; // al
  __int64 v4; // rsi

  if ( !a1 )
    return 0;
  if ( (a1 & 0x80000000) != 0 )
    return sub_12F9D30(0, 157, (a1 >> 1) | a1 & 1);
  _BitScanReverse(&v2, a1);
  v3 = v2 ^ 0x1F;
  v4 = 156LL - (char)((v2 ^ 0x1F) - 1);
  if ( (char)((v2 ^ 0x1F) - 1) > 6 && (unsigned int)v4 <= 0xFC )
    return ((_DWORD)v4 << 23) + (unsigned int)((unsigned __int64)a1 << (v3 - 8));
  else
    return sub_12F9D30(0, v4, (unsigned __int64)a1 << (v3 - 1));
}
