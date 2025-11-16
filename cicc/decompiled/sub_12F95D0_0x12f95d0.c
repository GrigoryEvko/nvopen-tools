// Function: sub_12F95D0
// Address: 0x12f95d0
//
unsigned __int64 __fastcall sub_12F95D0(unsigned int a1)
{
  unsigned __int64 result; // rax
  unsigned int v2; // ecx

  result = 0;
  if ( a1 )
  {
    _BitScanReverse(&v2, a1);
    return ((unsigned __int64)a1 << ((v2 ^ 0x1F) + 21))
         + ((unsigned __int64)(1074 - (unsigned int)(unsigned __int8)((v2 ^ 0x1F) + 21)) << 52);
  }
  return result;
}
