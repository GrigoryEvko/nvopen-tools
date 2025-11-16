// Function: sub_2B086B0
// Address: 0x2b086b0
//
__int64 __fastcall sub_2B086B0(unsigned int a1, unsigned int a2)
{
  unsigned int v2; // edx
  unsigned int v3; // eax
  __int64 result; // rax

  v2 = 1;
  v3 = (a1 != 0) + (a1 - (a1 != 0)) / a2;
  if ( v3 > 1 )
  {
    _BitScanReverse(&v3, v3 - 1);
    v2 = 1 << (32 - (v3 ^ 0x1F));
  }
  result = v2;
  if ( a1 <= v2 )
    return a1;
  return result;
}
