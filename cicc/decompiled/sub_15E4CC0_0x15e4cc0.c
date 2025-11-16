// Function: sub_15E4CC0
// Address: 0x15e4cc0
//
__int64 __fastcall sub_15E4CC0(__int64 a1, unsigned int a2)
{
  unsigned int v2; // eax
  __int64 result; // rax

  if ( a2 )
  {
    _BitScanReverse(&v2, a2);
    a2 = 32 - (v2 ^ 0x1F);
  }
  result = (*(_DWORD *)(a1 + 32) >> 15) & 0xFFFFFFE0;
  *(_DWORD *)(a1 + 32) = *(_DWORD *)(a1 + 32) & 0x7FFF | (((unsigned int)result | a2) << 15);
  return result;
}
