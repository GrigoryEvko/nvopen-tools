// Function: sub_15F8A20
// Address: 0x15f8a20
//
__int64 __fastcall sub_15F8A20(__int64 a1, unsigned int a2)
{
  int v2; // eax
  unsigned int v3; // ecx
  __int64 result; // rax

  v2 = *(unsigned __int16 *)(a1 + 18);
  v3 = v2 & 0xFFFF7FE0;
  if ( a2 )
  {
    _BitScanReverse(&a2, a2);
    v3 |= 31 - (a2 ^ 0x1F) + 1;
  }
  LOWORD(v2) = v2 & 0x8000;
  result = v3 | v2;
  *(_WORD *)(a1 + 18) = result;
  return result;
}
