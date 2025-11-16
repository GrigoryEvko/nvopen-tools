// Function: sub_B30310
// Address: 0xb30310
//
__int64 __fastcall sub_B30310(__int64 a1, int a2)
{
  int v2; // esi
  __int64 result; // rax

  v2 = (a2 + 1) << 6;
  result = *(_WORD *)(a1 + 34) & 1 | (2 * (v2 | (*(_WORD *)(a1 + 34) >> 1) & 0x7E3Fu));
  *(_WORD *)(a1 + 34) = *(_WORD *)(a1 + 34) & 1 | (2 * (v2 | (*(_WORD *)(a1 + 34) >> 1) & 0x7E3F));
  return result;
}
