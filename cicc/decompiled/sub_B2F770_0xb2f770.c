// Function: sub_B2F770
// Address: 0xb2f770
//
__int64 __fastcall sub_B2F770(__int64 a1, unsigned __int8 a2)
{
  int v2; // esi
  __int64 result; // rax

  v2 = a2 + 1;
  result = *(_WORD *)(a1 + 34) & 1 | (2 * (v2 | (*(_WORD *)(a1 + 34) >> 1) & 0x7FC0u));
  *(_WORD *)(a1 + 34) = *(_WORD *)(a1 + 34) & 1 | (2 * (v2 | (*(_WORD *)(a1 + 34) >> 1) & 0x7FC0));
  return result;
}
