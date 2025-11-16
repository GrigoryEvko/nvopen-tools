// Function: sub_38E2720
// Address: 0x38e2720
//
__int64 __fastcall sub_38E2720(__int64 a1, int a2)
{
  int v2; // esi
  __int64 result; // rax

  v2 = 32 * a2;
  result = v2 | *(_WORD *)(a1 + 12) & 0xFF9Fu;
  *(_WORD *)(a1 + 12) = v2 | *(_WORD *)(a1 + 12) & 0xFF9F;
  return result;
}
