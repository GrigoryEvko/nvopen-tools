// Function: sub_EA1660
// Address: 0xea1660
//
__int64 __fastcall sub_EA1660(__int64 a1, int a2)
{
  int v2; // esi
  __int64 result; // rax

  v2 = 32 * a2;
  result = v2 | *(_WORD *)(a1 + 12) & 0xFF9Fu;
  *(_WORD *)(a1 + 12) = v2 | *(_WORD *)(a1 + 12) & 0xFF9F;
  return result;
}
