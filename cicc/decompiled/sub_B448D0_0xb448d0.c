// Function: sub_B448D0
// Address: 0xb448d0
//
__int64 __fastcall sub_B448D0(__int64 a1, int a2)
{
  int v2; // edx
  __int64 result; // rax

  v2 = *(_BYTE *)(a1 + 1) & 1;
  result = v2 | (2 * (a2 | (*(_BYTE *)(a1 + 1) >> 1) & 0xFEu));
  *(_BYTE *)(a1 + 1) = v2 | (2 * (a2 | (*(_BYTE *)(a1 + 1) >> 1) & 0xFE));
  return result;
}
