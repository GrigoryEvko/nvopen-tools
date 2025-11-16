// Function: sub_15F2350
// Address: 0x15f2350
//
__int64 __fastcall sub_15F2350(__int64 a1, int a2)
{
  int v2; // edx
  __int64 result; // rax

  v2 = *(_BYTE *)(a1 + 17) & 1;
  result = v2 | (2 * (a2 | (*(_BYTE *)(a1 + 17) >> 1) & 0xFEu));
  *(_BYTE *)(a1 + 17) = v2 | (2 * (a2 | (*(_BYTE *)(a1 + 17) >> 1) & 0xFE));
  return result;
}
