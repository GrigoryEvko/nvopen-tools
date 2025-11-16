// Function: sub_15F2330
// Address: 0x15f2330
//
__int64 __fastcall sub_15F2330(__int64 a1, int a2)
{
  int v2; // esi
  int v3; // edx
  __int64 result; // rax

  v2 = 2 * a2;
  v3 = *(_BYTE *)(a1 + 17) & 1;
  result = v3 | (2 * (v2 | (*(_BYTE *)(a1 + 17) >> 1) & 0xFDu));
  *(_BYTE *)(a1 + 17) = v3 | (2 * (v2 | (*(_BYTE *)(a1 + 17) >> 1) & 0xFD));
  return result;
}
