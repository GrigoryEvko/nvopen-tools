// Function: sub_B450F0
// Address: 0xb450f0
//
__int64 __fastcall sub_B450F0(__int64 a1, int a2)
{
  int v2; // esi
  int v3; // edx
  __int64 result; // rax

  v2 = 16 * a2;
  v3 = *(_BYTE *)(a1 + 1) & 1;
  result = v3 | (2 * (v2 | (*(_BYTE *)(a1 + 1) >> 1) & 0xEFu));
  *(_BYTE *)(a1 + 1) = v3 | (2 * (v2 | (*(_BYTE *)(a1 + 1) >> 1) & 0xEF));
  return result;
}
