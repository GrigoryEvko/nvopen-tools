// Function: sub_B45110
// Address: 0xb45110
//
__int64 __fastcall sub_B45110(__int64 a1, int a2)
{
  int v2; // esi
  int v3; // edx
  __int64 result; // rax

  v2 = 32 * a2;
  v3 = *(_BYTE *)(a1 + 1) & 1;
  result = v3 | (2 * (v2 | (*(_BYTE *)(a1 + 1) >> 1) & 0xDFu));
  *(_BYTE *)(a1 + 1) = v3 | (2 * (v2 | (*(_BYTE *)(a1 + 1) >> 1) & 0xDF));
  return result;
}
