// Function: sub_B45150
// Address: 0xb45150
//
__int64 __fastcall sub_B45150(__int64 a1, int a2)
{
  __int64 result; // rax

  result = (2 * (a2 | (*(_BYTE *)(a1 + 1) >> 1))) | *(_BYTE *)(a1 + 1) & 1u;
  *(_BYTE *)(a1 + 1) = (2 * (a2 | (*(_BYTE *)(a1 + 1) >> 1))) | *(_BYTE *)(a1 + 1) & 1;
  return result;
}
