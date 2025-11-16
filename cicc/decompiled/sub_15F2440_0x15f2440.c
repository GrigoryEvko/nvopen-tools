// Function: sub_15F2440
// Address: 0x15f2440
//
__int64 __fastcall sub_15F2440(__int64 a1, int a2)
{
  __int64 result; // rax

  result = (2 * (a2 | (*(_BYTE *)(a1 + 17) >> 1))) | *(_BYTE *)(a1 + 17) & 1u;
  *(_BYTE *)(a1 + 17) = (2 * (a2 | (*(_BYTE *)(a1 + 17) >> 1))) | *(_BYTE *)(a1 + 17) & 1;
  return result;
}
