// Function: sub_15F2460
// Address: 0x15f2460
//
__int64 __fastcall sub_15F2460(__int64 a1, int a2)
{
  *(_BYTE *)(a1 + 17) = (2 * a2) | *(_BYTE *)(a1 + 17) & 1;
  return (unsigned int)(2 * a2);
}
