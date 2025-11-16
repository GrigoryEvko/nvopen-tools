// Function: sub_1E31360
// Address: 0x1e31360
//
__int64 __fastcall sub_1E31360(__int64 a1, int a2)
{
  *(_BYTE *)(a1 + 3) = ((_BYTE)a2 << 7) | *(_BYTE *)(a1 + 3) & 0x7F;
  return (unsigned int)(a2 << 7);
}
