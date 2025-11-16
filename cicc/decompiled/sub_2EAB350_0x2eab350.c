// Function: sub_2EAB350
// Address: 0x2eab350
//
__int64 __fastcall sub_2EAB350(__int64 a1, int a2)
{
  *(_BYTE *)(a1 + 3) = ((_BYTE)a2 << 7) | *(_BYTE *)(a1 + 3) & 0x7F;
  return (unsigned int)(a2 << 7);
}
