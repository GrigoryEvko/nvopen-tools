// Function: sub_B4DDE0
// Address: 0xb4dde0
//
__int64 __fastcall sub_B4DDE0(__int64 a1, int a2)
{
  *(_BYTE *)(a1 + 1) = (2 * a2) | *(_BYTE *)(a1 + 1) & 1;
  return (unsigned int)(2 * a2);
}
