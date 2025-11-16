// Function: sub_15FF6F0
// Address: 0x15ff6f0
//
__int64 __fastcall sub_15FF6F0(__int64 a1)
{
  *(_WORD *)(a1 + 18) = sub_15FF5D0(*(_WORD *)(a1 + 18) & 0x7FFF) | *(_WORD *)(a1 + 18) & 0x8000;
  return sub_16484A0(a1 - 48, a1 - 24);
}
