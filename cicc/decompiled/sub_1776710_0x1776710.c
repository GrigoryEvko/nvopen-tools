// Function: sub_1776710
// Address: 0x1776710
//
__int64 __fastcall sub_1776710(__int64 a1)
{
  unsigned int v1; // r8d

  v1 = 1;
  if ( (*(_BYTE *)(a1 + 8) & 0xFB) != 0xB )
    LOBYTE(v1) = (unsigned __int8)(*(_BYTE *)(a1 + 8) - 1) <= 5u;
  return v1;
}
