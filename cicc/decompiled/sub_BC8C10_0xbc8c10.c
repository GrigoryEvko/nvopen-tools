// Function: sub_BC8C10
// Address: 0xbc8c10
//
__int64 __fastcall sub_BC8C10(__int64 a1, __int64 a2)
{
  __int64 v2; // r8

  v2 = 0;
  if ( (*(_BYTE *)(a1 + 7) & 0x20) != 0 )
    v2 = sub_B91C10(a1, 2);
  return sub_BC8BD0(v2, a2);
}
