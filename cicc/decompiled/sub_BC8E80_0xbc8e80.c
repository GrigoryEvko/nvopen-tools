// Function: sub_BC8E80
// Address: 0xbc8e80
//
_BOOL8 __fastcall sub_BC8E80(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r8

  v2 = 0;
  if ( (*(_BYTE *)(a1 + 7) & 0x20) != 0 )
    v2 = sub_B91C10(a1, 2);
  return sub_BC8CE0(v2, a2);
}
