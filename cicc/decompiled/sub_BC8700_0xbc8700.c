// Function: sub_BC8700
// Address: 0xbc8700
//
__int64 __fastcall sub_BC8700(__int64 a1)
{
  __int64 v1; // rax

  if ( (*(_BYTE *)(a1 + 7) & 0x20) == 0 )
    return sub_BC8680(0);
  v1 = sub_B91C10(a1, 2);
  return sub_BC8680(v1);
}
