// Function: sub_BC89C0
// Address: 0xbc89c0
//
__int64 __fastcall sub_BC89C0(__int64 a1)
{
  __int64 v1; // r12

  v1 = 0;
  if ( (*(_BYTE *)(a1 + 7) & 0x20) != 0 )
    v1 = sub_B91C10(a1, 2);
  if ( !(unsigned __int8)sub_BC8680(v1) )
    return 0;
  return v1;
}
