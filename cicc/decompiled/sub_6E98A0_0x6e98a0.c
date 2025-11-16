// Function: sub_6E98A0
// Address: 0x6e98a0
//
_BOOL8 __fastcall sub_6E98A0(__int64 a1)
{
  char v2; // al

  if ( *(_BYTE *)(a1 + 17) != 2 )
    return 0;
  if ( !(unsigned int)sub_8D2E30(*(_QWORD *)a1) )
    return 0;
  v2 = *(_BYTE *)(a1 + 16);
  if ( v2 == 2 )
  {
    if ( !(unsigned int)sub_70FCE0(a1 + 144) )
      return 0;
    return (unsigned int)sub_710600(a1 + 144) == 0;
  }
  else
  {
    if ( v2 != 1 )
      return 0;
    return (unsigned int)sub_7311F0(*(_QWORD *)(a1 + 144)) != 0;
  }
}
