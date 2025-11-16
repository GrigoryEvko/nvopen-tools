// Function: sub_624240
// Address: 0x624240
//
__int64 __fastcall sub_624240(unsigned __int8 a1, __int64 a2, __int64 a3)
{
  __int64 i; // rbx
  char v5; // dl
  unsigned __int8 v7; // al

  for ( i = a2; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( !(unsigned int)sub_8D2E30(i) && (v5 = *(_BYTE *)(i + 140), (unsigned __int8)(v5 - 13) > 1u) && v5 )
  {
    sub_684B30(2786, a3);
    return 0;
  }
  else if ( *(_BYTE *)(a2 + 140) == 12 && (v7 = sub_8D4C10(a2, 1), (v7 & 0x70) != 0) && ((a1 ^ v7) & 0x70) != 0 )
  {
    sub_6851C0(2785, a3);
    return 0;
  }
  else
  {
    return 1;
  }
}
