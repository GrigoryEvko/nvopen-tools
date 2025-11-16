// Function: sub_327FC40
// Address: 0x327fc40
//
__int64 __fastcall sub_327FC40(_QWORD *a1, unsigned int a2)
{
  if ( a2 == 1 )
    return 2;
  if ( a2 == 2 )
    return 3;
  if ( a2 != 4 )
  {
    switch ( a2 )
    {
      case 8u:
        return 5;
      case 0x10u:
        return 6;
      case 0x20u:
        return 7;
      case 0x40u:
        return 8;
      case 0x80u:
        return 9;
      default:
        return sub_3007020(a1, a2);
    }
  }
  return 4;
}
