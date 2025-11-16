// Function: sub_C09F10
// Address: 0xc09f10
//
__int64 __fastcall sub_C09F10(_BYTE *a1, __int64 a2)
{
  if ( a2 == 1 )
  {
    switch ( *a1 )
    {
      case 'v':
        return 0;
      case 'l':
        return 1;
      case 'R':
        return 2;
      case 'L':
        return 3;
      case 'U':
        return 4;
      case 'u':
        return 9;
      default:
LABEL_24:
        BUG();
    }
  }
  else
  {
    if ( a2 != 2 )
      goto LABEL_24;
    switch ( *(_WORD *)a1 )
    {
      case 0x736C:
        return 5;
      case 0x734C:
        return 6;
      case 0x7352:
        return 7;
      case 0x7355:
        return 8;
      default:
        goto LABEL_24;
    }
  }
}
