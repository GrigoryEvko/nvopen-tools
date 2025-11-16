// Function: sub_9905C0
// Address: 0x9905c0
//
__int64 __fastcall sub_9905C0(unsigned int a1)
{
  if ( a1 == 329 )
    return 330;
  if ( a1 > 0x149 )
  {
    switch ( a1 )
    {
      case 0x16Du:
        return 366;
      case 0x16Eu:
        return 365;
      case 0x14Au:
        return 329;
      default:
        goto LABEL_19;
    }
  }
  else if ( a1 == 246 )
  {
    return 235;
  }
  else
  {
    if ( a1 <= 0xF6 )
    {
      if ( a1 == 235 )
        return 246;
      if ( a1 == 237 )
        return 248;
LABEL_19:
      BUG();
    }
    if ( a1 != 248 )
      goto LABEL_19;
    return 237;
  }
}
