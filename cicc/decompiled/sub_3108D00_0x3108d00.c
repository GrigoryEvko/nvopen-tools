// Function: sub_3108D00
// Address: 0x3108d00
//
__int64 __fastcall sub_3108D00(int a1)
{
  if ( a1 > 11 )
  {
    if ( (unsigned int)(a1 - 12) > 0xC )
      goto LABEL_8;
  }
  else
  {
    if ( a1 > 9 )
      return 1;
    if ( a1 <= 6 )
    {
      if ( a1 >= 0 )
        return 1;
LABEL_8:
      BUG();
    }
  }
  return 0;
}
