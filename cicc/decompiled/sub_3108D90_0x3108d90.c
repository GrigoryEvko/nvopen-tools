// Function: sub_3108D90
// Address: 0x3108d90
//
__int64 __fastcall sub_3108D90(int a1)
{
  if ( a1 > 8 )
  {
    if ( (unsigned int)(a1 - 9) > 0xF )
      goto LABEL_8;
  }
  else
  {
    if ( a1 > 3 )
      return 1;
    if ( a1 != 3 )
    {
      if ( a1 >= 0 )
        return 1;
LABEL_8:
      BUG();
    }
  }
  return 0;
}
