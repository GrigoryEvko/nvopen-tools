// Function: sub_3108D30
// Address: 0x3108d30
//
__int64 __fastcall sub_3108D30(int a1)
{
  if ( a1 == 6 )
    return 1;
  if ( a1 > 6 )
  {
    if ( (unsigned int)(a1 - 7) > 0x11 )
      goto LABEL_8;
  }
  else if ( a1 <= 2 )
  {
    if ( a1 >= 0 )
      return 1;
LABEL_8:
    BUG();
  }
  return 0;
}
