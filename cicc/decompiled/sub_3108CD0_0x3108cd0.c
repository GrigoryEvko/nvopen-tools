// Function: sub_3108CD0
// Address: 0x3108cd0
//
__int64 __fastcall sub_3108CD0(int a1)
{
  if ( a1 <= 6 )
  {
    if ( a1 >= 0 )
      return 1;
LABEL_6:
    BUG();
  }
  if ( (unsigned int)(a1 - 7) > 0x11 )
    goto LABEL_6;
  return 0;
}
