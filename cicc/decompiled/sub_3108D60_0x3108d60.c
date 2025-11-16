// Function: sub_3108D60
// Address: 0x3108d60
//
__int64 __fastcall sub_3108D60(int a1)
{
  if ( a1 == 5 )
    return 1;
  if ( a1 <= 5 )
  {
    if ( (unsigned int)a1 <= 4 )
      return 0;
LABEL_8:
    BUG();
  }
  if ( (unsigned int)(a1 - 6) > 0x12 )
    goto LABEL_8;
  return 0;
}
