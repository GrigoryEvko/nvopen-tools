// Function: sub_C7F6B0
// Address: 0xc7f6b0
//
__int64 __fastcall sub_C7F6B0(int a1)
{
  if ( a1 <= 1 )
  {
    if ( a1 >= 0 )
      return 6;
LABEL_6:
    BUG();
  }
  if ( (unsigned int)(a1 - 2) > 1 )
    goto LABEL_6;
  return 2;
}
