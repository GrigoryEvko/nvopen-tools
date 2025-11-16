// Function: sub_3108C50
// Address: 0x3108c50
//
__int64 __fastcall sub_3108C50(int a1)
{
  if ( a1 <= 1 )
  {
    if ( a1 >= 0 )
      return 1;
LABEL_6:
    BUG();
  }
  if ( (unsigned int)(a1 - 2) > 0x16 )
    goto LABEL_6;
  return 0;
}
