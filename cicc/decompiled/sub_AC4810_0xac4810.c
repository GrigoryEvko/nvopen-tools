// Function: sub_AC4810
// Address: 0xac4810
//
__int64 __fastcall sub_AC4810(unsigned int a1)
{
  if ( a1 > 0x2E )
  {
    if ( a1 - 47 > 3 )
      goto LABEL_7;
  }
  else
  {
    if ( a1 > 0x26 )
      return 0;
    if ( a1 != 38 )
LABEL_7:
      BUG();
  }
  return 1;
}
