// Function: sub_3108C80
// Address: 0x3108c80
//
__int64 __fastcall sub_3108C80(int a1)
{
  bool v1; // cc

  if ( a1 > 6 )
  {
    v1 = (unsigned int)(a1 - 7) <= 0x11;
  }
  else
  {
    v1 = (unsigned int)a1 <= 4;
    if ( a1 > 4 )
      return 1;
  }
  if ( !v1 )
    BUG();
  return 0;
}
