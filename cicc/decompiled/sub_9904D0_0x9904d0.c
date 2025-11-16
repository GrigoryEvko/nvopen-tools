// Function: sub_9904D0
// Address: 0x9904d0
//
__int64 __fastcall sub_9904D0(int a1, char a2)
{
  switch ( a1 )
  {
    case 1:
      return 40;
    case 2:
      return 36;
    case 3:
      return 38;
    case 4:
      return 34;
    case 5:
      return a2 == 0 ? 12 : 4;
  }
  if ( a1 != 6 )
    BUG();
  return a2 == 0 ? 10 : 2;
}
