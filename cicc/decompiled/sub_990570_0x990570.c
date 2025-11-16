// Function: sub_990570
// Address: 0x990570
//
__int64 __fastcall sub_990570(int a1)
{
  switch ( a1 )
  {
    case 1:
      return 3;
    case 2:
      return 4;
    case 3:
      return 1;
  }
  if ( a1 != 4 )
    BUG();
  return 2;
}
