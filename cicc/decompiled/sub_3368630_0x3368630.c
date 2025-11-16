// Function: sub_3368630
// Address: 0x3368630
//
__int64 __fastcall sub_3368630(int a1)
{
  switch ( a1 )
  {
    case 1:
      return 2;
    case 2:
      return 3;
    case 4:
      return 4;
    case 8:
      return 5;
    case 16:
      return 6;
    case 32:
      return 7;
    case 64:
      return 8;
  }
  return 9 * (unsigned int)(a1 == 128);
}
