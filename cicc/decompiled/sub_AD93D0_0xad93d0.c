// Function: sub_AD93D0
// Address: 0xad93d0
//
unsigned __int8 *__fastcall sub_AD93D0(unsigned int a1, __int64 a2, char a3, unsigned int a4)
{
  if ( a1 <= 0x1E && ((1LL << a1) & 0x70066000) != 0 )
  {
    switch ( a1 )
    {
      case 0xDu:
      case 0x1Du:
      case 0x1Eu:
        return (unsigned __int8 *)sub_AD6530(a2, a4);
      case 0xEu:
        return sub_AD9290(a2, (unsigned __int8)a4 ^ 1u);
      case 0x11u:
        return (unsigned __int8 *)sub_AD64C0(a2, 1, 0);
      case 0x12u:
        return sub_AD8DD0(a2, 1.0);
      case 0x1Cu:
        return (unsigned __int8 *)sub_AD62B0(a2);
      default:
        BUG();
    }
  }
  if ( a3 )
  {
    if ( a1 == 21 )
      return sub_AD8DD0(a2, 1.0);
    if ( a1 > 0x15 )
    {
      if ( a1 - 25 <= 2 )
        return (unsigned __int8 *)sub_AD6530(a2, a4);
    }
    else
    {
      if ( a1 > 0x10 )
      {
        if ( a1 - 19 <= 1 )
          return (unsigned __int8 *)sub_AD64C0(a2, 1, 0);
        return 0;
      }
      if ( a1 > 0xE )
        return (unsigned __int8 *)sub_AD6530(a2, a4);
    }
  }
  return 0;
}
