// Function: sub_6878E0
// Address: 0x6878e0
//
_BOOL8 __fastcall sub_6878E0(unsigned __int16 a1)
{
  if ( a1 > 0xBCu )
  {
    return a1 == 237 || a1 == 294;
  }
  else
  {
    if ( a1 > 0x7Au )
    {
      switch ( a1 )
      {
        case 0x7Bu:
        case 0x7Cu:
        case 0x8Au:
        case 0x8Bu:
        case 0x8Cu:
        case 0x8Du:
        case 0x92u:
        case 0x9Cu:
        case 0xA1u:
        case 0xB5u:
        case 0xB6u:
        case 0xBCu:
          return 1;
        default:
          return 0;
      }
    }
    return (unsigned __int16)(a1 - 1) <= 0x1Au && ((1LL << a1) & 0x86000FE) != 0;
  }
}
