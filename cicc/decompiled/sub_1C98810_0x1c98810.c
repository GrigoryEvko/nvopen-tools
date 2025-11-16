// Function: sub_1C98810
// Address: 0x1c98810
//
bool __fastcall sub_1C98810(__int64 a1, unsigned int a2)
{
  bool result; // al

  result = 0;
  if ( a2 <= 0xFC5 )
  {
    if ( a2 <= 0xF79 )
    {
      if ( a2 > 0xEA7 )
        return a2 == 3753;
      else
        return a2 > 0xEA4;
    }
    else
    {
      switch ( a2 )
      {
        case 0xF7Au:
        case 0xF7Bu:
        case 0xF7Cu:
        case 0xF7Du:
        case 0xF82u:
        case 0xF83u:
        case 0xF88u:
        case 0xF89u:
        case 0xF8Au:
        case 0xF8Bu:
        case 0xF90u:
        case 0xF91u:
        case 0xF92u:
        case 0xF93u:
        case 0xF94u:
        case 0xF95u:
        case 0xF9Au:
        case 0xF9Bu:
        case 0xFA6u:
        case 0xFA7u:
        case 0xFA8u:
        case 0xFA9u:
        case 0xFAAu:
        case 0xFADu:
        case 0xFAEu:
        case 0xFAFu:
        case 0xFB0u:
        case 0xFB1u:
        case 0xFB2u:
        case 0xFB5u:
        case 0xFB6u:
        case 0xFB7u:
        case 0xFB8u:
        case 0xFB9u:
        case 0xFBAu:
        case 0xFBDu:
        case 0xFBEu:
        case 0xFBFu:
        case 0xFC0u:
        case 0xFC1u:
        case 0xFC2u:
        case 0xFC5u:
          result = 1;
          break;
        default:
          result = 0;
          break;
      }
    }
  }
  return result;
}
