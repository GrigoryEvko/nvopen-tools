// Function: sub_2CE02B0
// Address: 0x2ce02b0
//
bool __fastcall sub_2CE02B0(__int64 a1, unsigned int a2)
{
  bool result; // al

  result = 0;
  if ( a2 <= 0x22D2 )
  {
    if ( a2 <= 0x227C )
    {
      if ( a2 > 0x2056 )
        return a2 == 8280;
      else
        return a2 > 0x2053;
    }
    else
    {
      switch ( a2 )
      {
        case 0x227Du:
        case 0x227Eu:
        case 0x227Fu:
        case 0x2280u:
        case 0x2285u:
        case 0x2286u:
        case 0x228Bu:
        case 0x228Cu:
        case 0x228Du:
        case 0x228Eu:
        case 0x2293u:
        case 0x2294u:
        case 0x2295u:
        case 0x2296u:
        case 0x2297u:
        case 0x2298u:
        case 0x229Du:
        case 0x229Eu:
        case 0x22B3u:
        case 0x22B4u:
        case 0x22B5u:
        case 0x22B6u:
        case 0x22B7u:
        case 0x22BAu:
        case 0x22BBu:
        case 0x22BCu:
        case 0x22BDu:
        case 0x22BEu:
        case 0x22BFu:
        case 0x22C2u:
        case 0x22C3u:
        case 0x22C4u:
        case 0x22C5u:
        case 0x22C6u:
        case 0x22C7u:
        case 0x22CAu:
        case 0x22CBu:
        case 0x22CCu:
        case 0x22CDu:
        case 0x22CEu:
        case 0x22CFu:
        case 0x22D2u:
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
