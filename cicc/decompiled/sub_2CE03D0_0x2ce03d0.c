// Function: sub_2CE03D0
// Address: 0x2ce03d0
//
bool __fastcall sub_2CE03D0(__int64 a1, unsigned int a2)
{
  bool result; // al

  result = sub_CEA230(a2);
  if ( !result )
  {
    if ( a2 <= 0xF3 )
    {
      if ( a2 > 0xED )
        return ((1LL << ((unsigned __int8)a2 + 18)) & 0x29) != 0;
    }
    else
    {
      switch ( a2 )
      {
        case 0x2285u:
        case 0x2286u:
        case 0x2293u:
        case 0x2294u:
        case 0x229Du:
        case 0x229Eu:
        case 0x22BAu:
        case 0x22C2u:
        case 0x22CAu:
          result = 1;
          break;
        default:
          return result;
      }
    }
  }
  return result;
}
