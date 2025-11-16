// Function: sub_1C98920
// Address: 0x1c98920
//
bool __fastcall sub_1C98920(__int64 a1, unsigned int a2)
{
  bool result; // al

  result = sub_1C302A0(a2);
  if ( !result )
  {
    if ( a2 <= 0x89 )
    {
      if ( a2 > 0x84 )
        return ((1LL << ((unsigned __int8)a2 + 123)) & 0x15) != 0;
    }
    else
    {
      switch ( a2 )
      {
        case 0xF82u:
        case 0xF83u:
        case 0xF90u:
        case 0xF91u:
        case 0xF9Au:
        case 0xF9Bu:
        case 0xFADu:
        case 0xFB5u:
        case 0xFBDu:
          result = 1;
          break;
        default:
          return result;
      }
    }
  }
  return result;
}
