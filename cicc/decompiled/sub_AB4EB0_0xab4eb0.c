// Function: sub_AB4EB0
// Address: 0xab4eb0
//
bool __fastcall sub_AB4EB0(unsigned int a1)
{
  bool result; // al

  result = 0;
  if ( a1 <= 0x173 )
  {
    if ( a1 > 0x136 )
    {
      switch ( a1 )
      {
        case 0x137u:
        case 0x149u:
        case 0x14Au:
        case 0x152u:
        case 0x167u:
        case 0x16Du:
        case 0x16Eu:
        case 0x173u:
          return 1;
        default:
          result = 0;
          break;
      }
    }
    else
    {
      return a1 == 1 || a1 - 65 < 3;
    }
  }
  return result;
}
