// Function: sub_36F7480
// Address: 0x36f7480
//
__int64 __fastcall sub_36F7480(unsigned int a1)
{
  if ( a1 > 0x1054 )
  {
    switch ( a1 )
    {
      case 0x15A6u:
        return 5541;
      case 0x15A8u:
        return 5543;
      case 0x15AAu:
        return 5545;
      case 0x15ACu:
        return 5547;
      case 0x15AEu:
        return 5549;
      case 0x15B0u:
        return 5551;
      case 0x15B2u:
        return 5553;
      case 0x15B4u:
        return 5555;
      default:
        goto LABEL_19;
    }
  }
  if ( a1 > 0x1049 )
  {
    switch ( a1 )
    {
      case 0x104Au:
        return 4169;
      case 0x104Cu:
        return 4171;
      case 0x104Eu:
        return 4173;
      case 0x1050u:
        return 4175;
      case 0x1052u:
        return 4177;
      case 0x1054u:
        return 4179;
      default:
        break;
    }
  }
LABEL_19:
  BUG();
}
