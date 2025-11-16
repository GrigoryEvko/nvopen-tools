// Function: sub_64EDF0
// Address: 0x64edf0
//
__int64 __fastcall sub_64EDF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax

  if ( word_4F06418[0] > 0xB4u )
  {
    if ( word_4F06418[0] == 239 )
    {
      return sub_72BA30(11);
    }
    else
    {
      switch ( word_4F06418[0] )
      {
        case 0x14Bu:
          result = sub_72C610(11);
          break;
        case 0x14Cu:
          result = sub_72C610(3);
          break;
        case 0x14Du:
          result = sub_72C610(12);
          break;
        case 0x14Eu:
          result = sub_72C610(5);
          break;
        case 0x14Fu:
          result = sub_72C610(13);
          break;
        default:
          return 0;
      }
    }
  }
  else if ( word_4F06418[0] <= 0x4Fu )
  {
    return 0;
  }
  else
  {
    switch ( word_4F06418[0] )
    {
      case 0x50u:
        result = sub_72BA30(0);
        break;
      case 0x55u:
        result = sub_72C610(4);
        break;
      case 0x59u:
        result = sub_72C610(2);
        break;
      case 0x5Du:
      case 0x62u:
        result = sub_72BA30(5);
        break;
      case 0x5Eu:
        result = sub_72BA30(7);
        break;
      case 0x61u:
        result = sub_72BA30(3);
        break;
      case 0x69u:
        result = sub_72BA30(6);
        break;
      case 0x6Au:
        result = sub_72CBE0(a1, a2, a3, a4, a5, a6);
        break;
      case 0x7Eu:
        result = sub_72C0F0();
        break;
      case 0x7Fu:
        result = sub_72C1B0();
        break;
      case 0x80u:
        result = sub_72C030();
        break;
      case 0x85u:
        result = sub_72BA30(unk_4F06AD1);
        break;
      case 0x86u:
        result = sub_72BA30(unk_4F06ACF);
        break;
      case 0x87u:
        result = sub_72BA30(unk_4F06ACD);
        break;
      case 0x88u:
        result = sub_72BA30(unk_4F06ACB);
        break;
      case 0xA5u:
        result = sub_72BF70();
        break;
      case 0xB4u:
        result = sub_72C390();
        break;
      default:
        return 0;
    }
  }
  return result;
}
