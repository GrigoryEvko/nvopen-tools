// Function: sub_1C303A0
// Address: 0x1c303a0
//
__int64 __fastcall sub_1C303A0(unsigned int a1)
{
  __int64 result; // rax

  result = 0;
  if ( a1 <= 0x1421 )
  {
    if ( a1 > 0x13DC )
    {
      switch ( a1 )
      {
        case 0x13DDu:
        case 0x13DFu:
        case 0x13E1u:
        case 0x13E3u:
        case 0x13E5u:
        case 0x13E7u:
        case 0x13E9u:
        case 0x13EBu:
        case 0x13EDu:
        case 0x13EFu:
        case 0x13F1u:
        case 0x13F3u:
        case 0x13F5u:
        case 0x13F7u:
        case 0x13F9u:
        case 0x13FBu:
        case 0x13FDu:
        case 0x13FFu:
        case 0x1401u:
        case 0x1403u:
        case 0x1405u:
        case 0x1407u:
        case 0x1409u:
        case 0x140Bu:
        case 0x140Du:
        case 0x140Fu:
        case 0x1411u:
        case 0x1413u:
        case 0x1415u:
        case 0x1417u:
        case 0x1419u:
        case 0x141Bu:
        case 0x141Du:
        case 0x141Fu:
        case 0x1421u:
LABEL_6:
          result = 1;
          break;
        default:
LABEL_7:
          result = 0;
          break;
      }
    }
    else if ( a1 > 0x1175 )
    {
      switch ( a1 )
      {
        case 0x1272u:
        case 0x1275u:
        case 0x1276u:
        case 0x127Bu:
        case 0x127Eu:
        case 0x1280u:
        case 0x1283u:
          goto LABEL_6;
        case 0x1273u:
        case 0x1274u:
        case 0x1277u:
        case 0x1278u:
        case 0x1279u:
        case 0x127Au:
        case 0x127Cu:
        case 0x127Du:
        case 0x127Fu:
        case 0x1281u:
        case 0x1282u:
          goto LABEL_7;
        default:
          return result;
      }
    }
    else if ( a1 > 0x1142 )
    {
      switch ( a1 )
      {
        case 0x1143u:
        case 0x1146u:
        case 0x1149u:
        case 0x114Bu:
        case 0x114Du:
        case 0x114Fu:
        case 0x1151u:
        case 0x1153u:
        case 0x1155u:
        case 0x1157u:
        case 0x1159u:
        case 0x115Bu:
        case 0x115Du:
        case 0x115Fu:
        case 0x1161u:
        case 0x1163u:
        case 0x1165u:
        case 0x1167u:
        case 0x1169u:
        case 0x116Bu:
        case 0x116Du:
        case 0x116Fu:
        case 0x1171u:
        case 0x1173u:
        case 0x1175u:
          goto LABEL_6;
        default:
          goto LABEL_7;
      }
    }
  }
  return result;
}
