// Function: sub_36E0C40
// Address: 0x36e0c40
//
__int64 __fastcall sub_36E0C40(__int16 a1, __int16 a2, __int64 a3)
{
  bool v3; // cl
  __int64 result; // rax

  v3 = 0;
  if ( a3 )
    v3 = ((*(_BYTE *)(a3 + 33) >> 2) & 3) == 2;
  switch ( a2 )
  {
    case 5:
      if ( a1 != 7 )
      {
        if ( a1 == 8 )
          return !v3 ? 1218 : 1156;
        if ( a1 == 6 )
          return !v3 ? 1194 : 1132;
LABEL_32:
        BUG();
      }
      result = !v3 ? 1206 : 1144;
      break;
    case 6:
      switch ( a1 )
      {
        case 7:
          result = !v3 ? 1203 : 1141;
          break;
        case 8:
          return !v3 ? 1215 : 1153;
        case 5:
          result = !v3 ? 1227 : 1165;
          break;
        default:
          goto LABEL_32;
      }
      break;
    case 7:
      if ( a1 == 6 )
        return !v3 ? 1192 : 1130;
      if ( a1 == 8 )
        return !v3 ? 1216 : 1154;
      if ( a1 != 5 )
        goto LABEL_32;
      return !v3 ? 1228 : 1166;
    case 8:
      if ( a1 == 6 )
        return !v3 ? 1193 : 1131;
      if ( a1 == 7 )
        return !v3 ? 1205 : 1143;
      if ( a1 != 5 )
        goto LABEL_32;
      return !v3 ? 1229 : 1167;
    case 11:
      if ( a1 != 12 )
      {
        if ( a1 == 13 )
          return 1114;
        goto LABEL_32;
      }
      return 1102;
    default:
      goto LABEL_32;
  }
  return result;
}
