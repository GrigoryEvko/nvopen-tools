// Function: sub_305B520
// Address: 0x305b520
//
bool __fastcall sub_305B520(_DWORD *a1, int a2)
{
  bool result; // al

  result = 0;
  if ( a1[86] > 0x4Fu )
  {
    if ( a2 <= 98 )
    {
      if ( a2 > 95 )
      {
LABEL_7:
        result = 0;
        if ( a1[85] > 0x383u )
          return a1[84] > 0x4Du;
      }
      else
      {
        return 1;
      }
    }
    else
    {
      switch ( a2 )
      {
        case 205:
        case 207:
        case 208:
        case 266:
        case 268:
        case 269:
        case 270:
        case 271:
        case 273:
        case 274:
          goto LABEL_7;
        case 279:
        case 280:
        case 281:
        case 282:
        case 283:
        case 284:
          result = 0;
          if ( a1[85] > 0x31Fu )
            result = a1[84] > 0x45u;
          break;
        default:
          return 1;
      }
    }
  }
  return result;
}
