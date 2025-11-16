// Function: sub_14EA280
// Address: 0x14ea280
//
__int64 __fastcall sub_14EA280(int a1, __int64 a2)
{
  char v2; // al
  __int64 result; // rax

  v2 = *(_BYTE *)(a2 + 8);
  if ( v2 == 16 )
    v2 = *(_BYTE *)(**(_QWORD **)(a2 + 16) + 8LL);
  if ( (unsigned __int8)(v2 - 1) > 5u )
  {
    if ( v2 == 11 )
    {
      switch ( a1 )
      {
        case 0:
          result = 11;
          break;
        case 1:
          result = 13;
          break;
        case 2:
          result = 15;
          break;
        case 3:
          result = 17;
          break;
        case 4:
          result = 18;
          break;
        case 5:
          result = 20;
          break;
        case 6:
          result = 21;
          break;
        case 7:
          result = 23;
          break;
        case 8:
          result = 24;
          break;
        case 9:
          result = 25;
          break;
        case 10:
          result = 26;
          break;
        case 11:
          result = 27;
          break;
        case 12:
          result = 28;
          break;
        default:
          result = 0xFFFFFFFFLL;
          break;
      }
    }
    else
    {
      return 0xFFFFFFFFLL;
    }
  }
  else
  {
    switch ( a1 )
    {
      case 0:
        result = 12;
        break;
      case 1:
        result = 14;
        break;
      case 2:
        result = 16;
        break;
      case 4:
        result = 19;
        break;
      case 6:
        result = 22;
        break;
      default:
        return 0xFFFFFFFFLL;
    }
  }
  return result;
}
