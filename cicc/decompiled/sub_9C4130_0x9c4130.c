// Function: sub_9C4130
// Address: 0x9c4130
//
__int64 __fastcall sub_9C4130(int a1, __int64 a2)
{
  int v2; // eax
  __int64 result; // rax

  v2 = *(unsigned __int8 *)(a2 + 8);
  if ( (unsigned int)(v2 - 17) <= 1 )
    LOBYTE(v2) = *(_BYTE *)(**(_QWORD **)(a2 + 16) + 8LL);
  if ( (unsigned __int8)v2 <= 3u || (_BYTE)v2 == 5 )
  {
    switch ( a1 )
    {
      case 0:
        goto LABEL_12;
      case 1:
        goto LABEL_13;
      case 2:
        goto LABEL_14;
      case 4:
        goto LABEL_11;
      case 6:
        goto LABEL_10;
      default:
        return 0xFFFFFFFFLL;
    }
  }
  if ( (v2 & 0xFD) == 4 )
  {
    switch ( a1 )
    {
      case 0:
LABEL_12:
        result = 14;
        break;
      case 1:
LABEL_13:
        result = 16;
        break;
      case 2:
LABEL_14:
        result = 18;
        break;
      case 4:
LABEL_11:
        result = 21;
        break;
      case 6:
LABEL_10:
        result = 24;
        break;
      default:
        return 0xFFFFFFFFLL;
    }
  }
  else if ( (_BYTE)v2 == 12 )
  {
    switch ( a1 )
    {
      case 0:
        result = 13;
        break;
      case 1:
        result = 15;
        break;
      case 2:
        result = 17;
        break;
      case 3:
        result = 19;
        break;
      case 4:
        result = 20;
        break;
      case 5:
        result = 22;
        break;
      case 6:
        result = 23;
        break;
      case 7:
        result = 25;
        break;
      case 8:
        result = 26;
        break;
      case 9:
        result = 27;
        break;
      case 10:
        result = 28;
        break;
      case 11:
        result = 29;
        break;
      case 12:
        result = 30;
        break;
      default:
        return 0xFFFFFFFFLL;
    }
  }
  else
  {
    return 0xFFFFFFFFLL;
  }
  return result;
}
