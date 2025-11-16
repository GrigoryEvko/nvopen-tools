// Function: sub_34CC720
// Address: 0x34cc720
//
__int64 __fastcall sub_34CC720(__int16 a1, char a2)
{
  __int64 result; // rax

  result = ~a1 & 0x3FF;
  if ( (unsigned int)result > 0x108 )
  {
    if ( (_DWORD)result != 448 )
    {
      switch ( ~a1 & 0x3FF )
      {
        case 0x1F8:
        case 0x200:
        case 0x204:
          return result;
        case 0x203:
        case 0x207:
LABEL_17:
          if ( !a2 )
            result = 0;
          break;
        default:
          return 0;
      }
    }
  }
  else if ( (unsigned int)result > 0xEF )
  {
    if ( ((1LL << (15 - (unsigned __int8)a1)) & 0x1010009) == 0 )
      return 0;
  }
  else if ( (unsigned int)result > 0x40 )
  {
    if ( (unsigned int)(result - 96) > 0x30 )
    {
      return 0;
    }
    else if ( ((1LL << (-(char)a1 - 97)) & 0x1000100000009LL) == 0 )
    {
      return 0;
    }
  }
  else if ( (~a1 & 0x3FF) != 0 )
  {
    switch ( ~a1 & 0x3FF )
    {
      case 1:
      case 2:
      case 3:
      case 4:
      case 8:
      case 0x10:
      case 0x20:
      case 0x38:
      case 0x40:
        return result;
      case 7:
        goto LABEL_17;
      default:
        return 0;
    }
  }
  return result;
}
