// Function: sub_7176C0
// Address: 0x7176c0
//
__int64 __fastcall sub_7176C0(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax
  unsigned __int64 v3; // rcx
  unsigned __int16 v4; // cx
  unsigned __int16 v5; // cx
  __int64 v6; // rdx
  unsigned __int16 v7; // cx

  if ( a2 )
    *a2 = 0;
  result = 0;
  if ( !a1 )
    return result;
  if ( *(_BYTE *)(a1 + 174) )
    return result;
  v3 = *(unsigned __int16 *)(a1 + 176);
  if ( !(_WORD)v3 )
    return result;
  if ( (unsigned __int16)v3 > 0x27F6u )
  {
    if ( (_WORD)v3 != 16714 )
    {
      if ( (unsigned __int16)v3 <= 0x414Au )
      {
        if ( (unsigned __int16)v3 > 0x3D02u )
          return (unsigned __int16)(v3 - 15752) < 3u;
        if ( (unsigned __int16)v3 > 0x3CDAu )
        {
          switch ( (__int16)v3 )
          {
            case 15579:
            case 15593:
            case 15596:
            case 15597:
            case 15598:
            case 15601:
            case 15602:
            case 15604:
            case 15608:
            case 15618:
              return 1;
            default:
LABEL_17:
              result = 0;
              break;
          }
        }
        else
        {
          v7 = v3 - 12474;
          if ( v7 <= 0x13u )
            return ((1LL << v7) & 0x80D03) != 0;
        }
        return result;
      }
      if ( (unsigned __int16)v3 > 0x619Fu )
      {
        if ( (_WORD)v3 != 25766 )
          return result;
      }
      else
      {
        if ( (unsigned __int16)v3 <= 0x617Bu )
          return (_WORD)v3 == 16727;
        v6 = 0xC00000079LL;
        LOWORD(v3) = v3 - 24956;
        if ( !_bittest64(&v6, v3) )
          return result;
      }
    }
LABEL_39:
    result = 1;
    if ( a2 )
      *a2 = 1;
    return result;
  }
  if ( (unsigned __int16)v3 > 0x27C5u )
  {
    switch ( (__int16)v3 )
    {
      case 10182:
      case 10183:
      case 10190:
      case 10203:
      case 10214:
      case 10219:
      case 10221:
      case 10222:
      case 10227:
      case 10228:
      case 10229:
      case 10230:
        return 1;
      default:
        goto LABEL_17;
    }
  }
  if ( (unsigned __int16)v3 <= 0x1177u )
  {
    if ( (unsigned __int16)v3 > 0x112Au )
    {
      switch ( (__int16)v3 )
      {
        case 4395:
        case 4396:
        case 4403:
        case 4451:
        case 4454:
        case 4455:
        case 4463:
        case 4464:
        case 4471:
          return 1;
        case 4432:
        case 4461:
          goto LABEL_39;
        default:
          goto LABEL_17;
      }
    }
    if ( (_WORD)v3 != 4174 )
    {
      result = 1;
      if ( (unsigned __int16)v3 <= 0x104Eu )
      {
        if ( (_WORD)v3 != 3387 )
          return (_WORD)v3 == 4139;
      }
      else if ( (_WORD)v3 != 4235 )
      {
        return (unsigned __int16)(v3 - 4286) < 3u;
      }
      return result;
    }
    return 1;
  }
  if ( (unsigned __int16)v3 > 0x1285u )
  {
    v5 = v3 - 4802;
    if ( v5 <= 0x33u )
      return ((1LL << v5) & 0x8180000000001LL) != 0;
  }
  else
  {
    if ( (unsigned __int16)v3 > 0x124Du )
    {
      switch ( (__int16)v3 )
      {
        case 4686:
        case 4687:
        case 4707:
        case 4708:
        case 4715:
        case 4737:
        case 4740:
        case 4741:
          return 1;
        default:
          goto LABEL_17;
      }
    }
    if ( (_WORD)v3 == 4586 )
      return 1;
    if ( (unsigned __int16)v3 <= 0x11EAu )
      v4 = v3 - 4526;
    else
      v4 = v3 - 4589;
    return v4 < 2u;
  }
  return result;
}
