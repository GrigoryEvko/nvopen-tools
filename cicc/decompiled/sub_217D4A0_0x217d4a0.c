// Function: sub_217D4A0
// Address: 0x217d4a0
//
__int64 __fastcall sub_217D4A0(unsigned __int16 *a1)
{
  unsigned __int16 v1; // ax
  __int64 result; // rax
  _BOOL4 v3; // eax
  _BOOL4 v4; // eax

  v1 = *a1;
  if ( *a1 > 0x20Fu )
  {
    if ( v1 > 0xC3Eu )
    {
      if ( v1 > 0xD11u )
        return (unsigned __int16)(v1 - 3348) < 0x18u ? 4 : 1;
      if ( v1 <= 0xD08u )
      {
        if ( v1 > 0xC91u )
          return 3 * (unsigned int)(v1 == 3242) + 1;
        else
          return v1 < 0xC74u ? 1 : 4;
      }
    }
    else if ( v1 <= 0xC32u )
    {
      if ( v1 > 0xBE8u )
        v3 = (unsigned __int16)(v1 - 3061) < 6u;
      else
        v3 = v1 > 0xBE2u;
      return (unsigned int)(v3 + 1);
    }
    return 4;
  }
  if ( v1 > 0x1C3u )
  {
    switch ( v1 )
    {
      case 0x1C4u:
      case 0x1C5u:
      case 0x1C6u:
      case 0x1C7u:
      case 0x1C8u:
      case 0x1C9u:
      case 0x1CAu:
      case 0x1CBu:
      case 0x1CCu:
      case 0x1CDu:
      case 0x1CEu:
      case 0x1CFu:
        return 4;
      case 0x1E6u:
      case 0x1E7u:
      case 0x1E8u:
      case 0x1E9u:
      case 0x1EAu:
      case 0x1EBu:
      case 0x1F0u:
      case 0x1F1u:
      case 0x1F2u:
      case 0x1F3u:
      case 0x1F4u:
      case 0x1F5u:
        result = 3;
        break;
      case 0x200u:
      case 0x201u:
      case 0x202u:
      case 0x203u:
      case 0x204u:
      case 0x205u:
      case 0x20Au:
      case 0x20Bu:
      case 0x20Cu:
      case 0x20Du:
      case 0x20Eu:
      case 0x20Fu:
LABEL_26:
        result = 2;
        break;
      default:
LABEL_6:
        result = 1;
        break;
    }
  }
  else
  {
    if ( v1 <= 0x13Au )
    {
      if ( v1 <= 0xE4u )
      {
        switch ( v1 )
        {
          case 0x94u:
          case 0x95u:
          case 0x96u:
          case 0x97u:
            goto LABEL_26;
          case 0x9Bu:
          case 0xA1u:
          case 0xA5u:
          case 0xC1u:
            return 4;
          default:
            goto LABEL_6;
        }
      }
      switch ( v1 )
      {
        case 0xE5u:
        case 0xE6u:
        case 0xE7u:
        case 0xE8u:
        case 0xE9u:
        case 0xEAu:
        case 0xECu:
        case 0xEDu:
        case 0xEEu:
        case 0x106u:
        case 0x107u:
        case 0x108u:
        case 0x109u:
        case 0x10Au:
        case 0x10Bu:
        case 0x10Du:
        case 0x10Eu:
        case 0x10Fu:
        case 0x132u:
        case 0x133u:
        case 0x134u:
        case 0x135u:
        case 0x136u:
        case 0x137u:
        case 0x139u:
        case 0x13Au:
          return 4;
        default:
          goto LABEL_6;
      }
    }
    if ( v1 > 0x1A0u )
      v4 = (unsigned __int16)(v1 - 421) < 6u;
    else
      v4 = v1 > 0x19Au;
    return (unsigned int)(v4 + 1);
  }
  return result;
}
