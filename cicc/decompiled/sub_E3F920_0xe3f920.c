// Function: sub_E3F920
// Address: 0xe3f920
//
__int64 __fastcall sub_E3F920(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx
  unsigned int v3; // edx

  switch ( *(_BYTE *)a1 )
  {
    case '+':
      result = 102;
      break;
    case '-':
      result = 115;
      break;
    case '/':
      result = 108;
      break;
    case '2':
      result = 105;
      break;
    case '5':
      result = 114;
      break;
    case 'F':
      result = 112;
      break;
    case 'G':
      result = 111;
      break;
    case 'H':
      result = 141;
      break;
    case 'I':
      result = 136;
      break;
    case 'J':
      result = 113;
      break;
    case 'K':
      result = 110;
      break;
    case 'S':
      result = 103;
      break;
    case 'U':
      v2 = *(_QWORD *)(a1 - 32);
      result = 0;
      if ( v2 )
      {
        if ( !*(_BYTE *)v2 && *(_QWORD *)(v2 + 24) == *(_QWORD *)(a1 + 80) && (*(_BYTE *)(v2 + 33) & 0x20) != 0 )
        {
          v3 = *(_DWORD *)(v2 + 36);
          if ( v3 <= 0x163 )
          {
            if ( v3 <= 0xAB )
            {
              if ( v3 > 0x15 )
              {
                result = 100;
                if ( v3 != 88 )
                {
                  if ( v3 <= 0x58 )
                  {
                    result = 98;
                    if ( v3 != 63 )
                    {
                      result = 0;
                      if ( v3 == 64 )
                        result = 99;
                    }
                  }
                  else
                  {
                    result = 0;
                    if ( v3 == 90 )
                      result = 101;
                  }
                }
              }
              else if ( v3 > 1 )
              {
                switch ( v3 )
                {
                  case 2u:
                    result = 93;
                    break;
                  case 0xAu:
                    result = 94;
                    break;
                  case 0xCu:
                    result = 95;
                    break;
                  case 0xDu:
                    result = 96;
                    break;
                  case 0x15u:
                    result = 97;
                    break;
                  default:
                    goto LABEL_2;
                }
              }
            }
            else
            {
              switch ( v3 )
              {
                case 0xACu:
                  result = 106;
                  break;
                case 0xADu:
                  result = 107;
                  break;
                case 0xAEu:
                  result = 109;
                  break;
                case 0xD1u:
                  result = 116;
                  break;
                case 0xD4u:
                  result = 117;
                  break;
                case 0xD5u:
                  result = 118;
                  break;
                case 0xDAu:
                  result = 119;
                  break;
                case 0xDBu:
                  result = 120;
                  break;
                case 0xDCu:
                  result = 121;
                  break;
                case 0xDFu:
                  result = 122;
                  break;
                case 0xE0u:
                  result = 123;
                  break;
                case 0xEBu:
                  result = 124;
                  break;
                case 0xEDu:
                  result = 125;
                  break;
                case 0xF6u:
                  result = 126;
                  break;
                case 0xF8u:
                  result = 127;
                  break;
                case 0xFAu:
                  result = 128;
                  break;
                case 0x11Cu:
                  result = 129;
                  break;
                case 0x11Du:
                  result = 130;
                  break;
                case 0x134u:
                  result = 131;
                  break;
                case 0x135u:
                  result = 132;
                  break;
                case 0x136u:
                  result = 133;
                  break;
                case 0x145u:
                  result = 134;
                  break;
                case 0x148u:
                  result = 135;
                  break;
                case 0x14Fu:
                  result = 137;
                  break;
                case 0x15Cu:
                  result = 138;
                  break;
                case 0x15Du:
                  result = 139;
                  break;
                case 0x163u:
                  result = 140;
                  break;
                default:
                  goto LABEL_2;
              }
            }
          }
        }
      }
      break;
    default:
LABEL_2:
      result = 0;
      break;
  }
  return result;
}
