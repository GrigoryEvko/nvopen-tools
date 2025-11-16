// Function: sub_2D05690
// Address: 0x2d05690
//
char __fastcall sub_2D05690(_BYTE *a1, __int64 a2)
{
  char result; // al
  __int64 v3; // rdx
  __int64 v4; // rdx
  unsigned int v5; // ecx
  __int64 v6; // rdx
  _QWORD *v7; // rax

  result = sub_2D04210(a2);
  if ( !result && *(_BYTE *)a2 > 0x1Cu )
  {
    switch ( *(_BYTE *)a2 )
    {
      case '*':
      case ',':
      case '.':
      case '6':
      case '7':
      case '8':
      case '9':
      case ':':
      case ';':
      case '?':
      case 'C':
      case 'D':
      case 'E':
      case 'L':
      case 'M':
      case 'N':
      case 'O':
      case 'R':
      case 'S':
      case 'V':
      case 'Z':
      case '[':
      case '\\':
      case ']':
      case '^':
        goto LABEL_4;
      case '+':
      case '-':
      case '/':
      case '2':
      case '5':
      case 'F':
      case 'G':
      case 'H':
      case 'I':
      case 'J':
      case 'K':
        result = a1[61];
        break;
      case '0':
      case '1':
      case '3':
      case '4':
        result = a1[60];
        break;
      case '=':
        v4 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
        if ( *(_BYTE *)(v4 + 8) == 14 )
          result = *(_DWORD *)(v4 + 8) >> 8 == 4 || *(_DWORD *)(v4 + 8) >> 8 == 101;
        break;
      case 'U':
        v3 = *(_QWORD *)(a2 - 32);
        if ( *(_BYTE *)v3 == 25 )
        {
          result = a1[62];
          if ( result )
            result = *(_BYTE *)(v3 + 96) ^ 1;
        }
        else if ( !*(_BYTE *)v3 && *(_QWORD *)(v3 + 24) == *(_QWORD *)(a2 + 80) && (*(_BYTE *)(v3 + 33) & 0x20) != 0 )
        {
          v5 = *(_DWORD *)(v3 + 36);
          if ( v5 > 0x24D5 )
          {
            if ( v5 > 0x256A )
            {
              result = v5 == 10078;
            }
            else if ( v5 > 0x2539 )
            {
              result = ((1LL << ((unsigned __int8)v5 - 58)) & 0x140000000000FLL) != 0;
            }
            else
            {
              result = v5 - 9466 <= 3;
            }
          }
          else if ( v5 > 0x248A )
          {
            switch ( v5 )
            {
              case 0x248Bu:
              case 0x248Cu:
              case 0x248Du:
              case 0x2490u:
              case 0x2491u:
              case 0x2492u:
              case 0x249Au:
              case 0x249Bu:
              case 0x249Cu:
              case 0x249Eu:
              case 0x24C2u:
              case 0x24C3u:
              case 0x24C4u:
              case 0x24C6u:
              case 0x24C7u:
              case 0x24C8u:
              case 0x24C9u:
              case 0x24CAu:
              case 0x24CBu:
              case 0x24CCu:
              case 0x24CDu:
              case 0x24CEu:
              case 0x24CFu:
              case 0x24D0u:
              case 0x24D1u:
              case 0x24D2u:
              case 0x24D3u:
              case 0x24D4u:
              case 0x24D5u:
                goto LABEL_4;
              case 0x24A0u:
                v6 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
                v7 = *(_QWORD **)(v6 + 24);
                if ( *(_DWORD *)(v6 + 32) > 0x40u )
                  v7 = (_QWORD *)*v7;
                result = (unsigned int)((_DWORD)v7 - 38) <= 0xB;
                break;
              default:
                return result;
            }
          }
          else if ( v5 > 0x2245 )
          {
            if ( v5 > 0x2431 )
            {
              result = v5 - 9307 <= 2;
            }
            else if ( v5 > 0x23C1 )
            {
              switch ( v5 )
              {
                case 0x23C2u:
                case 0x23C4u:
                case 0x23C6u:
                case 0x23C7u:
                case 0x23C9u:
                case 0x23CAu:
                case 0x23CBu:
                case 0x23CCu:
                case 0x23CDu:
                case 0x23CEu:
                case 0x23CFu:
                case 0x23D0u:
                case 0x23D1u:
                case 0x23D2u:
                case 0x23D3u:
                case 0x23D4u:
                case 0x23D5u:
                case 0x23D6u:
                case 0x23D8u:
                case 0x23D9u:
                case 0x23DAu:
                case 0x23DBu:
                case 0x23DCu:
                case 0x23DDu:
                case 0x23DEu:
                case 0x23DFu:
                case 0x2412u:
                case 0x2422u:
                case 0x242Bu:
                case 0x242Eu:
                case 0x2430u:
                case 0x2431u:
                  goto LABEL_4;
                default:
                  return result;
              }
            }
            else if ( v5 > 0x232B )
            {
              if ( v5 > 0x2334 )
                result = v5 - 9060 <= 1;
              else
                result = v5 >= 0x2333;
            }
            else if ( v5 > 0x2323 || v5 == 8923 )
            {
LABEL_4:
              result = 1;
            }
            else if ( v5 <= 0x22DB )
            {
              result = v5 == 8817;
            }
            else
            {
              result = v5 - 8962 <= 4;
            }
          }
          else if ( v5 > 0x214B )
          {
            switch ( v5 )
            {
              case 0x214Cu:
              case 0x214Du:
              case 0x2151u:
              case 0x218Du:
              case 0x218Eu:
              case 0x2190u:
              case 0x21C1u:
              case 0x21C2u:
              case 0x21C7u:
              case 0x21C9u:
              case 0x21CDu:
              case 0x21CEu:
              case 0x21D0u:
              case 0x21D6u:
              case 0x21D7u:
              case 0x21DDu:
              case 0x21F3u:
              case 0x21F4u:
              case 0x21F6u:
              case 0x21FAu:
              case 0x21FBu:
              case 0x21FDu:
              case 0x2200u:
              case 0x2205u:
              case 0x2206u:
              case 0x2207u:
              case 0x220Bu:
              case 0x220Eu:
              case 0x2237u:
              case 0x223Cu:
              case 0x223Du:
              case 0x223Eu:
              case 0x2242u:
              case 0x2245u:
                goto LABEL_4;
              default:
                return result;
            }
          }
          else if ( v5 > 0x1FEA )
          {
            if ( v5 > 0x2076 )
              result = v5 - 8453 <= 2;
            else
              result = v5 >= 0x2073;
          }
          else if ( v5 > 0x1FC2 )
          {
            switch ( v5 )
            {
              case 0x1FC3u:
              case 0x1FC5u:
              case 0x1FC9u:
              case 0x1FCAu:
              case 0x1FCCu:
              case 0x1FD0u:
              case 0x1FD1u:
              case 0x1FD3u:
              case 0x1FD7u:
              case 0x1FD8u:
              case 0x1FDAu:
              case 0x1FDEu:
              case 0x1FDFu:
              case 0x1FE1u:
              case 0x1FEAu:
                goto LABEL_4;
              default:
                return result;
            }
          }
          else if ( v5 <= 0x173 )
          {
            if ( v5 > 0x136 )
            {
              result = ((1LL << ((unsigned __int8)v5 - 55)) & 0x1001000008000001LL) != 0;
            }
            else
            {
              result = 1;
              if ( v5 != 14 && v5 != 173 )
                result = v5 == 1;
            }
          }
        }
        break;
      default:
        return result;
    }
  }
  return result;
}
