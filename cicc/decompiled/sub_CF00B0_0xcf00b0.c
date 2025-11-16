// Function: sub_CF00B0
// Address: 0xcf00b0
//
__int64 __fastcall sub_CF00B0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rsi
  __int64 v4; // rdi
  __int64 result; // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  bool v8; // zf
  __int64 v9; // rdx
  __int64 v10; // rdx
  unsigned int v11; // ecx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned int v15; // eax
  _BYTE **v16; // rdx
  unsigned int v17; // ecx
  __int64 v18; // [rsp+0h] [rbp-20h] BYREF
  __int64 v19; // [rsp+8h] [rbp-18h]

  v2 = sub_B43CA0(a1);
  v3 = *(_QWORD *)(a1 + 8);
  v4 = v2 + 312;
  switch ( *(_BYTE *)a1 )
  {
    case '"':
    case 'U':
      v10 = *(_QWORD *)(a1 - 32);
      if ( *(_BYTE *)v10 == 25 )
        return 10;
      result = 5000;
      if ( *(_BYTE *)v10 || *(_QWORD *)(v10 + 24) != *(_QWORD *)(a1 + 80) || (*(_BYTE *)(v10 + 33) & 0x20) == 0 )
        return result;
      v11 = *(_DWORD *)(v10 + 36);
      if ( v11 > 0x2334 )
      {
        if ( v11 > 0x249E )
        {
          if ( v11 > 0x256A )
          {
            result = 4;
            if ( v11 != 10078 )
            {
              result = 16;
              if ( v11 > 0x275D )
              {
                result = 1;
                if ( v11 != 10641 )
                {
                  result = 16;
                  if ( v11 == 10648 )
                    return 1;
                }
              }
            }
          }
          else if ( v11 > 0x2538 )
          {
            switch ( v11 )
            {
              case 0x2539u:
              case 0x253Eu:
              case 0x2540u:
              case 0x2541u:
              case 0x2542u:
              case 0x2543u:
              case 0x2544u:
              case 0x2545u:
              case 0x2546u:
              case 0x2547u:
              case 0x2548u:
              case 0x2549u:
              case 0x254Au:
              case 0x254Bu:
              case 0x254Cu:
LABEL_41:
                result = 32;
                break;
              case 0x253Cu:
              case 0x253Du:
                return 4;
              case 0x2568u:
              case 0x256Au:
                return 1;
              default:
LABEL_39:
                result = 16;
                break;
            }
          }
          else
          {
            switch ( v11 )
            {
              case 0x24C3u:
              case 0x24C6u:
              case 0x24C7u:
              case 0x24C8u:
              case 0x24FAu:
              case 0x24FBu:
              case 0x24FCu:
              case 0x24FDu:
                return 4;
              case 0x24C9u:
              case 0x24CAu:
              case 0x24CBu:
              case 0x24CCu:
              case 0x24CDu:
              case 0x24CEu:
              case 0x24CFu:
              case 0x24D0u:
              case 0x24D1u:
              case 0x24D3u:
              case 0x24D4u:
              case 0x24D5u:
                return 1;
              case 0x24D2u:
LABEL_50:
                result = 2;
                break;
              default:
                goto LABEL_39;
            }
          }
        }
        else
        {
          if ( v11 > 0x245A )
          {
            switch ( v11 )
            {
              case 0x245Bu:
              case 0x245Cu:
              case 0x245Du:
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
                return 6;
              default:
                goto LABEL_39;
            }
          }
          if ( v11 > 0x2365 )
          {
            switch ( v11 )
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
                return 1;
              case 0x23D7u:
              case 0x242Au:
              case 0x2432u:
              case 0x2433u:
              case 0x2434u:
              case 0x2435u:
              case 0x2436u:
              case 0x2437u:
              case 0x2438u:
              case 0x2439u:
              case 0x243Au:
              case 0x243Bu:
              case 0x243Cu:
              case 0x243Du:
              case 0x243Eu:
                goto LABEL_41;
              case 0x242Eu:
              case 0x2430u:
              case 0x2431u:
                return 4;
              default:
                goto LABEL_39;
            }
          }
          return v11 < 0x2364 ? 16 : 1;
        }
      }
      else
      {
        if ( v11 > 0x2302 )
        {
          switch ( v11 )
          {
            case 0x2303u:
            case 0x2305u:
            case 0x2306u:
              return 4;
            case 0x2324u:
            case 0x2325u:
            case 0x2326u:
            case 0x2327u:
            case 0x2328u:
            case 0x2329u:
            case 0x232Au:
            case 0x232Bu:
            case 0x2333u:
            case 0x2334u:
              return 1;
            case 0x232Cu:
              goto LABEL_41;
            default:
              goto LABEL_39;
          }
        }
        if ( v11 > 0x2245 )
        {
          result = 16;
          if ( v11 == 8817 )
            return 1;
        }
        else
        {
          if ( v11 > 0x214B )
          {
            switch ( v11 )
            {
              case 0x214Cu:
              case 0x2151u:
                return 4;
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
              case 0x2207u:
              case 0x220Bu:
              case 0x220Eu:
              case 0x2237u:
              case 0x223Eu:
              case 0x2242u:
              case 0x2245u:
                return 1;
              case 0x2205u:
              case 0x2206u:
              case 0x223Cu:
              case 0x223Du:
                goto LABEL_50;
              default:
                goto LABEL_39;
            }
          }
          if ( v11 > 0x1FEA )
          {
            if ( v11 > 0x2076 )
              return v11 - 8453 < 3 ? 1 : 16;
            else
              return v11 < 0x2073 ? 16 : 4;
          }
          else
          {
            if ( v11 > 0x1FC2 )
            {
              switch ( v11 )
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
                  return 1;
                case 0x1FEAu:
                  return 0;
                default:
                  goto LABEL_39;
              }
            }
            if ( v11 == 1 )
            {
              return 1;
            }
            else
            {
              v17 = v11 - 311;
              result = 16;
              if ( v17 <= 0x3C )
                return ((1LL << v17) & 0x1001000008000001LL) == 0 ? 16 : 1;
            }
          }
        }
      }
      return result;
    case '$':
    case '<':
    case 'L':
    case 'M':
    case 'N':
    case 'Z':
    case '[':
    case '\\':
    case ']':
    case '^':
      return 0;
    case '+':
    case '-':
    case '/':
      v18 = sub_9208B0(v4, v3);
      v19 = v9;
      return 4 * (unsigned int)(sub_CA1930(&v18) == 64) + 1;
    case '0':
    case '1':
    case '3':
    case '4':
      v6 = sub_9208B0(v4, v3);
      v19 = v7;
      v18 = v6;
      v8 = sub_CA1930(&v18) == 64;
      result = 64;
      if ( !v8 )
        return 32;
      return result;
    case '2':
    case '5':
      v12 = sub_9208B0(v4, v3);
      v19 = v13;
      v18 = v12;
      v8 = sub_CA1930(&v18) == 64;
      result = 200;
      if ( !v8 )
        return 100;
      return result;
    case '=':
      v14 = *(_QWORD *)(*(_QWORD *)(a1 - 32) + 8LL);
      if ( (unsigned int)*(unsigned __int8 *)(v14 + 8) - 17 <= 1 )
        v14 = **(_QWORD **)(v14 + 16);
      v15 = *(_DWORD *)(v14 + 8) >> 8;
      if ( v15 == 3 )
        return 10;
      if ( v15 <= 3 )
      {
        if ( v15 == 2 )
          goto LABEL_69;
      }
      else if ( v15 != 5 )
      {
        if ( v15 == 101 || v15 == 4 )
          return 6;
LABEL_69:
        BUG();
      }
      return 36;
    case '>':
      return 6;
    case '?':
      result = 1;
      v16 = (_BYTE **)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
      if ( (_BYTE **)a1 != v16 )
      {
        do
        {
          result = (unsigned int)result - ((**v16 < 0x1Du) - 1);
          v16 += 4;
        }
        while ( v16 != (_BYTE **)a1 );
      }
      return result;
    case 'A':
    case 'B':
      return 75;
    case 'F':
    case 'G':
    case 'H':
    case 'I':
    case 'J':
    case 'K':
      return 4;
    default:
      return 1;
  }
}
