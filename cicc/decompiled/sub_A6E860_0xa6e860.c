// Function: sub_A6E860
// Address: 0xa6e860
//
__int64 __fastcall sub_A6E860(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  int v3; // r8d

  switch ( a2 )
  {
    case 10LL:
      if ( *(_QWORD *)a1 == 0x696C61636F6C6C61LL && *(_WORD *)(a1 + 8) == 28263 )
        return 1;
      if ( *(_QWORD *)a1 == 0x65677265766E6F63LL && *(_WORD *)(a1 + 8) == 29806 )
        return 6;
      if ( *(_QWORD *)a1 == 0x6968656E696C6E69LL && *(_WORD *)(a1 + 8) == 29806 )
        return 16;
      if ( *(_QWORD *)a1 == 0x61626C6C61636F6ELL && *(_WORD *)(a1 + 8) == 27491 )
        return 24;
      if ( *(_QWORD *)a1 == 0x6568635F66636F6ELL && *(_WORD *)(a1 + 8) == 27491 )
        return 25;
      if ( *(_QWORD *)a1 == 0x7973617466697773LL && *(_WORD *)(a1 + 8) == 25454 )
        return 73;
      if ( *(_QWORD *)a1 == 0x7272657466697773LL && *(_WORD *)(a1 + 8) == 29295 )
        return 74;
      if ( *(_QWORD *)a1 == 0x757465726C6C6977LL && *(_WORD *)(a1 + 8) == 28274 )
        return 76;
      if ( *(_QWORD *)a1 == 0x6174736E67696C61LL && *(_WORD *)(a1 + 8) == 27491 )
        return 94;
      return 0;
    case 8LL:
      result = 2;
      if ( *(_QWORD *)a1 != 0x727470636F6C6C61LL )
      {
        result = 31;
        if ( *(_QWORD *)a1 != 0x656E696C6E696F6ELL )
        {
          result = 36;
          if ( *(_QWORD *)a1 != 0x6E72757465726F6ELL )
          {
            result = 41;
            if ( *(_QWORD *)a1 != 0x646E69776E756F6ELL )
            {
              result = 46;
              if ( *(_QWORD *)a1 != 0x677562656474706FLL )
              {
                result = 50;
                if ( *(_QWORD *)a1 != 0x656E6F6E64616572LL )
                {
                  result = 51;
                  if ( *(_QWORD *)a1 != 0x796C6E6F64616572LL )
                  {
                    result = 52;
                    if ( *(_QWORD *)a1 != 0x64656E7275746572LL )
                    {
                      result = 72;
                      if ( *(_QWORD *)a1 != 0x7066746369727473LL )
                      {
                        result = 77;
                        if ( *(_QWORD *)a1 != 0x656C626174697277LL )
                        {
                          result = 83;
                          if ( *(_QWORD *)a1 != 0x61636F6C6C616E69LL )
                          {
                            result = 89;
                            if ( *(_QWORD *)a1 != 0x7365727574706163LL )
                              return 0;
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      break;
    case 12LL:
      if ( *(_QWORD *)a1 == 0x6E69737961776C61LL && *(_DWORD *)(a1 + 8) == 1701734764 )
        return 3;
      if ( *(_QWORD *)a1 == 0x676F72707473756DLL && *(_DWORD *)(a1 + 8) == 1936942450 )
        return 19;
      if ( *(_QWORD *)a1 == 0x74616C7563657073LL && *(_DWORD *)(a1 + 8) == 1701601889 )
        return 67;
      if ( *(_QWORD *)a1 == 0x636F6C6C61657270LL && *(_DWORD *)(a1 + 8) == 1684370529 )
        return 84;
      if ( *(_QWORD *)a1 == 0x725F656C61637376LL && *(_DWORD *)(a1 + 8) == 1701277281 )
        return 96;
      return 0;
    case 7LL:
      if ( *(_DWORD *)a1 == 1818850658 && *(_WORD *)(a1 + 4) == 26996 && *(_BYTE *)(a1 + 6) == 110 )
        return 4;
      if ( *(_DWORD *)a1 == 1936615789 && *(_WORD *)(a1 + 4) == 31337 && *(_BYTE *)(a1 + 6) == 101 )
        return 18;
      if ( *(_DWORD *)a1 == 1818324846 && *(_WORD *)(a1 + 4) == 24937 && *(_BYTE *)(a1 + 6) == 115 )
        return 22;
      if ( *(_DWORD *)a1 == 1701670766 && *(_WORD *)(a1 + 4) == 26482 && *(_BYTE *)(a1 + 6) == 101 )
        return 32;
      if ( *(_DWORD *)a1 == 1853189998 && *(_WORD *)(a1 + 4) == 25956 && *(_BYTE *)(a1 + 6) == 102 )
        return 40;
      if ( *(_DWORD *)a1 == 1852731246 && *(_WORD *)(a1 + 4) == 27765 && *(_BYTE *)(a1 + 6) == 108 )
        return 43;
      if ( *(_DWORD *)a1 == 1937010799 && *(_WORD *)(a1 + 4) == 31337 && *(_BYTE *)(a1 + 6) == 101 )
        return 47;
      if ( *(_DWORD *)a1 == 1853124719 && *(_WORD *)(a1 + 4) == 28271 && *(_BYTE *)(a1 + 6) == 101 )
        return 48;
      if ( *(_DWORD *)a1 == 1852270963 && *(_WORD *)(a1 + 4) == 30821 && *(_BYTE *)(a1 + 6) == 116 )
        return 54;
      if ( *(_DWORD *)a1 == 1869768058 && *(_WORD *)(a1 + 4) == 30821 && *(_BYTE *)(a1 + 6) == 116 )
        return 79;
      if ( *(_DWORD *)a1 == 1635022709 && *(_WORD *)(a1 + 4) == 27746 && *(_BYTE *)(a1 + 6) == 101 )
        return 95;
      return 0;
    case 4LL:
      result = 5;
      if ( *(_DWORD *)a1 != 1684828003 )
      {
        result = 21;
        if ( *(_DWORD *)a1 != 1953719662 )
        {
          result = 85;
          if ( *(_DWORD *)a1 != 1952805491 )
            return 0;
        }
      }
      break;
    case 31LL:
      v3 = memcmp((const void *)a1, "coro_only_destroy_when_complete", 0x1Fu);
      result = 7;
      if ( v3 )
        return 0;
      break;
    default:
      switch ( a2 )
      {
        case 15LL:
          if ( *(_QWORD *)a1 == 0x696C655F6F726F63LL
            && *(_DWORD *)(a1 + 8) == 1935631716
            && *(_WORD *)(a1 + 12) == 26209
            && *(_BYTE *)(a1 + 14) == 101 )
          {
            return 8;
          }
          if ( *(_QWORD *)a1 == 0x63696C706D696F6ELL
            && *(_DWORD *)(a1 + 8) == 1818653801
            && *(_WORD *)(a1 + 12) == 24943
            && *(_BYTE *)(a1 + 14) == 116 )
          {
            return 30;
          }
          if ( *(_QWORD *)a1 == 0x657A6974696E6173LL
            && *(_DWORD *)(a1 + 8) == 1835363679
            && *(_WORD *)(a1 + 12) == 24948
            && *(_BYTE *)(a1 + 14) == 103 )
          {
            return 58;
          }
          if ( *(_QWORD *)a1 == 0x657A6974696E6173LL
            && *(_DWORD *)(a1 + 8) == 1835363679
            && *(_WORD *)(a1 + 12) == 29295
            && *(_BYTE *)(a1 + 14) == 121 )
          {
            return 59;
          }
          if ( *(_QWORD *)a1 == 0x657A6974696E6173LL
            && *(_DWORD *)(a1 + 8) == 1919448159
            && *(_WORD *)(a1 + 12) == 24933
            && *(_BYTE *)(a1 + 14) == 100 )
          {
            return 63;
          }
          if ( *(_QWORD *)a1 == 0x6163776F64616873LL
            && *(_DWORD *)(a1 + 8) == 1953721452
            && *(_WORD *)(a1 + 12) == 25441
            && *(_BYTE *)(a1 + 14) == 107 )
          {
            return 65;
          }
          if ( *(_QWORD *)a1 == 0x6572656665726564LL
            && *(_DWORD *)(a1 + 8) == 1634034542
            && *(_WORD *)(a1 + 12) == 27746
            && *(_BYTE *)(a1 + 14) == 101 )
          {
            return 90;
          }
          return 0;
        case 14LL:
          if ( *(_QWORD *)a1 == 0x5F6E6F5F64616564LL
            && *(_DWORD *)(a1 + 8) == 1769434741
            && *(_WORD *)(a1 + 12) == 25710 )
          {
            return 9;
          }
          return 0;
        case 33LL:
          if ( !(*(_QWORD *)a1 ^ 0x5F656C6261736964LL | *(_QWORD *)(a1 + 8) ^ 0x657A6974696E6173LL)
            && !(*(_QWORD *)(a1 + 16) ^ 0x757274736E695F72LL | *(_QWORD *)(a1 + 24) ^ 0x6F697461746E656DLL)
            && *(_BYTE *)(a1 + 32) == 110 )
          {
            return 10;
          }
          return 0;
        case 19LL:
          if ( !(*(_QWORD *)a1 ^ 0x745F7465725F6E66LL | *(_QWORD *)(a1 + 8) ^ 0x7478655F6B6E7568LL)
            && *(_WORD *)(a1 + 16) == 29285
            && *(_BYTE *)(a1 + 18) == 110 )
          {
            return 11;
          }
          if ( !(*(_QWORD *)a1 ^ 0x6974696E61736F6ELL | *(_QWORD *)(a1 + 8) ^ 0x7265766F635F657ALL)
            && *(_WORD *)(a1 + 16) == 26465
            && *(_BYTE *)(a1 + 18) == 101 )
          {
            return 38;
          }
          return 0;
        case 3LL:
          if ( *(_WORD *)a1 == 28520 && *(_BYTE *)(a1 + 2) == 116 )
            return 12;
          if ( *(_WORD *)a1 == 29555 && *(_BYTE *)(a1 + 2) == 112 )
            return 69;
          return 0;
        case 16LL:
          if ( !(*(_QWORD *)a1 ^ 0x705F646972627968LL | *(_QWORD *)(a1 + 8) ^ 0x656C626168637461LL) )
            return 13;
          if ( !(*(_QWORD *)a1 ^ 0x657A6974696E6173LL | *(_QWORD *)(a1 + 8) ^ 0x737365726464615FLL) )
            return 56;
          return 0;
        case 6LL:
          if ( *(_DWORD *)a1 == 1634561385 && *(_WORD *)(a1 + 4) == 26482 )
            return 14;
          if ( *(_DWORD *)a1 == 1919315822 && *(_WORD *)(a1 + 4) == 25957 )
            return 29;
          if ( *(_DWORD *)a1 == 2037608302 && *(_WORD *)(a1 + 4) == 25454 )
            return 39;
          if ( *(_DWORD *)a1 == 1919972211 && *(_WORD *)(a1 + 4) == 29029 )
            return 70;
          if ( *(_DWORD *)a1 == 1869440365 && *(_WORD *)(a1 + 4) == 31090 )
            return 92;
          return 0;
        case 5LL:
          if ( *(_DWORD *)a1 == 1701998185 && *(_BYTE *)(a1 + 4) == 103 )
            return 15;
          if ( *(_DWORD *)a1 == 1701536110 && *(_BYTE *)(a1 + 4) == 100 )
            return 20;
          if ( *(_DWORD *)a1 == 2019913582 && *(_BYTE *)(a1 + 4) == 116 )
            return 28;
          if ( *(_DWORD *)a1 == 1702000994 && *(_BYTE *)(a1 + 4) == 102 )
            return 80;
          if ( *(_DWORD *)a1 == 1635154274 && *(_BYTE *)(a1 + 4) == 108 )
            return 81;
          if ( *(_DWORD *)a1 == 1734962273 && *(_BYTE *)(a1 + 4) == 110 )
            return 86;
          if ( *(_DWORD *)a1 == 1735287154 && *(_BYTE *)(a1 + 4) == 101 )
            return 97;
          return 0;
      }
      if ( a2 != 9 )
      {
        switch ( a2 )
        {
          case 18LL:
            if ( !(*(_QWORD *)a1 ^ 0x6772657669646F6ELL | *(_QWORD *)(a1 + 8) ^ 0x72756F7365636E65LL)
              && *(_WORD *)(a1 + 16) == 25955 )
            {
              return 26;
            }
            if ( !(*(_QWORD *)a1 ^ 0x657A6974696E6173LL | *(_QWORD *)(a1 + 8) ^ 0x657264646177685FLL)
              && *(_WORD *)(a1 + 16) == 29555 )
            {
              return 57;
            }
            break;
          case 11LL:
            if ( *(_QWORD *)a1 == 0x63696C7075646F6ELL && *(_WORD *)(a1 + 8) == 29793 && *(_BYTE *)(a1 + 10) == 101 )
              return 27;
            if ( *(_QWORD *)a1 == 0x62797A616C6E6F6ELL && *(_WORD *)(a1 + 8) == 28265 && *(_BYTE *)(a1 + 10) == 100 )
              return 42;
            if ( *(_QWORD *)a1 == 0x666F727070696B73LL && *(_WORD *)(a1 + 8) == 27753 && *(_BYTE *)(a1 + 10) == 101 )
              return 66;
            if ( *(_QWORD *)a1 == 0x74746E656D656C65LL && *(_WORD *)(a1 + 8) == 28793 && *(_BYTE *)(a1 + 10) == 101 )
              return 82;
            if ( *(_QWORD *)a1 == 0x696C616974696E69LL && *(_WORD *)(a1 + 8) == 25978 && *(_BYTE *)(a1 + 10) == 115 )
              return 98;
            break;
          case 17LL:
            if ( !(*(_QWORD *)a1 ^ 0x6974696E61736F6ELL | *(_QWORD *)(a1 + 8) ^ 0x646E756F625F657ALL)
              && *(_BYTE *)(a1 + 16) == 115 )
            {
              return 37;
            }
            if ( !(*(_QWORD *)a1 ^ 0x74696C7073657270LL | *(_QWORD *)(a1 + 8) ^ 0x6E6974756F726F63LL)
              && *(_BYTE *)(a1 + 16) == 101 )
            {
              return 49;
            }
            if ( !(*(_QWORD *)a1 ^ 0x657A6974696E6173LL | *(_QWORD *)(a1 + 8) ^ 0x6D69746C6165725FLL)
              && *(_BYTE *)(a1 + 16) == 101 )
            {
              return 61;
            }
            break;
          case 21LL:
            if ( !(*(_QWORD *)a1 ^ 0x696F705F6C6C756ELL | *(_QWORD *)(a1 + 8) ^ 0x5F73695F7265746ELL)
              && *(_DWORD *)(a1 + 16) == 1768710518
              && *(_BYTE *)(a1 + 20) == 100 )
            {
              return 44;
            }
            break;
          case 13LL:
            if ( *(_QWORD *)a1 == 0x7566726F6674706FLL
              && *(_DWORD *)(a1 + 8) == 1852406394
              && *(_BYTE *)(a1 + 12) == 103 )
            {
              return 45;
            }
            if ( *(_QWORD *)a1 == 0x5F736E7275746572LL
              && *(_DWORD *)(a1 + 8) == 1667856244
              && *(_BYTE *)(a1 + 12) == 101 )
            {
              return 53;
            }
            if ( *(_QWORD *)a1 == 0x657A6974696E6173LL
              && *(_DWORD *)(a1 + 8) == 1887007839
              && *(_BYTE *)(a1 + 12) == 101 )
            {
              return 64;
            }
            break;
          case 28LL:
            if ( !(*(_QWORD *)a1 ^ 0x657A6974696E6173LL | *(_QWORD *)(a1 + 8) ^ 0x636972656D756E5FLL)
              && *(_QWORD *)(a1 + 16) == 0x69626174735F6C61LL
              && *(_DWORD *)(a1 + 24) == 2037672300 )
            {
              return 60;
            }
            break;
          case 26LL:
            if ( !(*(_QWORD *)a1 ^ 0x657A6974696E6173LL | *(_QWORD *)(a1 + 8) ^ 0x6D69746C6165725FLL)
              && *(_QWORD *)(a1 + 16) == 0x696B636F6C625F65LL
              && *(_WORD *)(a1 + 24) == 26478 )
            {
              return 62;
            }
            if ( !(*(_QWORD *)a1 ^ 0x74616C7563657073LL | *(_QWORD *)(a1 + 8) ^ 0x64616F6C5F657669LL)
              && *(_QWORD *)(a1 + 16) == 0x696E65647261685FLL
              && *(_WORD *)(a1 + 24) == 26478 )
            {
              return 68;
            }
            break;
          default:
            if ( a2 == 23
              && !(*(_QWORD *)a1 ^ 0x6572656665726564LL | *(_QWORD *)(a1 + 8) ^ 0x5F656C626165636ELL)
              && *(_DWORD *)(a1 + 16) == 1851748975
              && *(_WORD *)(a1 + 20) == 27765
              && *(_BYTE *)(a1 + 22) == 108 )
            {
              return 91;
            }
            break;
        }
        return 0;
      }
      if ( *(_QWORD *)a1 == 0x6C626174706D756ALL && *(_BYTE *)(a1 + 8) == 101 )
        return 17;
      if ( *(_QWORD *)a1 == 0x69746C6975626F6ELL && *(_BYTE *)(a1 + 8) == 110 )
        return 23;
      if ( *(_QWORD *)a1 == 0x6C69666F72706F6ELL && *(_BYTE *)(a1 + 8) == 101 )
        return 33;
      if ( *(_QWORD *)a1 != 0x7372756365726F6ELL || *(_BYTE *)(a1 + 8) != 101 )
      {
        if ( *(_QWORD *)a1 == 0x6E6F7A6465726F6ELL && *(_BYTE *)(a1 + 8) == 101 )
          return 35;
        if ( *(_QWORD *)a1 == 0x6361747365666173LL && *(_BYTE *)(a1 + 8) == 107 )
          return 55;
        if ( *(_QWORD *)a1 == 0x6E6F727473707373LL && *(_BYTE *)(a1 + 8) == 103 )
          return 71;
        if ( *(_QWORD *)a1 == 0x6C65737466697773LL && *(_BYTE *)(a1 + 8) == 102 )
          return 75;
        if ( *(_QWORD *)a1 == 0x6C6E6F6574697277LL && *(_BYTE *)(a1 + 8) == 121 )
          return 78;
        if ( *(_QWORD *)a1 == 0x6E696B636F6C6C61LL && *(_BYTE *)(a1 + 8) == 100 )
          return 87;
        if ( *(_QWORD *)a1 == 0x7A6973636F6C6C61LL && *(_BYTE *)(a1 + 8) == 101 )
          return 88;
        if ( *(_QWORD *)a1 == 0x73616C6370666F6ELL && *(_BYTE *)(a1 + 8) == 115 )
          return 93;
        return 0;
      }
      return 34;
  }
  return result;
}
