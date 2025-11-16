// Function: sub_CC4B20
// Address: 0xcc4b20
//
__int64 __fastcall sub_CC4B20(__int64 a1, unsigned __int64 a2)
{
  __int64 result; // rax

  if ( a2 <= 5 )
  {
    if ( a2 <= 3 || *(_DWORD *)a1 != 1768055141 )
      goto LABEL_9;
    return 15;
  }
  if ( *(_DWORD *)a1 == 1768055141 && *(_WORD *)(a1 + 4) == 26216 )
    return 16;
  if ( *(_DWORD *)a1 == 1768055141 )
    return 15;
  if ( a2 <= 8 )
  {
    if ( a2 <= 7 )
      goto LABEL_9;
  }
  else if ( *(_QWORD *)a1 == 0x336E696261756E67LL && *(_BYTE *)(a1 + 8) == 50 )
  {
    return 3;
  }
  if ( *(_QWORD *)a1 == 0x3436696261756E67LL )
    return 4;
  if ( a2 > 0xB )
  {
    if ( *(_QWORD *)a1 == 0x6869626165756E67LL && *(_DWORD *)(a1 + 8) == 875983974 )
      return 8;
    if ( *(_QWORD *)a1 == 0x6869626165756E67LL && *(_BYTE *)(a1 + 8) == 102 )
      return 7;
LABEL_108:
    if ( *(_QWORD *)a1 == 0x7469626165756E67LL && *(_WORD *)(a1 + 8) == 13366 )
      return 6;
    goto LABEL_109;
  }
  if ( a2 > 8 )
  {
    if ( *(_QWORD *)a1 == 0x6869626165756E67LL && *(_BYTE *)(a1 + 8) == 102 )
      return 7;
    if ( a2 == 9 )
      goto LABEL_109;
    goto LABEL_108;
  }
LABEL_9:
  if ( a2 <= 6 )
  {
    if ( a2 != 6 )
      goto LABEL_11;
    goto LABEL_110;
  }
LABEL_109:
  if ( *(_DWORD *)a1 == 1702194791 && *(_WORD *)(a1 + 4) == 25185 && *(_BYTE *)(a1 + 6) == 105 )
    return 5;
LABEL_110:
  if ( *(_DWORD *)a1 == 1718972007 && *(_WORD *)(a1 + 4) == 12851 )
    return 9;
  if ( *(_DWORD *)a1 == 1718972007 && *(_WORD *)(a1 + 4) == 13366 )
    return 10;
LABEL_11:
  if ( a2 > 4 )
  {
    if ( *(_DWORD *)a1 == 1937075815 && *(_BYTE *)(a1 + 4) == 102 )
      return 11;
    if ( a2 == 5 )
      goto LABEL_20;
    if ( *(_DWORD *)a1 == 2020961895 && *(_WORD *)(a1 + 4) == 12851 )
      return 12;
    if ( a2 > 8 && *(_QWORD *)a1 == 0x33706C695F756E67LL && *(_BYTE *)(a1 + 8) == 50 )
      return 13;
    if ( *(_DWORD *)a1 == 1701080931 && *(_WORD *)(a1 + 4) == 13873 )
      return 14;
    if ( *(_DWORD *)a1 == 1953853031 && *(_WORD *)(a1 + 4) == 13366 )
      return 2;
  }
  if ( a2 <= 2 )
    goto LABEL_54;
LABEL_20:
  if ( *(_WORD *)a1 == 28263 && *(_BYTE *)(a1 + 2) == 117 )
    return 1;
  if ( a2 <= 6 )
    goto LABEL_54;
  if ( *(_DWORD *)a1 == 1919184481 && *(_WORD *)(a1 + 4) == 26991 && *(_BYTE *)(a1 + 6) == 100 )
    return 17;
  if ( a2 <= 9 )
  {
LABEL_54:
    if ( a2 <= 8 )
      goto LABEL_55;
    goto LABEL_25;
  }
  if ( *(_QWORD *)a1 == 0x6E6962616C73756DLL && *(_WORD *)(a1 + 8) == 12851 )
    return 19;
LABEL_25:
  if ( *(_QWORD *)a1 == 0x366962616C73756DLL && *(_BYTE *)(a1 + 8) == 52 )
    return 20;
  if ( a2 > 9 )
  {
    if ( *(_QWORD *)a1 == 0x696261656C73756DLL && *(_WORD *)(a1 + 8) == 26216 )
      return 22;
    goto LABEL_28;
  }
LABEL_55:
  if ( a2 > 7 )
  {
LABEL_28:
    if ( *(_QWORD *)a1 == 0x696261656C73756DLL )
      return 21;
    goto LABEL_29;
  }
  if ( a2 == 7 )
  {
LABEL_29:
    if ( *(_DWORD *)a1 == 1819506029 && *(_WORD *)(a1 + 4) == 13158 && *(_BYTE *)(a1 + 6) == 50 )
      return 23;
    if ( *(_DWORD *)a1 == 1819506029 && *(_WORD *)(a1 + 4) == 26227 )
      return 24;
    if ( *(_DWORD *)a1 == 1819506029 && *(_WORD *)(a1 + 4) == 13176 && *(_BYTE *)(a1 + 6) == 50 )
      return 25;
    goto LABEL_32;
  }
  if ( a2 == 6 )
  {
    if ( *(_DWORD *)a1 == 1819506029 && *(_WORD *)(a1 + 4) == 26227 )
      return 24;
  }
  else if ( a2 <= 3 )
  {
    goto LABEL_59;
  }
LABEL_32:
  if ( *(_DWORD *)a1 == 1819506029 )
    return 18;
  result = 27;
  if ( *(_DWORD *)a1 != 1668707181 )
  {
    if ( a2 > 6 )
    {
      if ( *(_DWORD *)a1 == 1851880553 && *(_WORD *)(a1 + 4) == 30057 && *(_BYTE *)(a1 + 6) == 109 )
        return 28;
      if ( *(_DWORD *)a1 == 1852275043 && *(_WORD *)(a1 + 4) == 29557 )
        return 29;
      if ( *(_DWORD *)a1 == 1701998435 && *(_WORD *)(a1 + 4) == 27747 && *(_BYTE *)(a1 + 6) == 114 )
        return 30;
      if ( a2 > 8 && *(_QWORD *)a1 == 0x6F74616C756D6973LL && *(_BYTE *)(a1 + 8) == 114 )
        return 31;
      goto LABEL_40;
    }
    if ( a2 == 6 )
    {
      if ( *(_DWORD *)a1 == 1852275043 && *(_WORD *)(a1 + 4) == 29557 )
        return 29;
LABEL_40:
      if ( *(_DWORD *)a1 == 1633902957 && *(_WORD *)(a1 + 4) == 26978 )
        return 32;
LABEL_41:
      if ( *(_DWORD *)a1 == 1702390128 && *(_BYTE *)(a1 + 4) == 108 )
        return 33;
      if ( a2 > 5 )
      {
        if ( *(_DWORD *)a1 == 1953654134 && *(_WORD *)(a1 + 4) == 30821 )
          return 34;
        if ( a2 > 7 )
        {
          if ( *(_QWORD *)a1 == 0x797274656D6F6567LL )
            return 35;
          if ( *(_DWORD *)a1 != 1819047272 )
          {
LABEL_47:
            if ( *(_DWORD *)a1 == 1634561892 && *(_WORD *)(a1 + 4) == 28265 )
              return 37;
            if ( a2 > 6 )
            {
              if ( *(_DWORD *)a1 == 1886220131 && *(_WORD *)(a1 + 4) == 29813 && *(_BYTE *)(a1 + 6) == 101 )
                return 38;
              if ( *(_DWORD *)a1 == 1919052140 && *(_WORD *)(a1 + 4) == 29281 && *(_BYTE *)(a1 + 6) == 121 )
                return 39;
            }
LABEL_61:
            if ( a2 <= 0xC )
            {
              if ( a2 != 12 )
              {
                if ( a2 <= 5 )
                  goto LABEL_73;
                goto LABEL_64;
              }
            }
            else if ( *(_QWORD *)a1 == 0x72656E6567796172LL
                   && *(_DWORD *)(a1 + 8) == 1869182049
                   && *(_BYTE *)(a1 + 12) == 110 )
            {
              return 40;
            }
            if ( *(_QWORD *)a1 == 0x6365737265746E69LL && *(_DWORD *)(a1 + 8) == 1852795252 )
              return 41;
LABEL_64:
            if ( *(_DWORD *)a1 == 1752788577 && *(_WORD *)(a1 + 4) == 29801 )
              return 42;
            if ( a2 > 9 )
            {
              if ( *(_QWORD *)a1 == 0x68747365736F6C63LL && *(_WORD *)(a1 + 8) == 29801 )
                return 43;
              if ( *(_DWORD *)a1 != 1936943469 )
              {
LABEL_68:
                if ( *(_QWORD *)a1 == 0x656C62616C6C6163LL )
                  return 45;
LABEL_69:
                if ( *(_DWORD *)a1 == 1752393069 )
                  return 46;
                if ( a2 <= 0xC )
                {
                  if ( a2 <= 5 )
                  {
                    if ( *(_DWORD *)a1 == 1936681071 )
                      return 49;
LABEL_116:
                    if ( *(_DWORD *)a1 == 1836477548 )
                      return 26;
                    return 0;
                  }
                }
                else if ( *(_QWORD *)a1 == 0x636966696C706D61LL
                       && *(_DWORD *)(a1 + 8) == 1869182049
                       && *(_BYTE *)(a1 + 12) == 110 )
                {
                  return 47;
                }
                if ( *(_DWORD *)a1 == 1852141679 && *(_WORD *)(a1 + 4) == 27747 )
                  return 48;
                if ( *(_DWORD *)a1 == 1936681071 )
                  return 49;
                if ( a2 > 8 && *(_QWORD *)a1 == 0x7365746874756170LL && *(_BYTE *)(a1 + 8) == 116 )
                  return 50;
                goto LABEL_116;
              }
              return 44;
            }
LABEL_73:
            if ( a2 <= 3 )
              return 0;
            if ( *(_DWORD *)a1 != 1936943469 )
            {
              if ( a2 <= 7 )
                goto LABEL_69;
              goto LABEL_68;
            }
            return 44;
          }
          return 36;
        }
LABEL_88:
        if ( *(_DWORD *)a1 != 1819047272 )
        {
          if ( a2 <= 5 )
            goto LABEL_61;
          goto LABEL_47;
        }
        return 36;
      }
LABEL_60:
      if ( a2 <= 3 )
        goto LABEL_61;
      goto LABEL_88;
    }
LABEL_59:
    if ( a2 != 5 )
      goto LABEL_60;
    goto LABEL_41;
  }
  return result;
}
