// Function: sub_E03450
// Address: 0xe03450
//
__int64 __fastcall sub_E03450(_BYTE *s1, __int64 a2)
{
  __int64 result; // rax
  int v3; // r8d
  __int64 v4; // rax
  int v5; // r8d
  int v6; // r8d

  switch ( a2 )
  {
    case 11LL:
      if ( *(_QWORD *)s1 == 0x6E5F4741545F5744LL && *((_WORD *)s1 + 4) == 27765 )
      {
        result = 0;
        if ( s1[10] == 108 )
          return result;
      }
      return 0xFFFFFFFFLL;
    case 17LL:
      if ( *(_QWORD *)s1 ^ 0x615F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x7079745F79617272LL
        || (result = 1, s1[16] != 101) )
      {
        if ( !(*(_QWORD *)s1 ^ 0x635F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x7079745F7373616CLL) && s1[16] == 101 )
          return 2;
        if ( *(_QWORD *)s1 ^ 0x755F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x7079745F6E6F696ELL
          || (result = 23, s1[16] != 101) )
        {
          if ( *(_QWORD *)s1 ^ 0x635F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x7079745F74736E6FLL
            || (result = 38, s1[16] != 101) )
          {
            if ( *(_QWORD *)s1 ^ 0x655F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x6F746172656D756ELL
              || (result = 40, s1[16] != 114) )
            {
              if ( *(_QWORD *)s1 ^ 0x735F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x6172676F72706275LL
                || (result = 46, s1[16] != 109) )
              {
                if ( *(_QWORD *)s1 ^ 0x415F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x6F725F4D5549544CLL )
                  return 0xFFFFFFFFLL;
                result = 20753;
                if ( s1[16] != 109 )
                  return 0xFFFFFFFFLL;
              }
            }
          }
        }
      }
      break;
    case 18LL:
      if ( *(_QWORD *)s1 ^ 0x655F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x696F705F7972746ELL
        || (result = 3, *((_WORD *)s1 + 8) != 29806) )
      {
        if ( *(_QWORD *)s1 ^ 0x735F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x79745F676E697274LL
          || (result = 18, *((_WORD *)s1 + 8) != 25968) )
        {
          if ( *(_QWORD *)s1 ^ 0x695F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x6E6174697265686ELL
            || (result = 28, *((_WORD *)s1 + 8) != 25955) )
          {
            if ( *(_QWORD *)s1 ^ 0x635F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x6F6C625F68637461LL
              || (result = 37, *((_WORD *)s1 + 8) != 27491) )
            {
              if ( *(_QWORD *)s1 ^ 0x705F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x79745F64656B6361LL
                || (result = 45, *((_WORD *)s1 + 8) != 25968) )
              {
                if ( *(_QWORD *)s1 ^ 0x745F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x79745F6E776F7268LL
                  || (result = 49, *((_WORD *)s1 + 8) != 25968) )
                {
                  if ( *(_QWORD *)s1 ^ 0x735F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x79745F6465726168LL
                    || (result = 64, *((_WORD *)s1 + 8) != 25968) )
                  {
                    if ( *(_QWORD *)s1 ^ 0x615F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x79745F63696D6F74LL
                      || (result = 71, *((_WORD *)s1 + 8) != 25968) )
                    {
                      if ( *(_QWORD *)s1 ^ 0x555F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x78616C65725F4350LL )
                        return 0xFFFFFFFFLL;
                      result = 34663;
                      if ( *((_WORD *)s1 + 8) != 25701 )
                        return 0xFFFFFFFFLL;
                    }
                  }
                }
              }
            }
          }
        }
      }
      break;
    case 23LL:
      if ( *(_QWORD *)s1 ^ 0x655F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x69746172656D756ELL
        || *((_DWORD *)s1 + 4) != 1952411247
        || *((_WORD *)s1 + 10) != 28793
        || (result = 4, s1[22] != 101) )
      {
        if ( !(*(_QWORD *)s1 ^ 0x665F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x61705F6C616D726FLL)
          && *((_DWORD *)s1 + 4) == 1701667186
          && *((_WORD *)s1 + 10) == 25972
          && s1[22] == 114 )
        {
          return 5;
        }
        if ( *(_QWORD *)s1 ^ 0x635F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x6E695F6E6F6D6D6FLL
          || *((_DWORD *)s1 + 4) != 1937075299
          || *((_WORD *)s1 + 10) != 28521
          || (result = 27, s1[22] != 110) )
        {
          if ( *(_QWORD *)s1 ^ 0x755F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x696669636570736ELL
            || *((_DWORD *)s1 + 4) != 1952408677
            || *((_WORD *)s1 + 10) != 28793
            || (result = 59, s1[22] != 101) )
          {
            if ( *(_QWORD *)s1 ^ 0x675F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x735F636972656E65LL
              || *((_DWORD *)s1 + 4) != 1634886261
              || *((_WORD *)s1 + 10) != 26478
              || (result = 69, s1[22] != 101) )
            {
              if ( *(_QWORD *)s1 ^ 0x415F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x69635F4D5549544CLL
                || *((_DWORD *)s1 + 4) != 1952408434
                || *((_WORD *)s1 + 10) != 28793
                || (result = 20737, s1[22] != 101) )
              {
                if ( *(_QWORD *)s1 ^ 0x425F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x705F444E414C524FLL )
                  return 0xFFFFFFFFLL;
                if ( *((_DWORD *)s1 + 4) != 1701867378 )
                  return 0xFFFFFFFFLL;
                if ( *((_WORD *)s1 + 10) != 29810 )
                  return 0xFFFFFFFFLL;
                result = 45056;
                if ( s1[22] != 121 )
                  return 0xFFFFFFFFLL;
              }
            }
          }
        }
      }
      break;
    case 27LL:
      if ( *(_QWORD *)s1 ^ 0x695F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x5F646574726F706DLL
        || *((_QWORD *)s1 + 2) != 0x746172616C636564LL
        || *((_WORD *)s1 + 12) != 28521
        || (result = 8, s1[26] != 110) )
      {
        if ( *(_QWORD *)s1 ^ 0x415F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x776D5F4D5549544CLL )
          return 0xFFFFFFFFLL;
        if ( *((_QWORD *)s1 + 2) != 0x745F637269635F61LL )
          return 0xFFFFFFFFLL;
        if ( *((_WORD *)s1 + 12) != 28793 )
          return 0xFFFFFFFFLL;
        result = 20738;
        if ( s1[26] != 101 )
          return 0xFFFFFFFFLL;
      }
      break;
    case 12LL:
      if ( *(_QWORD *)s1 != 0x6C5F4741545F5744LL )
        return 0xFFFFFFFFLL;
      result = 10;
      if ( *((_DWORD *)s1 + 2) != 1818583649 )
        return 0xFFFFFFFFLL;
      break;
    case 20LL:
      if ( *(_QWORD *)s1 ^ 0x6C5F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x625F6C6163697865LL
        || (result = 11, *((_DWORD *)s1 + 4) != 1801678700) )
      {
        if ( *(_QWORD *)s1 ^ 0x735F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x5F65676E61726275LL
          || (result = 33, *((_DWORD *)s1 + 4) != 1701869940) )
        {
          if ( *(_QWORD *)s1 ^ 0x6E5F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x5F7473696C656D61LL
            || (result = 44, *((_DWORD *)s1 + 4) != 1835365481) )
          {
            if ( *(_QWORD *)s1 ^ 0x765F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x5F656C6974616C6FLL
              || (result = 53, *((_DWORD *)s1 + 4) != 1701869940) )
            {
              if ( *(_QWORD *)s1 ^ 0x725F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x5F74636972747365LL
                || (result = 55, *((_DWORD *)s1 + 4) != 1701869940) )
              {
                if ( *(_QWORD *)s1 ^ 0x695F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x5F646574726F706DLL
                  || (result = 61, *((_DWORD *)s1 + 4) != 1953066613) )
                {
                  if ( *(_QWORD *)s1 ^ 0x735F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x5F6E6F74656C656BLL
                    || (result = 74, *((_DWORD *)s1 + 4) != 1953066613) )
                  {
                    if ( *(_QWORD *)s1 ^ 0x475F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x5F6C6C61635F554ELL
                      || (result = 16649, *((_DWORD *)s1 + 4) != 1702127987) )
                    {
                      if ( *(_QWORD *)s1 ^ 0x535F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x6665646F635F4E55LL
                        || (result = 16902, *((_DWORD *)s1 + 4) != 1936154988) )
                      {
                        if ( *(_QWORD *)s1 ^ 0x535F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x5F726F74645F4E55LL
                          || (result = 16906, *((_DWORD *)s1 + 4) != 1868983913) )
                        {
                          if ( *(_QWORD *)s1 ^ 0x475F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x73656D616E5F5348LL )
                            return 0xFFFFFFFFLL;
                          result = 32772;
                          if ( *((_DWORD *)s1 + 4) != 1701011824 )
                            return 0xFFFFFFFFLL;
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
    case 13LL:
      if ( *(_QWORD *)s1 != 0x6D5F4741545F5744LL || *((_DWORD *)s1 + 2) != 1700949349 || (result = 13, s1[12] != 114) )
      {
        if ( *(_QWORD *)s1 != 0x6D5F4741545F5744LL || *((_DWORD *)s1 + 2) != 1819632751 || (result = 30, s1[12] != 101) )
        {
          if ( *(_QWORD *)s1 != 0x665F4741545F5744LL
            || *((_DWORD *)s1 + 2) != 1852139890
            || (result = 42, s1[12] != 100) )
          {
            if ( *(_QWORD *)s1 != 0x535F4741545F5744LL )
              return 0xFFFFFFFFLL;
            if ( *((_DWORD *)s1 + 2) != 1751076437 )
              return 0xFFFFFFFFLL;
            result = 17151;
            if ( s1[12] != 105 )
              return 0xFFFFFFFFLL;
          }
        }
      }
      break;
    case 19LL:
      if ( *(_QWORD *)s1 ^ 0x705F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x745F7265746E696FLL
        || *((_WORD *)s1 + 8) != 28793
        || (result = 15, s1[18] != 101) )
      {
        if ( *(_QWORD *)s1 ^ 0x635F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x755F656C69706D6FLL
          || *((_WORD *)s1 + 8) != 26990
          || (result = 17, s1[18] != 116) )
        {
          if ( *(_QWORD *)s1 ^ 0x635F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x6C625F6E6F6D6D6FLL
            || *((_WORD *)s1 + 8) != 25455
            || (result = 26, s1[18] != 107) )
          {
            if ( *(_QWORD *)s1 ^ 0x765F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x705F746E61697261LL
              || *((_WORD *)s1 + 8) != 29281
              || (result = 51, s1[18] != 116) )
            {
              if ( *(_QWORD *)s1 ^ 0x705F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x755F6C6169747261LL
                || *((_WORD *)s1 + 8) != 26990
                || (result = 60, s1[18] != 116) )
              {
                if ( *(_QWORD *)s1 ^ 0x635F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x745F79617272616FLL
                  || *((_WORD *)s1 + 8) != 28793
                  || (result = 68, s1[18] != 101) )
                {
                  if ( *(_QWORD *)s1 ^ 0x645F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x745F63696D616E79LL
                    || *((_WORD *)s1 + 8) != 28793
                    || (result = 70, s1[18] != 101) )
                  {
                    if ( *(_QWORD *)s1 ^ 0x665F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x616C5F74616D726FLL )
                      return 0xFFFFFFFFLL;
                    if ( *((_WORD *)s1 + 8) != 25954 )
                      return 0xFFFFFFFFLL;
                    result = 16641;
                    if ( s1[18] != 108 )
                      return 0xFFFFFFFFLL;
                  }
                }
              }
            }
          }
        }
      }
      break;
    case 21LL:
      if ( *(_QWORD *)s1 ^ 0x725F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x65636E6572656665LL
        || *((_DWORD *)s1 + 4) != 1887007839
        || (result = 16, s1[20] != 101) )
      {
        if ( *(_QWORD *)s1 ^ 0x735F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x6572757463757274LL
          || *((_DWORD *)s1 + 4) != 1887007839
          || (result = 19, s1[20] != 101) )
        {
          if ( *(_QWORD *)s1 ^ 0x695F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x656361667265746ELL
            || *((_DWORD *)s1 + 4) != 1887007839
            || (result = 56, s1[20] != 101) )
          {
            if ( *(_QWORD *)s1 ^ 0x745F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x5F6574616C706D65LL
              || *((_DWORD *)s1 + 4) != 1634298977
              || (result = 67, s1[20] != 115) )
            {
              if ( *(_QWORD *)s1 ^ 0x695F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x656C626174756D6DLL
                || *((_DWORD *)s1 + 4) != 1887007839
                || (result = 75, s1[20] != 101) )
              {
                if ( *(_QWORD *)s1 ^ 0x635F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x6D65745F7373616CLL
                  || *((_DWORD *)s1 + 4) != 1952541808
                  || (result = 16643, s1[20] != 101) )
                {
                  if ( *(_QWORD *)s1 ^ 0x415F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x6F72705F454C5050LL
                    || *((_DWORD *)s1 + 4) != 1953654128
                    || (result = 16896, s1[20] != 121) )
                  {
                    if ( *(_QWORD *)s1 ^ 0x535F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x706F6D656D5F4E55LL
                      || *((_DWORD *)s1 + 4) != 1718511967
                      || (result = 16903, s1[20] != 111) )
                    {
                      if ( *(_QWORD *)s1 ^ 0x505F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x696A6E616B5F4947LL )
                        return 0xFFFFFFFFLL;
                      if ( *((_DWORD *)s1 + 4) != 1887007839 )
                        return 0xFFFFFFFFLL;
                      result = 40960;
                      if ( s1[20] != 101 )
                        return 0xFFFFFFFFLL;
                    }
                  }
                }
              }
            }
          }
        }
      }
      break;
    case 22LL:
      if ( *(_QWORD *)s1 ^ 0x735F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x6E6974756F726275LL
        || *((_DWORD *)s1 + 4) != 2037669733
        || (result = 21, *((_WORD *)s1 + 10) != 25968) )
      {
        if ( *(_QWORD *)s1 ^ 0x645F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x6F72705F66726177LL
          || *((_DWORD *)s1 + 4) != 1969513827
          || (result = 54, *((_WORD *)s1 + 10) != 25970) )
        {
          if ( *(_QWORD *)s1 ^ 0x695F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x5F646574726F706DLL
            || *((_DWORD *)s1 + 4) != 1969516397
            || (result = 58, *((_WORD *)s1 + 10) != 25964) )
          {
            if ( *(_QWORD *)s1 ^ 0x4C5F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x6F6E6E615F4D564CLL
              || *((_DWORD *)s1 + 4) != 1769234804
              || (result = 24576, *((_WORD *)s1 + 10) != 28271) )
            {
              if ( *(_QWORD *)s1 ^ 0x555F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x65726168735F4350LL
                || *((_DWORD *)s1 + 4) != 2037669732
                || (result = 34661, *((_WORD *)s1 + 10) != 25968) )
              {
                if ( !(*(_QWORD *)s1 ^ 0x555F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x63697274735F4350LL)
                  && *((_DWORD *)s1 + 4) == 2037669748
                  && *((_WORD *)s1 + 10) == 25968 )
                {
                  return 34662;
                }
                return 0xFFFFFFFFLL;
              }
            }
          }
        }
      }
      break;
    case 14LL:
      if ( *(_QWORD *)s1 != 0x745F4741545F5744LL
        || *((_DWORD *)s1 + 2) != 1684369529
        || (result = 22, *((_WORD *)s1 + 6) != 26213) )
      {
        if ( *(_QWORD *)s1 != 0x765F4741545F5744LL )
          return 0xFFFFFFFFLL;
        if ( *((_DWORD *)s1 + 2) != 1634300513 )
          return 0xFFFFFFFFLL;
        result = 25;
        if ( *((_WORD *)s1 + 6) != 29806 )
          return 0xFFFFFFFFLL;
      }
      break;
    case 29LL:
      if ( *(_QWORD *)s1 ^ 0x755F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x696669636570736ELL
        || *((_QWORD *)s1 + 2) != 0x6D617261705F6465LL
        || *((_DWORD *)s1 + 6) != 1919251557
        || (result = 24, s1[28] != 115) )
      {
        if ( *(_QWORD *)s1 ^ 0x425F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x445F444E414C524FLL )
          return 0xFFFFFFFFLL;
        if ( *((_QWORD *)s1 + 2) != 0x61765F6968706C65LL )
          return 0xFFFFFFFFLL;
        if ( *((_DWORD *)s1 + 6) != 1851877746 )
          return 0xFFFFFFFFLL;
        result = 45060;
        if ( s1[28] != 116 )
          return 0xFFFFFFFFLL;
      }
      break;
    case 25LL:
      if ( *(_QWORD *)s1 ^ 0x695F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x735F64656E696C6ELL
        || *((_QWORD *)s1 + 2) != 0x6E6974756F726275LL
        || (result = 29, s1[24] != 101) )
      {
        if ( *(_QWORD *)s1 ^ 0x705F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x656D5F6F745F7274LL
          || *((_QWORD *)s1 + 2) != 0x7079745F7265626DLL
          || (result = 31, s1[24] != 101) )
        {
          if ( *(_QWORD *)s1 ^ 0x615F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x65645F7373656363LL
            || *((_QWORD *)s1 + 2) != 0x6F69746172616C63LL
            || (result = 35, s1[24] != 110) )
          {
            if ( *(_QWORD *)s1 ^ 0x535F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x7373616C635F4E55LL
              || *((_QWORD *)s1 + 2) != 0x74616C706D65745FLL
              || (result = 16898, s1[24] != 101) )
            {
              if ( *(_QWORD *)s1 ^ 0x535F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x6E6F696E755F4E55LL
                || *((_QWORD *)s1 + 2) != 0x74616C706D65745FLL
                || (result = 16900, s1[24] != 101) )
              {
                if ( *(_QWORD *)s1 ^ 0x535F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x635F706D6F5F4E55LL
                  || *((_QWORD *)s1 + 2) != 0x6E75665F646C6968LL
                  || (result = 16904, s1[24] != 99) )
                {
                  if ( *(_QWORD *)s1 ^ 0x425F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x445F444E414C524FLL )
                    return 0xFFFFFFFFLL;
                  if ( *((_QWORD *)s1 + 2) != 0x65735F6968706C65LL )
                    return 0xFFFFFFFFLL;
                  result = 45059;
                  if ( s1[24] != 116 )
                    return 0xFFFFFFFFLL;
                }
              }
            }
          }
        }
      }
      break;
    case 15LL:
      if ( *(_QWORD *)s1 != 0x735F4741545F5744LL
        || *((_DWORD *)s1 + 2) != 1952412773
        || *((_WORD *)s1 + 6) != 28793
        || (result = 32, s1[14] != 101) )
      {
        if ( *(_QWORD *)s1 != 0x635F4741545F5744LL
          || *((_DWORD *)s1 + 2) != 1953721967
          || *((_WORD *)s1 + 6) != 28257
          || (result = 39, s1[14] != 116) )
        {
          if ( *(_QWORD *)s1 != 0x6E5F4741545F5744LL
            || *((_DWORD *)s1 + 2) != 1818586465
            || *((_WORD *)s1 + 6) != 29545
            || (result = 43, s1[14] != 116) )
          {
            if ( *(_QWORD *)s1 != 0x765F4741545F5744LL
              || *((_DWORD *)s1 + 2) != 1634300513
              || *((_WORD *)s1 + 6) != 27746
              || (result = 52, s1[14] != 101) )
            {
              if ( *(_QWORD *)s1 != 0x535F4741545F5744LL )
                return 0xFFFFFFFFLL;
              if ( *((_DWORD *)s1 + 2) != 1683967573 )
                return 0xFFFFFFFFLL;
              if ( *((_WORD *)s1 + 6) != 28532 )
                return 0xFFFFFFFFLL;
              result = 16907;
              if ( s1[14] != 114 )
                return 0xFFFFFFFFLL;
            }
          }
        }
      }
      break;
    case 16LL:
      result = 34;
      if ( *(_QWORD *)s1 ^ 0x775F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x746D74735F687469LL )
      {
        result = 36;
        if ( *(_QWORD *)s1 ^ 0x625F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x657079745F657361LL )
        {
          result = 41;
          if ( *(_QWORD *)s1 ^ 0x665F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x657079745F656C69LL )
          {
            result = 50;
            if ( *(_QWORD *)s1 ^ 0x745F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x6B636F6C625F7972LL )
            {
              result = 57;
              if ( *(_QWORD *)s1 ^ 0x6E5F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x6563617073656D61LL )
              {
                result = 63;
                if ( *(_QWORD *)s1 ^ 0x635F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x6E6F697469646E6FLL )
                {
                  result = 65;
                  if ( *(_QWORD *)s1 ^ 0x745F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x74696E755F657079LL )
                  {
                    result = 72;
                    if ( *(_QWORD *)s1 ^ 0x635F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x657469735F6C6C61LL )
                    {
                      result = 16513;
                      if ( *(_QWORD *)s1 ^ 0x4D5F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x706F6F6C5F535049LL )
                      {
                        v4 = *(_QWORD *)s1 ^ 0x475F4741545F5744LL;
                        if ( !(v4 | *((_QWORD *)s1 + 1) ^ 0x4C434E49425F554ELL) )
                          return 16644;
                        if ( !(v4 | *((_QWORD *)s1 + 1) ^ 0x4C434E49455F554ELL) )
                          return 16645;
                        return 0xFFFFFFFFLL;
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
    case 30LL:
      if ( *(_QWORD *)s1 ^ 0x745F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x5F6574616C706D65LL
        || *((_QWORD *)s1 + 2) != 0x7261705F65707974LL
        || *((_DWORD *)s1 + 6) != 1952804193
        || (result = 47, *((_WORD *)s1 + 14) != 29285) )
      {
        if ( *(_QWORD *)s1 ^ 0x475F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x5F6C6C61635F554ELL )
          return 0xFFFFFFFFLL;
        if ( *((_QWORD *)s1 + 2) != 0x7261705F65746973LL )
          return 0xFFFFFFFFLL;
        if ( *((_DWORD *)s1 + 6) != 1952804193 )
          return 0xFFFFFFFFLL;
        result = 16650;
        if ( *((_WORD *)s1 + 14) != 29285 )
          return 0xFFFFFFFFLL;
      }
      break;
    case 31LL:
      v3 = memcmp(s1, "DW_TAG_template_value_parameter", 0x1Fu);
      result = 48;
      if ( v3 )
      {
        v5 = memcmp(s1, "DW_TAG_SUN_indirect_inheritance", 0x1Fu);
        result = 16901;
        if ( v5 )
        {
          v6 = memcmp(s1, "DW_TAG_GHS_template_templ_param", 0x1Fu);
          result = 32775;
          if ( v6 )
            return 0xFFFFFFFFLL;
        }
      }
      break;
    case 28LL:
      if ( *(_QWORD *)s1 ^ 0x725F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x65725F65756C6176LL
        || *((_QWORD *)s1 + 2) != 0x5F65636E65726566LL
        || (result = 66, *((_DWORD *)s1 + 6) != 1701869940) )
      {
        if ( *(_QWORD *)s1 ^ 0x535F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x74636E75665F4E55LL
          || *((_QWORD *)s1 + 2) != 0x706D65745F6E6F69LL
          || (result = 16897, *((_DWORD *)s1 + 6) != 1702125932) )
        {
          if ( *(_QWORD *)s1 ^ 0x415F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x65725F4D5549544CLL
            || *((_QWORD *)s1 + 2) != 0x5F79727261635F76LL
            || (result = 20739, *((_DWORD *)s1 + 6) != 1701869940) )
          {
            if ( *(_QWORD *)s1 ^ 0x475F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x676E6973755F5348LL
              || *((_QWORD *)s1 + 2) != 0x6172616C6365645FLL
              || (result = 32774, *((_DWORD *)s1 + 6) != 1852795252) )
            {
              if ( *(_QWORD *)s1 ^ 0x425F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x445F444E414C524FLL )
                return 0xFFFFFFFFLL;
              if ( *((_QWORD *)s1 + 2) != 0x74735F6968706C65LL )
                return 0xFFFFFFFFLL;
              result = 45057;
              if ( *((_DWORD *)s1 + 6) != 1735289202 )
                return 0xFFFFFFFFLL;
            }
          }
        }
      }
      break;
    case 26LL:
      if ( *(_QWORD *)s1 ^ 0x635F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x657469735F6C6C61LL
        || *((_QWORD *)s1 + 2) != 0x74656D617261705FLL
        || (result = 73, *((_WORD *)s1 + 12) != 29285) )
      {
        if ( *(_QWORD *)s1 ^ 0x535F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x63757274735F4E55LL
          || *((_QWORD *)s1 + 2) != 0x616C706D65745F74LL
          || (result = 16899, *((_WORD *)s1 + 12) != 25972) )
        {
          if ( *(_QWORD *)s1 ^ 0x535F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x5F697474725F4E55LL
            || *((_QWORD *)s1 + 2) != 0x7470697263736564LL
            || (result = 16905, *((_WORD *)s1 + 12) != 29295) )
          {
            if ( *(_QWORD *)s1 ^ 0x475F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x676E6973755F5348LL
              || *((_QWORD *)s1 + 2) != 0x617073656D616E5FLL
              || (result = 32773, *((_WORD *)s1 + 12) != 25955) )
            {
              if ( *(_QWORD *)s1 ^ 0x505F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x7265746E695F4947LL )
                return 0xFFFFFFFFLL;
              if ( *((_QWORD *)s1 + 2) != 0x6F6C625F65636166LL )
                return 0xFFFFFFFFLL;
              result = 40992;
              if ( *((_WORD *)s1 + 12) != 27491 )
                return 0xFFFFFFFFLL;
            }
          }
        }
      }
      break;
    case 24LL:
      if ( !(*(_QWORD *)s1 ^ 0x665F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x5F6E6F6974636E75LL)
        && *((_QWORD *)s1 + 2) == 0x6574616C706D6574LL )
      {
        return 16642;
      }
      if ( !(*(_QWORD *)s1 ^ 0x535F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x695F3039665F4E55LL)
        && *((_QWORD *)s1 + 2) == 0x656361667265746ELL )
      {
        return 16908;
      }
      if ( !(*(_QWORD *)s1 ^ 0x4C5F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x617274705F4D564CLL)
        && *((_QWORD *)s1 + 2) == 0x657079745F687475LL )
      {
        return 17152;
      }
      return 0xFFFFFFFFLL;
    case 34LL:
      if ( *(_QWORD *)s1 ^ 0x475F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x6C706D65745F554ELL
        || *((_QWORD *)s1 + 2) ^ 0x706D65745F657461LL | *((_QWORD *)s1 + 3) ^ 0x7261705F6574616CLL
        || (result = 16646, *((_WORD *)s1 + 16) != 28001) )
      {
        if ( !(*(_QWORD *)s1 ^ 0x475F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x6C706D65745F554ELL)
          && !(*((_QWORD *)s1 + 2) ^ 0x617261705F657461LL | *((_QWORD *)s1 + 3) ^ 0x61705F726574656DLL)
          && *((_WORD *)s1 + 16) == 27491 )
        {
          return 16647;
        }
        return 0xFFFFFFFFLL;
      }
      break;
    case 32LL:
      if ( *(_QWORD *)s1 ^ 0x475F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x616D726F665F554ELL
        || (result = 16648, *((_QWORD *)s1 + 2) ^ 0x656D617261705F6CLL | *((_QWORD *)s1 + 3) ^ 0x6B6361705F726574LL) )
      {
        if ( *(_QWORD *)s1 ^ 0x535F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x7274726F665F4E55LL )
          return 0xFFFFFFFFLL;
        result = 16909;
        if ( *((_QWORD *)s1 + 2) ^ 0x735F7861765F6E61LL | *((_QWORD *)s1 + 3) ^ 0x6572757463757274LL )
          return 0xFFFFFFFFLL;
      }
      break;
    case 35LL:
      if ( *(_QWORD *)s1 ^ 0x425F4741545F5744LL | *((_QWORD *)s1 + 1) ^ 0x445F444E414C524FLL )
        return 0xFFFFFFFFLL;
      if ( *((_QWORD *)s1 + 2) ^ 0x79645F6968706C65LL | *((_QWORD *)s1 + 3) ^ 0x72615F63696D616ELL )
        return 0xFFFFFFFFLL;
      if ( *((_WORD *)s1 + 16) != 24946 )
        return 0xFFFFFFFFLL;
      result = 45058;
      if ( s1[34] != 121 )
        return 0xFFFFFFFFLL;
      break;
    default:
      return 0xFFFFFFFFLL;
  }
  return result;
}
