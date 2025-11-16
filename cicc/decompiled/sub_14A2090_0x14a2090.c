// Function: sub_14A2090
// Address: 0x14a2090
//
char __fastcall sub_14A2090(__int64 a1, _BYTE *a2)
{
  char result; // al
  _BYTE *v3; // rax
  __int64 v4; // rdx
  _BYTE *v5; // r12

  if ( (a2[33] & 0x20) != 0 )
    return 0;
  if ( (a2[32] & 0xFu) - 7 <= 1 || (a2[23] & 0x20) == 0 )
    return 1;
  v3 = (_BYTE *)sub_1649960(a2);
  v5 = v3;
  switch ( v4 )
  {
    case 8LL:
      return *(_QWORD *)v3 != 0x6E67697379706F63LL;
    case 9LL:
      if ( *(_QWORD *)v3 == 0x6E67697379706F63LL && v3[8] == 102 )
      {
        return 0;
      }
      else
      {
        if ( *(_QWORD *)v3 != 0x6E67697379706F63LL )
          return 1;
        result = 0;
        if ( v5[8] != 108 )
          return 1;
      }
      break;
    case 4LL:
      return *(_DWORD *)v3 != 1935827302
          && *(_DWORD *)v3 != 1852403046
          && *(_DWORD *)v3 != 2019650918
          && *(_DWORD *)v3 != 1718511987
          && *(_DWORD *)v3 != 1819175283
          && *(_DWORD *)v3 != 1718841187
          && *(_DWORD *)v3 != 1819504483
          && *(_DWORD *)v3 != 1953657203
          && *(_DWORD *)v3 != 1719103344
          && *(_DWORD *)v3 != 1819766640
          && *(_DWORD *)v3 != 846231653
          && *(_DWORD *)v3 != 1818846563
          && *(_DWORD *)v3 != 1819502182
          && *(_DWORD *)v3 != 1935827308;
    case 5LL:
      if ( *(_DWORD *)v3 == 1935827302 && v3[4] == 102 )
        return 0;
      if ( *(_DWORD *)v3 == 1935827302 && v3[4] == 108 )
        return 0;
      if ( *(_DWORD *)v3 == 1852403046 && v3[4] == 102 )
        return 0;
      if ( *(_DWORD *)v3 == 1852403046 && v3[4] == 108 )
        return 0;
      if ( *(_DWORD *)v3 == 2019650918 && v3[4] == 102 )
        return 0;
      if ( *(_DWORD *)v3 == 2019650918 && v3[4] == 108 )
        return 0;
      if ( *(_DWORD *)v3 == 1953657203 && v3[4] == 102 )
      {
        return 0;
      }
      else
      {
        if ( *(_DWORD *)v3 != 1953657203 || v3[4] != 108 )
        {
          if ( memcmp(v3, "exp2l", 5u) && memcmp(v5, "exp2f", 5u) && memcmp(v5, "floor", 5u) && memcmp(v5, "round", 5u) )
            return memcmp(v5, "llabs", 5u) != 0;
          return 0;
        }
        return 0;
      }
    case 3LL:
      if ( *(_WORD *)v3 == 26995 && v3[2] == 110 )
        return 0;
      if ( *(_WORD *)v3 == 28515 && v3[2] == 115 )
        return 0;
      return memcmp(v3, "pow", 3u) && memcmp(v5, "ffs", 3u) && memcmp(v5, "abs", 3u);
    default:
      return v4 != 6 || memcmp(v3, "floorf", 6u);
  }
  return result;
}
