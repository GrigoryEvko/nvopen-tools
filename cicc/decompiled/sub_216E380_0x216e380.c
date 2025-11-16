// Function: sub_216E380
// Address: 0x216e380
//
char __fastcall sub_216E380(_BYTE *a1)
{
  char result; // al
  const char *v2; // rax
  __int64 v3; // rdx
  const char *v4; // r12

  if ( (a1[33] & 0x20) != 0 )
    return 0;
  if ( (a1[32] & 0xFu) - 7 <= 1 || (a1[23] & 0x20) == 0 )
    return 1;
  v2 = sub_1649960((__int64)a1);
  v4 = v2;
  switch ( v3 )
  {
    case 8LL:
      return *(_QWORD *)v2 != 0x6E67697379706F63LL;
    case 9LL:
      if ( *(_QWORD *)v2 == 0x6E67697379706F63LL && v2[8] == 102 )
      {
        return 0;
      }
      else
      {
        if ( *(_QWORD *)v2 != 0x6E67697379706F63LL )
          return 1;
        result = 0;
        if ( v4[8] != 108 )
          return 1;
      }
      break;
    case 4LL:
      return *(_DWORD *)v2 != 1935827302
          && *(_DWORD *)v2 != 1852403046
          && *(_DWORD *)v2 != 2019650918
          && *(_DWORD *)v2 != 1718511987
          && *(_DWORD *)v2 != 1819175283
          && *(_DWORD *)v2 != 1718841187
          && *(_DWORD *)v2 != 1819504483
          && *(_DWORD *)v2 != 1953657203
          && *(_DWORD *)v2 != 1719103344
          && *(_DWORD *)v2 != 1819766640
          && *(_DWORD *)v2 != 846231653
          && *(_DWORD *)v2 != 1818846563
          && *(_DWORD *)v2 != 1819502182
          && *(_DWORD *)v2 != 1935827308;
    case 5LL:
      if ( *(_DWORD *)v2 == 1935827302 && v2[4] == 102 )
        return 0;
      if ( *(_DWORD *)v2 == 1935827302 && v2[4] == 108 )
        return 0;
      if ( *(_DWORD *)v2 == 1852403046 && v2[4] == 102 )
        return 0;
      if ( *(_DWORD *)v2 == 1852403046 && v2[4] == 108 )
        return 0;
      if ( *(_DWORD *)v2 == 2019650918 && v2[4] == 102 )
        return 0;
      if ( *(_DWORD *)v2 == 2019650918 && v2[4] == 108 )
        return 0;
      if ( *(_DWORD *)v2 == 1953657203 && v2[4] == 102 )
      {
        return 0;
      }
      else
      {
        if ( *(_DWORD *)v2 != 1953657203 || v2[4] != 108 )
        {
          if ( memcmp(v2, "exp2l", 5u) && memcmp(v4, "exp2f", 5u) && memcmp(v4, "floor", 5u) && memcmp(v4, "round", 5u) )
            return memcmp(v4, "llabs", 5u) != 0;
          return 0;
        }
        return 0;
      }
    case 3LL:
      if ( *(_WORD *)v2 == 26995 && v2[2] == 110 )
        return 0;
      if ( *(_WORD *)v2 == 28515 && v2[2] == 115 )
        return 0;
      return memcmp(v2, "pow", 3u) && memcmp(v4, "ffs", 3u) && memcmp(v4, "abs", 3u);
    default:
      return v3 != 6 || memcmp(v2, "floorf", 6u);
  }
  return result;
}
