// Function: sub_10603A0
// Address: 0x10603a0
//
__int64 sub_10603A0()
{
  __int64 v0; // rax
  __int64 v1; // rdx
  unsigned int v2; // r8d

  v0 = sub_109C800();
  switch ( v1 )
  {
    case 7LL:
      if ( *(_DWORD *)v0 == 1701733735 && *(_WORD *)(v0 + 4) == 26994 && *(_BYTE *)(v0 + 6) == 99 )
        return 3;
LABEL_3:
      if ( v1 == 7 )
      {
        if ( *(_DWORD *)v0 == 912486512 && *(_WORD *)(v0 + 4) == 27700 )
        {
          v2 = 25;
          if ( *(_BYTE *)(v0 + 6) == 101 )
            return v2;
        }
        return 0;
      }
      if ( v1 == 6 && *(_DWORD *)v0 == 1970566502 && *(_WORD *)(v0 + 4) == 25970 )
        return 27;
      return 0;
    case 3LL:
      if ( *(_WORD *)v0 == 20291 && *(_BYTE *)(v0 + 2) == 77 )
        return 3;
      if ( *(_WORD *)v0 == 12342 )
      {
        v2 = 6;
        if ( *(_BYTE *)(v0 + 2) == 49 )
          return v2;
      }
      if ( (*(_WORD *)v0 != 12342 || *(_BYTE *)(v0 + 2) != 50) && (*(_WORD *)v0 != 12342 || *(_BYTE *)(v0 + 2) != 51) )
      {
        if ( *(_WORD *)v0 == 12342 )
        {
          v2 = 8;
          if ( *(_BYTE *)(v0 + 2) == 52 )
            return v2;
        }
        if ( *(_WORD *)v0 == 12854 )
        {
          v2 = 16;
          if ( *(_BYTE *)(v0 + 2) == 48 )
            return v2;
        }
        if ( *(_WORD *)v0 == 14137 && *(_BYTE *)(v0 + 2) == 48 )
          return 19;
        if ( *(_WORD *)v0 == 28784 && *(_BYTE *)(v0 + 2) == 99 || *(_WORD *)v0 == 20560 && *(_BYTE *)(v0 + 2) == 67 )
          return 3;
        if ( *(_WORD *)v0 == 28257 && *(_BYTE *)(v0 + 2) == 121 || *(_WORD *)v0 == 20033 && *(_BYTE *)(v0 + 2) == 89 )
          return 5;
        return 0;
      }
      break;
    case 4LL:
      if ( *(_DWORD *)v0 != 1697853494 )
      {
        v2 = 8;
        if ( *(_DWORD *)v0 == 1697919030 )
          return v2;
        v2 = 3;
        switch ( *(_DWORD *)v0 )
        {
          case 0x30303565:
            return v2;
          case 0x33727770:
            return v2;
          case 0x34727770:
            return v2;
        }
        v2 = 18;
        if ( *(_DWORD *)v0 == 896694128 )
          return v2;
        if ( *(_DWORD *)v0 == 894588752 )
          return v2;
        v2 = 20;
        if ( *(_DWORD *)v0 == 913471344 )
          return v2;
        if ( *(_DWORD *)v0 == 911365968 )
          return v2;
        v2 = 24;
        if ( *(_DWORD *)v0 == 930248560 )
          return v2;
        if ( *(_DWORD *)v0 == 928143184 )
          return v2;
        v2 = 25;
        if ( *(_DWORD *)v0 == 947025776 )
          return v2;
        if ( *(_DWORD *)v0 == 944920400 )
          return v2;
        v2 = 26;
        if ( *(_DWORD *)v0 == 963802992 || *(_DWORD *)v0 == 961697616 )
          return v2;
        return 0;
      }
      break;
    case 5LL:
      if ( *(_DWORD *)v0 != 1697853494 || *(_BYTE *)(v0 + 4) != 118 )
      {
        if ( *(_DWORD *)v0 == 896694128 && *(_BYTE *)(v0 + 4) == 120
          || *(_DWORD *)v0 == 894588752 && *(_BYTE *)(v0 + 4) == 88 )
        {
          return 22;
        }
        if ( *(_DWORD *)v0 == 913471344 && *(_BYTE *)(v0 + 4) == 120
          || *(_DWORD *)v0 == 911365968 && *(_BYTE *)(v0 + 4) == 69 )
        {
          return 23;
        }
        if ( *(_DWORD *)v0 == 829585264 && *(_BYTE *)(v0 + 4) == 48
          || *(_DWORD *)v0 == 827479888 && *(_BYTE *)(v0 + 4) == 48 )
        {
          return 27;
        }
        if ( *(_DWORD *)v0 == 862154864 && *(_BYTE *)(v0 + 4) == 50
          || *(_DWORD *)v0 == 912486512 && *(_BYTE *)(v0 + 4) == 52 )
        {
          return 3;
        }
        return 0;
      }
      break;
    case 2LL:
      v2 = 3;
      switch ( *(_WORD *)v0 )
      {
        case 0x3261:
          return v2;
        case 0x3367:
          return v2;
        case 0x3467:
          return v2;
      }
      v2 = 3;
      if ( *(_WORD *)v0 == 13671 )
        return v2;
      return 0;
    default:
      goto LABEL_3;
  }
  return 7;
}
