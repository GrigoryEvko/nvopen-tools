// Function: sub_109C800
// Address: 0x109c800
//
char *__fastcall sub_109C800(__int64 a1, __int64 a2)
{
  char *result; // rax

  result = (char *)a1;
  switch ( a2 )
  {
    case 6LL:
      if ( *(_DWORD *)a1 == 1835888483 && *(_WORD *)(a1 + 4) == 28271 )
        return "generic";
      if ( *(_DWORD *)a1 == 878932080 && *(_WORD *)(a1 + 4) == 12340 )
        return "440";
      if ( *(_DWORD *)a1 == 1702326128 && *(_WORD *)(a1 + 4) == 13170 )
        return "pwr3";
      if ( *(_DWORD *)a1 == 962818160 && *(_WORD *)(a1 + 4) == 12343 )
      {
        return "970";
      }
      else if ( *(_DWORD *)a1 == 1702326128 && *(_WORD *)(a1 + 4) == 13426 )
      {
        return "pwr4";
      }
      else if ( *(_DWORD *)a1 == 1702326128 && *(_WORD *)(a1 + 4) == 13682 )
      {
        return "pwr5";
      }
      else if ( *(_DWORD *)a1 == 1702326128 && *(_WORD *)(a1 + 4) == 13938 )
      {
        return "pwr6";
      }
      else if ( *(_DWORD *)a1 == 1702326128 && *(_WORD *)(a1 + 4) == 14194 )
      {
        return "pwr7";
      }
      else if ( *(_DWORD *)a1 == 1702326128 && *(_WORD *)(a1 + 4) == 14450 )
      {
        return "pwr8";
      }
      else if ( *(_DWORD *)a1 == 1702326128 )
      {
        result = "pwr9";
        if ( *(_WORD *)(a1 + 4) != 14706 )
          return (char *)a1;
      }
      break;
    case 3LL:
      if ( *(_WORD *)a1 == 12340 && *(_BYTE *)(a1 + 2) == 53 )
        return "generic";
      if ( *(_WORD *)a1 == 13110 && *(_BYTE *)(a1 + 2) == 48 )
        return "pwr3";
      if ( *(_WORD *)a1 == 13383 )
      {
        result = "g4+";
        if ( *(_BYTE *)(a1 + 2) != 43 )
          return (char *)a1;
      }
      break;
    case 5LL:
      if ( *(_DWORD *)a1 == 1714435124 && *(_BYTE *)(a1 + 4) == 112 )
      {
        return "440";
      }
      else if ( *(_DWORD *)a1 == 1633906800 )
      {
        result = "a2";
        if ( *(_BYTE *)(a1 + 4) != 50 )
          return (char *)a1;
      }
      break;
    case 2LL:
      if ( *(_WORD *)a1 == 13127 )
      {
        return "g3";
      }
      else if ( *(_WORD *)a1 == 13383 )
      {
        return "g4";
      }
      else
      {
        result = "g5";
        if ( *(_WORD *)a1 != 13639 )
          return (char *)a1;
      }
      break;
    case 4LL:
      if ( *(_DWORD *)a1 == 942945592 )
        return "e500";
      break;
    case 7LL:
      if ( *(_DWORD *)a1 == 1702326128 && *(_WORD *)(a1 + 4) == 13682 && *(_BYTE *)(a1 + 6) == 120 )
      {
        return "pwr5x";
      }
      else if ( *(_DWORD *)a1 == 1702326128 && *(_WORD *)(a1 + 4) == 13682 && *(_BYTE *)(a1 + 6) == 43 )
      {
        return "pwr5+";
      }
      else if ( *(_DWORD *)a1 == 1702326128 && *(_WORD *)(a1 + 4) == 13938 && *(_BYTE *)(a1 + 6) == 120 )
      {
        return "pwr6x";
      }
      else if ( *(_DWORD *)a1 == 1702326128 && *(_WORD *)(a1 + 4) == 12658 && *(_BYTE *)(a1 + 6) == 48 )
      {
        return "pwr10";
      }
      else if ( *(_DWORD *)a1 == 1702326128 && *(_WORD *)(a1 + 4) == 12658 && *(_BYTE *)(a1 + 6) == 49 )
      {
        return "pwr11";
      }
      else if ( *(_DWORD *)a1 == 1702326128 && *(_WORD *)(a1 + 4) == 28786 )
      {
        result = "ppc";
        if ( *(_BYTE *)(a1 + 6) != 99 )
          return (char *)a1;
      }
      break;
    case 9LL:
      if ( *(_QWORD *)a1 == 0x3363707265776F70LL && *(_BYTE *)(a1 + 8) == 50 )
      {
        return "ppc";
      }
      else if ( *(_QWORD *)a1 == 0x3663707265776F70LL )
      {
        result = "ppc64";
        if ( *(_BYTE *)(a1 + 8) != 52 )
          return (char *)a1;
      }
      break;
    default:
      if ( a2 == 11 && *(_QWORD *)a1 == 0x3663707265776F70LL && *(_WORD *)(a1 + 8) == 27700 )
      {
        result = "ppc64le";
        if ( *(_BYTE *)(a1 + 10) != 101 )
          return (char *)a1;
      }
      break;
  }
  return result;
}
