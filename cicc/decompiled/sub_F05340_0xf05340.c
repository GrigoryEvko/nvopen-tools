// Function: sub_F05340
// Address: 0xf05340
//
char *__fastcall sub_F05340(__int64 a1, __int64 a2)
{
  char *result; // rax

  result = (char *)a1;
  if ( a2 == 2 )
  {
    if ( *(_WORD *)a1 == 13686 )
      return (char *)&unk_3F88AB3;
    if ( *(_WORD *)a1 != 14198 )
    {
      if ( *(_WORD *)a1 != 14454 )
      {
        result = "v9-a";
        if ( *(_WORD *)a1 != 14710 )
          return (char *)a1;
        return result;
      }
      return "v8-a";
    }
    return "v7-a";
  }
  if ( a2 == 3 )
  {
    if ( *(_WORD *)a1 == 13686 && *(_BYTE *)(a1 + 2) == 101 )
      return "v5te";
    if ( *(_WORD *)a1 == 13942 && *(_BYTE *)(a1 + 2) == 106 )
      return "v6";
    if ( *(_WORD *)a1 == 13942 && *(_BYTE *)(a1 + 2) == 109 )
      return "v6-m";
    if ( *(_WORD *)a1 == 13942 && *(_BYTE *)(a1 + 2) == 122 )
      return "v6kz";
    if ( *(_WORD *)a1 == 14198 && *(_BYTE *)(a1 + 2) == 97 || *(_WORD *)a1 == 14198 && *(_BYTE *)(a1 + 2) == 108 )
      return "v7-a";
    if ( *(_WORD *)a1 == 14198 && *(_BYTE *)(a1 + 2) == 114 )
      return "v7-r";
    if ( *(_WORD *)a1 == 14198 && *(_BYTE *)(a1 + 2) == 109 )
      return "v7-m";
    if ( *(_WORD *)a1 == 14454 && *(_BYTE *)(a1 + 2) == 97 || *(_WORD *)a1 == 14454 && *(_BYTE *)(a1 + 2) == 108 )
      return "v8-a";
    if ( *(_WORD *)a1 == 14454 && *(_BYTE *)(a1 + 2) == 114 )
    {
      return "v8-r";
    }
    else if ( *(_WORD *)a1 == 14710 )
    {
      result = "v9-a";
      if ( *(_BYTE *)(a1 + 2) != 97 )
        return (char *)a1;
    }
  }
  else if ( a2 == 4 )
  {
    switch ( *(_DWORD *)a1 )
    {
      case 0x6C683676:
        return "v6k";
      case 0x6D733676:
        return "v6-m";
      case 0x6B7A3676:
        return "v6kz";
      case 0x6C683776:
        return "v7-a";
      case 0x6D653776:
        return "v7e-m";
    }
  }
  else
  {
    if ( a2 == 5 )
    {
      if ( *(_DWORD *)a1 == 762525302 && *(_BYTE *)(a1 + 4) == 109 )
        return "v6-m";
    }
    else if ( a2 == 7 )
    {
      if ( *(_DWORD *)a1 == 1668440417 && *(_WORD *)(a1 + 4) == 13928 && *(_BYTE *)(a1 + 6) == 52 )
        return "v8-a";
      return result;
    }
    if ( a2 == 5 )
    {
      if ( *(_DWORD *)a1 == 913142369 && *(_BYTE *)(a1 + 4) == 52 )
        return "v8-a";
      if ( *(_DWORD *)a1 == 825112694 && *(_BYTE *)(a1 + 4) == 97 )
      {
        return "v8.1-a";
      }
      else if ( *(_DWORD *)a1 == 841889910 && *(_BYTE *)(a1 + 4) == 97 )
      {
        return "v8.2-a";
      }
      else if ( *(_DWORD *)a1 == 858667126 && *(_BYTE *)(a1 + 4) == 97 )
      {
        return "v8.3-a";
      }
      else if ( *(_DWORD *)a1 == 875444342 && *(_BYTE *)(a1 + 4) == 97 )
      {
        return "v8.4-a";
      }
      else if ( *(_DWORD *)a1 == 892221558 && *(_BYTE *)(a1 + 4) == 97 )
      {
        return "v8.5-a";
      }
      else if ( *(_DWORD *)a1 == 908998774 && *(_BYTE *)(a1 + 4) == 97 )
      {
        return "v8.6-a";
      }
      else if ( *(_DWORD *)a1 == 925775990 && *(_BYTE *)(a1 + 4) == 97 )
      {
        return "v8.7-a";
      }
      else if ( *(_DWORD *)a1 == 942553206 && *(_BYTE *)(a1 + 4) == 97 )
      {
        return "v8.8-a";
      }
      else if ( *(_DWORD *)a1 == 959330422 && *(_BYTE *)(a1 + 4) == 97 )
      {
        return "v8.9-a";
      }
      else if ( *(_DWORD *)a1 == 825112950 && *(_BYTE *)(a1 + 4) == 97 )
      {
        return "v9.1-a";
      }
      else if ( *(_DWORD *)a1 == 841890166 && *(_BYTE *)(a1 + 4) == 97 )
      {
        return "v9.2-a";
      }
      else if ( *(_DWORD *)a1 == 858667382 && *(_BYTE *)(a1 + 4) == 97 )
      {
        return "v9.3-a";
      }
      else if ( *(_DWORD *)a1 == 875444598 && *(_BYTE *)(a1 + 4) == 97 )
      {
        return "v9.4-a";
      }
      else if ( *(_DWORD *)a1 == 892221814 && *(_BYTE *)(a1 + 4) == 97 )
      {
        return "v9.5-a";
      }
      else if ( *(_DWORD *)a1 == 908999030 )
      {
        result = "v9.6-a";
        if ( *(_BYTE *)(a1 + 4) != 97 )
          return (char *)a1;
      }
    }
    else if ( a2 == 8 )
    {
      result = "v8-m.base";
      if ( *(_QWORD *)a1 != 0x657361622E6D3876LL )
      {
        result = "v8-m.main";
        if ( *(_QWORD *)a1 != 0x6E69616D2E6D3876LL )
          return (char *)a1;
      }
    }
    else if ( a2 == 10 && *(_QWORD *)a1 == 0x616D2E6D312E3876LL )
    {
      result = "v8.1-m.main";
      if ( *(_WORD *)(a1 + 8) != 28265 )
        return (char *)a1;
    }
  }
  return result;
}
