// Function: sub_CC4400
// Address: 0xcc4400
//
__int64 __fastcall sub_CC4400(__int64 a1, unsigned __int64 a2)
{
  __int64 result; // rax

  if ( a2 <= 5 )
  {
    if ( a2 <= 2 )
      goto LABEL_22;
  }
  else
  {
    if ( *(_DWORD *)a1 == 2003984740 && *(_WORD *)(a1 + 4) == 28265 )
      return 1;
    if ( a2 <= 8 )
    {
      if ( a2 == 6 )
        goto LABEL_7;
    }
    else if ( *(_QWORD *)a1 == 0x6C666E6F67617264LL && *(_BYTE *)(a1 + 8) == 121 )
    {
      return 2;
    }
    if ( *(_DWORD *)a1 == 1701147238 && *(_WORD *)(a1 + 4) == 29538 && *(_BYTE *)(a1 + 6) == 100 )
      return 3;
    if ( *(_DWORD *)a1 == 1751348582 && *(_WORD *)(a1 + 4) == 26995 && *(_BYTE *)(a1 + 6) == 97 )
      return 4;
  }
LABEL_7:
  if ( *(_WORD *)a1 == 28521 && *(_BYTE *)(a1 + 2) == 115 )
    return 5;
  if ( a2 <= 7 )
  {
    if ( a2 <= 4 )
      goto LABEL_11;
  }
  else if ( *(_QWORD *)a1 == 0x647362656572666BLL )
  {
    return 6;
  }
  if ( *(_DWORD *)a1 == 1970170220 && *(_BYTE *)(a1 + 4) == 120 )
    return 7;
LABEL_11:
  if ( *(_WORD *)a1 == 30316 && *(_BYTE *)(a1 + 2) == 50 )
    return 8;
  if ( a2 > 4 )
  {
    if ( *(_DWORD *)a1 == 1868783981 && *(_BYTE *)(a1 + 4) == 115 )
      return 9;
    if ( a2 != 5 )
    {
      if ( *(_DWORD *)a1 == 1651795310 && *(_WORD *)(a1 + 4) == 25715 )
        return 10;
      if ( a2 != 6 )
      {
        if ( *(_DWORD *)a1 == 1852141679 && *(_WORD *)(a1 + 4) == 29538 && *(_BYTE *)(a1 + 6) == 100 )
          return 11;
        if ( *(_DWORD *)a1 == 1634496371 && *(_WORD *)(a1 + 4) == 26994 && *(_BYTE *)(a1 + 6) == 115 )
          return 12;
      }
    }
    goto LABEL_19;
  }
LABEL_22:
  if ( a2 <= 3 )
  {
    if ( a2 <= 2 )
      goto LABEL_24;
    goto LABEL_94;
  }
LABEL_19:
  if ( *(_DWORD *)a1 == 1768318325 )
    return 13;
  if ( a2 != 4 )
  {
    if ( *(_DWORD *)a1 == 862873975 && *(_BYTE *)(a1 + 4) == 50 )
      return 14;
    if ( a2 > 6 )
    {
      if ( *(_DWORD *)a1 == 1684957559 && *(_WORD *)(a1 + 4) == 30575 && *(_BYTE *)(a1 + 6) == 115 )
        return 14;
      if ( *(_WORD *)a1 == 28538 && *(_BYTE *)(a1 + 2) == 115 )
        return 15;
LABEL_36:
      if ( *(_DWORD *)a1 == 1802068328 && *(_BYTE *)(a1 + 4) == 117 )
        return 16;
      if ( *(_DWORD *)a1 == 1835365490 && *(_BYTE *)(a1 + 4) == 115 )
        return 17;
LABEL_38:
      if ( *(_DWORD *)a1 == 1818452334 )
        return 18;
      if ( *(_WORD *)a1 == 26977 && *(_BYTE *)(a1 + 2) == 120 )
        return 19;
      result = 20;
      if ( *(_DWORD *)a1 == 1633973603 )
        return result;
      result = 21;
      if ( *(_DWORD *)a1 == 1818457710 )
        return result;
      if ( a2 == 7 )
      {
        if ( *(_DWORD *)a1 == 1701996900 && *(_WORD *)(a1 + 4) == 29795 && *(_BYTE *)(a1 + 6) == 120 )
          return 22;
LABEL_44:
        if ( *(_DWORD *)a1 == 1751412065 && *(_WORD *)(a1 + 4) == 24947 )
          return 23;
        goto LABEL_45;
      }
LABEL_25:
      if ( a2 <= 5 )
      {
        if ( a2 <= 2 )
          goto LABEL_27;
        goto LABEL_45;
      }
      goto LABEL_44;
    }
  }
LABEL_94:
  if ( *(_WORD *)a1 == 28538 && *(_BYTE *)(a1 + 2) == 115 )
    return 15;
  if ( a2 > 4 )
    goto LABEL_36;
  if ( a2 > 3 )
    goto LABEL_38;
LABEL_24:
  if ( a2 != 3 )
    goto LABEL_25;
  if ( *(_WORD *)a1 == 26977 && *(_BYTE *)(a1 + 2) == 120 )
    return 19;
LABEL_45:
  if ( *(_WORD *)a1 == 29552 && *(_BYTE *)(a1 + 2) == 52 )
    return 24;
  if ( *(_WORD *)a1 == 29552 && *(_BYTE *)(a1 + 2) == 53 )
    return 25;
  if ( a2 > 7 )
  {
    if ( *(_QWORD *)a1 == 0x75636D6169666C65LL )
      return 26;
    goto LABEL_49;
  }
LABEL_27:
  if ( a2 <= 3 )
    return 0;
LABEL_49:
  if ( *(_DWORD *)a1 == 1936684660 )
    return 27;
  if ( a2 <= 6 )
    goto LABEL_77;
  if ( *(_DWORD *)a1 == 1668571511 && *(_WORD *)(a1 + 4) == 28520 && *(_BYTE *)(a1 + 6) == 115 )
    return 28;
  if ( a2 == 7 )
    goto LABEL_77;
  if ( *(_QWORD *)a1 == 0x736F656764697262LL )
    return 29;
  if ( a2 == 8 )
  {
LABEL_77:
    if ( *(_DWORD *)a1 == 1936683640 )
      return 31;
    if ( a2 <= 7 )
    {
      if ( a2 <= 5 )
        goto LABEL_66;
      goto LABEL_63;
    }
  }
  else
  {
    if ( *(_QWORD *)a1 == 0x696B726576697264LL && *(_BYTE *)(a1 + 8) == 116 )
      return 30;
    if ( *(_DWORD *)a1 == 1936683640 )
      return 31;
  }
  if ( *(_QWORD *)a1 == 0x736F6E6F69736976LL )
    return 31;
LABEL_63:
  if ( *(_DWORD *)a1 == 1634952557 && *(_WORD *)(a1 + 4) == 25651 )
    return 32;
  if ( *(_DWORD *)a1 == 1885629793 && *(_WORD *)(a1 + 4) == 27745 )
    return 33;
  if ( *(_DWORD *)a1 == 1836213608 && *(_WORD *)(a1 + 4) == 29801 )
    return 34;
LABEL_66:
  if ( *(_DWORD *)a1 == 1685222760 )
    return 35;
  result = 36;
  if ( *(_DWORD *)a1 != 1769169271 )
  {
    if ( a2 <= 9 )
    {
      if ( a2 <= 5 )
        return 0;
    }
    else
    {
      if ( *(_QWORD *)a1 == 0x7470697263736D65LL && *(_WORD *)(a1 + 8) == 28261 )
        return 37;
      if ( a2 != 10 )
      {
        if ( *(_QWORD *)a1 == 0x6F6D726564616873LL && *(_WORD *)(a1 + 8) == 25956 && *(_BYTE *)(a1 + 10) == 108 )
          return 38;
        if ( *(_DWORD *)a1 == 1702127980 && *(_WORD *)(a1 + 4) == 29551 )
          return 39;
LABEL_73:
        if ( *(_QWORD *)a1 == 0x7974696E65726573LL )
          return 40;
LABEL_74:
        if ( *(_DWORD *)a1 == 1802270070 && *(_WORD *)(a1 + 4) == 28257 )
          return 41;
        return 0;
      }
    }
    if ( *(_DWORD *)a1 == 1702127980 && *(_WORD *)(a1 + 4) == 29551 )
      return 39;
    if ( a2 <= 7 )
      goto LABEL_74;
    goto LABEL_73;
  }
  return result;
}
