// Function: sub_CC67C0
// Address: 0xcc67c0
//
__int64 __fastcall sub_CC67C0(__int64 a1, unsigned __int64 a2)
{
  __int64 result; // rax

  result = sub_CC4190(a1, a2);
  switch ( a2 )
  {
    case 7uLL:
      if ( *(_DWORD *)a1 == 1668440417 && *(_WORD *)(a1 + 4) == 13928 && *(_BYTE *)(a1 + 6) == 52 )
        return 3;
      goto LABEL_10;
    case 0xAuLL:
      if ( *(_QWORD *)a1 == 0x5F34366863726161LL && *(_WORD *)(a1 + 8) == 25954 )
        return 4;
      if ( *(_QWORD *)a1 != 0x5F34366863726161LL || *(_WORD *)(a1 + 8) != 12851 )
        goto LABEL_10;
      return 5;
    case 3uLL:
      if ( *(_WORD *)a1 == 29281 && *(_BYTE *)(a1 + 2) == 99 )
        return 6;
      if ( *(_WORD *)a1 == 29281 && *(_BYTE *)(a1 + 2) == 109 )
        return 1;
      if ( *(_WORD *)a1 == 30305 && *(_BYTE *)(a1 + 2) == 114 )
        return 7;
      goto LABEL_10;
    case 5uLL:
      if ( *(_DWORD *)a1 == 913142369 && *(_BYTE *)(a1 + 4) == 52 )
        return 3;
      if ( *(_DWORD *)a1 == 1701671521 && *(_BYTE *)(a1 + 4) == 98 )
        return 2;
      goto LABEL_10;
  }
  if ( a2 != 8 )
  {
    if ( a2 <= 2 )
      goto LABEL_35;
    goto LABEL_10;
  }
  if ( *(_QWORD *)a1 == 0x32335F34366D7261LL )
    return 5;
LABEL_10:
  if ( *(_WORD *)a1 == 28770 && *(_BYTE *)(a1 + 2) == 102 )
    return result;
  if ( a2 == 4 )
  {
    result = 15;
    if ( *(_DWORD *)a1 != 1798846061 )
    {
      result = 16;
      if ( *(_DWORD *)a1 != 1936746861 )
      {
        result = 26;
        if ( *(_DWORD *)a1 != 808466034 )
        {
          result = 38;
          if ( *(_DWORD *)a1 != 909652841 )
          {
            result = 48;
            if ( *(_DWORD *)a1 != 1919512691 )
            {
              result = 10;
              if ( *(_DWORD *)a1 != 2037085027 )
              {
                result = 11;
                if ( *(_DWORD *)a1 != 1818851428 )
                  return 0;
              }
            }
          }
        }
      }
    }
    return result;
  }
LABEL_35:
  if ( a2 == 6 )
  {
    if ( *(_DWORD *)a1 == 1936746861 )
    {
      result = 17;
      if ( *(_WORD *)(a1 + 4) == 27749 )
        return result;
    }
    if ( *(_DWORD *)a1 == 1936746861 && *(_WORD *)(a1 + 4) == 13366 )
      return 18;
    if ( *(_DWORD *)a1 == 879784813 )
    {
      result = 20;
      if ( *(_WORD *)(a1 + 4) == 12339 )
        return result;
    }
    if ( *(_DWORD *)a1 == 1734634849 )
    {
      result = 27;
      if ( *(_WORD *)(a1 + 4) == 28259 )
        return result;
    }
    if ( *(_DWORD *)a1 == 758528120 )
    {
      result = 39;
      if ( *(_WORD *)(a1 + 4) == 13366 )
        return result;
    }
    if ( *(_DWORD *)a1 == 1919512691 )
    {
      result = 49;
      if ( *(_WORD *)(a1 + 4) == 13366 )
        return result;
    }
    if ( *(_DWORD *)a1 == 1836278135 )
    {
      result = 56;
      if ( *(_WORD *)(a1 + 4) == 12851 )
        return result;
    }
    if ( *(_DWORD *)a1 == 1836278135 )
    {
      result = 57;
      if ( *(_WORD *)(a1 + 4) == 13366 )
        return result;
    }
    if ( *(_DWORD *)a1 == 1634956910 && *(_WORD *)(a1 + 4) == 29555 )
      return 58;
    if ( *(_DWORD *)a1 == 1852142712 )
    {
      result = 41;
      if ( *(_WORD *)(a1 + 4) != 24947 )
        return 0;
      return result;
    }
    return 0;
  }
  if ( a2 == 8 )
  {
    result = 19;
    if ( *(_QWORD *)a1 == 0x6C6534367370696DLL )
      return result;
    return 0;
  }
  if ( a2 != 5 )
  {
    if ( a2 == 3 )
    {
      if ( *(_WORD *)a1 != 28784 || *(_BYTE *)(a1 + 2) != 99 )
      {
        if ( *(_WORD *)a1 == 25460 )
        {
          result = 34;
          if ( *(_BYTE *)(a1 + 2) == 101 )
            return result;
        }
        if ( *(_WORD *)a1 == 14456 )
        {
          result = 38;
          if ( *(_BYTE *)(a1 + 2) != 54 )
            return 0;
          return result;
        }
        return 0;
      }
      return 22;
    }
    if ( a2 != 7 )
    {
      switch ( a2 )
      {
        case 0xEuLL:
          if ( *(_QWORD *)a1 == 0x63737265646E6572LL && *(_DWORD *)(a1 + 8) == 1953524082 )
          {
            result = 59;
            if ( *(_WORD *)(a1 + 12) == 12851 )
              return result;
          }
          if ( *(_QWORD *)a1 == 0x63737265646E6572LL
            && *(_DWORD *)(a1 + 8) == 1953524082
            && *(_WORD *)(a1 + 12) == 13366 )
          {
            return 60;
          }
          break;
        case 2uLL:
          result = 61;
          if ( *(_WORD *)a1 != 25974 )
            return 0;
          return result;
        case 0xBuLL:
          if ( *(_QWORD *)a1 == 0x637261676E6F6F6CLL && *(_WORD *)(a1 + 8) == 13160 )
          {
            result = 13;
            if ( *(_BYTE *)(a1 + 10) == 50 )
              return result;
          }
          if ( *(_QWORD *)a1 == 0x637261676E6F6F6CLL && *(_WORD *)(a1 + 8) == 13928 )
          {
            result = 14;
            if ( *(_BYTE *)(a1 + 10) != 52 )
              return 0;
            return result;
          }
          break;
      }
      return 0;
    }
    if ( *(_DWORD *)a1 == 862154864 && *(_WORD *)(a1 + 4) == 27698 && *(_BYTE *)(a1 + 6) == 101 )
      return 23;
    if ( *(_DWORD *)a1 == 912486512 && *(_WORD *)(a1 + 4) == 27700 )
    {
      result = 25;
      if ( *(_BYTE *)(a1 + 6) == 101 )
        return result;
    }
    if ( *(_DWORD *)a1 == 1668508018 && *(_WORD *)(a1 + 4) == 13174 )
    {
      result = 28;
      if ( *(_BYTE *)(a1 + 6) == 50 )
        return result;
    }
    if ( *(_DWORD *)a1 == 1668508018 && *(_WORD *)(a1 + 4) == 13942 )
    {
      result = 29;
      if ( *(_BYTE *)(a1 + 6) == 52 )
        return result;
    }
    if ( *(_DWORD *)a1 == 1635280232 && *(_WORD *)(a1 + 4) == 28519 )
    {
      result = 12;
      if ( *(_BYTE *)(a1 + 6) == 110 )
        return result;
    }
    if ( *(_DWORD *)a1 == 1918988403 && *(_WORD *)(a1 + 4) == 25955 )
    {
      result = 32;
      if ( *(_BYTE *)(a1 + 6) == 108 )
        return result;
    }
    if ( *(_DWORD *)a1 == 1918988403 && *(_WORD *)(a1 + 4) == 30307 )
    {
      result = 31;
      if ( *(_BYTE *)(a1 + 6) == 57 )
        return result;
    }
    if ( *(_DWORD *)a1 != 1953724787 || *(_WORD *)(a1 + 4) != 28005 || *(_BYTE *)(a1 + 6) != 122 )
    {
      if ( *(_DWORD *)a1 == 1836410996 && *(_WORD *)(a1 + 4) == 25954 )
      {
        result = 37;
        if ( *(_BYTE *)(a1 + 6) == 98 )
          return result;
      }
      if ( *(_DWORD *)a1 == 1953527406 && *(_WORD *)(a1 + 4) == 13944 )
      {
        result = 43;
        if ( *(_BYTE *)(a1 + 6) == 52 )
          return result;
      }
      if ( *(_DWORD *)a1 == 1768189281 && *(_WORD *)(a1 + 4) == 13932 )
      {
        result = 45;
        if ( *(_BYTE *)(a1 + 6) == 52 )
          return result;
      }
      if ( *(_DWORD *)a1 == 1767994216 && *(_WORD *)(a1 + 4) == 13932 )
      {
        result = 47;
        if ( *(_BYTE *)(a1 + 6) == 52 )
          return result;
      }
      if ( *(_DWORD *)a1 == 1919512691 && *(_WORD *)(a1 + 4) == 13174 )
      {
        result = 51;
        if ( *(_BYTE *)(a1 + 6) == 50 )
          return result;
      }
      if ( *(_DWORD *)a1 == 1919512691 && *(_WORD *)(a1 + 4) == 13942 )
      {
        result = 52;
        if ( *(_BYTE *)(a1 + 6) == 52 )
          return result;
      }
      if ( *(_DWORD *)a1 == 1768710507 && *(_WORD *)(a1 + 4) == 25197 && *(_BYTE *)(a1 + 6) == 97 )
        return 53;
      return 0;
    }
    return 33;
  }
  if ( *(_DWORD *)a1 == 1885828718 )
  {
    result = 21;
    if ( *(_BYTE *)(a1 + 4) == 117 )
      return result;
  }
  if ( *(_DWORD *)a1 == 912486512 )
  {
    result = 24;
    if ( *(_BYTE *)(a1 + 4) == 52 )
      return result;
  }
  if ( *(_DWORD *)a1 == 862154864 && *(_BYTE *)(a1 + 4) == 50 )
    return 22;
  if ( *(_DWORD *)a1 == 1818456176 && *(_BYTE *)(a1 + 4) == 101 )
    return 23;
  if ( *(_DWORD *)a1 == 1918988403 )
  {
    result = 30;
    if ( *(_BYTE *)(a1 + 4) == 99 )
      return result;
  }
  if ( *(_DWORD *)a1 == 809055091 && *(_BYTE *)(a1 + 4) == 120 )
    return 33;
  if ( *(_DWORD *)a1 != 1818583924 || (result = 35, *(_BYTE *)(a1 + 4) != 101) )
  {
    if ( *(_DWORD *)a1 == 1836410996 && *(_BYTE *)(a1 + 4) == 98 )
      return 36;
    if ( *(_DWORD *)a1 != 1919902584 || (result = 40, *(_BYTE *)(a1 + 4) != 101) )
    {
      if ( *(_DWORD *)a1 != 1953527406 || (result = 42, *(_BYTE *)(a1 + 4) != 120) )
      {
        if ( *(_DWORD *)a1 != 1768189281 || (result = 44, *(_BYTE *)(a1 + 4) != 108) )
        {
          if ( *(_DWORD *)a1 != 1767994216 || (result = 46, *(_BYTE *)(a1 + 4) != 108) )
          {
            if ( *(_DWORD *)a1 != 1919512691 || (result = 50, *(_BYTE *)(a1 + 4) != 118) )
            {
              if ( *(_DWORD *)a1 != 1634623852 || (result = 55, *(_BYTE *)(a1 + 4) != 105) )
              {
                if ( *(_DWORD *)a1 == 1986095219 && *(_BYTE *)(a1 + 4) == 101 )
                  return 54;
                return 0;
              }
            }
          }
        }
      }
    }
  }
  return result;
}
