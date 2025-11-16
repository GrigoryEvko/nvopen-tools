// Function: sub_A7CD60
// Address: 0xa7cd60
//
__int64 __fastcall sub_A7CD60(__int64 a1, unsigned __int64 a2)
{
  __int64 result; // rax

  if ( a2 <= 3 )
    return 0;
  if ( *(_DWORD *)a1 == 779313761 )
  {
    if ( a2 == 8 )
    {
      if ( *(_DWORD *)(a1 + 4) == 909207138 )
        return 8127;
      return 0;
    }
    if ( a2 != 10 || *(_DWORD *)(a1 + 4) != 909207138 || *(_WORD *)(a1 + 8) != 12920 )
      return 0;
    return 8128;
  }
  if ( a2 <= 6 )
  {
    if ( a2 == 4 )
      return 0;
  }
  else if ( *(_DWORD *)a1 == 778136934 && *(_WORD *)(a1 + 4) == 28274 && *(_BYTE *)(a1 + 6) == 46 )
  {
    switch ( a2 )
    {
      case 0xBuLL:
        if ( *(_DWORD *)(a1 + 7) == 909207138 )
          return 8659;
        return 0;
      case 0xDuLL:
        if ( *(_DWORD *)(a1 + 7) != 909207138 || *(_WORD *)(a1 + 11) != 12920 )
          return 0;
        return 8661;
      case 0xFuLL:
        result = 8667;
        if ( *(_QWORD *)(a1 + 7) != 0x363166622E7A7466LL )
        {
          result = 8684;
          if ( *(_QWORD *)(a1 + 7) != 0x363166622E746173LL )
            return 0;
        }
        break;
      case 0x11uLL:
        if ( *(_QWORD *)(a1 + 7) == 0x363166622E7A7466LL && *(_WORD *)(a1 + 15) == 12920 )
        {
          return 8668;
        }
        else
        {
          if ( *(_QWORD *)(a1 + 7) != 0x363166622E746173LL || *(_WORD *)(a1 + 15) != 12920 )
            return 0;
          return 8686;
        }
      case 0x14uLL:
        if ( *(_QWORD *)(a1 + 7) != 0x756C65722E7A7466LL
          || *(_DWORD *)(a1 + 15) != 828793390
          || *(_BYTE *)(a1 + 19) != 54 )
        {
          return 0;
        }
        return 8672;
      case 0x16uLL:
        if ( *(_QWORD *)(a1 + 7) != 0x756C65722E7A7466LL
          || *(_DWORD *)(a1 + 15) != 828793390
          || *(_WORD *)(a1 + 19) != 30774
          || *(_BYTE *)(a1 + 21) != 50 )
        {
          return 0;
        }
        return 8673;
      case 0x13uLL:
        if ( *(_QWORD *)(a1 + 7) != 0x2E7461732E7A7466LL || *(_DWORD *)(a1 + 15) != 909207138 )
          return 0;
        return 8676;
      case 0x15uLL:
        if ( *(_QWORD *)(a1 + 7) != 0x2E7461732E7A7466LL
          || *(_DWORD *)(a1 + 15) != 909207138
          || *(_WORD *)(a1 + 19) != 12920 )
        {
          return 0;
        }
        return 8677;
      case 0x10uLL:
        if ( *(_QWORD *)(a1 + 7) != 0x3166622E756C6572LL || *(_BYTE *)(a1 + 15) != 54 )
          return 0;
        return 8680;
      default:
        if ( a2 != 18
          || *(_QWORD *)(a1 + 7) != 0x3166622E756C6572LL
          || *(_WORD *)(a1 + 15) != 30774
          || *(_BYTE *)(a1 + 17) != 50 )
        {
          return 0;
        }
        return 8681;
    }
    return result;
  }
  if ( *(_DWORD *)a1 == 2019650918 && *(_BYTE *)(a1 + 4) == 46 )
  {
    switch ( a2 )
    {
      case 9uLL:
        if ( *(_DWORD *)(a1 + 5) != 909207138 )
          return 0;
        return 8705;
      case 0xBuLL:
        if ( *(_DWORD *)(a1 + 5) != 909207138 || *(_WORD *)(a1 + 9) != 12920 )
          return 0;
        return 8707;
      case 0xDuLL:
        result = 8716;
        if ( *(_QWORD *)(a1 + 5) != 0x363166622E7A7466LL )
        {
          result = 8736;
          if ( *(_QWORD *)(a1 + 5) != 0x363166622E6E616ELL )
            return 0;
        }
        break;
      case 0xFuLL:
        if ( *(_QWORD *)(a1 + 5) == 0x363166622E7A7466LL && *(_WORD *)(a1 + 13) == 12920 )
        {
          return 8717;
        }
        else
        {
          if ( *(_QWORD *)(a1 + 5) != 0x363166622E6E616ELL || *(_WORD *)(a1 + 13) != 12920 )
            return 0;
          return 8737;
        }
      case 0x11uLL:
        if ( *(_QWORD *)(a1 + 5) != 0x2E6E616E2E7A7466LL || *(_DWORD *)(a1 + 13) != 909207138 )
          return 0;
        return 8721;
      case 0x13uLL:
        if ( *(_QWORD *)(a1 + 5) != 0x2E6E616E2E7A7466LL
          || *(_DWORD *)(a1 + 13) != 909207138
          || *(_WORD *)(a1 + 17) != 12920 )
        {
          return 0;
        }
        return 8722;
      case 0x1DuLL:
        if ( *(_QWORD *)(a1 + 5) ^ 0x2E6E616E2E7A7466LL | *(_QWORD *)(a1 + 13) ^ 0x2E6E676973726F78LL
          || *(_QWORD *)(a1 + 21) != 0x363166622E736261LL )
        {
          return 0;
        }
        return 8726;
      case 0x1FuLL:
        if ( *(_QWORD *)(a1 + 5) ^ 0x2E6E616E2E7A7466LL | *(_QWORD *)(a1 + 13) ^ 0x2E6E676973726F78LL
          || *(_QWORD *)(a1 + 21) != 0x363166622E736261LL
          || *(_WORD *)(a1 + 29) != 12920 )
        {
          return 0;
        }
        return 8727;
      case 0x19uLL:
        if ( !(*(_QWORD *)(a1 + 5) ^ 0x73726F782E7A7466LL | *(_QWORD *)(a1 + 13) ^ 0x2E7362612E6E6769LL)
          && *(_DWORD *)(a1 + 21) == 909207138 )
        {
          return 8731;
        }
        else
        {
          if ( *(_QWORD *)(a1 + 5) ^ 0x73726F782E6E616ELL | *(_QWORD *)(a1 + 13) ^ 0x2E7362612E6E6769LL
            || *(_DWORD *)(a1 + 21) != 909207138 )
          {
            return 0;
          }
          return 8741;
        }
      case 0x1BuLL:
        if ( !(*(_QWORD *)(a1 + 5) ^ 0x73726F782E7A7466LL | *(_QWORD *)(a1 + 13) ^ 0x2E7362612E6E6769LL)
          && *(_DWORD *)(a1 + 21) == 909207138
          && *(_WORD *)(a1 + 25) == 12920 )
        {
          return 8732;
        }
        else
        {
          if ( *(_QWORD *)(a1 + 5) ^ 0x73726F782E6E616ELL | *(_QWORD *)(a1 + 13) ^ 0x2E7362612E6E6769LL
            || *(_DWORD *)(a1 + 21) != 909207138
            || *(_WORD *)(a1 + 25) != 12920 )
          {
            return 0;
          }
          return 8742;
        }
      case 0x15uLL:
        if ( *(_QWORD *)(a1 + 5) ^ 0x2E6E676973726F78LL | *(_QWORD *)(a1 + 13) ^ 0x363166622E736261LL )
          return 0;
        return 8746;
      default:
        if ( a2 != 23
          || *(_QWORD *)(a1 + 5) ^ 0x2E6E676973726F78LL | *(_QWORD *)(a1 + 13) ^ 0x363166622E736261LL
          || *(_WORD *)(a1 + 21) != 12920 )
        {
          return 0;
        }
        return 8747;
    }
  }
  else if ( *(_DWORD *)a1 == 1852403046 && *(_BYTE *)(a1 + 4) == 46 )
  {
    switch ( a2 )
    {
      case 9uLL:
        if ( *(_DWORD *)(a1 + 5) != 909207138 )
          return 0;
        return 8760;
      case 0xBuLL:
        if ( *(_DWORD *)(a1 + 5) != 909207138 || *(_WORD *)(a1 + 9) != 12920 )
          return 0;
        return 8762;
      case 0xDuLL:
        result = 8771;
        if ( *(_QWORD *)(a1 + 5) != 0x363166622E7A7466LL )
        {
          result = 8791;
          if ( *(_QWORD *)(a1 + 5) != 0x363166622E6E616ELL )
            return 0;
        }
        break;
      case 0xFuLL:
        if ( *(_QWORD *)(a1 + 5) == 0x363166622E7A7466LL && *(_WORD *)(a1 + 13) == 12920 )
        {
          return 8772;
        }
        else
        {
          if ( *(_QWORD *)(a1 + 5) != 0x363166622E6E616ELL || *(_WORD *)(a1 + 13) != 12920 )
            return 0;
          return 8792;
        }
      case 0x11uLL:
        if ( *(_QWORD *)(a1 + 5) != 0x2E6E616E2E7A7466LL || *(_DWORD *)(a1 + 13) != 909207138 )
          return 0;
        return 8776;
      case 0x13uLL:
        if ( *(_QWORD *)(a1 + 5) != 0x2E6E616E2E7A7466LL
          || *(_DWORD *)(a1 + 13) != 909207138
          || *(_WORD *)(a1 + 17) != 12920 )
        {
          return 0;
        }
        return 8777;
      case 0x1DuLL:
        if ( *(_QWORD *)(a1 + 5) ^ 0x2E6E616E2E7A7466LL | *(_QWORD *)(a1 + 13) ^ 0x2E6E676973726F78LL
          || *(_QWORD *)(a1 + 21) != 0x363166622E736261LL )
        {
          return 0;
        }
        return 8781;
      case 0x1FuLL:
        if ( *(_QWORD *)(a1 + 5) ^ 0x2E6E616E2E7A7466LL | *(_QWORD *)(a1 + 13) ^ 0x2E6E676973726F78LL
          || *(_QWORD *)(a1 + 21) != 0x363166622E736261LL
          || *(_WORD *)(a1 + 29) != 12920 )
        {
          return 0;
        }
        return 8782;
      case 0x19uLL:
        if ( !(*(_QWORD *)(a1 + 5) ^ 0x73726F782E7A7466LL | *(_QWORD *)(a1 + 13) ^ 0x2E7362612E6E6769LL)
          && *(_DWORD *)(a1 + 21) == 909207138 )
        {
          return 8786;
        }
        else
        {
          if ( *(_QWORD *)(a1 + 5) ^ 0x73726F782E6E616ELL | *(_QWORD *)(a1 + 13) ^ 0x2E7362612E6E6769LL
            || *(_DWORD *)(a1 + 21) != 909207138 )
          {
            return 0;
          }
          return 8796;
        }
      case 0x1BuLL:
        if ( !(*(_QWORD *)(a1 + 5) ^ 0x73726F782E7A7466LL | *(_QWORD *)(a1 + 13) ^ 0x2E7362612E6E6769LL)
          && *(_DWORD *)(a1 + 21) == 909207138
          && *(_WORD *)(a1 + 25) == 12920 )
        {
          return 8787;
        }
        else
        {
          if ( *(_QWORD *)(a1 + 5) ^ 0x73726F782E6E616ELL | *(_QWORD *)(a1 + 13) ^ 0x2E7362612E6E6769LL
            || *(_DWORD *)(a1 + 21) != 909207138
            || *(_WORD *)(a1 + 25) != 12920 )
          {
            return 0;
          }
          return 8797;
        }
      case 0x15uLL:
        if ( *(_QWORD *)(a1 + 5) ^ 0x2E6E676973726F78LL | *(_QWORD *)(a1 + 13) ^ 0x363166622E736261LL )
          return 0;
        return 8801;
      default:
        if ( a2 != 23
          || *(_QWORD *)(a1 + 5) ^ 0x2E6E676973726F78LL | *(_QWORD *)(a1 + 13) ^ 0x363166622E736261LL
          || *(_WORD *)(a1 + 21) != 12920 )
        {
          return 0;
        }
        return 8802;
    }
  }
  else
  {
    if ( *(_DWORD *)a1 != 778528110 )
      return 0;
    if ( a2 == 8 )
    {
      if ( *(_DWORD *)(a1 + 4) == 909207138 )
        return 9211;
      return 0;
    }
    if ( a2 != 10 || *(_DWORD *)(a1 + 4) != 909207138 || *(_WORD *)(a1 + 8) != 12920 )
      return 0;
    return 9212;
  }
  return result;
}
