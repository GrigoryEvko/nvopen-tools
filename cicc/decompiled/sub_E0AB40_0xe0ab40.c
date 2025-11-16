// Function: sub_E0AB40
// Address: 0xe0ab40
//
__int64 __fastcall sub_E0AB40(__int64 a1, __int64 a2)
{
  switch ( a2 )
  {
    case 11LL:
      if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_WORD *)(a1 + 8) == 14403 && *(_BYTE *)(a1 + 10) == 57 )
        return 1;
      if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_WORD *)(a1 + 8) == 14659 && *(_BYTE *)(a1 + 10) == 57 )
        return 12;
      if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_WORD *)(a1 + 8) == 19536 && *(_BYTE *)(a1 + 10) == 73 )
        return 15;
      if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_WORD *)(a1 + 8) == 20565 && *(_BYTE *)(a1 + 10) == 67 )
        return 18;
      if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_WORD *)(a1 + 8) == 12611 && *(_BYTE *)(a1 + 10) == 49 )
        return 29;
      if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_WORD *)(a1 + 8) == 26970 && *(_BYTE *)(a1 + 10) == 103 )
        return 39;
      if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_WORD *)(a1 + 8) == 12611 && *(_BYTE *)(a1 + 10) == 55 )
        return 44;
      if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_WORD *)(a1 + 8) == 18760 && *(_BYTE *)(a1 + 10) == 80 )
        return 48;
      return 0;
    case 9LL:
      if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_BYTE *)(a1 + 8) == 67 )
        return 2;
      if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_BYTE *)(a1 + 8) == 68 )
        return 19;
      return 0;
    case 13LL:
      if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_DWORD *)(a1 + 8) == 945906753 && *(_BYTE *)(a1 + 12) == 51 )
        return 3;
      if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_DWORD *)(a1 + 8) == 962683969 && *(_BYTE *)(a1 + 12) == 53 )
        return 13;
      if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_DWORD *)(a1 + 8) == 1835090767 && *(_BYTE *)(a1 + 12) == 108 )
        return 27;
      if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_DWORD *)(a1 + 8) == 1718187859 && *(_BYTE *)(a1 + 12) == 116 )
      {
        return 30;
      }
      else if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_DWORD *)(a1 + 8) == 1768715594 && *(_BYTE *)(a1 + 12) == 97 )
      {
        return 31;
      }
      else
      {
        if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_DWORD *)(a1 + 8) == 1634498884 && *(_BYTE *)(a1 + 12) == 110 )
          return 32;
        if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_DWORD *)(a1 + 8) == 1397312578 && *(_BYTE *)(a1 + 12) == 83 )
          return 37;
        if ( *(_QWORD *)a1 != 0x5F474E414C5F5744LL || *(_DWORD *)(a1 + 8) != 1635018061 || *(_BYTE *)(a1 + 12) != 108 )
          return 0;
        return 61;
      }
    case 19LL:
      if ( *(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x705F73756C705F43LL
        || *(_WORD *)(a1 + 16) != 30060
        || *(_BYTE *)(a1 + 18) != 115 )
      {
        return 0;
      }
      return 4;
    case 15LL:
      if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL
        && *(_DWORD *)(a1 + 8) == 1868721987
        && *(_WORD *)(a1 + 12) == 14188
        && *(_BYTE *)(a1 + 14) == 52 )
      {
        return 5;
      }
      else
      {
        if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL
          && *(_DWORD *)(a1 + 8) == 1868721987
          && *(_WORD *)(a1 + 12) == 14444
          && *(_BYTE *)(a1 + 14) == 53 )
        {
          return 6;
        }
        if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL
          && *(_DWORD *)(a1 + 8) == 1969516365
          && *(_WORD *)(a1 + 12) == 24940
          && *(_BYTE *)(a1 + 14) == 50 )
        {
          return 10;
        }
        if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL
          && *(_DWORD *)(a1 + 8) == 1969516365
          && *(_WORD *)(a1 + 12) == 24940
          && *(_BYTE *)(a1 + 14) == 51 )
        {
          return 23;
        }
        else
        {
          if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL
            && *(_DWORD *)(a1 + 8) == 1802723656
            && *(_WORD *)(a1 + 12) == 27749
            && *(_BYTE *)(a1 + 14) == 108 )
          {
            return 24;
          }
          if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL
            && *(_DWORD *)(a1 + 8) == 1937338947
            && *(_WORD *)(a1 + 12) == 24948
            && *(_BYTE *)(a1 + 14) == 108 )
          {
            return 40;
          }
          if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL
            && *(_DWORD *)(a1 + 8) == 845243457
            && *(_WORD *)(a1 + 12) == 12336
            && *(_BYTE *)(a1 + 14) == 53 )
          {
            return 46;
          }
          else
          {
            if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL
              && *(_DWORD *)(a1 + 8) == 845243457
              && *(_WORD *)(a1 + 12) == 12592
              && *(_BYTE *)(a1 + 14) == 50 )
            {
              return 47;
            }
            if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL
              && *(_DWORD *)(a1 + 8) == 1752391491
              && *(_WORD *)(a1 + 12) == 29281
              && *(_BYTE *)(a1 + 14) == 112 )
            {
              return 50;
            }
            else
            {
              if ( *(_QWORD *)a1 != 0x5F474E414C5F5744LL
                || *(_DWORD *)(a1 + 8) != 1280527431
                || *(_WORD *)(a1 + 12) != 17759
                || *(_BYTE *)(a1 + 14) != 83 )
              {
                return 0;
              }
              return 53;
            }
          }
        }
      }
    case 17LL:
      if ( *(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x376E617274726F46LL
        || *(_BYTE *)(a1 + 16) != 55 )
      {
        if ( !(*(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x396E617274726F46LL)
          && *(_BYTE *)(a1 + 16) == 48 )
        {
          return 8;
        }
        if ( !(*(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x396E617274726F46LL)
          && *(_BYTE *)(a1 + 16) == 53 )
        {
          return 14;
        }
        if ( *(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x306E617274726F46LL
          || *(_BYTE *)(a1 + 16) != 51 )
        {
          if ( !(*(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x306E617274726F46LL)
            && *(_BYTE *)(a1 + 16) == 56 )
          {
            return 35;
          }
          if ( !(*(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x316E617274726F46LL)
            && *(_BYTE *)(a1 + 16) == 56 )
          {
            return 45;
          }
          return 0;
        }
        return 34;
      }
      else
      {
        return 7;
      }
    case 16LL:
      if ( !(*(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x33386C6163736150LL) )
        return 9;
      if ( *(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x796C626D65737341LL )
        return 0;
      return 49;
    case 12LL:
      if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_DWORD *)(a1 + 8) == 1635148106 )
        return 11;
      if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_DWORD *)(a1 + 8) == 1131045455 )
        return 16;
      if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_DWORD *)(a1 + 8) == 1953723730 )
        return 28;
      if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_DWORD *)(a1 + 8) == 1869246285 )
      {
        return 51;
      }
      else if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_DWORD *)(a1 + 8) == 1280527431 )
      {
        return 52;
      }
      else if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_DWORD *)(a1 + 8) == 1280527432 )
      {
        return 54;
      }
      else if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_DWORD *)(a1 + 8) == 1279482195 )
      {
        return 57;
      }
      else if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_DWORD *)(a1 + 8) == 2036495698 )
      {
        return 64;
      }
      else if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_DWORD *)(a1 + 8) == 1702260557 )
      {
        return 65;
      }
      else
      {
        if ( *(_QWORD *)a1 != 0x5F474E414C5F5744LL || *(_DWORD *)(a1 + 8) != 1869379912 )
          return 0;
        return 66;
      }
    case 22LL:
      if ( !(*(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x756C705F436A624FLL)
        && *(_DWORD *)(a1 + 16) == 1819303795
        && *(_WORD *)(a1 + 20) == 29557 )
      {
        return 17;
      }
      if ( !(*(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x705F73756C705F43LL)
        && *(_DWORD *)(a1 + 16) == 1601402220
        && *(_WORD *)(a1 + 20) == 13104 )
      {
        return 25;
      }
      else
      {
        if ( !(*(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x705F73756C705F43LL)
          && *(_DWORD *)(a1 + 16) == 1601402220
          && *(_WORD *)(a1 + 20) == 12593 )
        {
          return 26;
        }
        if ( !(*(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x705F73756C705F43LL)
          && *(_DWORD *)(a1 + 16) == 1601402220
          && *(_WORD *)(a1 + 20) == 13361 )
        {
          return 33;
        }
        if ( !(*(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x705F73756C705F43LL)
          && *(_DWORD *)(a1 + 16) == 1601402220
          && *(_WORD *)(a1 + 20) == 14129 )
        {
          return 42;
        }
        else
        {
          if ( !(*(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x705F73756C705F43LL)
            && *(_DWORD *)(a1 + 16) == 1601402220
            && *(_WORD *)(a1 + 20) == 12338 )
          {
            return 43;
          }
          if ( !(*(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x5F726F665F505043LL)
            && *(_DWORD *)(a1 + 16) == 1852141647
            && *(_WORD *)(a1 + 20) == 19523 )
          {
            return 56;
          }
          else if ( !(*(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x7373415F7370694DLL)
                 && *(_DWORD *)(a1 + 16) == 1818389861
                 && *(_WORD *)(a1 + 20) == 29285 )
          {
            return 32769;
          }
          else
          {
            if ( *(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x5F444E414C524F42LL
              || *(_DWORD *)(a1 + 16) != 1886152004
              || *(_WORD *)(a1 + 20) != 26984 )
            {
              return 0;
            }
            return 45056;
          }
        }
      }
    case 14LL:
      if ( *(_QWORD *)a1 != 0x5F474E414C5F5744LL || *(_DWORD *)(a1 + 8) != 1752463696 || *(_WORD *)(a1 + 12) != 28271 )
      {
        if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_DWORD *)(a1 + 8) == 1852141647 && *(_WORD *)(a1 + 12) == 19523 )
          return 21;
        if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_DWORD *)(a1 + 8) == 1819569995 && *(_WORD *)(a1 + 12) == 28265 )
          return 38;
        return 0;
      }
      return 20;
    case 10LL:
      if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_WORD *)(a1 + 8) == 28487 )
        return 22;
      return 0;
    case 20LL:
      if ( !(*(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x63537265646E6552LL)
        && *(_DWORD *)(a1 + 16) == 1953524082 )
      {
        return 36;
      }
      return 0;
    case 18LL:
      if ( *(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x435F4C436E65704FLL
        || *(_WORD *)(a1 + 16) != 20560 )
      {
        return 0;
      }
      return 55;
    case 27LL:
      if ( *(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x525F454C474F4F47LL
        || *(_QWORD *)(a1 + 16) != 0x7263537265646E65LL
        || *(_WORD *)(a1 + 24) != 28777
        || *(_BYTE *)(a1 + 26) != 116 )
      {
        return 0;
      }
      return 36439;
    default:
      return 0;
  }
}
