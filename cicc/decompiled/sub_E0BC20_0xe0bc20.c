// Function: sub_E0BC20
// Address: 0xe0bc20
//
__int64 __fastcall sub_E0BC20(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  int v3; // r8d

  switch ( a2 )
  {
    case 12LL:
      if ( *(_QWORD *)a1 == 0x6F6E5F43435F5744LL && *(_DWORD *)(a1 + 8) == 1818324338 )
        return 1;
      if ( *(_QWORD *)a1 == 0x6F6E5F43435F5744LL && *(_DWORD *)(a1 + 8) == 1819042147 )
        return 3;
      return 0;
    case 13LL:
      if ( *(_QWORD *)a1 != 0x72705F43435F5744LL || *(_DWORD *)(a1 + 8) != 1634887535 || *(_BYTE *)(a1 + 12) != 109 )
        return 0;
      return 2;
    case 23LL:
      if ( !(*(_QWORD *)a1 ^ 0x61705F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x65725F79625F7373LL)
        && *(_DWORD *)(a1 + 16) == 1701995878
        && *(_WORD *)(a1 + 20) == 25454
        && *(_BYTE *)(a1 + 22) == 101 )
      {
        return 4;
      }
      if ( !(*(_QWORD *)a1 ^ 0x4C4C5F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6C65746E495F4D56LL)
        && *(_DWORD *)(a1 + 16) == 1114399567
        && *(_WORD *)(a1 + 20) == 25449
        && *(_BYTE *)(a1 + 22) == 99 )
      {
        return 197;
      }
      if ( !(*(_QWORD *)a1 ^ 0x4C4C5F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x46726970535F4D56LL)
        && *(_DWORD *)(a1 + 16) == 1952673397
        && *(_WORD *)(a1 + 20) == 28521
        && *(_BYTE *)(a1 + 22) == 110 )
      {
        return 198;
      }
      else
      {
        if ( *(_QWORD *)a1 ^ 0x4C4C5F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x436E65704F5F4D56LL
          || *(_DWORD *)(a1 + 16) != 1919241036
          || *(_WORD *)(a1 + 20) != 25966
          || *(_BYTE *)(a1 + 22) != 108 )
        {
          if ( !(*(_QWORD *)a1 ^ 0x4C4C5F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x65736572505F4D56LL)
            && *(_DWORD *)(a1 + 16) == 1298495090
            && *(_WORD *)(a1 + 20) == 29551
            && *(_BYTE *)(a1 + 22) == 116 )
          {
            return 201;
          }
          if ( !(*(_QWORD *)a1 ^ 0x4C4C5F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x65736572505F4D56LL)
            && *(_DWORD *)(a1 + 16) == 1315272306
            && *(_WORD *)(a1 + 20) == 28271
            && *(_BYTE *)(a1 + 22) == 101 )
          {
            return 205;
          }
          if ( !(*(_QWORD *)a1 ^ 0x4C4C5F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x56435349525F4D56LL)
            && *(_DWORD *)(a1 + 16) == 1129532502
            && *(_WORD *)(a1 + 20) == 27745
            && *(_BYTE *)(a1 + 22) == 108 )
          {
            return 208;
          }
          return 0;
        }
        return 199;
      }
    case 19LL:
      if ( !(*(_QWORD *)a1 ^ 0x61705F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x61765F79625F7373LL)
        && *(_WORD *)(a1 + 16) == 30060
        && *(_BYTE *)(a1 + 18) == 101 )
      {
        return 5;
      }
      return 0;
    case 20LL:
      if ( !(*(_QWORD *)a1 ^ 0x4E475F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6173656E65725F55LL)
        && *(_DWORD *)(a1 + 16) == 1752391539 )
      {
        return 64;
      }
      if ( !(*(_QWORD *)a1 ^ 0x4F425F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x61705F444E414C52LL)
        && *(_DWORD *)(a1 + 16) == 1818321779 )
      {
        return 178;
      }
      if ( !(*(_QWORD *)a1 ^ 0x4C4C5F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x53435041415F4D56LL)
        && *(_DWORD *)(a1 + 16) == 1346786911 )
      {
        return 196;
      }
      if ( !(*(_QWORD *)a1 ^ 0x4C4C5F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x74666977535F4D56LL)
        && *(_DWORD *)(a1 + 16) == 1818845524 )
      {
        return 207;
      }
      if ( !(*(_QWORD *)a1 ^ 0x44475F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x704F5F4D42495F42LL)
        && *(_DWORD *)(a1 + 16) == 1279487589 )
      {
        return 255;
      }
      return 0;
    case 31LL:
      v3 = memcmp((const void *)a1, "DW_CC_GNU_borland_fastcall_i386", 0x1Fu);
      result = 65;
      if ( v3 )
        return 0;
      break;
    case 22LL:
      if ( !(*(_QWORD *)a1 ^ 0x4F425F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x61735F444E414C52LL)
        && *(_DWORD *)(a1 + 16) == 1633903974
        && *(_WORD *)(a1 + 20) == 27756 )
      {
        return 176;
      }
      if ( !(*(_QWORD *)a1 ^ 0x4F425F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x736D5F444E414C52LL)
        && *(_DWORD *)(a1 + 16) == 1970562418
        && *(_WORD *)(a1 + 20) == 28274 )
      {
        return 180;
      }
      if ( *(_QWORD *)a1 ^ 0x4F425F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x68745F444E414C52LL
        || *(_DWORD *)(a1 + 16) != 1633907561
        || *(_WORD *)(a1 + 20) != 27756 )
      {
        if ( !(*(_QWORD *)a1 ^ 0x4F425F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x61665F444E414C52LL)
          && *(_DWORD *)(a1 + 16) == 1633907827
          && *(_WORD *)(a1 + 20) == 27756 )
        {
          return 182;
        }
        if ( !(*(_QWORD *)a1 ^ 0x4C4C5F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x65736572505F4D56LL)
          && *(_DWORD *)(a1 + 16) == 1097168498
          && *(_WORD *)(a1 + 20) == 27756 )
        {
          return 202;
        }
        return 0;
      }
      return 181;
    default:
      switch ( a2 )
      {
        case 21LL:
          if ( !(*(_QWORD *)a1 ^ 0x4F425F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x74735F444E414C52LL)
            && *(_DWORD *)(a1 + 16) == 1818321764
            && *(_BYTE *)(a1 + 20) == 108 )
          {
            return 177;
          }
          if ( !(*(_QWORD *)a1 ^ 0x4C4C5F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6F746365765F4D56LL)
            && *(_DWORD *)(a1 + 16) == 1818321778
            && *(_BYTE *)(a1 + 20) == 108 )
          {
            return 192;
          }
          if ( !(*(_QWORD *)a1 ^ 0x4C4C5F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x365F3638585F4D56LL)
            && *(_DWORD *)(a1 + 16) == 1937330996
            && *(_BYTE *)(a1 + 20) == 86 )
          {
            return 194;
          }
          if ( !(*(_QWORD *)a1 ^ 0x4C4C5F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x65523638585F4D56LL)
            && *(_DWORD *)(a1 + 16) == 1818313575
            && *(_BYTE *)(a1 + 20) == 108 )
          {
            return 203;
          }
          break;
        case 24LL:
          if ( !(*(_QWORD *)a1 ^ 0x4F425F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x736D5F444E414C52LL)
            && *(_QWORD *)(a1 + 16) == 0x6C6C616374736166LL )
          {
            return 179;
          }
          break;
        case 16LL:
          if ( !(*(_QWORD *)a1 ^ 0x4C4C5F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x34366E69575F4D56LL) )
            return 193;
          if ( !(*(_QWORD *)a1 ^ 0x4C4C5F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x53435041415F4D56LL) )
            return 195;
          if ( !(*(_QWORD *)a1 ^ 0x4C4C5F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x74666977535F4D56LL) )
            return 200;
          break;
        case 18LL:
          if ( !(*(_QWORD *)a1 ^ 0x4C4C5F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x526B38364D5F4D56LL)
            && *(_WORD *)(a1 + 16) == 17492 )
          {
            return 204;
          }
          break;
        case 26LL:
          if ( !(*(_QWORD *)a1 ^ 0x4C4C5F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x56435349525F4D56LL)
            && *(_QWORD *)(a1 + 16) == 0x6143726F74636556LL
            && *(_WORD *)(a1 + 24) == 27756 )
          {
            return 206;
          }
          break;
        default:
          return 0;
      }
      return 0;
  }
  return result;
}
