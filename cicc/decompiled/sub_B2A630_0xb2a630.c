// Function: sub_B2A630
// Address: 0xb2a630
//
__int64 __fastcall sub_B2A630(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rdx

  if ( !a1 )
    return 0;
  v1 = sub_BD3990(a1);
  v2 = v1;
  if ( *(_BYTE *)v1 > 3u )
    return 0;
  v3 = *(_QWORD *)(v1 + 24);
  if ( !v3 || *(_BYTE *)(v3 + 8) != 13 )
    return 0;
  v5 = sub_BD5D20(v2);
  v7 = *(_QWORD *)(v2 + 40);
  if ( *(_DWORD *)(v7 + 264) == 3 && *(_DWORD *)(v7 + 268) == 36 )
  {
    if ( !v6 )
      return 0;
    if ( *(_BYTE *)v5 == 35 )
    {
      --v6;
      ++v5;
    }
  }
  switch ( v6 )
  {
    case 21LL:
      if ( !(*(_QWORD *)v5 ^ 0x655F74616E675F5FLL | *(_QWORD *)(v5 + 8) ^ 0x6E6F737265705F68LL)
        && *(_DWORD *)(v5 + 16) == 1953066081
        && *(_BYTE *)(v5 + 20) == 121 )
      {
        return 1;
      }
      else if ( !(*(_QWORD *)v5 ^ 0x65705F7878675F5FLL | *(_QWORD *)(v5 + 8) ^ 0x74696C616E6F7372LL)
             && *(_DWORD *)(v5 + 16) == 1785945977
             && *(_BYTE *)(v5 + 20) == 48 )
      {
        return 5;
      }
      else if ( !(*(_QWORD *)v5 ^ 0x65705F6363675F5FLL | *(_QWORD *)(v5 + 8) ^ 0x74696C616E6F7372LL)
             && *(_DWORD *)(v5 + 16) == 1785945977
             && *(_BYTE *)(v5 + 20) == 48 )
      {
        return 3;
      }
      else
      {
        if ( *(_QWORD *)v5 ^ 0x705F636A626F5F5FLL | *(_QWORD *)(v5 + 8) ^ 0x696C616E6F737265LL
          || *(_DWORD *)(v5 + 16) != 1985968500
          || *(_BYTE *)(v5 + 20) != 48 )
        {
          return 0;
        }
        return 6;
      }
    case 20LL:
      if ( *(_QWORD *)v5 ^ 0x65705F7878675F5FLL | *(_QWORD *)(v5 + 8) ^ 0x74696C616E6F7372LL
        || *(_DWORD *)(v5 + 16) != 813064057 )
      {
        if ( *(_QWORD *)v5 ^ 0x65705F6363675F5FLL | *(_QWORD *)(v5 + 8) ^ 0x74696C616E6F7372LL
          || *(_DWORD *)(v5 + 16) != 813064057 )
        {
          if ( !(*(_QWORD *)v5 ^ 0x636570735F435F5FLL | *(_QWORD *)(v5 + 8) ^ 0x6E61685F63696669LL)
            && *(_DWORD *)(v5 + 16) == 1919249508 )
          {
            return 8;
          }
          return 0;
        }
        return 2;
      }
      return 4;
    case 22LL:
      if ( !(*(_QWORD *)v5 ^ 0x65705F7878675F5FLL | *(_QWORD *)(v5 + 8) ^ 0x74696C616E6F7372LL)
        && *(_DWORD *)(v5 + 16) == 1702059897
        && *(_WORD *)(v5 + 20) == 12392 )
      {
        return 4;
      }
      if ( !(*(_QWORD *)v5 ^ 0x65705F6363675F5FLL | *(_QWORD *)(v5 + 8) ^ 0x74696C616E6F7372LL)
        && *(_DWORD *)(v5 + 16) == 1702059897
        && *(_WORD *)(v5 + 20) == 12392 )
      {
        return 2;
      }
      if ( *(_QWORD *)v5 ^ 0x5F7878636C785F5FLL | *(_QWORD *)(v5 + 8) ^ 0x6C616E6F73726570LL
        || *(_DWORD *)(v5 + 16) != 1601795177
        || *(_WORD *)(v5 + 20) != 12662 )
      {
        return 0;
      }
      return 13;
    case 16LL:
      v8 = *(_QWORD *)v5 ^ 0x5F7470656378655FLL;
      if ( v8 | *(_QWORD *)(v5 + 8) ^ 0x3372656C646E6168LL && v8 | *(_QWORD *)(v5 + 8) ^ 0x3472656C646E6168LL )
        return 0;
      return 7;
    case 18LL:
      if ( *(_QWORD *)v5 ^ 0x6172467878435F5FLL | *(_QWORD *)(v5 + 8) ^ 0x656C646E6148656DLL
        || *(_WORD *)(v5 + 16) != 13170 )
      {
        return 0;
      }
      return 9;
    case 19LL:
      if ( !(*(_QWORD *)v5 ^ 0x43737365636F7250LL | *(_QWORD *)(v5 + 8) ^ 0x747065637845524CLL)
        && *(_WORD *)(v5 + 16) == 28521
        && *(_BYTE *)(v5 + 18) == 110 )
      {
        return 10;
      }
      else
      {
        if ( *(_QWORD *)v5 ^ 0x5F68655F74737572LL | *(_QWORD *)(v5 + 8) ^ 0x6C616E6F73726570LL
          || *(_WORD *)(v5 + 16) != 29801
          || *(_BYTE *)(v5 + 18) != 121 )
        {
          return 0;
        }
        return 11;
      }
    case 25LL:
      if ( *(_QWORD *)v5 ^ 0x61775F7878675F5FLL | *(_QWORD *)(v5 + 8) ^ 0x6F737265705F6D73LL
        || *(_QWORD *)(v5 + 16) != 0x765F7974696C616ELL
        || *(_BYTE *)(v5 + 24) != 48 )
      {
        return 0;
      }
      return 12;
    default:
      if ( v6 != 24
        || *(_QWORD *)v5 ^ 0x78635F736F7A5F5FLL | *(_QWORD *)(v5 + 8) ^ 0x6E6F737265705F78LL
        || *(_QWORD *)(v5 + 16) != 0x32765F7974696C61LL )
      {
        return 0;
      }
      return 14;
  }
}
