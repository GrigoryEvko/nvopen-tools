// Function: sub_14DD7D0
// Address: 0x14dd7d0
//
__int64 __fastcall sub_14DD7D0(__int64 a1)
{
  __int64 v1; // rax
  char v2; // bl
  __int64 v3; // rax
  __int64 v4; // rdx
  char v5; // si
  char v6; // di
  char v7; // cl
  unsigned int v8; // r8d
  __int64 v10; // rcx

  if ( !a1 )
    return 0;
  v1 = sub_1649C60(a1);
  v2 = *(_BYTE *)(v1 + 16);
  if ( v2 )
    return 0;
  v3 = sub_1649960(v1);
  if ( v4 != 21 )
  {
    if ( v4 == 20 )
    {
      if ( *(_QWORD *)v3 ^ 0x65705F7878675F5FLL | *(_QWORD *)(v3 + 8) ^ 0x74696C616E6F7372LL
        || *(_DWORD *)(v3 + 16) != 813064057 )
      {
        v7 = 1;
        if ( *(_QWORD *)v3 ^ 0x65705F6363675F5FLL | *(_QWORD *)(v3 + 8) ^ 0x74696C616E6F7372LL
          || *(_DWORD *)(v3 + 16) != 813064057 )
        {
          v8 = 4;
          goto LABEL_10;
        }
        return 2;
      }
    }
    else
    {
      if ( v4 != 22 )
      {
        v5 = 0;
        v6 = 0;
        goto LABEL_7;
      }
      if ( *(_QWORD *)v3 ^ 0x65705F7878675F5FLL | *(_QWORD *)(v3 + 8) ^ 0x74696C616E6F7372LL
        || *(_DWORD *)(v3 + 16) != 1702059897
        || *(_WORD *)(v3 + 20) != 12392 )
      {
        if ( *(_QWORD *)v3 ^ 0x65705F6363675F5FLL | *(_QWORD *)(v3 + 8) ^ 0x74696C616E6F7372LL
          || *(_DWORD *)(v3 + 16) != 1702059897
          || *(_WORD *)(v3 + 20) != 12392 )
        {
          v8 = 1;
          v7 = 0;
          goto LABEL_10;
        }
        return 2;
      }
    }
    return 4;
  }
  if ( *(_QWORD *)v3 ^ 0x655F74616E675F5FLL | *(_QWORD *)(v3 + 8) ^ 0x6E6F737265705F68LL
    || *(_DWORD *)(v3 + 16) != 1953066081
    || *(_BYTE *)(v3 + 20) != 121 )
  {
    if ( !(*(_QWORD *)v3 ^ 0x65705F7878675F5FLL | *(_QWORD *)(v3 + 8) ^ 0x74696C616E6F7372LL)
      && *(_DWORD *)(v3 + 16) == 1785945977 )
    {
      v8 = 5;
      if ( *(_BYTE *)(v3 + 20) == 48 )
        return v8;
    }
    v7 = 0;
    v8 = 1;
LABEL_34:
    if ( !(*(_QWORD *)v3 ^ 0x65705F6363675F5FLL | *(_QWORD *)(v3 + 8) ^ 0x74696C616E6F7372LL)
      && *(_DWORD *)(v3 + 16) == 1785945977
      && *(_BYTE *)(v3 + 20) == 48 )
    {
      return 3;
    }
    if ( !(*(_QWORD *)v3 ^ 0x705F636A626F5F5FLL | *(_QWORD *)(v3 + 8) ^ 0x696C616E6F737265LL)
      && *(_DWORD *)(v3 + 16) == 1985968500
      && *(_BYTE *)(v3 + 20) == 48 )
    {
      return 6;
    }
    v2 = 0;
LABEL_10:
    if ( v2 != 1 && v7 )
    {
      if ( !(*(_QWORD *)v3 ^ 0x636570735F435F5FLL | *(_QWORD *)(v3 + 8) ^ 0x6E61685F63696669LL) )
      {
        v8 = 8;
        if ( *(_DWORD *)(v3 + 16) == 1919249508 )
          return v8;
      }
LABEL_13:
      if ( v4 == 19 )
      {
        if ( !(*(_QWORD *)v3 ^ 0x43737365636F7250LL | *(_QWORD *)(v3 + 8) ^ 0x747065637845524CLL)
          && *(_WORD *)(v3 + 16) == 28521 )
        {
          v8 = 10;
          if ( *(_BYTE *)(v3 + 18) == 110 )
            return v8;
        }
        if ( !(*(_QWORD *)v3 ^ 0x5F68655F74737572LL | *(_QWORD *)(v3 + 8) ^ 0x6C616E6F73726570LL)
          && *(_WORD *)(v3 + 16) == 29801 )
        {
          v8 = 11;
          if ( *(_BYTE *)(v3 + 18) == 121 )
            return v8;
        }
      }
      else if ( v4 == 25
             && !(*(_QWORD *)v3 ^ 0x61775F7878675F5FLL | *(_QWORD *)(v3 + 8) ^ 0x6F737265705F6D73LL)
             && *(_QWORD *)(v3 + 16) == 0x765F7974696C616ELL )
      {
        v8 = 12;
        if ( *(_BYTE *)(v3 + 24) == 48 )
          return v8;
      }
      return 0;
    }
    if ( v4 == 18 && v2 != 1 )
    {
      if ( !(*(_QWORD *)v3 ^ 0x6172467878435F5FLL | *(_QWORD *)(v3 + 8) ^ 0x656C646E6148656DLL) )
      {
        v8 = 9;
        if ( *(_WORD *)(v3 + 16) == 13170 )
          return v8;
      }
      return 0;
    }
    goto LABEL_45;
  }
  v5 = 1;
  v6 = 1;
  v2 = 1;
LABEL_7:
  v7 = 0;
  v8 = 1;
  if ( v5 )
    return v8;
  if ( v6 )
    goto LABEL_34;
  if ( v4 != 16 )
    goto LABEL_10;
  v10 = *(_QWORD *)v3 ^ 0x5F7470656378655FLL;
  if ( !(v10 | *(_QWORD *)(v3 + 8) ^ 0x3372656C646E6168LL) || !(v10 | *(_QWORD *)(v3 + 8) ^ 0x3472656C646E6168LL) )
    return 7;
LABEL_45:
  if ( !v2 )
    goto LABEL_13;
  return v8;
}
