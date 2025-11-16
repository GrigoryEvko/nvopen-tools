// Function: sub_14E8B30
// Address: 0x14e8b30
//
__int64 __fastcall sub_14E8B30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  bool v5; // r10
  bool v6; // r15
  char v7; // r13
  char v8; // r12
  char v9; // r11
  bool v10; // cl
  char v11; // al
  char v12; // r14
  char v13; // al
  bool v14; // al
  char v15; // dl
  char v16; // dl
  char v17; // r10
  char v18; // al
  __int64 v19; // rdx
  char v20; // cl
  int v22; // eax
  unsigned int v23; // [rsp+0h] [rbp-40h]
  char v24; // [rsp+4h] [rbp-3Ch]

  v5 = a2 == 23;
  if ( a2 == 12 )
  {
    if ( *(_QWORD *)a1 == 0x6F6E5F43435F5744LL && *(_DWORD *)(a1 + 8) == 1818324338 )
    {
      v6 = 0;
      v7 = 1;
      a5 = 1;
      v8 = 0;
      v9 = 1;
    }
    else
    {
      if ( *(_QWORD *)a1 != 0x6F6E5F43435F5744LL || *(_DWORD *)(a1 + 8) != 1819042147 )
      {
        v6 = 0;
        v8 = 1;
        v7 = 0;
        v9 = 0;
        a5 = 1;
        goto LABEL_7;
      }
      v7 = 1;
      a5 = 3;
      v6 = 0;
      v8 = 0;
      v9 = 1;
    }
    goto LABEL_6;
  }
  if ( a2 != 13 )
  {
    if ( a2 == 23 )
    {
      if ( !(*(_QWORD *)a1 ^ 0x61705F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x65725F79625F7373LL)
        && *(_DWORD *)(a1 + 16) == 1701995878
        && *(_WORD *)(a1 + 20) == 25454
        && *(_BYTE *)(a1 + 22) == 101 )
      {
        v6 = 0;
        v7 = 1;
        v8 = 0;
        v9 = 1;
        a5 = 4;
      }
      else
      {
        v6 = 0;
        v8 = 1;
        v7 = 0;
        v9 = 0;
      }
      goto LABEL_7;
    }
    v6 = a2 == 20;
    if ( a2 == 19 )
    {
      if ( !(*(_QWORD *)a1 ^ 0x61705F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x61765F79625F7373LL)
        && *(_WORD *)(a1 + 16) == 30060
        && *(_BYTE *)(a1 + 18) == 101 )
      {
        v7 = 1;
        v10 = 0;
        a5 = 5;
        v8 = 0;
        v9 = 1;
      }
      else
      {
        v10 = 0;
        v7 = 0;
        v8 = 1;
        v9 = 0;
      }
LABEL_8:
      v11 = v10 & v8;
      v12 = a2 == 21;
      if ( (v10 & (unsigned __int8)v8) == 0 )
        goto LABEL_9;
      goto LABEL_52;
    }
    goto LABEL_5;
  }
  if ( *(_QWORD *)a1 != 0x72705F43435F5744LL || *(_DWORD *)(a1 + 8) != 1634887535 || *(_BYTE *)(a1 + 12) != 109 )
  {
    a5 = 2;
    v6 = 0;
LABEL_5:
    v7 = 0;
    v8 = 1;
    v9 = 0;
    goto LABEL_6;
  }
  v6 = 0;
  v7 = 1;
  a5 = 2;
  v8 = 0;
  v9 = 1;
LABEL_6:
  if ( ((unsigned __int8)v8 & v6) != 0 )
  {
    v10 = a2 == 22;
    if ( !(*(_QWORD *)a1 ^ 0x4E475F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6173656E65725F55LL)
      && *(_DWORD *)(a1 + 16) == 1752391539 )
    {
      v7 = v8 & v6;
      a5 = 64;
      v9 = 1;
      v12 = a2 == 21;
      v8 = 0;
      goto LABEL_9;
    }
    v11 = v8 & v10;
    v12 = a2 == 21;
    if ( ((unsigned __int8)v8 & v10) == 0 )
    {
LABEL_9:
      v13 = v8 & v12;
      goto LABEL_10;
    }
LABEL_52:
    if ( !(*(_QWORD *)a1 ^ 0x4F425F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x61735F444E414C52LL)
      && *(_DWORD *)(a1 + 16) == 1633903974
      && *(_WORD *)(a1 + 20) == 27756 )
    {
      v7 = v11;
      a5 = 176;
      v8 = 0;
      v9 = 1;
      goto LABEL_12;
    }
    goto LABEL_11;
  }
LABEL_7:
  v10 = a2 == 22;
  if ( ((unsigned __int8)v8 & (a2 == 31)) == 0 )
    goto LABEL_8;
  v12 = a2 == 21;
  v23 = a5;
  v24 = v9;
  v22 = memcmp((const void *)a1, "DW_CC_GNU_borland_fastcall_i386", 0x1Fu);
  v5 = a2 == 23;
  v10 = a2 == 22;
  if ( !v22 )
  {
    v7 = v8 & (a2 == 31);
    v9 = 1;
    v8 = 0;
    a5 = 65;
    goto LABEL_11;
  }
  v9 = v24;
  a5 = v23;
  v13 = v12 & v8;
LABEL_10:
  if ( v13 )
  {
    if ( !(*(_QWORD *)a1 ^ 0x4F425F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x74735F444E414C52LL)
      && *(_DWORD *)(a1 + 16) == 1818321764
      && *(_BYTE *)(a1 + 20) == 108 )
    {
      v9 = 1;
      a5 = 177;
      goto LABEL_44;
    }
    goto LABEL_12;
  }
LABEL_11:
  if ( (v6 & (unsigned __int8)v8) != 0 )
  {
    if ( !(*(_QWORD *)a1 ^ 0x4F425F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x61705F444E414C52LL)
      && *(_DWORD *)(a1 + 16) == 1818321779 )
    {
      v9 = 1;
      a5 = 178;
      goto LABEL_44;
    }
LABEL_13:
    v14 = a2 == 16;
    if ( !v7 )
    {
      v8 = 1;
      if ( !v10 )
      {
LABEL_15:
        v14 = a2 == 16;
        v15 = v8 & (a2 == 16);
        v7 = v12 & v8;
        if ( ((unsigned __int8)v12 & (unsigned __int8)v8) == 0 )
          goto LABEL_16;
        if ( *(_QWORD *)a1 ^ 0x4C4C5F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6F746365765F4D56LL
          || *(_DWORD *)(a1 + 16) != 1818321778
          || *(_BYTE *)(a1 + 20) != 108 )
        {
          v7 = 0;
          goto LABEL_93;
        }
        v12 &= v8;
        v8 = 0;
        v9 = 1;
        a5 = 192;
LABEL_19:
        v16 = v6 & v8;
        goto LABEL_20;
      }
      if ( !(*(_QWORD *)a1 ^ 0x4F425F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x736D5F444E414C52LL)
        && *(_DWORD *)(a1 + 16) == 1970562418
        && *(_WORD *)(a1 + 20) == 28274 )
      {
        v9 = 1;
        a5 = 180;
      }
      else if ( !(*(_QWORD *)a1 ^ 0x4F425F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x68745F444E414C52LL)
             && *(_DWORD *)(a1 + 16) == 1633907561
             && *(_WORD *)(a1 + 20) == 27756 )
      {
        v9 = 1;
        a5 = 181;
      }
      else
      {
        if ( *(_QWORD *)a1 ^ 0x4F425F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x61665F444E414C52LL
          || *(_DWORD *)(a1 + 16) != 1633907827
          || *(_WORD *)(a1 + 20) != 27756 )
        {
          v8 = v10;
          goto LABEL_15;
        }
        v9 = 1;
        a5 = 182;
        v14 = a2 == 16;
      }
LABEL_45:
      v7 = 1;
      v8 = 0;
      goto LABEL_19;
    }
LABEL_44:
    v14 = a2 == 16;
    goto LABEL_45;
  }
LABEL_12:
  if ( ((unsigned __int8)v8 & (a2 == 24)) == 0 )
    goto LABEL_13;
  if ( !(*(_QWORD *)a1 ^ 0x4F425F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x736D5F444E414C52LL)
    && *(_QWORD *)(a1 + 16) == 0x6C6C616374736166LL )
  {
    v9 = 1;
    a5 = 179;
    v14 = a2 == 16;
    goto LABEL_45;
  }
  if ( !v7 )
    goto LABEL_15;
  v14 = a2 == 16;
  v15 = (a2 == 16) & v8;
LABEL_16:
  if ( v15 )
  {
    v19 = *(_QWORD *)a1 ^ 0x4C4C5F43435F5744LL;
    if ( !(v19 | *(_QWORD *)(a1 + 8) ^ 0x34366E69575F4D56LL) )
    {
      a5 = 193;
      goto LABEL_33;
    }
    if ( !(v19 | *(_QWORD *)(a1 + 8) ^ 0x53435041415F4D56LL) )
    {
      a5 = 195;
      goto LABEL_33;
    }
LABEL_21:
    if ( !v7 )
    {
      if ( a2 != 23 )
      {
LABEL_23:
        v7 = 1;
        goto LABEL_24;
      }
      if ( !(*(_QWORD *)a1 ^ 0x4C4C5F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6C65746E495F4D56LL)
        && *(_DWORD *)(a1 + 16) == 1114399567
        && *(_WORD *)(a1 + 20) == 25449
        && *(_BYTE *)(a1 + 22) == 99 )
      {
        a5 = 197;
        v18 = 1;
      }
      else if ( !(*(_QWORD *)a1 ^ 0x4C4C5F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x46726970535F4D56LL)
             && *(_DWORD *)(a1 + 16) == 1952673397
             && *(_WORD *)(a1 + 20) == 28521
             && *(_BYTE *)(a1 + 22) == 110 )
      {
        a5 = 198;
        v18 = 1;
      }
      else
      {
        if ( *(_QWORD *)a1 ^ 0x4C4C5F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x436E65704F5F4D56LL
          || *(_DWORD *)(a1 + 16) != 1919241036
          || *(_WORD *)(a1 + 20) != 25966
          || *(_BYTE *)(a1 + 22) != 108 )
        {
          goto LABEL_23;
        }
        a5 = 199;
        v18 = 1;
      }
LABEL_34:
      v20 = v7 & v10;
      goto LABEL_35;
    }
LABEL_33:
    v7 = 0;
    v18 = 1;
    goto LABEL_34;
  }
  if ( !v8 || !v12 )
    goto LABEL_19;
LABEL_93:
  if ( !(*(_QWORD *)a1 ^ 0x4C4C5F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x365F3638585F4D56LL)
    && *(_DWORD *)(a1 + 16) == 1937330996
    && *(_BYTE *)(a1 + 20) == 86 )
  {
    v12 = 1;
    a5 = 194;
    goto LABEL_33;
  }
  v16 = v6;
  v8 = 1;
  v12 = 1;
LABEL_20:
  if ( !v16 )
    goto LABEL_21;
  if ( !(*(_QWORD *)a1 ^ 0x4C4C5F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x53435041415F4D56LL)
    && *(_DWORD *)(a1 + 16) == 1346786911 )
  {
    a5 = 196;
    goto LABEL_33;
  }
  if ( v7 )
  {
    v17 = v8 & v5;
    v18 = v7;
    v7 = v8;
    if ( v17 )
      goto LABEL_26;
    goto LABEL_34;
  }
  v7 = v8;
LABEL_24:
  v17 = v7 & v5;
  v18 = v7 & v14;
  if ( !v18 )
  {
    if ( v17 )
    {
LABEL_26:
      if ( !(*(_QWORD *)a1 ^ 0x4C4C5F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x65736572505F4D56LL)
        && *(_DWORD *)(a1 + 16) == 1298495090
        && *(_WORD *)(a1 + 20) == 29551
        && *(_BYTE *)(a1 + 22) == 116 )
      {
        v18 = v17;
        a5 = 201;
        v7 = 0;
        goto LABEL_37;
      }
      goto LABEL_36;
    }
    goto LABEL_34;
  }
  if ( !(*(_QWORD *)a1 ^ 0x4C4C5F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x74666977535F4D56LL) )
  {
    a5 = 200;
    v7 = 0;
    goto LABEL_36;
  }
  v18 = v9;
  v7 = v9 ^ 1;
  v20 = (v9 ^ 1) & v10;
LABEL_35:
  if ( v20 )
  {
    if ( !(*(_QWORD *)a1 ^ 0x4C4C5F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x65736572505F4D56LL)
      && *(_DWORD *)(a1 + 16) == 1097168498
      && *(_WORD *)(a1 + 20) == 27756 )
    {
      return 202;
    }
    goto LABEL_37;
  }
LABEL_36:
  if ( ((unsigned __int8)v7 & (unsigned __int8)v12) != 0 )
  {
    if ( !(*(_QWORD *)a1 ^ 0x4C4C5F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x65523638585F4D56LL)
      && *(_DWORD *)(a1 + 16) == 1818313575
      && *(_BYTE *)(a1 + 20) == 108 )
    {
      return 203;
    }
    goto LABEL_38;
  }
LABEL_37:
  if ( (v6 & (unsigned __int8)v7) == 0 )
  {
LABEL_38:
    if ( !v18 )
      return 0;
    return a5;
  }
  if ( *(_QWORD *)a1 ^ 0x44475F43435F5744LL | *(_QWORD *)(a1 + 8) ^ 0x704F5F4D42495F42LL )
    return 0;
  a5 = 255;
  if ( *(_DWORD *)(a1 + 16) != 1279487589 )
    return 0;
  return a5;
}
