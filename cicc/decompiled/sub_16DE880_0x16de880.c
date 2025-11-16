// Function: sub_16DE880
// Address: 0x16de880
//
__int64 __fastcall sub_16DE880(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  char v5; // r10
  bool v6; // bl
  char v7; // r9
  char v8; // dl
  char v9; // r13
  char v10; // al
  char v11; // cl
  char v12; // r12
  char v13; // r11
  char v14; // al
  char v15; // r13
  char v16; // r11
  char v17; // al
  bool v18; // dl
  unsigned __int8 v19; // al
  char v20; // r12
  char v21; // r9
  char v22; // bl
  char v23; // dl
  unsigned int v24; // r8d
  int v26; // eax
  char v27; // al
  char v28; // dl

  v5 = a2 > 5;
  v6 = a2 > 7;
  if ( a2 <= 5 )
  {
    v5 = 0;
  }
  else
  {
    if ( *(_DWORD *)a1 == 1851879009 && *(_WORD *)(a1 + 4) == 29537 )
    {
      v5 = 1;
      a5 = 1;
      goto LABEL_90;
    }
    if ( a2 > 7 )
    {
      if ( *(_QWORD *)a1 == 0x69626164756F6C63LL )
      {
        a5 = 2;
LABEL_90:
        v11 = 1;
        v9 = 1;
        v7 = a2 > 2;
        v10 = 0;
LABEL_91:
        v8 = v6 & v10;
        goto LABEL_12;
      }
    }
    else
    {
      v5 = 1;
    }
    if ( *(_DWORD *)a1 == 2003984740 && *(_WORD *)(a1 + 4) == 28265 )
    {
      a5 = 3;
      goto LABEL_90;
    }
    if ( a2 > 8 )
    {
      if ( *(_QWORD *)a1 == 0x6C666E6F67617264LL && *(_BYTE *)(a1 + 8) == 121 )
      {
        v11 = 1;
        a5 = 4;
        v9 = 1;
        v7 = a2 > 2;
        v10 = 0;
        goto LABEL_91;
      }
LABEL_55:
      if ( *(_DWORD *)a1 != 1701147238 || *(_WORD *)(a1 + 4) != 29538 || (v26 = 0, *(_BYTE *)(a1 + 6) != 100) )
        v26 = 1;
      v7 = a2 > 2;
      if ( v26 )
      {
        if ( *(_DWORD *)a1 != 1751348582 || *(_WORD *)(a1 + 4) != 26995 || *(_BYTE *)(a1 + 6) != 97 )
          goto LABEL_10;
        a5 = 6;
      }
      else
      {
        a5 = 5;
      }
      v8 = 1;
      v11 = 1;
      v12 = a2 > 4;
      v10 = 0;
      goto LABEL_60;
    }
  }
  if ( a2 > 6 )
    goto LABEL_55;
  v7 = a2 > 2;
  if ( a2 <= 2 )
  {
    v11 = 0;
    v10 = 1;
    v9 = 0;
    goto LABEL_91;
  }
  v7 = 1;
LABEL_10:
  if ( *(_WORD *)a1 == 28521 && *(_BYTE *)(a1 + 2) == 115 )
  {
    v8 = 1;
    v11 = 1;
    a5 = 7;
    v12 = a2 > 4;
    v10 = 0;
    goto LABEL_60;
  }
  v8 = a2 > 7;
  v9 = 0;
  v10 = 1;
  v11 = 0;
LABEL_12:
  v12 = a2 > 4;
  if ( v8 )
  {
    if ( *(_QWORD *)a1 == 0x647362656572666BLL )
    {
      v10 = 0;
      v11 = 1;
      a5 = 8;
      goto LABEL_60;
    }
    v13 = a2 > 4;
  }
  else
  {
    v13 = v10 & v12;
    if ( ((unsigned __int8)v10 & (unsigned __int8)v12) == 0 )
    {
      if ( v10 && v7 )
      {
LABEL_16:
        if ( *(_WORD *)a1 == 30316 && *(_BYTE *)(a1 + 2) == 50 )
        {
          v7 = 1;
          v11 = 1;
          a5 = 10;
          goto LABEL_63;
        }
        v8 = v11;
        v14 = v11 ^ 1;
        v7 = v12 & (v11 ^ 1);
        if ( v7 )
        {
          v12 &= v11 ^ 1;
          goto LABEL_19;
        }
        v13 = v12;
        v7 = 1;
        goto LABEL_108;
      }
      v8 = v9;
LABEL_60:
      v27 = v5 & v10;
      goto LABEL_61;
    }
  }
  if ( *(_DWORD *)a1 != 1970170220 || *(_BYTE *)(a1 + 4) != 120 )
  {
    if ( v11 || !v7 )
    {
      v12 = v13;
LABEL_19:
      if ( *(_DWORD *)a1 == 1868783981 && *(_BYTE *)(a1 + 4) == 115 )
      {
        v11 = 1;
        a5 = 11;
        goto LABEL_63;
      }
      if ( ((unsigned __int8)v5 & ((unsigned __int8)v11 ^ 1)) == 0 )
        goto LABEL_21;
      v5 &= v11 ^ 1;
      goto LABEL_127;
    }
    v12 = v13;
    goto LABEL_16;
  }
  v14 = 0;
  v8 = 1;
  a5 = 9;
  v11 = 1;
LABEL_108:
  v27 = v5 & v14;
  v12 = v13;
LABEL_61:
  if ( v27 )
  {
LABEL_127:
    if ( *(_DWORD *)a1 == 1651795310 && *(_WORD *)(a1 + 4) == 25715 )
    {
      v11 = 1;
      a5 = 12;
      goto LABEL_63;
    }
    goto LABEL_21;
  }
  if ( v8 )
  {
LABEL_63:
    v16 = a2 > 6;
LABEL_64:
    v15 = v12;
LABEL_65:
    v17 = 1;
LABEL_66:
    v18 = a2 > 3;
    goto LABEL_67;
  }
LABEL_21:
  if ( a2 <= 6 )
  {
    v15 = v11;
    v16 = v12 & (v11 ^ 1);
    if ( v16 )
    {
      v16 = 0;
      if ( *(_DWORD *)a1 == 862873975 && *(_BYTE *)(a1 + 4) == 50 )
        goto LABEL_190;
LABEL_28:
      if ( *(_DWORD *)a1 == 1802068328 && *(_BYTE *)(a1 + 4) == 117 )
      {
        v17 = 1;
        v15 = 1;
        v11 = 1;
        a5 = 16;
        goto LABEL_66;
      }
      if ( *(_DWORD *)a1 == 1768843629 && *(_BYTE *)(a1 + 4) == 120 )
      {
        v17 = 1;
        v15 = 1;
        v11 = 1;
        a5 = 17;
        goto LABEL_66;
      }
      if ( *(_DWORD *)a1 == 1835365490 && *(_BYTE *)(a1 + 4) == 115 )
      {
        v17 = 1;
        v15 = 1;
        v11 = 1;
        a5 = 18;
        goto LABEL_66;
      }
      v17 = v11;
      v18 = a2 > 3;
      v15 = (a2 > 3) & (v11 ^ 1);
      if ( !v15 )
      {
        v15 = 1;
LABEL_67:
        if ( !v17 )
          goto LABEL_68;
LABEL_34:
        v19 = 0;
        v20 = 1;
        goto LABEL_35;
      }
      if ( *(_DWORD *)a1 == 1818452334 )
        goto LABEL_33;
      goto LABEL_146;
    }
  }
  else
  {
    if ( *(_DWORD *)a1 == 1852141679 && *(_WORD *)(a1 + 4) == 29538 && *(_BYTE *)(a1 + 6) == 100 )
    {
      a5 = 13;
      goto LABEL_174;
    }
    if ( *(_DWORD *)a1 == 1634496371 && *(_WORD *)(a1 + 4) == 26994 && *(_BYTE *)(a1 + 6) == 115 )
    {
      a5 = 14;
LABEL_174:
      v15 = v12;
      v16 = 1;
      v11 = 1;
      goto LABEL_65;
    }
    v15 = v11;
    v16 = v12 & (v11 ^ 1);
    if ( v16 )
    {
      if ( *(_DWORD *)a1 == 862873975 && *(_BYTE *)(a1 + 4) == 50 )
      {
LABEL_190:
        a5 = 15;
        v15 = 1;
        v11 = 1;
        goto LABEL_65;
      }
      if ( v11 )
      {
        v16 = v11;
        goto LABEL_28;
      }
      v12 = v16;
      goto LABEL_151;
    }
    v16 = a2 > 6;
    if ( v11 != 1 )
    {
LABEL_151:
      if ( *(_DWORD *)a1 == 1684957559 && *(_WORD *)(a1 + 4) == 30575 && *(_BYTE *)(a1 + 6) == 115 )
      {
        v15 = v12;
        v17 = 1;
        v11 = 1;
        v16 = 1;
        a5 = 15;
        goto LABEL_66;
      }
      v15 = v11;
      v16 = 1;
      goto LABEL_136;
    }
  }
  if ( v11 )
    goto LABEL_64;
LABEL_136:
  if ( v12 )
    goto LABEL_28;
  v17 = v11;
  if ( v11 )
  {
    v15 = 0;
    goto LABEL_66;
  }
  v18 = a2 > 3;
  if ( a2 <= 3 )
  {
LABEL_68:
    if ( !v7 )
      goto LABEL_72;
    goto LABEL_69;
  }
  if ( *(_DWORD *)a1 == 1818452334 )
  {
LABEL_33:
    v18 = 1;
    v11 = 1;
    a5 = 19;
    goto LABEL_34;
  }
LABEL_146:
  if ( !v7 )
    goto LABEL_141;
  v18 = v7;
LABEL_69:
  if ( *(_WORD *)a1 == 28259 && *(_BYTE *)(a1 + 2) == 107 )
  {
    a5 = 20;
LABEL_185:
    v7 = 1;
    v11 = 1;
    goto LABEL_34;
  }
  if ( *(_WORD *)a1 == 26977 && *(_BYTE *)(a1 + 2) == 120 )
  {
    a5 = 21;
    goto LABEL_185;
  }
  v7 = 1;
LABEL_72:
  if ( !v18 )
  {
    v19 = v11 ^ 1;
    v20 = (v11 ^ 1) & v5;
    goto LABEL_74;
  }
LABEL_141:
  if ( *(_DWORD *)a1 == 1633973603 )
  {
    a5 = 22;
LABEL_143:
    v18 = 1;
    v20 = 1;
    v19 = 0;
    v11 = 1;
    goto LABEL_36;
  }
  if ( *(_DWORD *)a1 == 1818457710 )
  {
    a5 = 23;
    goto LABEL_143;
  }
  v18 = 1;
  v19 = v11 ^ 1;
  v20 = (v11 ^ 1) & v5;
LABEL_74:
  if ( !v20 )
  {
    v20 = v11;
LABEL_35:
    v21 = v19 & v7;
    if ( !v21 )
      goto LABEL_36;
LABEL_77:
    if ( *(_WORD *)a1 == 29552 && *(_BYTE *)(a1 + 2) == 52 )
    {
      v20 = v21;
      a5 = 27;
      v19 = 0;
      v11 = 1;
      goto LABEL_38;
    }
    goto LABEL_36;
  }
  if ( *(_DWORD *)a1 == 1751412065 && *(_WORD *)(a1 + 4) == 24947 )
  {
    v19 = 0;
    v11 = 1;
    a5 = 26;
    goto LABEL_37;
  }
  v21 = v19 & v7;
  v20 = v11;
  if ( v21 )
    goto LABEL_77;
LABEL_36:
  v22 = v19 & v6;
  if ( v22 && *(_QWORD *)a1 == 0x75636D6169666C65LL )
  {
    v19 = 0;
    v20 = v22;
    v11 = 1;
    a5 = 28;
    v28 = 0;
LABEL_117:
    if ( v28 )
      goto LABEL_118;
LABEL_43:
    if ( !v19 || !v5 )
      goto LABEL_122;
    goto LABEL_45;
  }
LABEL_37:
  v23 = v19 & v18;
  if ( v23 && *(_DWORD *)a1 == 1936684660 )
  {
    v20 = v23;
    v19 = 0;
    v11 = 1;
    a5 = 29;
LABEL_116:
    v28 = v16 & v19;
    goto LABEL_117;
  }
LABEL_38:
  if ( (v19 & (unsigned __int8)v16) != 0 )
  {
    if ( *(_DWORD *)a1 == 1668571511 && *(_WORD *)(a1 + 4) == 28520 && *(_BYTE *)(a1 + 6) == 115 )
      return 30;
    if ( !v5 || !v19 )
      goto LABEL_118;
  }
  else if ( !v5 || !v19 )
  {
    goto LABEL_116;
  }
  if ( *(_DWORD *)a1 == 1634952557 && *(_WORD *)(a1 + 4) == 25651 )
    return 31;
  v20 = v11;
  v5 = 1;
  v19 = v11 ^ 1;
  if ( ((unsigned __int8)v16 & ((unsigned __int8)v11 ^ 1)) == 0 )
    goto LABEL_43;
LABEL_118:
  if ( *(_DWORD *)a1 == 1953394531 && *(_WORD *)(a1 + 4) == 27497 && *(_BYTE *)(a1 + 6) == 105 )
    return 32;
  if ( v11 || !v5 )
    goto LABEL_48;
LABEL_45:
  if ( *(_DWORD *)a1 == 1885629793 && *(_WORD *)(a1 + 4) == 27745 )
    return 33;
  v20 = v11;
  v19 = v11 ^ 1;
  if ( v11 != 1 && v16 )
  {
LABEL_48:
    if ( *(_DWORD *)a1 == 1701996900 && *(_WORD *)(a1 + 4) == 29795 )
    {
      v24 = 24;
      if ( *(_BYTE *)(a1 + 6) == 120 )
        return v24;
    }
    if ( v11 || !v15 )
      return 0;
LABEL_111:
    if ( *(_DWORD *)a1 == 1919512691 )
    {
      v24 = 25;
      if ( *(_BYTE *)(a1 + 4) == 118 )
        return v24;
    }
    return 0;
  }
LABEL_122:
  if ( (v19 & (unsigned __int8)v15) != 0 )
    goto LABEL_111;
  if ( !v20 )
    return 0;
  return a5;
}
