// Function: sub_16DF2D0
// Address: 0x16df2d0
//
__int64 __fastcall sub_16DF2D0(__int64 a1, unsigned __int64 a2)
{
  unsigned int v4; // r14d
  char v5; // cl
  char v6; // di
  bool v7; // r11
  char v8; // dl
  char v9; // r9
  char v10; // al
  char v11; // si
  char v12; // r8
  char v13; // si
  bool v14; // r10
  char v15; // r8
  char v16; // r8
  char v17; // bl
  char v18; // al
  char v19; // r15
  bool v20; // r8
  bool v21; // bl
  char v22; // dl
  char v23; // dl
  char v24; // al
  char v25; // al
  char v26; // r10
  char v27; // dl
  bool v28; // dl
  char v30; // bl
  char v31; // r15
  char v32; // bl
  char v33; // r15
  char v34; // r15
  char v35; // si
  int v36; // r15d
  int v37; // eax
  int v38; // ebx
  __int64 v39; // rax
  _WORD *v40; // rax
  __int64 v41; // rdx
  _WORD *v42; // r13
  __int64 v43; // r12
  int v44; // r15d
  int v45; // eax
  __int64 v46; // rax

  if ( a2 != 4 )
  {
    if ( a2 != 5 )
    {
      if ( a2 == 6 )
      {
        if ( *(_DWORD *)a1 == 1597388920 && *(_WORD *)(a1 + 4) == 13366 )
        {
          v4 = 32;
          v5 = 0;
          v6 = 1;
          goto LABEL_20;
        }
        v4 = 31;
        v5 = 0;
        v6 = 1;
LABEL_6:
        v7 = 0;
        v8 = 0;
        v9 = 0;
        v10 = 1;
LABEL_21:
        v11 = v10 & (a2 == 9);
        if ( !v11 )
        {
LABEL_22:
          if ( ((unsigned __int8)v10 & v7) == 0 )
            goto LABEL_23;
          if ( *(_WORD *)a1 == 28784 && *(_BYTE *)(a1 + 2) == 117 )
          {
            v8 = v10 & v7;
            v9 = 1;
            v4 = 17;
            v13 = a2 == 7;
            v10 = 0;
            goto LABEL_25;
          }
LABEL_24:
          v13 = a2 == 7;
          if ( ((unsigned __int8)v10 & (a2 == 11)) != 0 )
          {
            if ( *(_QWORD *)a1 == 0x3663707265776F70LL && *(_WORD *)(a1 + 8) == 27700 && *(_BYTE *)(a1 + 10) == 101 )
            {
              v8 = v10 & (a2 == 11);
              v9 = 1;
              v4 = 18;
              v14 = a2 == 8;
              v10 = 0;
LABEL_27:
              v15 = v14 & v10;
              if ( (v14 & (unsigned __int8)v10) == 0 )
                goto LABEL_28;
              goto LABEL_203;
            }
            goto LABEL_26;
          }
          goto LABEL_25;
        }
        if ( *(_QWORD *)a1 == 0x3663707265776F70LL && *(_BYTE *)(a1 + 8) == 52 )
        {
          v8 = v11;
          v10 = 0;
          v9 = 1;
          v4 = 17;
          goto LABEL_24;
        }
LABEL_23:
        v12 = v5 & v10;
        if ( ((unsigned __int8)v5 & (unsigned __int8)v10) == 0 )
          goto LABEL_24;
        goto LABEL_128;
      }
      goto LABEL_14;
    }
    v5 = 1;
    v4 = 32;
    if ( *(_DWORD *)a1 != 912551265 || *(_BYTE *)(a1 + 4) != 52 )
    {
      v6 = 0;
      goto LABEL_125;
    }
LABEL_19:
    v6 = 0;
LABEL_20:
    v7 = 0;
    v9 = 1;
    v10 = 0;
    v8 = 1;
    goto LABEL_21;
  }
  if ( *(_DWORD *)a1 == 909652841
    || *(_DWORD *)a1 == 909653097
    || *(_DWORD *)a1 == 909653353
    || *(_DWORD *)a1 == 909653609
    || *(_DWORD *)a1 == 909653865
    || *(_DWORD *)a1 == 909654121
    || *(_DWORD *)a1 == 909654377 )
  {
    v5 = 0;
    v4 = 31;
    goto LABEL_19;
  }
LABEL_14:
  v7 = a2 == 3;
  if ( a2 == 7 )
  {
    if ( *(_DWORD *)a1 == 1597388920 && *(_WORD *)(a1 + 4) == 13366 && (v4 = 32, *(_BYTE *)(a1 + 6) == 104)
      || *(_DWORD *)a1 == 1702326128 && *(_WORD *)(a1 + 4) == 28786 && (v4 = 16, *(_BYTE *)(a1 + 6) == 99) )
    {
      v5 = 0;
      v6 = 0;
      v9 = 1;
      v10 = 0;
      v8 = 1;
    }
    else
    {
      v5 = 0;
      v6 = 0;
      v9 = 0;
      v4 = 31;
      v10 = 1;
      v8 = 0;
    }
    goto LABEL_21;
  }
  v5 = 0;
  v4 = 31;
  v6 = 0;
  if ( a2 == 3 )
  {
    if ( *(_WORD *)a1 == 28784 && *(_BYTE *)(a1 + 2) == 99 )
    {
      v7 = 1;
      v9 = 1;
      v4 = 16;
      v10 = 0;
      v8 = 1;
    }
    else
    {
      v10 = 1;
      v8 = 0;
      v7 = 1;
      v9 = 0;
    }
    goto LABEL_22;
  }
LABEL_125:
  if ( !v5 )
    goto LABEL_6;
  if ( *(_DWORD *)a1 == 862154864 && *(_BYTE *)(a1 + 4) == 50 )
  {
    v8 = v5;
    v7 = 0;
    v9 = 1;
    v10 = 0;
    v4 = 16;
    goto LABEL_23;
  }
  v10 = v5;
  v7 = 0;
  v9 = 0;
  v8 = 0;
  v12 = v5;
LABEL_128:
  v13 = a2 == 7;
  if ( *(_DWORD *)a1 == 912486512 && *(_BYTE *)(a1 + 4) == 52 )
  {
    v8 = v12;
    v10 = 0;
    v9 = 1;
    v4 = 17;
    goto LABEL_26;
  }
LABEL_25:
  if ( ((unsigned __int8)v10 & (unsigned __int8)v13) != 0 )
  {
    v14 = a2 == 8;
    if ( *(_DWORD *)a1 == 912486512 && *(_WORD *)(a1 + 4) == 27700 && *(_BYTE *)(a1 + 6) == 101 )
    {
      v8 = v10 & v13;
      v10 = 0;
      v9 = 1;
      v4 = 18;
    }
    else
    {
      v15 = v10 & v14;
      if ( ((unsigned __int8)v10 & v14) != 0 )
      {
LABEL_203:
        if ( *(_QWORD *)a1 == 0x6265656C61637378LL )
        {
          v8 = v15;
          v9 = 1;
          v4 = 2;
          v10 = 0;
LABEL_30:
          v16 = v10 & v7;
          goto LABEL_31;
        }
LABEL_29:
        if ( ((unsigned __int8)v10 & (a2 == 10)) != 0 )
        {
          if ( *(_QWORD *)a1 == 0x5F34366863726161LL && *(_WORD *)(a1 + 8) == 25954 )
          {
            v9 = 1;
            v4 = 4;
            goto LABEL_73;
          }
          goto LABEL_32;
        }
        goto LABEL_30;
      }
    }
    goto LABEL_28;
  }
LABEL_26:
  v14 = a2 == 8;
  if ( ((unsigned __int8)v6 & (unsigned __int8)v10) == 0 )
    goto LABEL_27;
  if ( *(_DWORD *)a1 == 1633907576 && *(_WORD *)(a1 + 4) == 25964 )
  {
    v8 = v6 & v10;
    v9 = 1;
    v10 = 0;
    v4 = 1;
    goto LABEL_29;
  }
LABEL_28:
  if ( ((unsigned __int8)v13 & (unsigned __int8)v10) == 0 )
    goto LABEL_29;
  if ( *(_DWORD *)a1 == 1668440417 && *(_WORD *)(a1 + 4) == 13928 && *(_BYTE *)(a1 + 6) == 52 )
  {
    v8 = v13 & v10;
    v9 = 1;
    v4 = 3;
    v10 = 0;
    goto LABEL_32;
  }
  v16 = v7 & v10;
LABEL_31:
  if ( !v16 )
  {
LABEL_32:
    if ( ((unsigned __int8)v5 & (unsigned __int8)v10) != 0 && *(_DWORD *)a1 == 913142369 && *(_BYTE *)(a1 + 4) == 52 )
    {
      v9 = 1;
      v4 = 3;
      goto LABEL_73;
    }
    if ( !v8 )
    {
      if ( v5 )
      {
        if ( *(_DWORD *)a1 == 1701671521 && *(_BYTE *)(a1 + 4) == 98 )
        {
          v4 = 2;
        }
        else
        {
          if ( *(_DWORD *)a1 != 1836410996 || *(_BYTE *)(a1 + 4) != 98 )
          {
            v30 = v7;
            v18 = 0;
            v8 = v5;
LABEL_75:
            if ( v30 )
            {
              v20 = a2 == 4;
              if ( *(_WORD *)a1 == 30305 && *(_BYTE *)(a1 + 2) == 114 )
              {
                v18 = v30;
                v8 = 0;
                v9 = 1;
                v4 = 6;
                goto LABEL_80;
              }
              v31 = v8 & v20;
              goto LABEL_78;
            }
LABEL_76:
            v19 = v6 & v8;
            v20 = a2 == 4;
            if ( ((unsigned __int8)v6 & (unsigned __int8)v8) == 0 )
              goto LABEL_77;
LABEL_39:
            if ( *(_DWORD *)a1 != 879784813 || *(_WORD *)(a1 + 4) != 12339 )
              goto LABEL_40;
            v18 = v19;
            v8 = 0;
            v9 = 1;
            v4 = 14;
            goto LABEL_80;
          }
          v4 = 29;
        }
        v18 = v5;
        v9 = 1;
        goto LABEL_76;
      }
      v17 = 0;
      v8 = 1;
      goto LABEL_36;
    }
LABEL_73:
    v18 = 1;
    v8 = 0;
LABEL_74:
    v30 = v8 & v7;
    goto LABEL_75;
  }
  if ( *(_WORD *)a1 == 29281 && *(_BYTE *)(a1 + 2) == 99 )
  {
    v9 = 1;
    v4 = 5;
    goto LABEL_73;
  }
  if ( *(_WORD *)a1 == 29281 && *(_BYTE *)(a1 + 2) == 109 )
  {
    v9 = 1;
    v4 = 1;
    goto LABEL_73;
  }
  v17 = v5;
  v5 = v8;
  v8 = v10;
LABEL_36:
  v18 = v13 & v8;
  if ( ((unsigned __int8)v13 & (unsigned __int8)v8) == 0 )
  {
    v18 = v5;
    v5 = v17;
    goto LABEL_74;
  }
  if ( *(_DWORD *)a1 == 1836410996 && *(_WORD *)(a1 + 4) == 25954 && *(_BYTE *)(a1 + 6) == 98 )
  {
    v13 &= v8;
    v5 = v17;
    v9 = 1;
    v20 = a2 == 4;
    v4 = 30;
    v8 = 0;
    goto LABEL_77;
  }
  v19 = v6;
  v8 &= v13;
  v13 = v18;
  v20 = a2 == 4;
  v18 = v5;
  v5 = v17;
  if ( v6 )
    goto LABEL_39;
LABEL_77:
  v31 = v20 & v8;
LABEL_78:
  if ( !v31 )
  {
    v32 = v6 & v8;
    if ( ((unsigned __int8)v6 & (unsigned __int8)v8) == 0 )
      goto LABEL_80;
    v6 &= v8;
    v8 = v32;
LABEL_40:
    v21 = a2 == 14;
    if ( *(_DWORD *)a1 == 1936746861 && *(_WORD *)(a1 + 4) == 25189 )
    {
      v8 = 0;
      v9 = 1;
      v4 = 10;
      v18 = 1;
      goto LABEL_82;
    }
    v14 = v18;
    v18 = v8;
    goto LABEL_42;
  }
  if ( *(_DWORD *)a1 == 1936746861 )
  {
    v18 = v31;
    v9 = 1;
    v4 = 10;
    v21 = a2 == 14;
    v8 = 0;
    goto LABEL_82;
  }
LABEL_80:
  if ( ((unsigned __int8)v8 & (a2 == 12)) == 0 )
  {
    v21 = a2 == 14;
    v33 = v6 & v8;
    if ( ((unsigned __int8)v6 & (unsigned __int8)v8) == 0 )
      goto LABEL_82;
    v14 = v18;
    v6 &= v8;
    v18 = v33;
LABEL_42:
    if ( *(_DWORD *)a1 == 1936746861 && *(_WORD *)(a1 + 4) == 27749 )
    {
LABEL_171:
      v9 = 1;
      v4 = 11;
      goto LABEL_46;
    }
    if ( *(_DWORD *)a1 != 1936746861 || *(_WORD *)(a1 + 4) != 13366 )
    {
LABEL_86:
      v34 = v5 & v18;
      if ( ((unsigned __int8)v5 & (unsigned __int8)v18) != 0 )
      {
        if ( *(_DWORD *)a1 == 1936681326 && *(_BYTE *)(a1 + 4) == 50 )
        {
          v5 &= v18;
          v9 = 1;
          v4 = 15;
          goto LABEL_92;
        }
        v18 &= v5;
        v5 = v34;
        if ( !v6 )
          goto LABEL_50;
LABEL_89:
        if ( *(_DWORD *)a1 == 1734634849 && *(_WORD *)(a1 + 4) == 28259 )
        {
          v9 = 1;
          v4 = 20;
          goto LABEL_92;
        }
        goto LABEL_52;
      }
      goto LABEL_47;
    }
    v9 = 1;
    v4 = 12;
LABEL_46:
    v14 = 1;
    v18 = 0;
LABEL_47:
    v22 = v20 & v18;
    goto LABEL_48;
  }
  v21 = a2 == 14;
  if ( *(_QWORD *)a1 == 0x656C6C617370696DLL && *(_DWORD *)(a1 + 8) == 2019914343 )
  {
    v9 = 1;
    v4 = 10;
    goto LABEL_46;
  }
LABEL_82:
  if ( (v21 & (unsigned __int8)v8) != 0
    && *(_QWORD *)a1 == 0x656C6C617370696DLL
    && *(_DWORD *)(a1 + 8) == 2019914343
    && *(_WORD *)(a1 + 12) == 27749 )
  {
    goto LABEL_171;
  }
  if ( v18 )
    goto LABEL_46;
  if ( !v14 )
  {
    v18 = 1;
    goto LABEL_86;
  }
  if ( *(_QWORD *)a1 == 0x626534367370696DLL )
  {
    v4 = 12;
LABEL_167:
    v9 = 1;
    goto LABEL_49;
  }
  if ( *(_QWORD *)a1 == 0x6C6534367370696DLL )
  {
    v4 = 13;
    goto LABEL_167;
  }
  v18 = v14;
  v22 = v20;
  v14 = 0;
LABEL_48:
  if ( v22 )
  {
    if ( *(_DWORD *)a1 == 808466034 )
    {
      v9 = 1;
      v4 = 19;
      goto LABEL_92;
    }
    goto LABEL_50;
  }
LABEL_49:
  if ( ((unsigned __int8)v18 & (unsigned __int8)v6) != 0 )
    goto LABEL_89;
LABEL_50:
  if ( v14 )
    goto LABEL_92;
  v18 = 1;
  if ( v13 )
  {
    if ( *(_DWORD *)a1 == 1668508018 && *(_WORD *)(a1 + 4) == 13174 && *(_BYTE *)(a1 + 6) == 50 )
    {
      v9 = 1;
      v4 = 21;
      goto LABEL_92;
    }
    if ( *(_DWORD *)a1 == 1668508018 && *(_WORD *)(a1 + 4) == 13942 && *(_BYTE *)(a1 + 6) == 52 )
    {
      v9 = 1;
      v4 = 22;
      goto LABEL_92;
    }
    if ( *(_DWORD *)a1 == 1635280232 && *(_WORD *)(a1 + 4) == 28519 && *(_BYTE *)(a1 + 6) == 110 )
    {
      v9 = 1;
      v4 = 9;
      goto LABEL_92;
    }
    v14 = v9;
    v18 = v9 ^ 1;
  }
LABEL_52:
  v23 = v5;
  v5 &= v18;
  if ( v5 )
  {
    if ( *(_DWORD *)a1 != 809055091 || *(_BYTE *)(a1 + 4) != 120 )
    {
      if ( *(_DWORD *)a1 == 1918988403 && *(_BYTE *)(a1 + 4) == 99 )
      {
        v4 = 23;
        v9 = 1;
        goto LABEL_92;
      }
      v23 = v5;
      goto LABEL_56;
    }
    goto LABEL_288;
  }
  v24 = v13 & v18;
  if ( v24 )
  {
    if ( *(_DWORD *)a1 != 1953724787 || *(_WORD *)(a1 + 4) != 28005 || *(_BYTE *)(a1 + 6) != 122 )
    {
      if ( !v14 )
        goto LABEL_134;
      v13 = v14;
LABEL_345:
      v5 = v23;
      goto LABEL_92;
    }
    v13 = v24;
    v5 = v23;
LABEL_288:
    v4 = 26;
    v9 = 1;
    goto LABEL_92;
  }
  if ( v14 )
    goto LABEL_345;
  if ( !v13 )
  {
LABEL_56:
    v5 = v23;
    goto LABEL_57;
  }
LABEL_134:
  if ( *(_DWORD *)a1 == 1918988403 && *(_WORD *)(a1 + 4) == 25955 && *(_BYTE *)(a1 + 6) == 108 )
  {
    v5 = v23;
    v13 = 1;
    v9 = 1;
    v4 = 25;
    goto LABEL_92;
  }
  if ( *(_DWORD *)a1 == 1918988403 && *(_WORD *)(a1 + 4) == 30307 && *(_BYTE *)(a1 + 6) == 57
    || *(_DWORD *)a1 == 1918988403 && *(_WORD *)(a1 + 4) == 13923 && *(_BYTE *)(a1 + 6) == 52 )
  {
    v5 = v23;
    v13 = 1;
    v9 = 1;
    v4 = 24;
    goto LABEL_92;
  }
  v14 = 0;
  v5 = v23;
  v13 = 1;
LABEL_57:
  if ( v7 )
  {
    if ( *(_WORD *)a1 == 25460 && *(_BYTE *)(a1 + 2) == 101 )
    {
      v9 = 1;
      v4 = 27;
      goto LABEL_92;
    }
  }
  else
  {
    if ( v14 )
      goto LABEL_92;
    if ( v23 )
    {
      if ( *(_DWORD *)a1 == 1818583924 && *(_BYTE *)(a1 + 4) == 101 )
      {
        v9 = 1;
        v4 = 28;
        goto LABEL_92;
      }
      if ( *(_DWORD *)a1 == 1919902584 && *(_BYTE *)(a1 + 4) == 101 )
      {
        v9 = 1;
        v4 = 33;
        goto LABEL_92;
      }
      if ( *(_DWORD *)a1 == 1953527406 && *(_BYTE *)(a1 + 4) == 120 )
      {
        v9 = 1;
        v4 = 34;
        goto LABEL_92;
      }
    }
  }
  v25 = v9;
  if ( ((unsigned __int8)v13 & ((unsigned __int8)v9 ^ 1)) == 0 )
  {
    if ( !v9 )
    {
      if ( v20 )
      {
        if ( *(_DWORD *)a1 == 842229100 )
        {
          v4 = 36;
        }
        else
        {
          if ( *(_DWORD *)a1 != 875980140 )
          {
            v26 = 0;
            v25 = 1;
            goto LABEL_95;
          }
          v4 = 37;
        }
        v26 = v20;
        v9 = 1;
        goto LABEL_96;
      }
      goto LABEL_63;
    }
LABEL_92:
    v25 = 0;
    v26 = 1;
    goto LABEL_93;
  }
  if ( *(_DWORD *)a1 == 1953527406 && *(_WORD *)(a1 + 4) == 13944 && *(_BYTE *)(a1 + 6) == 52 )
  {
    v13 &= v9 ^ 1;
    v4 = 35;
    v9 = 1;
    goto LABEL_92;
  }
  v13 &= v9 ^ 1;
LABEL_63:
  v26 = v9;
  v25 = v9 ^ 1;
  v27 = (v9 ^ 1) & v23;
  if ( v27 )
  {
    if ( *(_DWORD *)a1 == 1768189281 && *(_BYTE *)(a1 + 4) == 108 )
    {
      v5 = v27;
      v26 = v27;
      v4 = 38;
      v25 = 0;
      v9 = 1;
LABEL_96:
      if ( ((unsigned __int8)v25 & v20) == 0 )
        goto LABEL_97;
      goto LABEL_68;
    }
    v26 = 0;
    goto LABEL_66;
  }
LABEL_93:
  v35 = v25 & v13;
  if ( !v35 )
  {
    if ( !v25 )
      goto LABEL_96;
LABEL_95:
    if ( !v5 )
      goto LABEL_96;
LABEL_66:
    if ( *(_DWORD *)a1 != 1767994216 || *(_BYTE *)(a1 + 4) != 108 )
    {
      v25 = 1;
      v5 = 1;
      if ( !v20 )
        goto LABEL_97;
LABEL_68:
      v28 = a2 > 6;
      if ( *(_DWORD *)a1 == 1919512691 )
        return 42;
      goto LABEL_98;
    }
    v5 = 1;
    v4 = 40;
    v9 = 1;
    v25 = 0;
    v26 = 1;
LABEL_97:
    v28 = a2 > 6;
    if ( ((unsigned __int8)v25 & (unsigned __int8)v6) == 0 )
      goto LABEL_98;
LABEL_182:
    if ( *(_DWORD *)a1 == 1919512691 && *(_WORD *)(a1 + 4) == 13366 )
      return 43;
    goto LABEL_99;
  }
  if ( *(_DWORD *)a1 == 1768189281 && *(_WORD *)(a1 + 4) == 13932 && *(_BYTE *)(a1 + 6) == 52 )
  {
    v26 = v35;
    v4 = 39;
    v9 = 1;
    v25 = 0;
    goto LABEL_97;
  }
  if ( *(_DWORD *)a1 == 1767994216 && *(_WORD *)(a1 + 4) == 13932 && *(_BYTE *)(a1 + 6) == 52 )
  {
    v26 = v35;
    v9 = 1;
    v4 = 41;
    v28 = a2 > 6;
    v25 = 0;
  }
  else
  {
    v28 = a2 > 6;
    if ( ((unsigned __int8)v6 & (unsigned __int8)v25) != 0 )
      goto LABEL_182;
  }
LABEL_98:
  if ( (v28 & (unsigned __int8)v25) != 0 )
  {
    if ( *(_DWORD *)a1 == 1768710507 && *(_WORD *)(a1 + 4) == 25197 && *(_BYTE *)(a1 + 6) == 97 )
      return 44;
    goto LABEL_100;
  }
LABEL_99:
  if ( v26 )
    return v4;
LABEL_100:
  if ( v5 )
  {
    if ( *(_DWORD *)a1 == 1634623852 && *(_BYTE *)(a1 + 4) == 105 )
      return 46;
    if ( *(_DWORD *)a1 == 1986095219 && *(_BYTE *)(a1 + 4) == 101 )
      return 45;
  }
  else if ( v6 )
  {
    if ( *(_DWORD *)a1 == 1836278135 && *(_WORD *)(a1 + 4) == 12851 )
      return 47;
    if ( *(_DWORD *)a1 == 1836278135 && *(_WORD *)(a1 + 4) == 13366 )
      return 48;
    if ( v9 )
      return v4;
    if ( *(_DWORD *)a1 == 1634956910 )
    {
      v4 = 49;
      if ( *(_WORD *)(a1 + 4) == 29555 )
        return v4;
    }
    goto LABEL_103;
  }
  if ( v9 )
    return v4;
LABEL_103:
  if ( v21 )
  {
    if ( *(_QWORD *)a1 == 0x63737265646E6572LL && *(_DWORD *)(a1 + 8) == 1953524082 )
    {
      v4 = 50;
      if ( *(_WORD *)(a1 + 12) == 12851 )
        return v4;
    }
    if ( *(_QWORD *)a1 == 0x63737265646E6572LL && *(_DWORD *)(a1 + 8) == 1953524082 )
    {
      v4 = 51;
      if ( *(_WORD *)(a1 + 12) == 13366 )
        return v4;
    }
  }
  if ( a2 <= 2 )
    return 0;
  if ( *(_WORD *)a1 == 29281 && *(_BYTE *)(a1 + 2) == 109
    || a2 > 4
    && (*(_DWORD *)a1 == 1836410996 && *(_BYTE *)(a1 + 4) == 98
     || v28 && *(_DWORD *)a1 == 1668440417 && *(_WORD *)(a1 + 4) == 13928 && *(_BYTE *)(a1 + 6) == 52) )
  {
    v36 = sub_16F5F40(a1, a2);
    v37 = sub_16F5FC0(a1, a2);
    v38 = v37;
    if ( v37 == 1 )
    {
      v46 = (unsigned int)(v36 - 1);
      v4 = 0;
      if ( (unsigned int)v46 <= 2 )
        v4 = dword_42AF660[v46];
    }
    else
    {
      v4 = 0;
      if ( v37 == 2 )
      {
        v39 = (unsigned int)(v36 - 1);
        v4 = 0;
        if ( (unsigned int)v39 <= 2 )
          v4 = dword_42AF650[v39];
      }
    }
    v40 = (_WORD *)sub_16F5C40(a1, a2);
    v42 = v40;
    v43 = v41;
    if ( v41 && (v36 != 2 || v41 == 1 || *v40 != 12918 && *v40 != 13174) )
    {
      v44 = sub_16F6140(v40, v41);
      v45 = sub_16F6170(v42, v43);
      if ( v44 == 3 && v45 == 6 )
        return (unsigned int)(v38 == 2) + 29;
      return v4;
    }
    return 0;
  }
  if ( *(_WORD *)a1 != 28770 || *(_BYTE *)(a1 + 2) != 102 )
    return 0;
  return sub_16DDE90(a1, a2);
}
