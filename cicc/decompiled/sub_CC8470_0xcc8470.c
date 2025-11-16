// Function: sub_CC8470
// Address: 0xcc8470
//
__int64 __fastcall sub_CC8470(_DWORD *s1, unsigned __int64 a2)
{
  char v2; // al
  char v3; // al
  unsigned int v4; // r12d
  int v6; // r13d
  int v7; // eax
  int v8; // ebx
  __int64 v9; // rax
  _WORD *v10; // rax
  __int64 v11; // rdx
  _WORD *v12; // r15
  __int64 v13; // r14
  int v14; // r13d
  int v15; // eax
  __int64 v16; // rax
  int v17; // [rsp+2Ch] [rbp-54h] BYREF
  _DWORD *v18; // [rsp+30h] [rbp-50h] BYREF
  unsigned __int64 v19; // [rsp+38h] [rbp-48h]
  int v20; // [rsp+40h] [rbp-40h]
  int v21; // [rsp+44h] [rbp-3Ch]

  v18 = s1;
  v19 = a2;
  v21 = 0;
  v17 = 38;
  if ( a2 == 4 && (*s1 == 909652841 || *s1 == 909653097 || *s1 == 909653353) )
  {
    v20 = 38;
    LOBYTE(v21) = 1;
    v17 = 38;
  }
  else
  {
    sub_CC83E0((__int64)&v18, &v17, "i686", 4);
    v17 = 38;
    if ( !(_BYTE)v21 && v19 == 4 && (*v18 == 909653865 || *v18 == 909654121) )
    {
      v20 = 38;
      LOBYTE(v21) = 1;
      v17 = 39;
      goto LABEL_11;
    }
  }
  sub_CC83E0((__int64)&v18, &v17, "i986", 4);
  v17 = 39;
  if ( (_BYTE)v21 )
    goto LABEL_11;
  if ( v19 == 5 )
  {
    if ( *v18 != 912551265 || *((_BYTE *)v18 + 4) != 52 )
      goto LABEL_11;
LABEL_62:
    v20 = 39;
    LOBYTE(v21) = 1;
    v17 = 22;
    goto LABEL_15;
  }
  if ( v19 == 6 && *v18 == 1597388920 && *((_WORD *)v18 + 2) == 13366 )
    goto LABEL_62;
LABEL_11:
  sub_CC83E0((__int64)&v18, &v17, "x86_64h", 7);
  v17 = 22;
  if ( !(_BYTE)v21 )
  {
    if ( v19 == 7 )
    {
      if ( *v18 == 1702326128 && *((_WORD *)v18 + 2) == 28786 && *((_BYTE *)v18 + 6) == 99 )
      {
LABEL_297:
        v20 = 22;
        LOBYTE(v21) = 1;
        v17 = 23;
        goto LABEL_20;
      }
    }
    else if ( v19 == 10 && *(_QWORD *)v18 == 0x7363707265776F70LL && *((_WORD *)v18 + 4) == 25968 )
    {
      goto LABEL_297;
    }
  }
LABEL_15:
  if ( !(unsigned __int8)sub_CC83E0((__int64)&v18, &v17, "ppc", 3) )
    sub_CC83E0((__int64)&v18, &v17, "ppc32", 5);
  v17 = 23;
  if ( !(_BYTE)v21 )
  {
    if ( v19 == 9 )
    {
      if ( *(_QWORD *)v18 == 0x6C63707265776F70LL && *((_BYTE *)v18 + 8) == 101 )
      {
LABEL_173:
        v20 = 23;
        LOBYTE(v21) = 1;
        v17 = 24;
        goto LABEL_23;
      }
    }
    else if ( v19 == 5 && *v18 == 1818456176 && *((_BYTE *)v18 + 4) == 101 )
    {
      goto LABEL_173;
    }
  }
LABEL_20:
  sub_CC83E0((__int64)&v18, &v17, "ppc32le", 7);
  v17 = 24;
  if ( !(_BYTE)v21 )
  {
    if ( v19 == 9 )
    {
      if ( *(_QWORD *)v18 == 0x3663707265776F70LL && *((_BYTE *)v18 + 8) == 52 )
      {
LABEL_177:
        v20 = 24;
        LOBYTE(v21) = 1;
        goto LABEL_116;
      }
    }
    else if ( v19 == 3 && *(_WORD *)v18 == 28784 && *((_BYTE *)v18 + 2) == 117 )
    {
      goto LABEL_177;
    }
  }
LABEL_23:
  sub_CC83E0((__int64)&v18, &v17, "ppc64", 5);
  if ( (_BYTE)v21 )
  {
LABEL_116:
    v17 = 16;
    goto LABEL_28;
  }
  if ( v19 != 11 )
  {
    switch ( v19 )
    {
      case 7uLL:
        if ( *v18 == 912486512 && *((_WORD *)v18 + 2) == 27700 && *((_BYTE *)v18 + 6) == 101 )
          goto LABEL_278;
        if ( (*v18 != 1668440417 || *((_WORD *)v18 + 2) != 13928 || *((_BYTE *)v18 + 6) != 52)
          && (*v18 != 913142369 || *((_WORD *)v18 + 2) != 25908 || *((_BYTE *)v18 + 6) != 99) )
        {
          if ( *v18 == 1836410996 && *((_WORD *)v18 + 2) == 25954 && *((_BYTE *)v18 + 6) == 98 )
          {
            v20 = 37;
            LOBYTE(v21) = 1;
            goto LABEL_116;
          }
          goto LABEL_26;
        }
LABEL_188:
        v20 = 3;
        LOBYTE(v21) = 1;
        goto LABEL_116;
      case 6uLL:
        if ( *v18 == 1633907576 && *((_WORD *)v18 + 2) == 25964 )
          goto LABEL_267;
        if ( *v18 == 913142369 && *((_WORD *)v18 + 2) == 25908 )
          goto LABEL_188;
        if ( *v18 != 879784813 || *((_WORD *)v18 + 2) != 12339 )
          goto LABEL_26;
        v20 = 20;
LABEL_215:
        LOBYTE(v21) = 1;
        goto LABEL_116;
      case 8uLL:
        if ( *(_QWORD *)v18 != 0x6265656C61637378LL )
        {
          if ( *(_QWORD *)v18 != 0x32335F34366D7261LL )
            goto LABEL_26;
LABEL_115:
          v20 = 5;
          LOBYTE(v21) = 1;
          goto LABEL_116;
        }
        break;
      case 0xAuLL:
        if ( *(_QWORD *)v18 == 0x5F34366863726161LL && *((_WORD *)v18 + 4) == 25954 )
        {
          v20 = 4;
          LOBYTE(v21) = 1;
          goto LABEL_116;
        }
        if ( *(_QWORD *)v18 != 0x5F34366863726161LL || *((_WORD *)v18 + 4) != 12851 )
          goto LABEL_26;
        goto LABEL_115;
      case 3uLL:
        if ( *(_WORD *)v18 == 29281 && *((_BYTE *)v18 + 2) == 99 )
        {
          v20 = 6;
          LOBYTE(v21) = 1;
          goto LABEL_116;
        }
        if ( *(_WORD *)v18 != 29281 || *((_BYTE *)v18 + 2) != 109 )
        {
          if ( *(_WORD *)v18 == 30305 && *((_BYTE *)v18 + 2) == 114 )
          {
            v20 = 7;
            LOBYTE(v21) = 1;
            goto LABEL_116;
          }
          goto LABEL_26;
        }
LABEL_267:
        v20 = 1;
        LOBYTE(v21) = 1;
        goto LABEL_116;
      case 5uLL:
        if ( *v18 == 913142369 && *((_BYTE *)v18 + 4) == 52 )
          goto LABEL_188;
        if ( *v18 != 1701671521 || *((_BYTE *)v18 + 4) != 98 )
        {
          if ( *v18 == 1836410996 && *((_BYTE *)v18 + 4) == 98 )
          {
            v20 = 36;
            LOBYTE(v21) = 1;
            goto LABEL_116;
          }
          goto LABEL_26;
        }
        break;
      default:
        if ( v19 != 4 || *v18 != 1798846061 )
          goto LABEL_26;
        v20 = 15;
        goto LABEL_215;
    }
    v20 = 2;
    LOBYTE(v21) = 1;
    goto LABEL_116;
  }
  if ( *(_QWORD *)v18 == 0x3663707265776F70LL && *((_WORD *)v18 + 4) == 27700 && *((_BYTE *)v18 + 10) == 101 )
  {
LABEL_278:
    v20 = 25;
    LOBYTE(v21) = 1;
    goto LABEL_116;
  }
LABEL_26:
  v17 = 16;
  if ( v19 == 4 )
  {
    if ( *v18 != 1936746861 )
      goto LABEL_28;
LABEL_220:
    v20 = 16;
    LOBYTE(v21) = 1;
    v17 = 17;
    goto LABEL_32;
  }
  if ( v19 == 6 && *v18 == 1936746861 && *((_WORD *)v18 + 2) == 25189 )
    goto LABEL_220;
LABEL_28:
  if ( !(unsigned __int8)sub_CC83E0((__int64)&v18, &v17, "mipsallegrex", 12)
    && !(unsigned __int8)sub_CC83E0((__int64)&v18, &v17, "mipsisa32r6", 11) )
  {
    sub_CC83E0((__int64)&v18, &v17, "mipsr6", 6);
  }
  v17 = 17;
  if ( !(_BYTE)v21 )
  {
    if ( v19 == 6 )
    {
      if ( *v18 == 1936746861 && *((_WORD *)v18 + 2) == 27749 )
      {
LABEL_132:
        v20 = 17;
        LOBYTE(v21) = 1;
        v17 = 18;
        goto LABEL_35;
      }
    }
    else if ( v19 == 14 )
    {
      if ( *(_QWORD *)v18 == 0x656C6C617370696DLL && v18[2] == 2019914343 && *((_WORD *)v18 + 6) == 27749 )
        goto LABEL_132;
    }
    else if ( v19 == 13 && *(_QWORD *)v18 == 0x336173697370696DLL && v18[2] == 1698066994 && *((_BYTE *)v18 + 12) == 108 )
    {
      goto LABEL_132;
    }
  }
LABEL_32:
  sub_CC83E0((__int64)&v18, &v17, "mipsr6el", 8);
  v17 = 18;
  if ( !(_BYTE)v21 )
  {
    if ( v19 == 6 )
    {
      if ( *v18 == 1936746861 && *((_WORD *)v18 + 2) == 13366 )
      {
LABEL_223:
        v20 = 18;
        LOBYTE(v21) = 1;
        v17 = 19;
        goto LABEL_39;
      }
    }
    else if ( v19 == 8 && *(_QWORD *)v18 == 0x626534367370696DLL )
    {
      goto LABEL_223;
    }
  }
LABEL_35:
  if ( !(unsigned __int8)sub_CC83E0((__int64)&v18, &v17, "mipsn32", 7)
    && !(unsigned __int8)sub_CC83E0((__int64)&v18, &v17, "mipsisa64r6", 11)
    && !(unsigned __int8)sub_CC83E0((__int64)&v18, &v17, "mips64r6", 8) )
  {
    sub_CC83E0((__int64)&v18, &v17, "mipsn32r6", 9);
  }
  v17 = 19;
  if ( !(_BYTE)v21 )
  {
    if ( v19 == 8 )
    {
      if ( *(_QWORD *)v18 != 0x6C6534367370696DLL )
        goto LABEL_39;
LABEL_69:
      v20 = 19;
      LOBYTE(v21) = 1;
      goto LABEL_70;
    }
    if ( v19 == 9 )
    {
      if ( *(_QWORD *)v18 == 0x6532336E7370696DLL && *((_BYTE *)v18 + 8) == 108 )
        goto LABEL_69;
    }
    else if ( v19 == 13 && *(_QWORD *)v18 == 0x366173697370696DLL && v18[2] == 1698066996 && *((_BYTE *)v18 + 12) == 108 )
    {
      goto LABEL_69;
    }
  }
LABEL_39:
  if ( !(unsigned __int8)sub_CC83E0((__int64)&v18, &v17, "mips64r6el", 10) )
    sub_CC83E0((__int64)&v18, &v17, "mipsn32r6el", 11);
  if ( (_BYTE)v21 )
    goto LABEL_70;
  if ( v19 != 5 )
  {
    switch ( v19 )
    {
      case 4uLL:
        if ( *v18 != 808466034 )
        {
          if ( *v18 == 1919512691 )
          {
            v20 = 48;
            LOBYTE(v21) = 1;
            v17 = 50;
            goto LABEL_71;
          }
          goto LABEL_51;
        }
        v20 = 26;
        LOBYTE(v21) = 1;
        goto LABEL_70;
      case 6uLL:
        if ( *v18 == 1734634849 && *((_WORD *)v18 + 2) == 28259 )
        {
          v20 = 27;
          LOBYTE(v21) = 1;
          goto LABEL_70;
        }
        break;
      case 7uLL:
        if ( *v18 == 1668508018 && *((_WORD *)v18 + 2) == 13174 && *((_BYTE *)v18 + 6) == 50 )
        {
          v20 = 28;
          LOBYTE(v21) = 1;
          goto LABEL_70;
        }
        if ( *v18 == 1668508018 && *((_WORD *)v18 + 2) == 13942 && *((_BYTE *)v18 + 6) == 52 )
        {
          v20 = 29;
          LOBYTE(v21) = 1;
          goto LABEL_70;
        }
        if ( *v18 == 1635280232 && *((_WORD *)v18 + 2) == 28519 && *((_BYTE *)v18 + 6) == 110 )
        {
          v20 = 12;
          LOBYTE(v21) = 1;
          goto LABEL_70;
        }
        if ( *v18 == 1953724787 && *((_WORD *)v18 + 2) == 28005 && *((_BYTE *)v18 + 6) == 122 )
        {
LABEL_284:
          v20 = 33;
          LOBYTE(v21) = 1;
          goto LABEL_70;
        }
        if ( *v18 == 1918988403 && *((_WORD *)v18 + 2) == 25955 && *((_BYTE *)v18 + 6) == 108 )
        {
          v20 = 32;
          LOBYTE(v21) = 1;
        }
        else
        {
          if ( *v18 == 1918988403 && *((_WORD *)v18 + 2) == 30307 && *((_BYTE *)v18 + 6) == 57
            || *v18 == 1918988403 && *((_WORD *)v18 + 2) == 13923 && *((_BYTE *)v18 + 6) == 52 )
          {
            v20 = 31;
            LOBYTE(v21) = 1;
            goto LABEL_70;
          }
          if ( *v18 == 1953527406 && *((_WORD *)v18 + 2) == 13944 && *((_BYTE *)v18 + 6) == 52 )
          {
            v20 = 43;
            LOBYTE(v21) = 1;
            v17 = 50;
            goto LABEL_71;
          }
          if ( *v18 == 1768189281 && *((_WORD *)v18 + 2) == 13932 && *((_BYTE *)v18 + 6) == 52 )
          {
            v20 = 45;
            LOBYTE(v21) = 1;
            v17 = 50;
            goto LABEL_71;
          }
          if ( *v18 != 1767994216 || *((_WORD *)v18 + 2) != 13932 || *((_BYTE *)v18 + 6) != 52 )
            goto LABEL_51;
          v20 = 47;
          LOBYTE(v21) = 1;
        }
LABEL_70:
        v17 = 50;
        goto LABEL_71;
      case 3uLL:
        if ( *(_WORD *)v18 != 25460 || *((_BYTE *)v18 + 2) != 101 )
          goto LABEL_51;
        v20 = 34;
        LOBYTE(v21) = 1;
        goto LABEL_70;
    }
    if ( v19 == 6 && *v18 == 1919512691 && *((_WORD *)v18 + 2) == 13366 )
    {
      v20 = 49;
      LOBYTE(v21) = 1;
      v17 = 50;
      goto LABEL_71;
    }
    goto LABEL_51;
  }
  if ( *v18 == 1885828718 && *((_BYTE *)v18 + 4) == 117 )
  {
    v20 = 21;
    LOBYTE(v21) = 1;
    goto LABEL_70;
  }
  if ( *v18 == 809055091 && *((_BYTE *)v18 + 4) == 120 )
    goto LABEL_284;
  if ( *v18 == 1918988403 && *((_BYTE *)v18 + 4) == 99 )
  {
    v20 = 30;
    LOBYTE(v21) = 1;
    goto LABEL_70;
  }
  if ( *v18 == 1818583924 && *((_BYTE *)v18 + 4) == 101 )
  {
    v20 = 35;
    LOBYTE(v21) = 1;
    v17 = 50;
    goto LABEL_71;
  }
  if ( *v18 == 1919902584 && *((_BYTE *)v18 + 4) == 101 )
  {
    v20 = 40;
    LOBYTE(v21) = 1;
    v17 = 50;
    goto LABEL_71;
  }
  if ( *v18 == 1953527406 && *((_BYTE *)v18 + 4) == 120 )
  {
    v20 = 42;
    LOBYTE(v21) = 1;
    v17 = 50;
    goto LABEL_71;
  }
  if ( *v18 == 1768189281 && *((_BYTE *)v18 + 4) == 108 )
  {
    v20 = 44;
    LOBYTE(v21) = 1;
    v17 = 50;
    goto LABEL_71;
  }
  if ( *v18 == 1767994216 && *((_BYTE *)v18 + 4) == 108 )
  {
    v20 = 46;
    LOBYTE(v21) = 1;
    v17 = 50;
    goto LABEL_71;
  }
LABEL_51:
  v17 = 50;
  if ( v19 == 5 )
  {
    if ( *v18 == 1919512691 && *((_BYTE *)v18 + 4) == 118 )
    {
LABEL_54:
      v20 = 50;
      LOBYTE(v21) = 1;
      v17 = 51;
      goto LABEL_74;
    }
  }
  else if ( v19 == 8 && *(_QWORD *)v18 == 0x352E317672697073LL )
  {
    goto LABEL_54;
  }
LABEL_71:
  sub_CC83E0((__int64)&v18, &v17, "spirv1.6", 8);
  v17 = 51;
  if ( !(_BYTE)v21 )
  {
    if ( v19 == 7 )
    {
      if ( *v18 == 1919512691 && *((_WORD *)v18 + 2) == 13174 && *((_BYTE *)v18 + 6) == 50 )
      {
LABEL_289:
        v20 = 51;
        v2 = 1;
        LOBYTE(v21) = 1;
        goto LABEL_76;
      }
    }
    else if ( v19 == 11
           && (*(_QWORD *)v18 == 0x7632337672697073LL && *((_WORD *)v18 + 4) == 11825 && *((_BYTE *)v18 + 10) == 48
            || *(_QWORD *)v18 == 0x7632337672697073LL && *((_WORD *)v18 + 4) == 11825 && *((_BYTE *)v18 + 10) == 49) )
    {
      goto LABEL_289;
    }
  }
LABEL_74:
  if ( !(unsigned __int8)sub_CC83E0((__int64)&v18, &v17, "spirv32v1.2", 11)
    && !(unsigned __int8)sub_CC83E0((__int64)&v18, &v17, "spirv32v1.3", 11)
    && !(unsigned __int8)sub_CC83E0((__int64)&v18, &v17, "spirv32v1.4", 11)
    && !(unsigned __int8)sub_CC83E0((__int64)&v18, &v17, "spirv32v1.5", 11) )
  {
    sub_CC83E0((__int64)&v18, &v17, "spirv32v1.6", 11);
  }
  v2 = v21;
LABEL_76:
  v17 = 52;
  if ( v2 )
    goto LABEL_77;
  if ( v19 == 7 )
  {
    if ( *v18 == 1919512691 && *((_WORD *)v18 + 2) == 13942 && *((_BYTE *)v18 + 6) == 52 )
    {
LABEL_315:
      v20 = 52;
      LOBYTE(v21) = 1;
      goto LABEL_316;
    }
  }
  else if ( v19 == 11
         && *(_QWORD *)v18 == 0x7634367672697073LL
         && *((_WORD *)v18 + 4) == 11825
         && *((_BYTE *)v18 + 10) == 48 )
  {
    goto LABEL_315;
  }
LABEL_77:
  if ( !(unsigned __int8)sub_CC83E0((__int64)&v18, &v17, "spirv64v1.1", 11)
    && !(unsigned __int8)sub_CC83E0((__int64)&v18, &v17, "spirv64v1.2", 11)
    && !(unsigned __int8)sub_CC83E0((__int64)&v18, &v17, "spirv64v1.3", 11)
    && !(unsigned __int8)sub_CC83E0((__int64)&v18, &v17, "spirv64v1.4", 11)
    && !(unsigned __int8)sub_CC83E0((__int64)&v18, &v17, "spirv64v1.5", 11) )
  {
    sub_CC83E0((__int64)&v18, &v17, "spirv64v1.6", 11);
  }
  v3 = v21;
  if ( (_BYTE)v21 )
    goto LABEL_316;
  if ( v19 <= 6 )
  {
    if ( v19 != 5 || *v18 != 1634623852 || *((_BYTE *)v18 + 4) != 105 )
      goto LABEL_81;
    v20 = 55;
    LOBYTE(v21) = 1;
LABEL_316:
    v17 = 11;
    goto LABEL_100;
  }
  if ( *v18 == 1768710507 && *((_WORD *)v18 + 2) == 25197 && *((_BYTE *)v18 + 6) == 97 )
  {
    v20 = 53;
    LOBYTE(v21) = 1;
    goto LABEL_316;
  }
LABEL_81:
  if ( v19 == 14 && *(_QWORD *)v18 == 0x63737265646E6572LL && v18[2] == 1953524082 && *((_WORD *)v18 + 6) == 12851 )
  {
    v20 = 59;
    LOBYTE(v21) = 1;
    goto LABEL_316;
  }
  if ( v19 == 14 && *(_QWORD *)v18 == 0x63737265646E6572LL && v18[2] == 1953524082 && *((_WORD *)v18 + 6) == 13366 )
  {
    v20 = 60;
    LOBYTE(v21) = 1;
    goto LABEL_316;
  }
  if ( v19 == 5 && *v18 == 1986095219 && *((_BYTE *)v18 + 4) == 101 )
  {
    v20 = 54;
    LOBYTE(v21) = 1;
    goto LABEL_316;
  }
  if ( v19 == 2 && *(_WORD *)v18 == 25974 )
  {
    v20 = 61;
    LOBYTE(v21) = 1;
    goto LABEL_316;
  }
  if ( v19 == 6 && *v18 == 1836278135 && *((_WORD *)v18 + 2) == 12851 )
  {
    v20 = 56;
    LOBYTE(v21) = 1;
    goto LABEL_316;
  }
  if ( v19 == 6 && *v18 == 1836278135 && *((_WORD *)v18 + 2) == 13366 )
  {
    v20 = 57;
    LOBYTE(v21) = 1;
    goto LABEL_316;
  }
  if ( v19 == 4 && *v18 == 2037085027 )
  {
    v20 = 10;
    LOBYTE(v21) = 1;
    goto LABEL_316;
  }
  if ( v19 == 11 && *(_QWORD *)v18 == 0x637261676E6F6F6CLL && *((_WORD *)v18 + 4) == 13160 && *((_BYTE *)v18 + 10) == 50 )
  {
    v20 = 13;
    v3 = 1;
    LOBYTE(v21) = 1;
  }
  else if ( v19 == 11
         && *(_QWORD *)v18 == 0x637261676E6F6F6CLL
         && *((_WORD *)v18 + 4) == 13928
         && *((_BYTE *)v18 + 10) == 52 )
  {
    v20 = 14;
    LOBYTE(v21) = 1;
    goto LABEL_316;
  }
  v17 = 11;
  if ( !v3 )
  {
    if ( v19 == 4 )
    {
      if ( *v18 != 1818851428 )
        goto LABEL_100;
    }
    else if ( v19 != 8 || *(_QWORD *)v18 != 0x302E31766C697864LL && *(_QWORD *)v18 != 0x312E31766C697864LL )
    {
      goto LABEL_100;
    }
    v20 = 11;
    goto LABEL_102;
  }
LABEL_100:
  if ( !(unsigned __int8)sub_CC83E0((__int64)&v18, &v17, "dxilv1.2", 8)
    && !(unsigned __int8)sub_CC83E0((__int64)&v18, &v17, "dxilv1.3", 8)
    && !(unsigned __int8)sub_CC83E0((__int64)&v18, &v17, "dxilv1.4", 8)
    && !(unsigned __int8)sub_CC83E0((__int64)&v18, &v17, "dxilv1.5", 8)
    && !(unsigned __int8)sub_CC83E0((__int64)&v18, &v17, "dxilv1.6", 8)
    && !(unsigned __int8)sub_CC83E0((__int64)&v18, &v17, "dxilv1.7", 8) )
  {
    sub_CC83E0((__int64)&v18, &v17, "dxilv1.8", 8);
  }
  if ( (_BYTE)v21 )
  {
LABEL_102:
    v4 = v20;
    if ( v20 )
      return v4;
    goto LABEL_103;
  }
  if ( v19 == 6 && *v18 == 1852142712 && *((_WORD *)v18 + 2) == 24947 )
  {
    v20 = 41;
    goto LABEL_102;
  }
  if ( v19 == 6 && *v18 == 1634956910 )
  {
    v4 = 58;
    if ( *((_WORD *)v18 + 2) == 29555 )
      return v4;
  }
LABEL_103:
  if ( a2 <= 2 )
    return 0;
  if ( *(_WORD *)s1 == 29281 && *((_BYTE *)s1 + 2) == 109
    || a2 > 4
    && (*s1 == 1836410996 && *((_BYTE *)s1 + 4) == 98
     || a2 > 6 && *s1 == 1668440417 && *((_WORD *)s1 + 2) == 13928 && *((_BYTE *)s1 + 6) == 52) )
  {
    v6 = sub_F05D50(s1, a2);
    v7 = sub_F05DE0(s1, a2);
    v8 = v7;
    if ( v7 == 1 )
    {
      v16 = (unsigned int)(v6 - 1);
      v4 = 0;
      if ( (unsigned int)v16 <= 2 )
        v4 = dword_3F6C3B0[v16];
    }
    else
    {
      v4 = 0;
      if ( v7 == 2 )
      {
        v9 = (unsigned int)(v6 - 1);
        v4 = 0;
        if ( (unsigned int)v9 <= 2 )
          v4 = dword_3F6C3A0[v9];
      }
    }
    v10 = (_WORD *)sub_F05A00(s1);
    v12 = v10;
    v13 = v11;
    if ( v11 && (v6 != 2 || v11 == 1 || *v10 != 12918 && *v10 != 13174) )
    {
      v14 = sub_F05310(v10, v11);
      v15 = sub_F052E0(v12, v13);
      if ( v14 == 3 && v15 == 6 )
        return (unsigned int)(v8 == 2) + 36;
      return v4;
    }
    return 0;
  }
  if ( *(_WORD *)s1 != 28770 || *((_BYTE *)s1 + 2) != 102 )
    return 0;
  return sub_CC4190((__int64)s1, a2);
}
