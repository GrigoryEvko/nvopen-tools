// Function: sub_15AFD40
// Address: 0x15afd40
//
__int64 __fastcall sub_15AFD40(__int64 a1, __int64 a2)
{
  bool v2; // cl
  int *v3; // r11
  char v4; // dl
  bool v5; // r10
  char v6; // al
  char v7; // r10
  char v8; // r9
  char v9; // r12
  bool v10; // r9
  char v11; // r13
  bool v12; // r8
  char v13; // al
  char v14; // al
  char v16; // r12
  char v17; // r9

  v2 = a2 == 13;
  if ( a2 == 10 )
  {
    if ( *(_QWORD *)a1 != 0x655A67616C464944LL || (v6 = 0, v4 = 1, *(_WORD *)(a1 + 8) != 28530) )
    {
      v6 = 1;
      v4 = 0;
    }
    LODWORD(v3) = 0;
    goto LABEL_14;
  }
  if ( a2 == 13 )
  {
    if ( *(_QWORD *)a1 != 0x725067616C464944LL || *(_DWORD *)(a1 + 8) != 1952544361 || *(_BYTE *)(a1 + 12) != 101 )
    {
      v6 = 1;
      v4 = 0;
      LODWORD(v3) = 1;
      v8 = 1;
      v5 = 0;
      goto LABEL_66;
    }
    LODWORD(v3) = 1;
    v4 = 1;
    v6 = 0;
LABEL_14:
    v5 = a2 == 16;
    v8 = v2 & v6;
    if ( (v2 & (unsigned __int8)v6) == 0 )
      goto LABEL_15;
LABEL_66:
    if ( *(_QWORD *)a1 == 0x774667616C464944LL && *(_DWORD *)(a1 + 8) == 1667581028 && *(_BYTE *)(a1 + 12) == 108 )
    {
      v4 = v8;
      v6 = 0;
      LODWORD(v3) = 4;
      goto LABEL_17;
    }
    goto LABEL_16;
  }
  if ( a2 != 15 )
  {
    if ( a2 == 12 )
    {
      if ( *(_QWORD *)a1 != 0x755067616C464944LL || *(_DWORD *)(a1 + 8) != 1667853410 )
      {
        LODWORD(v3) = 0;
        v4 = 0;
        v5 = 0;
        v6 = 1;
        goto LABEL_17;
      }
      LODWORD(v3) = 3;
      v4 = 1;
      v6 = 0;
      v5 = 0;
      goto LABEL_16;
    }
    LODWORD(v3) = 0;
    v4 = 0;
    v6 = 1;
    goto LABEL_14;
  }
  if ( *(_QWORD *)a1 != 0x725067616C464944LL
    || *(_DWORD *)(a1 + 8) != 1667593327
    || *(_WORD *)(a1 + 12) != 25972
    || *(_BYTE *)(a1 + 14) != 100 )
  {
    v6 = 1;
    v5 = 0;
    v4 = 0;
    LODWORD(v3) = 0;
LABEL_16:
    if ( ((unsigned __int8)v6 & (a2 == 22)) != 0 )
    {
      if ( !(*(_QWORD *)a1 ^ 0x6C4267616C464944LL | *(_QWORD *)(a1 + 8) ^ 0x66657279426B636FLL)
        && *(_DWORD *)(a1 + 16) == 1970435155
        && *(_WORD *)(a1 + 20) == 29795 )
      {
        v4 = v6 & (a2 == 22);
        LODWORD(v3) = 16;
        v7 = a2 == 14;
        v6 = 0;
        goto LABEL_19;
      }
      goto LABEL_18;
    }
    goto LABEL_17;
  }
  v5 = 0;
  v4 = 1;
  v6 = 0;
  LODWORD(v3) = 2;
LABEL_15:
  if ( (v5 & (unsigned __int8)v6) == 0 )
    goto LABEL_16;
  if ( !(*(_QWORD *)a1 ^ 0x704167616C464944LL | *(_QWORD *)(a1 + 8) ^ 0x6B636F6C42656C70LL) )
  {
    v4 = v5 & v6;
    LODWORD(v3) = 8;
    v6 = 0;
LABEL_18:
    v9 = v6 & v5;
    v7 = a2 == 14;
    if ( v9 )
    {
      if ( *(_QWORD *)a1 ^ 0x724167616C464944LL | *(_QWORD *)(a1 + 8) ^ 0x6C61696369666974LL )
      {
        v10 = a2 == 23;
        if ( !(*(_QWORD *)a1 ^ 0x725067616C464944LL | *(_QWORD *)(a1 + 8) ^ 0x64657079746F746FLL) )
        {
          v4 = v9;
          LODWORD(v3) = 256;
          v6 = 0;
          goto LABEL_22;
        }
      }
      else
      {
        v4 = v9;
        LODWORD(v3) = 64;
        v10 = a2 == 23;
        v6 = 0;
      }
      goto LABEL_21;
    }
    goto LABEL_19;
  }
LABEL_17:
  if ( (v2 & (unsigned __int8)v6) == 0 )
    goto LABEL_18;
  v7 = a2 == 14;
  if ( *(_QWORD *)a1 == 0x695667616C464944LL && *(_DWORD *)(a1 + 8) == 1635087474 && *(_BYTE *)(a1 + 12) == 108 )
  {
    v4 = v2 & v6;
    LODWORD(v3) = 32;
    v10 = a2 == 23;
    v6 = 0;
    goto LABEL_20;
  }
LABEL_19:
  v10 = a2 == 23;
  if ( ((unsigned __int8)v7 & (unsigned __int8)v6) != 0
    && *(_QWORD *)a1 == 0x784567616C464944LL
    && *(_DWORD *)(a1 + 8) == 1667853424
    && *(_WORD *)(a1 + 12) == 29801 )
  {
    v4 = v7 & v6;
    LODWORD(v3) = 128;
    v6 = 0;
LABEL_21:
    if ( ((unsigned __int8)v6 & (a2 == 19)) != 0 )
    {
      if ( !(*(_QWORD *)a1 ^ 0x624F67616C464944LL | *(_QWORD *)(a1 + 8) ^ 0x6E696F507463656ALL)
        && *(_WORD *)(a1 + 16) == 25972
        && *(_BYTE *)(a1 + 18) == 114 )
      {
        LODWORD(v3) = 1024;
        goto LABEL_47;
      }
      goto LABEL_23;
    }
    goto LABEL_22;
  }
LABEL_20:
  if ( (v10 & (unsigned __int8)v6) == 0 )
    goto LABEL_21;
  if ( !(*(_QWORD *)a1 ^ 0x624F67616C464944LL | *(_QWORD *)(a1 + 8) ^ 0x437373616C43636ALL)
    && *(_DWORD *)(a1 + 16) == 1819307375
    && *(_WORD *)(a1 + 20) == 29797
    && *(_BYTE *)(a1 + 22) == 101 )
  {
    v4 = v10 & v6;
    v6 = 0;
    LODWORD(v3) = 512;
LABEL_23:
    if ( ((unsigned __int8)v6 & (a2 == 18)) != 0 )
    {
      if ( *(_QWORD *)a1 ^ 0x745367616C464944LL | *(_QWORD *)(a1 + 8) ^ 0x626D654D63697461LL
        || *(_WORD *)(a1 + 16) != 29285 )
      {
LABEL_26:
        v11 = v7 & v6;
        if ( ((unsigned __int8)v7 & (unsigned __int8)v6) != 0 )
        {
          v12 = a2 == 25;
          if ( *(_QWORD *)a1 == 0x655267616C464944LL
            && *(_DWORD *)(a1 + 8) == 1987208563
            && *(_WORD *)(a1 + 12) == 25701 )
          {
            v4 = v7 & v6;
            v7 &= v6;
            v6 = 0;
            LODWORD(v3) = 0x8000;
            goto LABEL_51;
          }
          v6 &= v7;
          v7 = v11;
          if ( a2 != 25 )
            goto LABEL_51;
LABEL_29:
          if ( !(*(_QWORD *)a1 ^ 0x754D67616C464944LL | *(_QWORD *)(a1 + 8) ^ 0x6E49656C7069746CLL)
            && *(_QWORD *)(a1 + 16) == 0x636E617469726568LL
            && *(_BYTE *)(a1 + 24) == 101 )
          {
            LODWORD(v3) = 0x20000;
LABEL_33:
            v7 = 1;
            v4 = 0;
            goto LABEL_34;
          }
          goto LABEL_52;
        }
        goto LABEL_48;
      }
      v4 = v6 & (a2 == 18);
      v6 = 0;
      LODWORD(v3) = 4096;
LABEL_48:
      v16 = v6 & v10;
      goto LABEL_49;
    }
    goto LABEL_24;
  }
LABEL_22:
  if ( ((unsigned __int8)v6 & (a2 == 12)) == 0 )
    goto LABEL_23;
  if ( *(_QWORD *)a1 == 0x655667616C464944LL && *(_DWORD *)(a1 + 8) == 1919906915 )
  {
    LODWORD(v3) = 2048;
    goto LABEL_47;
  }
LABEL_24:
  if ( v4 )
  {
LABEL_47:
    v4 = 1;
    v6 = 0;
    goto LABEL_48;
  }
  v6 = 1;
  if ( a2 != 21 )
    goto LABEL_26;
  if ( !(*(_QWORD *)a1 ^ 0x564C67616C464944LL | *(_QWORD *)(a1 + 8) ^ 0x6566655265756C61LL)
    && *(_DWORD *)(a1 + 16) == 1668179314
    && *(_BYTE *)(a1 + 20) == 101 )
  {
    LODWORD(v3) = 0x2000;
    goto LABEL_157;
  }
  if ( !(*(_QWORD *)a1 ^ 0x565267616C464944LL | *(_QWORD *)(a1 + 8) ^ 0x6566655265756C61LL)
    && *(_DWORD *)(a1 + 16) == 1668179314
    && *(_BYTE *)(a1 + 20) == 101 )
  {
    LODWORD(v3) = 0x4000;
LABEL_157:
    v12 = 0;
    v6 = 0;
    v4 = 1;
LABEL_50:
    if ( ((unsigned __int8)v6 & v12) == 0 )
      goto LABEL_51;
    goto LABEL_29;
  }
  v16 = v10;
  v6 = 1;
LABEL_49:
  v12 = a2 == 25;
  if ( !v16 )
    goto LABEL_50;
  if ( !(*(_QWORD *)a1 ^ 0x695367616C464944LL | *(_QWORD *)(a1 + 8) ^ 0x65686E49656C676ELL)
    && *(_DWORD *)(a1 + 16) == 1635019122
    && *(_WORD *)(a1 + 20) == 25454
    && *(_BYTE *)(a1 + 22) == 101 )
  {
    v4 = v16;
    v6 = 0;
    LODWORD(v3) = 0x10000;
LABEL_52:
    v17 = v6 & v10;
    if ( v17 )
    {
      if ( *(_QWORD *)a1 ^ 0x6E4967616C464944LL | *(_QWORD *)(a1 + 8) ^ 0x64656375646F7274LL
        || *(_DWORD *)(a1 + 16) != 1953655126
        || *(_WORD *)(a1 + 20) != 24949
        || *(_BYTE *)(a1 + 22) != 108 )
      {
        v7 = v4;
        v4 = v6;
LABEL_56:
        if ( ((unsigned __int8)v4 & (a2 == 20)) != 0 )
        {
          if ( !(*(_QWORD *)a1 ^ 0x614D67616C464944LL | *(_QWORD *)(a1 + 8) ^ 0x6F72706275536E69LL)
            && *(_DWORD *)(a1 + 16) == 1835102823 )
          {
            v7 = v4 & (a2 == 20);
            v4 = 0;
            LODWORD(v3) = 0x200000;
          }
          else
          {
            v4 &= a2 == 20;
            v14 = v12;
            if ( v12 )
              goto LABEL_60;
          }
          goto LABEL_37;
        }
        goto LABEL_34;
      }
      v7 = v17;
      LODWORD(v3) = 0x40000;
      v4 = 0;
LABEL_34:
      v13 = v4 & (a2 == 21);
      goto LABEL_35;
    }
    goto LABEL_53;
  }
LABEL_51:
  if ( ((unsigned __int8)v6 & (a2 == 24)) == 0 )
    goto LABEL_52;
  if ( !(*(_QWORD *)a1 ^ 0x695667616C464944LL | *(_QWORD *)(a1 + 8) ^ 0x686E496C61757472LL)
    && *(_QWORD *)(a1 + 16) == 0x65636E6174697265LL )
  {
    LODWORD(v3) = 196608;
    goto LABEL_33;
  }
LABEL_53:
  if ( v4 )
    goto LABEL_33;
  if ( !v7 )
  {
    v4 = 1;
    goto LABEL_56;
  }
  if ( *(_QWORD *)a1 == 0x694267616C464944LL && *(_DWORD *)(a1 + 8) == 1701398132 && *(_WORD *)(a1 + 12) == 25708 )
  {
    LODWORD(v3) = 0x80000;
    goto LABEL_36;
  }
  if ( *(_QWORD *)a1 == 0x6F4E67616C464944LL && *(_DWORD *)(a1 + 8) == 1970562386 && *(_WORD *)(a1 + 12) == 28274 )
  {
    LODWORD(v3) = 0x100000;
LABEL_36:
    v14 = v12 & v4;
    if ( (v12 & (unsigned __int8)v4) != 0 )
    {
LABEL_60:
      if ( !(*(_QWORD *)a1 ^ 0x795467616C464944LL | *(_QWORD *)(a1 + 8) ^ 0x7942737361506570LL)
        && *(_QWORD *)(a1 + 16) == 0x636E657265666552LL
        && *(_BYTE *)(a1 + 24) == 101 )
      {
        v7 = v14;
        v4 = 0;
        LODWORD(v3) = 0x800000;
        goto LABEL_39;
      }
      goto LABEL_38;
    }
    goto LABEL_37;
  }
  v4 = v7;
  v13 = a2 == 21;
  v7 = 0;
LABEL_35:
  if ( !v13 )
    goto LABEL_36;
  if ( !(*(_QWORD *)a1 ^ 0x795467616C464944LL | *(_QWORD *)(a1 + 8) ^ 0x7942737361506570LL)
    && *(_DWORD *)(a1 + 16) == 1970037078
    && *(_BYTE *)(a1 + 20) == 101 )
  {
    v7 = v13;
    v3 = &dword_400000;
    v4 = 0;
    goto LABEL_38;
  }
LABEL_37:
  if ( ((unsigned __int8)v4 & (a2 == 15)) != 0 )
  {
    if ( *(_QWORD *)a1 != 0x694667616C464944LL
      || *(_DWORD *)(a1 + 8) != 1164207480
      || *(_WORD *)(a1 + 12) != 30062
      || *(_BYTE *)(a1 + 14) != 109 )
    {
      goto LABEL_39;
    }
    v7 = v4 & (a2 == 15);
    v4 = 0;
    v3 = (int *)&loc_1000000;
    goto LABEL_40;
  }
LABEL_38:
  if ( ((unsigned __int8)v4 & (a2 == 11)) == 0 )
  {
LABEL_39:
    if ( ((unsigned __int8)v4 & v2) == 0 )
      goto LABEL_40;
    if ( *(_QWORD *)a1 != 0x725467616C464944LL || *(_DWORD *)(a1 + 8) != 1634301545 || *(_BYTE *)(a1 + 12) != 108 )
      goto LABEL_41;
    LODWORD(v3) = 0x4000000;
    return (unsigned int)v3;
  }
  if ( *(_QWORD *)a1 == 0x685467616C464944LL && *(_WORD *)(a1 + 8) == 28277 && *(_BYTE *)(a1 + 10) == 107 )
  {
    LODWORD(v3) = 0x2000000;
    return (unsigned int)v3;
  }
LABEL_40:
  if ( (v12 & (unsigned __int8)v4) == 0 )
  {
LABEL_41:
    if ( !v7 )
      LODWORD(v3) = 0;
    return (unsigned int)v3;
  }
  if ( !(*(_QWORD *)a1 ^ 0x6E4967616C464944LL | *(_QWORD *)(a1 + 8) ^ 0x6956746365726964LL)
    && *(_QWORD *)(a1 + 16) == 0x7361426C61757472LL )
  {
    LODWORD(v3) = 36;
    if ( *(_BYTE *)(a1 + 24) == 101 )
      return (unsigned int)v3;
  }
  return 0;
}
