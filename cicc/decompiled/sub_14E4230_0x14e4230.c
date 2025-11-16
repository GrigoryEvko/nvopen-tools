// Function: sub_14E4230
// Address: 0x14e4230
//
__int64 __fastcall sub_14E4230(__int64 a1, __int64 a2)
{
  char v2; // al
  char v3; // cl
  bool v4; // r11
  bool v5; // r9
  char v6; // dl
  char v7; // r12
  bool v8; // r8
  unsigned int v9; // r10d
  bool v10; // r12
  char v11; // r12
  bool v12; // r12
  char v13; // r13
  char v14; // r11
  bool v15; // r9
  bool v16; // r8
  char v17; // al
  char v18; // r14
  char v19; // r13
  char v20; // r9
  char v21; // cl
  char v22; // cl
  char v23; // r12
  char v24; // cl
  char v25; // r8
  char v27; // r14
  int v28; // eax

  if ( a2 == 10 )
  {
    if ( *(_QWORD *)a1 != 0x64615F504F5F5744LL || (v9 = 3, *(_WORD *)(a1 + 8) != 29284) )
    {
      if ( *(_QWORD *)a1 == 0x72645F504F5F5744LL && (v9 = 19, *(_WORD *)(a1 + 8) == 28783)
        || *(_QWORD *)a1 == 0x766F5F504F5F5744LL && (v9 = 20, *(_WORD *)(a1 + 8) == 29285)
        || *(_QWORD *)a1 == 0x69705F504F5F5744LL && (v9 = 21, *(_WORD *)(a1 + 8) == 27491)
        || *(_QWORD *)a1 == 0x77735F504F5F5744LL && (v9 = 22, *(_WORD *)(a1 + 8) == 28769) )
      {
        v3 = 0;
        v2 = 0;
        goto LABEL_296;
      }
      v3 = 0;
      v2 = 0;
LABEL_22:
      if ( v3 )
      {
        if ( *(_QWORD *)a1 == 0x64785F504F5F5744LL )
        {
          v9 = 24;
          if ( *(_DWORD *)(a1 + 8) == 1717924453 )
            goto LABEL_296;
        }
      }
      if ( a2 != 11 )
      {
        v2 = 0;
        goto LABEL_25;
      }
      goto LABEL_24;
    }
LABEL_294:
    v2 = a2 == 9;
    goto LABEL_295;
  }
  if ( a2 == 11 )
  {
    if ( *(_QWORD *)a1 != 0x65645F504F5F5744LL )
      goto LABEL_8;
    if ( *(_WORD *)(a1 + 8) != 25970 )
      goto LABEL_8;
    v9 = 6;
    if ( *(_BYTE *)(a1 + 10) != 102 )
      goto LABEL_8;
    goto LABEL_294;
  }
  v2 = a2 == 9;
  if ( a2 == 13 )
  {
    if ( *(_QWORD *)a1 == 0x6F635F504F5F5744LL && *(_DWORD *)(a1 + 8) == 829715310 )
    {
      v9 = 8;
      if ( *(_BYTE *)(a1 + 12) == 117 )
        goto LABEL_295;
    }
    if ( *(_QWORD *)a1 == 0x6F635F504F5F5744LL && *(_DWORD *)(a1 + 8) == 829715310 )
    {
      v9 = 9;
      if ( *(_BYTE *)(a1 + 12) == 115 )
        goto LABEL_295;
    }
    if ( *(_QWORD *)a1 == 0x6F635F504F5F5744LL && *(_DWORD *)(a1 + 8) == 846492526 )
    {
      v2 = 0;
      v9 = 10;
      if ( *(_BYTE *)(a1 + 12) == 117 )
        goto LABEL_295;
    }
    if ( *(_QWORD *)a1 == 0x6F635F504F5F5744LL && *(_DWORD *)(a1 + 8) == 846492526 )
    {
      v2 = 0;
      v9 = 11;
      if ( *(_BYTE *)(a1 + 12) == 115 )
        goto LABEL_295;
    }
    if ( *(_QWORD *)a1 == 0x6F635F504F5F5744LL && *(_DWORD *)(a1 + 8) == 880046958 )
    {
      v2 = 0;
      v9 = 12;
      if ( *(_BYTE *)(a1 + 12) == 117 )
        goto LABEL_295;
    }
    if ( *(_QWORD *)a1 == 0x6F635F504F5F5744LL && *(_DWORD *)(a1 + 8) == 880046958 )
    {
      v2 = 0;
      v9 = 13;
      if ( *(_BYTE *)(a1 + 12) == 115 )
        goto LABEL_295;
    }
    if ( *(_QWORD *)a1 == 0x6F635F504F5F5744LL && *(_DWORD *)(a1 + 8) == 947155822 )
    {
      v2 = 0;
      v9 = 14;
      if ( *(_BYTE *)(a1 + 12) == 117 )
        goto LABEL_295;
    }
    if ( *(_QWORD *)a1 != 0x6F635F504F5F5744LL || *(_DWORD *)(a1 + 8) != 947155822 || *(_BYTE *)(a1 + 12) != 115 )
      goto LABEL_8;
    v2 = 0;
    v9 = 15;
    goto LABEL_295;
  }
  if ( a2 == 12 )
  {
    if ( *(_QWORD *)a1 == 0x6F635F504F5F5744LL && (v9 = 16, *(_DWORD *)(a1 + 8) == 1970565998) )
    {
      v2 = 0;
    }
    else
    {
      if ( *(_QWORD *)a1 != 0x6F635F504F5F5744LL )
        goto LABEL_8;
      v9 = 17;
      if ( *(_DWORD *)(a1 + 8) != 1937011566 )
        goto LABEL_8;
      v2 = 0;
    }
LABEL_295:
    v3 = a2 == 12;
LABEL_296:
    v5 = a2 == 10;
LABEL_297:
    v6 = a2 == 17;
    goto LABEL_298;
  }
  if ( a2 != 9 )
  {
LABEL_8:
    v2 = 0;
    goto LABEL_9;
  }
  if ( *(_QWORD *)a1 == 0x75645F504F5F5744LL )
  {
    v9 = 18;
    v2 = 1;
    if ( *(_BYTE *)(a1 + 8) == 112 )
      goto LABEL_295;
  }
  v2 = 1;
LABEL_9:
  v3 = a2 == 12;
  if ( !v2 )
    goto LABEL_22;
  if ( *(_QWORD *)a1 == 0x6F725F504F5F5744LL )
  {
    v9 = 23;
    if ( *(_BYTE *)(a1 + 8) == 116 )
      goto LABEL_296;
  }
  if ( *(_QWORD *)a1 == 0x62615F504F5F5744LL )
  {
    v9 = 25;
    v2 = 1;
    if ( *(_BYTE *)(a1 + 8) == 115 )
      goto LABEL_296;
  }
  if ( *(_QWORD *)a1 == 0x6E615F504F5F5744LL )
  {
    v9 = 26;
    v2 = 1;
    if ( *(_BYTE *)(a1 + 8) == 100 )
      goto LABEL_296;
  }
  if ( *(_QWORD *)a1 == 0x69645F504F5F5744LL )
  {
    v9 = 27;
    v2 = 1;
    if ( *(_BYTE *)(a1 + 8) == 118 )
      goto LABEL_296;
  }
  if ( a2 != 11 )
  {
    v5 = a2 == 10;
    if ( *(_QWORD *)a1 == 0x6F6D5F504F5F5744LL )
    {
      v2 = 1;
      v9 = 29;
      if ( *(_BYTE *)(a1 + 8) == 100 )
        goto LABEL_297;
    }
    if ( *(_QWORD *)a1 == 0x756D5F504F5F5744LL )
    {
      v2 = 1;
      v9 = 30;
      if ( *(_BYTE *)(a1 + 8) == 108 )
        goto LABEL_297;
    }
    if ( *(_QWORD *)a1 != 0x656E5F504F5F5744LL || (v28 = 0, *(_BYTE *)(a1 + 8) != 103) )
      v28 = 1;
    v5 = a2 == 10;
    if ( !v28 )
    {
      v2 = 1;
      v9 = 31;
      goto LABEL_297;
    }
    v2 = 1;
    if ( *(_QWORD *)a1 == 0x6F6E5F504F5F5744LL )
    {
      v9 = 32;
      if ( *(_BYTE *)(a1 + 8) == 116 )
        goto LABEL_297;
    }
    goto LABEL_25;
  }
  v2 = 1;
LABEL_24:
  if ( *(_QWORD *)a1 == 0x696D5F504F5F5744LL && *(_WORD *)(a1 + 8) == 30062 )
  {
    v9 = 28;
    if ( *(_BYTE *)(a1 + 10) == 115 )
      goto LABEL_296;
  }
LABEL_25:
  v4 = a2 == 8;
  v5 = a2 == 10;
  if ( a2 == 8 )
  {
    v9 = 3;
    if ( *(_QWORD *)a1 == 0x726F5F504F5F5744LL )
      v9 = 33;
    v4 = 1;
    v6 = 0;
    v8 = *(_QWORD *)a1 == 0x726F5F504F5F5744LL;
    v5 = 0;
    if ( *(_QWORD *)a1 == 0x726F5F504F5F5744LL )
      goto LABEL_87;
  }
  else
  {
    v6 = a2 == 17;
    if ( a2 == 10 )
    {
      if ( *(_QWORD *)a1 == 0x6C705F504F5F5744LL && *(_WORD *)(a1 + 8) == 29557 )
      {
        v7 = 1;
        v8 = 1;
        v9 = 34;
      }
      else
      {
        v7 = 0;
        v8 = 0;
        v9 = 3;
      }
      v5 = 1;
      if ( v7 )
        goto LABEL_87;
    }
    else
    {
      if ( a2 == 17 )
      {
        if ( !(*(_QWORD *)a1 ^ 0x6C705F504F5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x736E6F63755F7375LL)
          && *(_BYTE *)(a1 + 16) == 116 )
        {
          v8 = 1;
          v9 = 35;
          v6 = 1;
          goto LABEL_87;
        }
        v8 = 0;
        v9 = 3;
        v6 = 1;
        goto LABEL_31;
      }
      v8 = 0;
      v9 = 3;
    }
  }
  if ( v2 )
  {
    if ( *(_QWORD *)a1 == 0x68735F504F5F5744LL && *(_BYTE *)(a1 + 8) == 108 )
    {
      v9 = 36;
      goto LABEL_298;
    }
    if ( *(_QWORD *)a1 == 0x68735F504F5F5744LL && *(_BYTE *)(a1 + 8) == 114 )
    {
      v9 = 37;
LABEL_298:
      v8 = 1;
LABEL_87:
      v4 = 1;
      goto LABEL_88;
    }
    goto LABEL_137;
  }
LABEL_31:
  if ( !v5 )
  {
    if ( !v2 )
      goto LABEL_33;
LABEL_137:
    if ( *(_QWORD *)a1 == 0x6F785F504F5F5744LL && *(_BYTE *)(a1 + 8) == 114 )
    {
      v9 = 39;
    }
    else
    {
      if ( *(_QWORD *)a1 != 0x72625F504F5F5744LL || *(_BYTE *)(a1 + 8) != 97 )
      {
        v2 = 1;
        v10 = 0;
LABEL_34:
        if ( !v5 )
        {
          v4 = v8;
          if ( v10 )
            goto LABEL_36;
          v5 = v8;
          goto LABEL_149;
        }
LABEL_263:
        if ( *(_QWORD *)a1 == 0x6B735F504F5F5744LL && *(_WORD *)(a1 + 8) == 28777 )
        {
          v4 = 1;
          v5 = 1;
          v8 = 1;
          v9 = 47;
          goto LABEL_88;
        }
        if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 12404 )
        {
          v4 = 1;
          v5 = 1;
          v8 = 1;
          v9 = 48;
          goto LABEL_88;
        }
        v5 = v8;
        if ( v8 )
        {
          v4 = v8;
          goto LABEL_152;
        }
        if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 12660 )
        {
          v4 = 1;
          v5 = 1;
          v8 = 1;
          v9 = 49;
          goto LABEL_88;
        }
        if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 12916 )
        {
          v4 = 1;
          v5 = 1;
          v8 = 1;
          v9 = 50;
          goto LABEL_88;
        }
        if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 13172 )
        {
          v4 = 1;
          v5 = 1;
          v8 = 1;
          v9 = 51;
          goto LABEL_88;
        }
        if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 13428 )
        {
          v4 = 1;
          v5 = 1;
          v8 = 1;
          v9 = 52;
          goto LABEL_88;
        }
        if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 13684 )
        {
          v4 = 1;
          v5 = 1;
          v8 = 1;
          v9 = 53;
          goto LABEL_88;
        }
        if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 13940 )
        {
          v4 = 1;
          v5 = 1;
          v8 = 1;
          v9 = 54;
          goto LABEL_88;
        }
        if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 14196 )
        {
          v4 = 1;
          v5 = 1;
          v8 = 1;
          v9 = 55;
          goto LABEL_88;
        }
        if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 14452 )
        {
          v4 = 1;
          v5 = 1;
          v8 = 1;
          v9 = 56;
          goto LABEL_88;
        }
        if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 14708 )
        {
          v4 = 1;
          v5 = 1;
          v8 = 1;
          v9 = 57;
          goto LABEL_88;
        }
        if ( a2 != 11 )
          goto LABEL_39;
        v5 = 1;
LABEL_168:
        if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 12660 && *(_BYTE *)(a1 + 10) == 48 )
        {
          v4 = 1;
          v8 = 1;
          v9 = 58;
          goto LABEL_88;
        }
        if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 12660 && *(_BYTE *)(a1 + 10) == 49 )
        {
          v4 = 1;
          v8 = 1;
          v9 = 59;
          goto LABEL_88;
        }
        v4 = v8;
        if ( v8 )
          goto LABEL_88;
        goto LABEL_171;
      }
      v9 = 40;
    }
    v4 = 1;
    v2 = 1;
    v8 = 1;
    goto LABEL_88;
  }
  if ( *(_QWORD *)a1 == 0x68735F504F5F5744LL && *(_WORD *)(a1 + 8) == 24946 )
  {
    v8 = 1;
    v9 = 38;
    goto LABEL_87;
  }
LABEL_33:
  v10 = v8;
  if ( !v4 )
    goto LABEL_34;
  if ( *(_QWORD *)a1 == 0x71655F504F5F5744LL )
  {
    v8 = 1;
    v9 = 41;
    goto LABEL_152;
  }
  if ( *(_QWORD *)a1 == 0x65675F504F5F5744LL )
  {
    v8 = 1;
    v9 = 42;
    goto LABEL_152;
  }
  if ( v8 )
  {
    v4 = v8;
LABEL_36:
    if ( v8 )
      goto LABEL_88;
    if ( a2 != 11 )
    {
      if ( v5 )
      {
LABEL_39:
        if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 12391 )
        {
          v4 = 1;
          v5 = 1;
          v8 = 1;
          v9 = 80;
          goto LABEL_88;
        }
        if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 12647 )
        {
          v4 = 1;
          v5 = 1;
          v8 = 1;
          v9 = 81;
          goto LABEL_88;
        }
        goto LABEL_41;
      }
      goto LABEL_154;
    }
LABEL_171:
    if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 12660 && *(_BYTE *)(a1 + 10) == 50 )
    {
      v4 = 1;
      v8 = 1;
      v9 = 60;
      goto LABEL_88;
    }
    if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 12660 && *(_BYTE *)(a1 + 10) == 51 )
    {
      v4 = 1;
      v8 = 1;
      v9 = 61;
      goto LABEL_88;
    }
    if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 12660 && *(_BYTE *)(a1 + 10) == 52 )
    {
      v4 = 1;
      v8 = 1;
      v9 = 62;
      goto LABEL_88;
    }
    if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 12660 && *(_BYTE *)(a1 + 10) == 53 )
    {
      v4 = 1;
      v8 = 1;
      v9 = 63;
      goto LABEL_88;
    }
    if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 12660 && *(_BYTE *)(a1 + 10) == 54 )
    {
      v4 = 1;
      v8 = 1;
      v9 = 64;
      goto LABEL_88;
    }
    if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 12660 && *(_BYTE *)(a1 + 10) == 55 )
    {
      v4 = 1;
      v8 = 1;
      v9 = 65;
      goto LABEL_88;
    }
    if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 12660 && *(_BYTE *)(a1 + 10) == 56 )
    {
      v4 = 1;
      v8 = 1;
      v9 = 66;
      goto LABEL_88;
    }
    if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 12660 && *(_BYTE *)(a1 + 10) == 57 )
    {
      v4 = 1;
      v8 = 1;
      v9 = 67;
      goto LABEL_88;
    }
    if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 12916 && *(_BYTE *)(a1 + 10) == 48 )
    {
      v4 = 1;
      v8 = 1;
      v9 = 68;
      goto LABEL_88;
    }
    if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 12916 && *(_BYTE *)(a1 + 10) == 49 )
    {
      v4 = 1;
      v8 = 1;
      v9 = 69;
      goto LABEL_88;
    }
    if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 12916 && *(_BYTE *)(a1 + 10) == 50 )
    {
      v4 = 1;
      v8 = 1;
      v9 = 70;
      goto LABEL_88;
    }
    if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 12916 && *(_BYTE *)(a1 + 10) == 51 )
    {
      v4 = 1;
      v8 = 1;
      v9 = 71;
      goto LABEL_88;
    }
    if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 12916 && *(_BYTE *)(a1 + 10) == 52 )
    {
      v4 = 1;
      v8 = 1;
      v9 = 72;
      goto LABEL_88;
    }
    if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 12916 && *(_BYTE *)(a1 + 10) == 53 )
    {
      v4 = 1;
      v8 = 1;
      v9 = 73;
      goto LABEL_88;
    }
    if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 12916 && *(_BYTE *)(a1 + 10) == 54 )
    {
      v4 = 1;
      v8 = 1;
      v9 = 74;
      goto LABEL_88;
    }
    if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 12916 && *(_BYTE *)(a1 + 10) == 55 )
    {
      v4 = 1;
      v8 = 1;
      v9 = 75;
      goto LABEL_88;
    }
    if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 12916 && *(_BYTE *)(a1 + 10) == 56 )
    {
      v4 = 1;
      v8 = 1;
      v9 = 76;
      goto LABEL_88;
    }
    if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 12916 && *(_BYTE *)(a1 + 10) == 57 )
    {
      v4 = 1;
      v8 = 1;
      v9 = 77;
      goto LABEL_88;
    }
    if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 13172 && *(_BYTE *)(a1 + 10) == 48 )
    {
      v4 = 1;
      v8 = 1;
      v9 = 78;
      goto LABEL_88;
    }
    if ( *(_QWORD *)a1 == 0x696C5F504F5F5744LL && *(_WORD *)(a1 + 8) == 13172 && *(_BYTE *)(a1 + 10) == 49 )
    {
      v4 = 1;
      v8 = 1;
      v9 = 79;
      goto LABEL_88;
    }
    if ( v5 )
      goto LABEL_39;
LABEL_192:
    v5 = 0;
    goto LABEL_193;
  }
  switch ( *(_QWORD *)a1 )
  {
    case 0x74675F504F5F5744LL:
      v8 = 1;
      v9 = 43;
      goto LABEL_152;
    case 0x656C5F504F5F5744LL:
      v8 = 1;
      v9 = 44;
      goto LABEL_152;
    case 0x746C5F504F5F5744LL:
      v8 = 1;
      v9 = 45;
      goto LABEL_152;
    case 0x656E5F504F5F5744LL:
      v8 = 1;
      v9 = 46;
      goto LABEL_152;
  }
  if ( v5 )
    goto LABEL_263;
LABEL_149:
  if ( a2 == 11 )
  {
    v5 = 0;
    goto LABEL_168;
  }
  v4 = v8;
  if ( !v8 )
    goto LABEL_154;
  v5 = 0;
LABEL_152:
  if ( v4 )
    goto LABEL_88;
  if ( !v5 )
  {
LABEL_154:
    if ( a2 != 11 )
    {
      v4 = v8;
      if ( v5 )
      {
        v5 = 0;
        goto LABEL_88;
      }
LABEL_51:
      if ( !v3 )
      {
        v11 = v8;
        if ( v4 )
          goto LABEL_53;
        goto LABEL_90;
      }
      goto LABEL_227;
    }
    goto LABEL_192;
  }
LABEL_41:
  if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 12903 )
  {
    v4 = 1;
    v5 = 1;
    v8 = 1;
    v9 = 82;
    goto LABEL_88;
  }
  v5 = v8;
  if ( !v8 )
  {
    if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 13159 )
    {
      v4 = 1;
      v5 = 1;
      v8 = 1;
      v9 = 83;
      goto LABEL_88;
    }
    if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 13415 )
    {
      v4 = 1;
      v5 = 1;
      v8 = 1;
      v9 = 84;
      goto LABEL_88;
    }
    if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 13671 )
    {
      v4 = 1;
      v5 = 1;
      v8 = 1;
      v9 = 85;
      goto LABEL_88;
    }
    if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 13927 )
    {
      v4 = 1;
      v5 = 1;
      v8 = 1;
      v9 = 86;
      goto LABEL_88;
    }
    if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 14183 )
    {
      v4 = 1;
      v5 = 1;
      v8 = 1;
      v9 = 87;
      goto LABEL_88;
    }
    if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 14439 )
    {
      v4 = 1;
      v5 = 1;
      v8 = 1;
      v9 = 88;
      goto LABEL_88;
    }
    if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 14695 )
    {
      v4 = 1;
      v5 = 1;
      v8 = 1;
      v9 = 89;
      goto LABEL_88;
    }
    v4 = 0;
    v5 = 1;
    if ( a2 != 11 )
      goto LABEL_51;
    v5 = 1;
LABEL_193:
    if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 12647 && *(_BYTE *)(a1 + 10) == 48 )
    {
      v4 = 1;
      v8 = 1;
      v9 = 90;
      goto LABEL_88;
    }
    if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 12647 && *(_BYTE *)(a1 + 10) == 49 )
    {
      v4 = 1;
      v8 = 1;
      v9 = 91;
      goto LABEL_88;
    }
    v4 = v8;
    if ( v8 )
      goto LABEL_88;
    if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 12647 && *(_BYTE *)(a1 + 10) == 50 )
    {
      v4 = 1;
      v9 = 92;
      goto LABEL_446;
    }
    if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 12647 && *(_BYTE *)(a1 + 10) == 51 )
    {
      v4 = 1;
      v9 = 93;
      goto LABEL_446;
    }
    if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 12647 && *(_BYTE *)(a1 + 10) == 52 )
    {
      v4 = 1;
      v9 = 94;
      goto LABEL_446;
    }
    if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 12647 && *(_BYTE *)(a1 + 10) == 53 )
    {
      v4 = 1;
      v9 = 95;
      goto LABEL_446;
    }
    if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 12647 && *(_BYTE *)(a1 + 10) == 54 )
    {
      v4 = 1;
      v9 = 96;
      goto LABEL_446;
    }
    if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 12647 && *(_BYTE *)(a1 + 10) == 55 )
    {
      v4 = 1;
      v9 = 97;
      goto LABEL_446;
    }
    if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 12647 && *(_BYTE *)(a1 + 10) == 56 )
    {
      v4 = 1;
      v9 = 98;
      goto LABEL_446;
    }
    if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 12647 && *(_BYTE *)(a1 + 10) == 57 )
    {
      v4 = 1;
      v9 = 99;
      goto LABEL_446;
    }
    if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 12903 && *(_BYTE *)(a1 + 10) == 48 )
    {
      v4 = 1;
      v9 = 100;
      goto LABEL_446;
    }
    if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 12903 && *(_BYTE *)(a1 + 10) == 49 )
    {
      v4 = 1;
      v9 = 101;
      goto LABEL_446;
    }
    if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 12903 && *(_BYTE *)(a1 + 10) == 50 )
    {
      v9 = 102;
      goto LABEL_142;
    }
    if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 12903 && *(_BYTE *)(a1 + 10) == 51 )
    {
      v9 = 103;
      goto LABEL_142;
    }
    if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 12903 && *(_BYTE *)(a1 + 10) == 52 )
    {
      v9 = 104;
      goto LABEL_142;
    }
    if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 12903 && *(_BYTE *)(a1 + 10) == 53 )
    {
      v9 = 105;
      goto LABEL_142;
    }
    if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 12903 && *(_BYTE *)(a1 + 10) == 54 )
    {
      v9 = 106;
      goto LABEL_142;
    }
    if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 12903 && *(_BYTE *)(a1 + 10) == 55 )
    {
      v9 = 107;
      goto LABEL_142;
    }
    if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 12903 && *(_BYTE *)(a1 + 10) == 56 )
    {
      v9 = 108;
      goto LABEL_142;
    }
    if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 12903 && *(_BYTE *)(a1 + 10) == 57 )
    {
      v9 = 109;
      goto LABEL_142;
    }
    if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 13159 && *(_BYTE *)(a1 + 10) == 48 )
    {
      v9 = 110;
      goto LABEL_142;
    }
    if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 13159 && *(_BYTE *)(a1 + 10) == 49 )
    {
      v9 = 111;
      goto LABEL_142;
    }
    if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_WORD *)(a1 + 8) == 26469 && *(_BYTE *)(a1 + 10) == 48 )
    {
      v9 = 112;
      goto LABEL_142;
    }
    if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_WORD *)(a1 + 8) == 26469 && *(_BYTE *)(a1 + 10) == 49 )
    {
      v9 = 113;
      goto LABEL_142;
    }
    if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_WORD *)(a1 + 8) == 26469 && *(_BYTE *)(a1 + 10) == 50 )
    {
      v9 = 114;
      goto LABEL_142;
    }
    if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_WORD *)(a1 + 8) == 26469 && *(_BYTE *)(a1 + 10) == 51 )
    {
      v9 = 115;
      goto LABEL_142;
    }
    if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_WORD *)(a1 + 8) == 26469 && *(_BYTE *)(a1 + 10) == 52 )
    {
      v9 = 116;
      goto LABEL_142;
    }
    if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_WORD *)(a1 + 8) == 26469 && *(_BYTE *)(a1 + 10) == 53 )
    {
      v9 = 117;
      goto LABEL_142;
    }
    if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_WORD *)(a1 + 8) == 26469 && *(_BYTE *)(a1 + 10) == 54 )
    {
      v9 = 118;
      goto LABEL_142;
    }
    if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_WORD *)(a1 + 8) == 26469 && *(_BYTE *)(a1 + 10) == 55 )
    {
      v9 = 119;
      goto LABEL_142;
    }
    if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_WORD *)(a1 + 8) == 26469 && *(_BYTE *)(a1 + 10) == 56 )
    {
      v9 = 120;
      goto LABEL_142;
    }
    if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_WORD *)(a1 + 8) == 26469 && *(_BYTE *)(a1 + 10) == 57 )
    {
      v9 = 121;
      goto LABEL_142;
    }
    v11 = 0;
    if ( !v3 )
    {
LABEL_90:
      v3 = v5 & (v11 ^ 1);
      if ( v3 )
      {
        v3 = 0;
LABEL_92:
        if ( *(_QWORD *)a1 == 0x65725F504F5F5744LL && *(_WORD *)(a1 + 8) == 30823 )
        {
          v9 = 144;
          goto LABEL_142;
        }
        goto LABEL_93;
      }
LABEL_53:
      if ( !v11 )
        goto LABEL_54;
LABEL_142:
      v14 = 1;
      v12 = a2 == 16;
      v13 = 0;
LABEL_143:
      v2 &= v13;
      goto LABEL_144;
    }
LABEL_227:
    if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_DWORD *)(a1 + 8) == 808544101 )
    {
      v3 = 1;
      v9 = 122;
      goto LABEL_142;
    }
    if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_DWORD *)(a1 + 8) == 825321317 )
    {
      v3 = 1;
      v9 = 123;
      goto LABEL_142;
    }
    v11 = v8;
    if ( v8 )
    {
      v3 = v8;
      goto LABEL_53;
    }
    if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_DWORD *)(a1 + 8) == 842098533 )
      goto LABEL_141;
    goto LABEL_231;
  }
  v4 = v8;
LABEL_88:
  v11 = v3;
  if ( v4 )
  {
LABEL_446:
    v11 = v4;
    goto LABEL_53;
  }
  if ( !v3 )
    goto LABEL_90;
  if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_DWORD *)(a1 + 8) == 842098533 )
  {
LABEL_141:
    v3 = 1;
    v9 = 124;
    goto LABEL_142;
  }
  v3 = v8;
  if ( v8 )
  {
    v11 = v8;
    goto LABEL_53;
  }
LABEL_231:
  if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_DWORD *)(a1 + 8) == 858875749 )
  {
    v3 = 1;
    v9 = 125;
    goto LABEL_142;
  }
  if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_DWORD *)(a1 + 8) == 875652965 )
  {
    v3 = 1;
    v9 = 126;
    goto LABEL_142;
  }
  if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_DWORD *)(a1 + 8) == 892430181 )
  {
    v9 = 127;
    v3 = 1;
    goto LABEL_142;
  }
  if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_DWORD *)(a1 + 8) == 909207397 )
  {
    v9 = 128;
    v3 = 1;
    goto LABEL_142;
  }
  if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_DWORD *)(a1 + 8) == 925984613 )
  {
    v9 = 129;
    v3 = 1;
    goto LABEL_142;
  }
  if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_DWORD *)(a1 + 8) == 942761829 )
  {
    v9 = 130;
    v3 = 1;
    goto LABEL_142;
  }
  if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_DWORD *)(a1 + 8) == 959539045 )
  {
    v9 = 131;
    v3 = 1;
    goto LABEL_142;
  }
  if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_DWORD *)(a1 + 8) == 808609637 )
  {
    v9 = 132;
    v3 = 1;
    goto LABEL_142;
  }
  if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_DWORD *)(a1 + 8) == 825386853 )
  {
    v9 = 133;
    v3 = 1;
    goto LABEL_142;
  }
  if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_DWORD *)(a1 + 8) == 842164069 )
  {
    v9 = 134;
    v3 = 1;
    goto LABEL_142;
  }
  if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_DWORD *)(a1 + 8) == 858941285 )
  {
    v9 = 135;
    v3 = 1;
    goto LABEL_142;
  }
  if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_DWORD *)(a1 + 8) == 875718501 )
  {
    v9 = 136;
    v3 = 1;
    goto LABEL_142;
  }
  if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_DWORD *)(a1 + 8) == 892495717 )
  {
    v9 = 137;
    v3 = 1;
    goto LABEL_142;
  }
  if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_DWORD *)(a1 + 8) == 909272933 )
  {
    v9 = 138;
    v3 = 1;
    goto LABEL_142;
  }
  if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_DWORD *)(a1 + 8) == 926050149 )
  {
    v9 = 139;
    v3 = 1;
    goto LABEL_142;
  }
  if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_DWORD *)(a1 + 8) == 942827365 )
  {
    v9 = 140;
    v3 = 1;
    goto LABEL_142;
  }
  if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_DWORD *)(a1 + 8) == 959604581 )
  {
    v9 = 141;
    v3 = 1;
    goto LABEL_142;
  }
  if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_DWORD *)(a1 + 8) == 808675173 )
  {
    v9 = 142;
    v3 = 1;
    goto LABEL_142;
  }
  if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_DWORD *)(a1 + 8) == 825452389 )
  {
    v9 = 143;
    v3 = 1;
    goto LABEL_142;
  }
  v3 = 1;
  if ( v5 )
  {
    v3 = v5;
    goto LABEL_92;
  }
LABEL_54:
  if ( a2 == 11 )
  {
    if ( *(_QWORD *)a1 == 0x62665F504F5F5744LL && *(_WORD *)(a1 + 8) == 25970 && *(_BYTE *)(a1 + 10) == 103 )
    {
      v9 = 145;
      goto LABEL_142;
    }
    if ( *(_QWORD *)a1 == 0x72625F504F5F5744LL && *(_WORD *)(a1 + 8) == 26469 && *(_BYTE *)(a1 + 10) == 120 )
    {
      v9 = 146;
      goto LABEL_142;
    }
    if ( *(_QWORD *)a1 == 0x69705F504F5F5744LL && *(_WORD *)(a1 + 8) == 25445 && *(_BYTE *)(a1 + 10) == 101 )
    {
      v9 = 147;
      goto LABEL_142;
    }
LABEL_58:
    if ( v6 )
    {
      if ( !(*(_QWORD *)a1 ^ 0x64785F504F5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x7A69735F66657265LL)
        && *(_BYTE *)(a1 + 16) == 101 )
      {
        v12 = 0;
        v9 = 149;
        goto LABEL_67;
      }
      v12 = 0;
      if ( a2 != 25 )
      {
LABEL_99:
        v13 = 1;
        v14 = 0;
        if ( a2 != 11 )
          goto LABEL_100;
        if ( *(_QWORD *)a1 == 0x61635F504F5F5744LL && *(_WORD *)(a1 + 8) == 27756 && *(_BYTE *)(a1 + 10) == 50 )
        {
          v9 = 152;
        }
        else
        {
          if ( *(_QWORD *)a1 != 0x61635F504F5F5744LL || *(_WORD *)(a1 + 8) != 27756 || *(_BYTE *)(a1 + 10) != 52 )
          {
            v14 = 0;
            v17 = 1;
LABEL_255:
            v16 = 0;
            v15 = 0;
            goto LABEL_68;
          }
          v9 = 153;
        }
        v14 = 1;
        v17 = 0;
        goto LABEL_255;
      }
      v13 = v6;
      v12 = 0;
      v14 = 0;
      goto LABEL_63;
    }
    v12 = 0;
    v14 = 0;
    v13 = 1;
    goto LABEL_143;
  }
LABEL_93:
  if ( a2 != 16 )
    goto LABEL_58;
  if ( *(_QWORD *)a1 ^ 0x65645F504F5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x657A69735F666572LL )
  {
    v14 = 0;
    v13 = 1;
    v12 = 1;
LABEL_144:
    if ( v2 )
    {
      if ( *(_QWORD *)a1 == 0x6F6E5F504F5F5744LL && *(_BYTE *)(a1 + 8) == 112 )
      {
        v9 = 150;
        goto LABEL_67;
      }
LABEL_98:
      if ( !v14 )
        goto LABEL_99;
LABEL_67:
      v14 = 1;
      v15 = a2 == 22;
      v16 = a2 == 20;
      v17 = 0;
      goto LABEL_68;
    }
    goto LABEL_96;
  }
  v12 = 1;
  v14 = 1;
  v13 = 0;
  v9 = 148;
LABEL_96:
  if ( a2 != 25 || !v13 )
    goto LABEL_98;
LABEL_63:
  if ( !(*(_QWORD *)a1 ^ 0x75705F504F5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x63656A626F5F6873LL)
    && *(_QWORD *)(a1 + 16) == 0x7365726464615F74LL
    && *(_BYTE *)(a1 + 24) == 115 )
  {
    v9 = 151;
    goto LABEL_67;
  }
LABEL_100:
  v15 = a2 == 22;
  v17 = v13 & (a2 == 14);
  if ( v17 )
  {
    if ( *(_QWORD *)a1 == 0x61635F504F5F5744LL && *(_DWORD *)(a1 + 8) == 1918856300 && *(_WORD *)(a1 + 12) == 26213 )
    {
      v14 = v13 & (a2 == 14);
      v9 = 154;
      v17 = 0;
    }
    v16 = 0;
    goto LABEL_103;
  }
  v16 = a2 == 20;
  v27 = v15 & v13;
  if ( (v15 & (unsigned __int8)v13) == 0 )
  {
    v17 = v13;
LABEL_68:
    v18 = v17 & v16;
    if ( ((unsigned __int8)v17 & v16) != 0 )
    {
      if ( !(*(_QWORD *)a1 ^ 0x61635F504F5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x656D6172665F6C6CLL)
        && *(_DWORD *)(a1 + 16) == 1634100063 )
      {
        v14 = v17 & v16;
        v17 = 0;
        v9 = 156;
        goto LABEL_105;
      }
      goto LABEL_104;
    }
LABEL_103:
    v18 = v16 & v17;
    v19 = v17 & (a2 == 15);
    if ( !v19 )
      goto LABEL_104;
LABEL_162:
    if ( *(_QWORD *)a1 == 0x69625F504F5F5744LL
      && *(_DWORD *)(a1 + 8) == 1768972148
      && *(_WORD *)(a1 + 12) == 25445
      && *(_BYTE *)(a1 + 14) == 101 )
    {
      v14 = v19;
      v17 = 0;
      v9 = 157;
      goto LABEL_106;
    }
    goto LABEL_105;
  }
  if ( !(*(_QWORD *)a1 ^ 0x6F665F504F5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x615F736C745F6D72LL)
    && *(_DWORD *)(a1 + 16) == 1701995620
    && *(_WORD *)(a1 + 20) == 29555 )
  {
    v19 = 0;
    v14 = v27;
    v9 = 155;
  }
  else
  {
    v17 = v15 & v13;
    v19 = a2 == 15;
  }
  v15 = v27;
  v18 = v16 & v17;
  if ( v19 )
    goto LABEL_162;
LABEL_104:
  if ( v18 )
  {
    if ( *(_QWORD *)a1 ^ 0x6D695F504F5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x765F746963696C70LL
      || *(_DWORD *)(a1 + 16) != 1702194273 )
    {
      goto LABEL_106;
    }
    v14 = v18;
    v17 = 0;
    v9 = 158;
LABEL_107:
    if ( ((unsigned __int8)v17 & (a2 == 11)) != 0 )
    {
      if ( *(_QWORD *)a1 == 0x64615F504F5F5744LL && *(_WORD *)(a1 + 8) == 29284 && *(_BYTE *)(a1 + 10) == 120 )
      {
        v14 = v17 & (a2 == 11);
        v17 = 0;
        v9 = 161;
        goto LABEL_110;
      }
      goto LABEL_109;
    }
    goto LABEL_108;
  }
LABEL_105:
  if ( ((unsigned __int8)v6 & (unsigned __int8)v17) != 0 )
  {
    if ( !(*(_QWORD *)a1 ^ 0x74735F504F5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x756C61765F6B6361LL)
      && *(_BYTE *)(a1 + 16) == 101 )
    {
      v14 = v6 & v17;
      v17 = 0;
      v9 = 159;
      goto LABEL_108;
    }
    goto LABEL_107;
  }
LABEL_106:
  v20 = v17 & v15;
  if ( !v20 )
    goto LABEL_107;
  if ( !(*(_QWORD *)a1 ^ 0x6D695F504F5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x705F746963696C70LL)
    && *(_DWORD *)(a1 + 16) == 1953393007
    && *(_WORD *)(a1 + 20) == 29285 )
  {
    v14 = v20;
    v9 = 160;
    v17 = 0;
    goto LABEL_109;
  }
LABEL_108:
  v21 = v17 & v3;
  if ( v21 )
  {
    if ( *(_QWORD *)a1 == 0x6F635F504F5F5744LL && *(_DWORD *)(a1 + 8) == 2020897646 )
    {
      v14 = v21;
      v17 = 0;
      v9 = 162;
      goto LABEL_112;
    }
    goto LABEL_110;
  }
LABEL_109:
  if ( ((unsigned __int8)v6 & (unsigned __int8)v17) != 0 )
  {
    if ( !(*(_QWORD *)a1 ^ 0x6E655F504F5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x756C61765F797274LL)
      && *(_BYTE *)(a1 + 16) == 101 )
    {
      v14 = v6 & v17;
      v9 = 163;
      v17 = 0;
LABEL_113:
      v22 = v17 & v6;
      if ( ((unsigned __int8)v17 & (unsigned __int8)v6) == 0 )
      {
LABEL_114:
        v24 = v17 & (a2 == 13);
        goto LABEL_115;
      }
LABEL_314:
      if ( *(_QWORD *)a1 ^ 0x64785F504F5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x7079745F66657265LL
        || *(_BYTE *)(a1 + 16) != 101 )
      {
        goto LABEL_316;
      }
      v17 = 0;
      v9 = 167;
      v14 = 1;
LABEL_117:
      if ( ((unsigned __int8)v17 & (a2 == 26)) != 0 )
      {
        if ( !(*(_QWORD *)a1 ^ 0x4E475F504F5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x745F687375705F55LL)
          && *(_QWORD *)(a1 + 16) == 0x65726464615F736CLL
          && *(_WORD *)(a1 + 24) == 29555 )
        {
          return 224;
        }
        goto LABEL_120;
      }
      goto LABEL_118;
    }
    v22 = v17;
LABEL_312:
    if ( !(*(_QWORD *)a1 ^ 0x65725F504F5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x7079745F6C617667LL)
      && *(_BYTE *)(a1 + 16) == 101 )
    {
      v17 = 0;
      v9 = 165;
      v14 = 1;
      goto LABEL_114;
    }
    goto LABEL_314;
  }
LABEL_110:
  if ( (v12 & (unsigned __int8)v17) != 0 )
  {
    if ( !(*(_QWORD *)a1 ^ 0x6F635F504F5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x657079745F74736ELL) )
    {
      v14 = v12 & v17;
      v17 = 0;
      v9 = 164;
      goto LABEL_114;
    }
    v23 = v17;
    goto LABEL_342;
  }
  v22 = v6 & v17;
  if ( ((unsigned __int8)v6 & (unsigned __int8)v17) != 0 )
  {
    v6 &= v17;
    goto LABEL_312;
  }
LABEL_112:
  v23 = v17 & v12;
  if ( !v23 )
    goto LABEL_113;
LABEL_342:
  if ( !(*(_QWORD *)a1 ^ 0x65645F504F5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x657079745F666572LL) )
  {
    v17 = 0;
    v9 = 166;
    v14 = 1;
    goto LABEL_117;
  }
  v17 = v23;
  v24 = v23 & (a2 == 13);
LABEL_115:
  if ( v24 )
  {
    if ( *(_QWORD *)a1 == 0x6F635F504F5F5744LL && *(_DWORD *)(a1 + 8) == 1919252078 && *(_BYTE *)(a1 + 12) == 116 )
    {
      v14 = v24;
      v9 = 168;
      v17 = 0;
LABEL_118:
      v25 = v17 & v16;
      goto LABEL_119;
    }
    goto LABEL_117;
  }
  v22 = v17 & v6;
  if ( ((unsigned __int8)v17 & (unsigned __int8)v6) == 0 )
    goto LABEL_117;
LABEL_316:
  if ( !(*(_QWORD *)a1 ^ 0x65725F504F5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6572707265746E69LL)
    && *(_BYTE *)(a1 + 16) == 116 )
  {
    return 169;
  }
  v25 = v22 & v16;
LABEL_119:
  if ( v25 )
  {
    if ( !(*(_QWORD *)a1 ^ 0x4E475F504F5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x695F726464615F55LL) )
    {
      v9 = 251;
      if ( *(_DWORD *)(a1 + 16) == 2019910766 )
        return v9;
    }
    goto LABEL_121;
  }
LABEL_120:
  if ( v14 )
    return v9;
LABEL_121:
  if ( a2 != 21 )
  {
    if ( a2 == 19
      && !(*(_QWORD *)a1 ^ 0x4C4C5F504F5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6D676172665F4D56LL)
      && *(_WORD *)(a1 + 16) == 28261 )
    {
      v9 = 4096;
      if ( *(_BYTE *)(a1 + 18) == 116 )
        return v9;
    }
    return 0;
  }
  if ( *(_QWORD *)a1 ^ 0x4E475F504F5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x5F74736E6F635F55LL )
    return 0;
  if ( *(_DWORD *)(a1 + 16) != 1701080681 )
    return 0;
  v9 = 252;
  if ( *(_BYTE *)(a1 + 20) != 120 )
    return 0;
  return v9;
}
