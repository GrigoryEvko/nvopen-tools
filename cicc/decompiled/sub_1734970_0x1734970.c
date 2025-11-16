// Function: sub_1734970
// Address: 0x1734970
//
_QWORD *__fastcall sub_1734970(__int64 *a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 v5; // rcx
  unsigned __int8 v6; // al
  int v7; // r8d
  int v8; // edx
  bool v9; // r9
  bool v10; // r10
  int v11; // r8d
  bool v12; // r11
  _QWORD *v13; // r15
  __int64 *v14; // rbx
  __int64 *v15; // r12
  __int64 v16; // rsi
  __int64 *v18; // rcx
  __int64 *v19; // rsi
  __int64 *v20; // [rsp+0h] [rbp-60h] BYREF
  __int64 v21; // [rsp+8h] [rbp-58h]
  _BYTE v22[80]; // [rsp+10h] [rbp-50h] BYREF

  v4 = *(_QWORD *)(a2 - 48);
  v5 = *(_QWORD *)(a2 - 24);
  v6 = *(_BYTE *)(v4 + 16);
  if ( v6 == 61 )
  {
    if ( (*(_BYTE *)(v4 + 23) & 0x40) != 0 )
      v19 = *(__int64 **)(v4 - 8);
    else
      v19 = (__int64 *)(v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF));
    v4 = *v19;
    v6 = *(_BYTE *)(v4 + 16);
  }
  if ( *(_BYTE *)(v5 + 16) == 61 )
  {
    if ( (*(_BYTE *)(v5 + 23) & 0x40) != 0 )
      v18 = *(__int64 **)(v5 - 8);
    else
      v18 = (__int64 *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF));
    v5 = *v18;
  }
  if ( v6 == 51 )
  {
    v10 = 0;
    v9 = 1;
    goto LABEL_16;
  }
  if ( v6 == 5 )
  {
    v7 = *(unsigned __int16 *)(v4 + 18);
    if ( (_WORD)v7 == 27 )
      goto LABEL_29;
    v8 = *(unsigned __int8 *)(v5 + 16);
    if ( (_BYTE)v8 == 51 )
    {
      if ( (unsigned int)(unsigned __int16)v7 - 23 > 1 )
      {
        v10 = 0;
        v9 = 1;
        goto LABEL_51;
      }
      v11 = 51;
      v9 = 1;
LABEL_12:
      v10 = (unsigned int)(v11 - 47) <= 1;
      goto LABEL_13;
    }
    if ( (_BYTE)v8 != 5 )
    {
      v9 = 0;
      goto LABEL_84;
    }
    goto LABEL_8;
  }
  v8 = *(unsigned __int8 *)(v5 + 16);
  v9 = 1;
  if ( (_BYTE)v8 == 51 )
  {
LABEL_9:
    if ( v6 > 0x17u )
      goto LABEL_10;
    goto LABEL_39;
  }
  if ( (_BYTE)v8 == 5 )
  {
LABEL_8:
    v9 = *(_WORD *)(v5 + 18) == 27;
    goto LABEL_9;
  }
  v9 = 0;
  if ( v6 <= 0x17u )
  {
LABEL_39:
    if ( v6 != 5 )
    {
LABEL_40:
      if ( !v9 )
        return 0;
      goto LABEL_29;
    }
    v7 = *(unsigned __int16 *)(v4 + 18);
LABEL_84:
    if ( (unsigned int)(unsigned __int16)v7 - 23 > 1 )
    {
      v10 = 0;
      goto LABEL_51;
    }
LABEL_11:
    v11 = (unsigned __int8)v8;
    if ( (unsigned __int8)v8 <= 0x17u )
    {
      v10 = 0;
      if ( (_BYTE)v8 == 5 )
        v10 = (unsigned int)*(unsigned __int16 *)(v5 + 18) - 23 <= 1;
      goto LABEL_13;
    }
    goto LABEL_12;
  }
LABEL_10:
  v10 = 0;
  if ( (unsigned int)v6 - 47 <= 1 )
    goto LABEL_11;
LABEL_13:
  if ( v6 == 50 )
  {
    if ( (_BYTE)v8 != 50 )
    {
      if ( (_BYTE)v8 != 5 )
      {
LABEL_16:
        v12 = 0;
LABEL_17:
        if ( (unsigned int)v6 - 47 > 1 )
        {
          if ( v6 != 50 )
            goto LABEL_26;
          v8 = *(unsigned __int8 *)(v5 + 16);
LABEL_46:
          if ( (unsigned __int8)v8 > 0x17u )
            goto LABEL_47;
          if ( (_BYTE)v8 != 5 )
            goto LABEL_26;
LABEL_68:
          if ( (unsigned int)*(unsigned __int16 *)(v5 + 18) - 23 > 1 )
            goto LABEL_26;
          goto LABEL_29;
        }
        v8 = *(unsigned __int8 *)(v5 + 16);
        goto LABEL_19;
      }
      goto LABEL_54;
    }
    goto LABEL_66;
  }
  if ( v6 != 5 )
    goto LABEL_16;
  v7 = *(unsigned __int16 *)(v4 + 18);
LABEL_51:
  if ( (_WORD)v7 == 26 )
  {
    if ( (_BYTE)v8 != 50 )
    {
      if ( (_BYTE)v8 != 5 )
      {
        v12 = 0;
        goto LABEL_79;
      }
LABEL_54:
      v12 = *(_WORD *)(v5 + 18) == 26;
      if ( v6 > 0x17u )
        goto LABEL_17;
      LOWORD(v7) = *(_WORD *)(v4 + 18);
LABEL_79:
      if ( (unsigned int)(unsigned __int16)v7 - 23 > 1 )
      {
LABEL_25:
        if ( (_WORD)v7 != 26 )
          goto LABEL_26;
        goto LABEL_46;
      }
      goto LABEL_19;
    }
LABEL_66:
    v12 = 1;
    goto LABEL_47;
  }
  if ( (unsigned int)(v7 - 23) > 1 )
  {
    if ( v10 )
      goto LABEL_29;
    goto LABEL_40;
  }
  v12 = 0;
LABEL_19:
  if ( (_BYTE)v8 == 50 )
    goto LABEL_29;
  if ( (_BYTE)v8 == 5 )
  {
    if ( *(_WORD *)(v5 + 18) == 26 )
      goto LABEL_29;
    if ( v6 == 50 )
      goto LABEL_68;
    goto LABEL_23;
  }
  if ( v6 != 50 )
  {
LABEL_23:
    if ( v6 != 5 )
    {
LABEL_26:
      if ( v10 || v9 || v12 )
        goto LABEL_29;
      return 0;
    }
    LOWORD(v7) = *(_WORD *)(v4 + 18);
    goto LABEL_25;
  }
  if ( (unsigned __int8)v8 <= 0x17u )
    goto LABEL_26;
LABEL_47:
  if ( (unsigned int)(v8 - 47) > 1 )
    goto LABEL_26;
LABEL_29:
  v20 = (__int64 *)v22;
  v21 = 0x400000000LL;
  if ( (unsigned __int8)sub_1AECB30(a2, 1, 0, &v20) )
  {
    v13 = (_QWORD *)v20[(unsigned int)v21 - 1];
    LODWORD(v21) = v21 - 1;
    sub_15F2070(v13);
    v14 = v20;
    v15 = &v20[(unsigned int)v21];
    if ( v20 != v15 )
    {
      do
      {
        v16 = *v14++;
        sub_170B990(*a1, v16);
      }
      while ( v15 != v14 );
      v15 = v20;
    }
  }
  else
  {
    v15 = v20;
    v13 = 0;
  }
  if ( v15 != (__int64 *)v22 )
    _libc_free((unsigned __int64)v15);
  return v13;
}
