// Function: sub_14B4200
// Address: 0x14b4200
//
__int64 __fastcall sub_14B4200(__int64 *a1, __int64 a2)
{
  char v4; // al
  unsigned int v5; // r8d
  __int64 v7; // r13
  __int64 v8; // r14
  unsigned __int8 v9; // al
  __int64 v10; // rax
  int v11; // edx
  int v12; // r15d
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rsi
  int v16; // ecx
  int v17; // ecx
  _QWORD *v18; // rsi
  int v19; // ecx
  _QWORD *v20; // rax
  int v21; // edx
  _QWORD *v22; // rax
  _QWORD *v23; // rdi
  int v24; // ecx
  _QWORD *v25; // rcx
  _QWORD *v26; // rcx

  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 == 50 )
  {
    v7 = *(_QWORD *)(a2 - 48);
    v8 = *a1;
    if ( v7 == *a1 )
      goto LABEL_15;
    v9 = *(_BYTE *)(v7 + 16);
    if ( v9 > 0x17u )
    {
      v12 = v9 - 24;
      if ( v9 == 69 && *(_QWORD *)sub_13CF970(*(_QWORD *)(a2 - 48)) == a1[1] )
        goto LABEL_15;
    }
    else
    {
      if ( v9 != 5 )
        goto LABEL_8;
      v12 = *(unsigned __int16 *)(v7 + 18);
      if ( (_WORD)v12 == 45 )
      {
        if ( a1[1] != *(_QWORD *)sub_13CF970(*(_QWORD *)(a2 - 48)) )
          goto LABEL_8;
        goto LABEL_15;
      }
    }
    if ( v12 != 47 || *(_QWORD *)sub_13CF970(v7) != a1[2] )
    {
LABEL_8:
      v10 = *(_QWORD *)(a2 - 24);
      if ( v10 == v8 )
        goto LABEL_42;
      goto LABEL_9;
    }
LABEL_15:
    v10 = *(_QWORD *)(a2 - 24);
    if ( v10 )
    {
LABEL_16:
      v5 = 1;
      *(_QWORD *)a1[3] = v10;
      return v5;
    }
    if ( !v8 )
      goto LABEL_41;
LABEL_9:
    v11 = *(unsigned __int8 *)(v10 + 16);
    if ( (unsigned __int8)v11 > 0x17u )
    {
      if ( (_BYTE)v11 != 69 )
        goto LABEL_36;
    }
    else
    {
      if ( (_BYTE)v11 != 5 )
        return 0;
      if ( *(_WORD *)(v10 + 18) != 45 )
      {
LABEL_66:
        v21 = *(unsigned __int16 *)(v10 + 18);
        goto LABEL_37;
      }
    }
    if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
      v26 = *(_QWORD **)(v10 - 8);
    else
      v26 = (_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
    if ( *v26 == a1[1] )
    {
LABEL_41:
      if ( !v7 )
        return 0;
LABEL_42:
      v5 = 1;
      *(_QWORD *)a1[3] = v7;
      return v5;
    }
    if ( (unsigned __int8)v11 <= 0x17u )
      goto LABEL_66;
LABEL_36:
    v21 = v11 - 24;
LABEL_37:
    v5 = 0;
    if ( v21 != 47 )
      return v5;
    v22 = (*(_BYTE *)(v10 + 23) & 0x40) != 0
        ? *(_QWORD **)(v10 - 8)
        : (_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
    v5 = 0;
    if ( *v22 != a1[2] )
      return v5;
    goto LABEL_41;
  }
  v5 = 0;
  if ( v4 != 5 || *(_WORD *)(a2 + 18) != 26 )
    return v5;
  v13 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v14 = *(_QWORD *)(a2 - 24 * v13);
  v15 = *a1;
  if ( v14 == *a1 )
    goto LABEL_53;
  v16 = *(unsigned __int8 *)(v14 + 16);
  if ( (unsigned __int8)v16 <= 0x17u )
  {
    if ( (_BYTE)v16 != 5 )
      goto LABEL_20;
    if ( *(_WORD *)(v14 + 18) != 45 )
      goto LABEL_70;
    goto LABEL_44;
  }
  if ( (_BYTE)v16 == 69 )
  {
LABEL_44:
    if ( (*(_BYTE *)(v14 + 23) & 0x40) != 0 )
      v23 = *(_QWORD **)(v14 - 8);
    else
      v23 = (_QWORD *)(v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF));
    if ( *v23 == a1[1] )
      goto LABEL_53;
    if ( (unsigned __int8)v16 > 0x17u )
      goto LABEL_48;
LABEL_70:
    v24 = *(unsigned __int16 *)(v14 + 18);
    goto LABEL_49;
  }
LABEL_48:
  v24 = v16 - 24;
LABEL_49:
  if ( v24 != 47
    || ((*(_BYTE *)(v14 + 23) & 0x40) == 0
      ? (v25 = (_QWORD *)(v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF)))
      : (v25 = *(_QWORD **)(v14 - 8)),
        *v25 != a1[2]) )
  {
LABEL_20:
    v10 = *(_QWORD *)(a2 + 24 * (1 - v13));
    if ( v15 == v10 )
    {
LABEL_34:
      v5 = 1;
      *(_QWORD *)a1[3] = v14;
      return v5;
    }
    goto LABEL_21;
  }
LABEL_53:
  v10 = *(_QWORD *)(a2 + 24 * (1 - v13));
  if ( v10 )
    goto LABEL_16;
  if ( !v15 )
    goto LABEL_33;
LABEL_21:
  v17 = *(unsigned __int8 *)(v10 + 16);
  if ( (unsigned __int8)v17 <= 0x17u )
  {
    if ( (_BYTE)v17 != 5 )
      return 0;
    if ( *(_WORD *)(v10 + 18) != 45 )
      goto LABEL_78;
    goto LABEL_24;
  }
  if ( (_BYTE)v17 == 69 )
  {
LABEL_24:
    if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
      v18 = *(_QWORD **)(v10 - 8);
    else
      v18 = (_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
    if ( *v18 == a1[1] )
    {
LABEL_33:
      if ( v14 )
        goto LABEL_34;
      return 0;
    }
    if ( (unsigned __int8)v17 > 0x17u )
      goto LABEL_28;
LABEL_78:
    v19 = *(unsigned __int16 *)(v10 + 18);
    goto LABEL_29;
  }
LABEL_28:
  v19 = v17 - 24;
LABEL_29:
  v5 = 0;
  if ( v19 == 47 )
  {
    v20 = (*(_BYTE *)(v10 + 23) & 0x40) != 0
        ? *(_QWORD **)(v10 - 8)
        : (_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
    v5 = 0;
    if ( *v20 == a1[2] )
      goto LABEL_33;
  }
  return v5;
}
