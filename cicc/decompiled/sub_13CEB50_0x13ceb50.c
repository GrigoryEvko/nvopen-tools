// Function: sub_13CEB50
// Address: 0x13ceb50
//
__int64 __fastcall sub_13CEB50(__int64 *a1, __int64 a2)
{
  __int64 v4; // rax
  char v5; // dl
  __int64 result; // rax
  __int64 v7; // r14
  _BYTE *v8; // rdi
  unsigned __int8 v9; // al
  _BYTE *v10; // r13
  _BYTE *v11; // rdi
  unsigned __int8 v12; // al
  _BYTE *v13; // rsi
  __int16 v14; // r15
  __int64 v15; // r14
  __int16 v16; // r12
  char v17; // bl
  unsigned int v18; // r12d
  unsigned int v19; // r15d
  _QWORD *v20; // r14
  __int64 v21; // rax
  bool v22; // al
  __int64 v23; // rax
  __int64 v24; // rax
  int v25; // eax
  _QWORD *v26; // rax
  __int64 v27; // [rsp-78h] [rbp-78h]
  __int64 v28; // [rsp-70h] [rbp-70h]
  char v29; // [rsp-65h] [rbp-65h]
  int v30; // [rsp-64h] [rbp-64h]
  int v31; // [rsp-60h] [rbp-60h]
  __int64 v32; // [rsp-60h] [rbp-60h]
  _QWORD *v33; // [rsp-58h] [rbp-58h] BYREF
  unsigned int v34; // [rsp-50h] [rbp-50h]
  _QWORD *v35; // [rsp-48h] [rbp-48h] BYREF
  unsigned int v36; // [rsp-40h] [rbp-40h]

  if ( !a1 )
    return 0;
  v4 = *(a1 - 6);
  v5 = *(_BYTE *)(v4 + 16);
  if ( v5 != 35 )
  {
    if ( v5 != 5 )
      return 0;
    if ( *(_WORD *)(v4 + 18) != 11 )
      return 0;
    v7 = *(_QWORD *)(v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF));
    if ( !v7 )
      return 0;
    v8 = *(_BYTE **)(v4 + 24 * (1LL - (*(_DWORD *)(v4 + 20) & 0xFFFFFFF)));
    if ( v8[16] != 13 )
    {
      if ( *(_BYTE *)(*(_QWORD *)v8 + 8LL) != 16 )
        return 0;
      goto LABEL_32;
    }
LABEL_9:
    v10 = v8 + 24;
    goto LABEL_10;
  }
  v7 = *(_QWORD *)(v4 - 48);
  if ( !v7 )
    return 0;
  v8 = *(_BYTE **)(v4 - 24);
  v9 = v8[16];
  if ( v9 == 13 )
    goto LABEL_9;
  if ( *(_BYTE *)(*(_QWORD *)v8 + 8LL) != 16 || v9 > 0x10u )
    return 0;
LABEL_32:
  v23 = sub_15A1020(v8);
  if ( !v23 || *(_BYTE *)(v23 + 16) != 13 )
    return 0;
  v10 = (_BYTE *)(v23 + 24);
LABEL_10:
  v11 = (_BYTE *)*(a1 - 3);
  v12 = v11[16];
  v13 = v11 + 24;
  if ( v12 != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v11 + 8LL) != 16 )
      return 0;
    if ( v12 > 0x10u )
      return 0;
    v24 = sub_15A1020(v11);
    if ( !v24 || *(_BYTE *)(v24 + 16) != 13 )
      return 0;
    v13 = (_BYTE *)(v24 + 24);
  }
  v14 = *((_WORD *)a1 + 9);
  result = 0;
  if ( !a2 )
    return result;
  if ( *(_QWORD *)(a2 - 48) != v7 )
    return result;
  v15 = *(a1 - 6);
  if ( *(_QWORD *)(a2 - 24) != *(_QWORD *)(v15 - 24) )
    return result;
  v16 = *(_WORD *)(a2 + 18);
  v28 = *a1;
  v29 = sub_15F2380(v15);
  v17 = sub_15F2370(v15);
  v36 = *((_DWORD *)v13 + 2);
  if ( v36 > 0x40 )
    sub_16A4FD0(&v35, v13);
  else
    v35 = *(_QWORD **)v13;
  v31 = v14 & 0x7FFF;
  v30 = v16 & 0x7FFF;
  sub_16A7590(&v35, v10);
  v18 = *((_DWORD *)v10 + 2);
  v19 = v36;
  v20 = v35;
  v34 = v36;
  v33 = v35;
  v21 = 1LL << ((unsigned __int8)v18 - 1);
  if ( v18 <= 0x40 )
  {
    if ( (*(_QWORD *)v10 & v21) != 0 || !*(_QWORD *)v10 )
      goto LABEL_18;
    goto LABEL_49;
  }
  v27 = *(_QWORD *)(*(_QWORD *)v10 + 8LL * ((v18 - 1) >> 6)) & v21;
  v25 = sub_16A57B0(v10);
  if ( v27 )
  {
LABEL_54:
    v22 = v18 == (unsigned int)sub_16A57B0(v10);
    goto LABEL_19;
  }
  if ( v25 != v18 )
  {
LABEL_49:
    if ( v19 > 0x40 )
    {
      if ( v19 - (unsigned int)sub_16A57B0(&v33) > 0x40 )
        goto LABEL_53;
      v26 = (_QWORD *)*v20;
      if ( *v20 != 2 )
        goto LABEL_52;
    }
    else if ( v20 != (_QWORD *)2 )
    {
LABEL_51:
      v26 = v20;
LABEL_52:
      if ( v26 != (_QWORD *)1 )
        goto LABEL_53;
      if ( v31 != 37 )
      {
        if ( v30 == 38 && v31 == 41 && v29 )
          goto LABEL_45;
        goto LABEL_53;
      }
      if ( v30 != 38 )
        goto LABEL_53;
LABEL_45:
      result = sub_15A0640(v28);
LABEL_22:
      if ( v19 <= 0x40 )
        return result;
      goto LABEL_23;
    }
    if ( v31 == 36 )
    {
      if ( v30 == 38 )
        goto LABEL_45;
    }
    else if ( v30 == 38 && v31 == 40 && v29 )
    {
      goto LABEL_45;
    }
    if ( v19 > 0x40 )
    {
      if ( v19 - (unsigned int)sub_16A57B0(&v33) > 0x40 )
        goto LABEL_53;
      v26 = (_QWORD *)*v20;
      goto LABEL_52;
    }
    goto LABEL_51;
  }
LABEL_53:
  if ( v18 > 0x40 )
    goto LABEL_54;
LABEL_18:
  v22 = *(_QWORD *)v10 == 0;
LABEL_19:
  if ( v22 || !v17 )
  {
    result = 0;
    goto LABEL_22;
  }
  if ( v19 <= 0x40 )
  {
    if ( v20 == (_QWORD *)2 )
    {
      if ( v30 == 34 && v31 == 36 )
        goto LABEL_45;
    }
    else if ( v20 == (_QWORD *)1 && v31 == 37 && v30 == 34 )
    {
      goto LABEL_45;
    }
    return 0;
  }
  if ( v19 - (unsigned int)sub_16A57B0(&v33) <= 0x40 )
  {
    if ( *v20 == 2 )
    {
      if ( v31 == 36 && v30 == 34 )
        goto LABEL_45;
LABEL_73:
      result = 0;
      goto LABEL_24;
    }
    if ( *v20 != 1 )
      goto LABEL_73;
    if ( v31 == 37 && v30 == 34 )
      goto LABEL_45;
  }
  result = 0;
LABEL_23:
  if ( v20 )
  {
LABEL_24:
    v32 = result;
    j_j___libc_free_0_0(v20);
    return v32;
  }
  return result;
}
