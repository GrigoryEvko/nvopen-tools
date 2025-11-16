// Function: sub_1719E80
// Address: 0x1719e80
//
bool __fastcall sub_1719E80(__int64 a1, _QWORD *a2, __int64 a3, _BYTE *a4)
{
  char v6; // al
  __int16 v7; // ax
  __int64 v8; // rax
  __int64 v9; // rdx
  _BYTE *v10; // rdi
  bool result; // al
  __int64 v12; // rax
  _BYTE *v13; // rdi
  unsigned __int8 v14; // al
  __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rax
  unsigned __int64 v18; // rsi
  __int64 v19; // rax
  _BYTE *v20; // rdi
  unsigned __int8 v21; // al
  __int64 v22; // rdx
  __int64 v23; // rax
  unsigned __int8 v24; // al
  _BYTE *v25; // r13
  unsigned int v26; // eax
  unsigned __int64 v27; // r14
  __int64 v28; // rdx
  char v29; // al
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  bool v34; // cc
  unsigned int v35; // ebx
  const void *v36; // r13
  __int64 v37; // rax
  __int64 v38; // rax
  _BYTE *v39; // [rsp+0h] [rbp-60h]
  bool v40; // [rsp+8h] [rbp-58h]
  unsigned __int64 v41; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v42; // [rsp+18h] [rbp-48h]
  const void *v43; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v44; // [rsp+28h] [rbp-38h]

  *a4 = 0;
  v6 = *(_BYTE *)(a1 + 16);
  if ( v6 == 45 )
  {
    v19 = *(_QWORD *)(a1 - 48);
    if ( !v19 )
      return 0;
    *a2 = v19;
    v20 = *(_BYTE **)(a1 - 24);
    v21 = v20[16];
    if ( v21 != 13 )
    {
      v22 = *(_QWORD *)v20;
      if ( *(_BYTE *)(*(_QWORD *)v20 + 8LL) != 16 || v21 > 0x10u )
        goto LABEL_27;
      goto LABEL_53;
    }
LABEL_21:
    v15 = (__int64)(v20 + 24);
LABEL_22:
    *a4 = 1;
    if ( *(_DWORD *)(a3 + 8) <= 0x40u && *(_DWORD *)(v15 + 8) <= 0x40u )
      goto LABEL_17;
    goto LABEL_24;
  }
  if ( v6 != 5 )
  {
    if ( v6 != 44 )
      goto LABEL_29;
    goto LABEL_12;
  }
  v7 = *(_WORD *)(a1 + 18);
  if ( v7 != 21 )
    goto LABEL_4;
  v32 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  if ( !v32 )
    return 0;
  *a2 = v32;
  v22 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  v20 = *(_BYTE **)(a1 + 24 * (1 - v22));
  if ( v20[16] == 13 )
    goto LABEL_21;
  if ( *(_BYTE *)(*(_QWORD *)v20 + 8LL) != 16 )
    goto LABEL_27;
LABEL_53:
  v39 = a4;
  v33 = sub_15A1020(v20, (__int64)a2, v22, (__int64)a4);
  if ( v33 && *(_BYTE *)(v33 + 16) == 13 )
  {
    a4 = v39;
    v15 = v33 + 24;
    goto LABEL_22;
  }
LABEL_27:
  v6 = *(_BYTE *)(a1 + 16);
  if ( v6 != 44 )
  {
    if ( v6 == 5 )
    {
      v7 = *(_WORD *)(a1 + 18);
LABEL_4:
      if ( v7 != 20 )
        goto LABEL_5;
      v30 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      if ( !v30 )
        return 0;
      *a2 = v30;
      v28 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
      v13 = *(_BYTE **)(a1 + 24 * (1 - v28));
      if ( v13[16] != 13 )
      {
        if ( *(_BYTE *)(*(_QWORD *)v13 + 8LL) == 16 )
          goto LABEL_47;
        goto LABEL_41;
      }
LABEL_14:
      v15 = (__int64)(v13 + 24);
LABEL_15:
      if ( *(_DWORD *)(a3 + 8) <= 0x40u && *(_DWORD *)(v15 + 8) <= 0x40u )
      {
LABEL_17:
        v16 = *(_QWORD *)v15;
        *(_QWORD *)a3 = *(_QWORD *)v15;
        v17 = *(unsigned int *)(v15 + 8);
        *(_DWORD *)(a3 + 8) = v17;
        v18 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v17;
        if ( (unsigned int)v17 > 0x40 )
        {
          v37 = (unsigned int)((unsigned __int64)(v17 + 63) >> 6) - 1;
          *(_QWORD *)(v16 + 8 * v37) &= v18;
          return 1;
        }
        else
        {
          *(_QWORD *)a3 = v18 & v16;
          return 1;
        }
      }
LABEL_24:
      sub_16A51C0(a3, v15);
      return 1;
    }
LABEL_29:
    if ( v6 != 50 )
      return 0;
    goto LABEL_30;
  }
LABEL_12:
  v12 = *(_QWORD *)(a1 - 48);
  if ( !v12 )
    return 0;
  *a2 = v12;
  v13 = *(_BYTE **)(a1 - 24);
  v14 = v13[16];
  if ( v14 == 13 )
    goto LABEL_14;
  v28 = *(_QWORD *)v13;
  if ( *(_BYTE *)(*(_QWORD *)v13 + 8LL) == 16 && v14 <= 0x10u )
  {
LABEL_47:
    v31 = sub_15A1020(v13, (__int64)a2, v28, (__int64)a4);
    if ( v31 && *(_BYTE *)(v31 + 16) == 13 )
    {
      v15 = v31 + 24;
      goto LABEL_15;
    }
  }
LABEL_41:
  v29 = *(_BYTE *)(a1 + 16);
  if ( v29 != 50 )
  {
    if ( v29 != 5 )
      return 0;
    v7 = *(_WORD *)(a1 + 18);
LABEL_5:
    if ( v7 != 26 )
      return 0;
    v8 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    if ( !v8 )
      return 0;
    *a2 = v8;
    v9 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
    v10 = *(_BYTE **)(a1 + 24 * (1 - v9));
    if ( v10[16] != 13 )
    {
      if ( *(_BYTE *)(*(_QWORD *)v10 + 8LL) != 16 )
        return 0;
      goto LABEL_73;
    }
LABEL_32:
    v25 = v10 + 24;
    goto LABEL_33;
  }
LABEL_30:
  v23 = *(_QWORD *)(a1 - 48);
  if ( !v23 )
    return 0;
  *a2 = v23;
  v10 = *(_BYTE **)(a1 - 24);
  v24 = v10[16];
  if ( v24 == 13 )
    goto LABEL_32;
  v9 = *(_QWORD *)v10;
  if ( *(_BYTE *)(*(_QWORD *)v10 + 8LL) != 16 || v24 > 0x10u )
    return 0;
LABEL_73:
  v38 = sub_15A1020(v10, (__int64)a2, v9, (__int64)a4);
  if ( !v38 || *(_BYTE *)(v38 + 16) != 13 )
    return 0;
  v25 = (_BYTE *)(v38 + 24);
LABEL_33:
  v42 = *((_DWORD *)v25 + 2);
  if ( v42 > 0x40 )
    sub_16A4FD0((__int64)&v41, (const void **)v25);
  else
    v41 = *(_QWORD *)v25;
  sub_16A7490((__int64)&v41, 1);
  v26 = v42;
  v27 = v41;
  v42 = 0;
  v44 = v26;
  v43 = (const void *)v41;
  if ( v26 <= 0x40 )
  {
    if ( !v41 || (v41 & (v41 - 1)) != 0 )
      return 0;
    goto LABEL_61;
  }
  result = (unsigned int)sub_16A5940((__int64)&v43) == 1;
  if ( v27 )
  {
    v40 = result;
    j_j___libc_free_0_0(v27);
    result = v40;
    if ( v42 > 0x40 )
    {
      if ( v41 )
      {
        j_j___libc_free_0_0(v41);
        result = v40;
      }
    }
  }
  if ( result )
  {
LABEL_61:
    v44 = *((_DWORD *)v25 + 2);
    if ( v44 > 0x40 )
      sub_16A4FD0((__int64)&v43, (const void **)v25);
    else
      v43 = *(const void **)v25;
    sub_16A7490((__int64)&v43, 1);
    v34 = *(_DWORD *)(a3 + 8) <= 0x40u;
    v35 = v44;
    v44 = 0;
    v36 = v43;
    if ( v34 || !*(_QWORD *)a3 )
    {
      *(_QWORD *)a3 = v43;
      *(_DWORD *)(a3 + 8) = v35;
    }
    else
    {
      j_j___libc_free_0_0(*(_QWORD *)a3);
      v34 = v44 <= 0x40;
      *(_QWORD *)a3 = v36;
      *(_DWORD *)(a3 + 8) = v35;
      if ( !v34 )
      {
        if ( v43 )
          j_j___libc_free_0_0(v43);
      }
    }
    return 1;
  }
  return result;
}
