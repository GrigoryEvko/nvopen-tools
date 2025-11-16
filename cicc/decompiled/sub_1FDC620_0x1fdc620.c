// Function: sub_1FDC620
// Address: 0x1fdc620
//
__int64 __fastcall sub_1FDC620(__int64 **a1, __int64 a2)
{
  __int64 *v2; // rax
  __int64 *v4; // rax
  unsigned int v5; // eax
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 *v8; // rbx
  __int64 v9; // r12
  unsigned __int64 v10; // r14
  __int64 v11; // rax
  unsigned __int64 v12; // r12
  __int64 v13; // r13
  __int64 v14; // r15
  __int64 v15; // r15
  char v16; // al
  __int64 v17; // r11
  __int64 *v18; // rsi
  int v19; // eax
  __int64 v20; // r11
  __int64 v21; // r13
  unsigned __int64 v22; // r14
  unsigned __int64 v23; // r14
  unsigned __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r8
  unsigned int v27; // eax
  __int64 (*v28)(); // rax
  bool v29; // zf
  char v30; // al
  int v31; // [rsp+8h] [rbp-78h]
  __int64 v32; // [rsp+8h] [rbp-78h]
  __int64 v33; // [rsp+10h] [rbp-70h]
  __int64 v34; // [rsp+10h] [rbp-70h]
  __int64 v35; // [rsp+10h] [rbp-70h]
  __int64 v36; // [rsp+18h] [rbp-68h]
  unsigned __int64 v37; // [rsp+18h] [rbp-68h]
  __int64 v38; // [rsp+18h] [rbp-68h]
  __int64 v39; // [rsp+18h] [rbp-68h]
  __int64 v41; // [rsp+28h] [rbp-58h]
  unsigned int v42; // [rsp+38h] [rbp-48h]
  unsigned __int8 v43; // [rsp+3Dh] [rbp-43h]
  unsigned __int8 v44; // [rsp+3Eh] [rbp-42h]
  char v45; // [rsp+3Fh] [rbp-41h]
  __int64 *v46; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v47; // [rsp+48h] [rbp-38h]

  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v2 = *(__int64 **)(a2 - 8);
  else
    v2 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v42 = sub_1FD8F60(a1, *v2);
  if ( !v42 )
    return 0;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v4 = *(__int64 **)(a2 - 8);
  else
    v4 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v43 = sub_1FD4DC0((__int64)a1, *v4);
  v5 = 8 * sub_15A9520((__int64)a1[12], 0);
  if ( v5 == 32 )
  {
    v44 = 5;
  }
  else if ( v5 > 0x20 )
  {
    v44 = 6;
    if ( v5 != 64 )
    {
      v29 = v5 == 128;
      v30 = 7;
      if ( !v29 )
        v30 = 0;
      v44 = v30;
      v6 = a2;
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        goto LABEL_13;
LABEL_64:
      v7 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      goto LABEL_14;
    }
  }
  else
  {
    v44 = 3;
    if ( v5 != 8 )
      v44 = 4 * (v5 == 16);
  }
  v6 = a2;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) == 0 )
    goto LABEL_64;
LABEL_13:
  v7 = *(_QWORD *)(v6 - 8);
LABEL_14:
  v8 = (__int64 *)(v7 + 24);
  v41 = a2;
  v9 = sub_16348C0(a2) | 4;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v41 = *(_QWORD *)(a2 - 8) + 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  if ( v8 == (__int64 *)v41 )
  {
LABEL_31:
    sub_1FD5CC0((__int64)a1, a2, v42, 1);
    return 1;
  }
  v10 = 0;
  do
  {
    while ( 1 )
    {
      v11 = v9;
      v12 = v9 & 0xFFFFFFFFFFFFFFF8LL;
      v13 = *v8;
      v14 = v12;
      LODWORD(v11) = (v11 >> 2) & 1;
      v45 = v11;
      if ( (_DWORD)v11 )
      {
        v17 = v12;
        if ( v12 )
        {
          if ( *(_BYTE *)(v13 + 16) != 13 )
            goto LABEL_45;
          goto LABEL_34;
        }
      }
      else if ( v12 )
      {
        v15 = *(_QWORD *)(v13 + 24);
        if ( *(_DWORD *)(v13 + 32) > 0x40u )
          v15 = **(_QWORD **)(v13 + 24);
        if ( v15 )
        {
          v10 += *(_QWORD *)(sub_15A9930((__int64)a1[12], v12) + 8LL * (unsigned int)v15 + 16);
          if ( v10 <= 0x7FF )
          {
LABEL_43:
            v13 = *v8;
          }
          else
          {
            v42 = sub_1FDC040(a1, v44, 0x34u, v42, v43, v10, v44);
            if ( !v42 )
              return 0;
            v43 = 1;
            v13 = *v8;
            v10 = 0;
          }
        }
        v14 = sub_1643D30(v12, v13);
        goto LABEL_25;
      }
      v17 = sub_1643D30(0, v13);
      if ( *(_BYTE *)(v13 + 16) != 13 )
      {
LABEL_45:
        if ( v10 )
        {
          v38 = v17;
          v42 = sub_1FDC040(a1, v44, 0x34u, v42, v43, v10, v44);
          if ( !v42 )
            return 0;
          v43 = 1;
          v17 = v38;
        }
        v35 = v17;
        v39 = (__int64)a1[12];
        v22 = (unsigned int)sub_15A9FE0(v39, v17);
        v23 = (v22 + ((unsigned __int64)(sub_127FA20(v39, v35) + 7) >> 3) - 1) / v22 * v22;
        v24 = sub_1FD9270(a1, (__int64 *)v13);
        v25 = (unsigned int)v24;
        if ( !(_DWORD)v24 )
          return 0;
        v26 = BYTE4(v24);
        if ( v23 != 1 )
        {
          v27 = sub_1FDC040(a1, v44, 0x36u, v24, BYTE4(v24), v23, v44);
          v25 = v27;
          if ( !v27 )
            return 0;
          v26 = 1;
        }
        v28 = (__int64 (*)())(*a1)[9];
        if ( v28 == sub_1FD34D0 )
          return 0;
        v42 = ((__int64 (__fastcall *)(__int64 **, _QWORD, _QWORD, __int64, _QWORD, _QWORD, __int64, __int64))v28)(
                a1,
                v44,
                v44,
                52,
                v42,
                v43,
                v25,
                v26);
        if ( !v42 )
          return 0;
        v10 = 0;
        goto LABEL_41;
      }
LABEL_34:
      v18 = (__int64 *)(v13 + 24);
      if ( *(_DWORD *)(v13 + 32) <= 0x40u )
      {
        if ( !*(_QWORD *)(v13 + 24) )
          goto LABEL_41;
      }
      else
      {
        v31 = *(_DWORD *)(v13 + 32);
        v33 = v17;
        v19 = sub_16A57B0(v13 + 24);
        v18 = (__int64 *)(v13 + 24);
        v17 = v33;
        if ( v31 == v19 )
          goto LABEL_41;
      }
      v36 = v17;
      sub_16A5D70((__int64)&v46, v18, 0x40u);
      v20 = v36;
      if ( v47 > 0x40 )
      {
        v21 = *v46;
        j_j___libc_free_0_0(v46);
        v20 = v36;
      }
      else
      {
        v21 = (__int64)((_QWORD)v46 << (64 - (unsigned __int8)v47)) >> (64 - (unsigned __int8)v47);
      }
      v32 = v20;
      v34 = (__int64)a1[12];
      v37 = (unsigned int)sub_15A9FE0(v34, v20);
      v10 += v21 * v37 * ((v37 + ((unsigned __int64)(sub_127FA20(v34, v32) + 7) >> 3) - 1) / v37);
      if ( v10 > 0x7FF )
      {
        v42 = sub_1FDC040(a1, v44, 0x34u, v42, v43, v10, v44);
        if ( !v42 )
          return 0;
        v43 = 1;
        v10 = 0;
      }
LABEL_41:
      if ( !v45 || !v12 )
        goto LABEL_43;
LABEL_25:
      v16 = *(_BYTE *)(v14 + 8);
      if ( ((v16 - 14) & 0xFD) != 0 )
        break;
      v8 += 3;
      v9 = *(_QWORD *)(v14 + 24) | 4LL;
      if ( (__int64 *)v41 == v8 )
        goto LABEL_29;
    }
    v9 = 0;
    if ( v16 == 13 )
      v9 = v14;
    v8 += 3;
  }
  while ( (__int64 *)v41 != v8 );
LABEL_29:
  if ( !v10 )
    goto LABEL_31;
  v42 = sub_1FDC040(a1, v44, 0x34u, v42, v43, v10, v44);
  if ( v42 )
    goto LABEL_31;
  return 0;
}
