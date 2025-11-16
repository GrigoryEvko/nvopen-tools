// Function: sub_1022080
// Address: 0x1022080
//
__int64 __fastcall sub_1022080(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rax
  char v10; // al
  _QWORD *v11; // rcx
  _QWORD *v12; // rsi
  __int64 v13; // rdi
  _BYTE *v14; // rsi
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // r14
  __int64 *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r15
  __int64 v22; // rax
  __int64 v23; // r13
  unsigned int v24; // eax
  unsigned int v25; // r13d
  __int64 v26; // r14
  unsigned int v27; // r14d
  const void *v28; // r8
  bool v29; // al
  unsigned int v30; // eax
  const void *v31; // rdi
  char v32; // r13
  _BYTE **v33; // rdx
  _BYTE *v34; // rax
  bool v35; // zf
  int v36; // edi
  __int64 v37; // rcx
  const void *v39; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v40; // [rsp+10h] [rbp-D0h] BYREF
  unsigned int v41; // [rsp+18h] [rbp-C8h]
  const void *v42; // [rsp+20h] [rbp-C0h] BYREF
  unsigned int v43; // [rsp+28h] [rbp-B8h]
  const void *v44; // [rsp+30h] [rbp-B0h] BYREF
  unsigned int v45; // [rsp+38h] [rbp-A8h]
  const void *v46; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v47; // [rsp+48h] [rbp-98h]
  const void *v48; // [rsp+50h] [rbp-90h] BYREF
  unsigned int v49; // [rsp+58h] [rbp-88h]
  const void *v50; // [rsp+60h] [rbp-80h] BYREF
  unsigned int v51; // [rsp+68h] [rbp-78h]
  const void *v52; // [rsp+70h] [rbp-70h] BYREF
  unsigned int v53; // [rsp+78h] [rbp-68h]
  __int64 v54; // [rsp+80h] [rbp-60h] BYREF
  unsigned int v55; // [rsp+88h] [rbp-58h]
  __int64 v56; // [rsp+90h] [rbp-50h] BYREF
  unsigned int v57; // [rsp+98h] [rbp-48h]
  __int64 v58; // [rsp+A0h] [rbp-40h]
  unsigned int v59; // [rsp+A8h] [rbp-38h]

  v7 = *(_QWORD *)(a3 + 16);
  if ( !v7 || *(_QWORD *)(v7 + 8) || *(_BYTE *)a4 != 86 )
    goto LABEL_2;
  v10 = *(_BYTE *)(a4 + 7) & 0x40;
  if ( v10 )
  {
    v11 = *(_QWORD **)(a4 - 8);
    v12 = v11;
    v13 = *(_QWORD *)(*v11 + 16LL);
    if ( v13 )
    {
      if ( *(_QWORD *)(v13 + 8) || (unsigned __int8)(*(_BYTE *)*v11 - 82) > 1u )
        goto LABEL_9;
      goto LABEL_68;
    }
LABEL_2:
    *(_BYTE *)a1 = 0;
    *(_QWORD *)(a1 + 8) = a4;
    *(_DWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 24) = 0;
    return a1;
  }
  v36 = *(_DWORD *)(a4 + 4);
  v12 = (_QWORD *)(a4 - 32LL * (v36 & 0x7FFFFFF));
  v37 = *(_QWORD *)(*v12 + 16LL);
  if ( !v37 )
    goto LABEL_2;
  if ( *(_QWORD *)(v37 + 8) || (unsigned __int8)(*(_BYTE *)*v12 - 82) > 1u )
    goto LABEL_65;
LABEL_68:
  v16 = v12[4];
  if ( v16 && a3 == v12[8] )
    goto LABEL_15;
  if ( !v10 )
  {
    v36 = *(_DWORD *)(a4 + 4);
LABEL_65:
    v11 = (_QWORD *)(a4 - 32LL * (v36 & 0x7FFFFFF));
    v14 = (_BYTE *)*v11;
    v15 = *(_QWORD *)(*v11 + 16LL);
    goto LABEL_10;
  }
  v11 = *(_QWORD **)(a4 - 8);
LABEL_9:
  v14 = (_BYTE *)*v11;
  v15 = *(_QWORD *)(*v11 + 16LL);
LABEL_10:
  if ( !v15 )
    goto LABEL_2;
  if ( *(_QWORD *)(v15 + 8) )
    goto LABEL_2;
  if ( (unsigned __int8)(*v14 - 82) > 1u )
    goto LABEL_2;
  if ( a3 != v11[4] )
    goto LABEL_2;
  v16 = v11[8];
  if ( !v16 )
    goto LABEL_2;
LABEL_15:
  v17 = *(_QWORD *)(v16 + 8);
  if ( !sub_D97040(a5, v17) )
    goto LABEL_2;
  v18 = sub_DD8400(a5, v16);
  v21 = (__int64)v18;
  if ( *((_WORD *)v18 + 12) != 8 )
    goto LABEL_2;
  if ( a2 != v18[6] )
    goto LABEL_2;
  v22 = sub_D33D80(v18, a5, v19, v20, a5);
  if ( !(unsigned __int8)sub_DBEDC0(a5, v22) )
    goto LABEL_2;
  v23 = sub_DBB9F0(a5, v21, 1u, 0);
  v53 = *(_DWORD *)(v23 + 8);
  if ( v53 > 0x40 )
    sub_C43780((__int64)&v52, (const void **)v23);
  else
    v52 = *(const void **)v23;
  v55 = *(_DWORD *)(v23 + 24);
  if ( v55 > 0x40 )
    sub_C43780((__int64)&v54, (const void **)(v23 + 16));
  else
    v54 = *(_QWORD *)(v23 + 16);
  v24 = *(_DWORD *)(v17 + 8) >> 8;
  v25 = v24 - 1;
  v41 = v24;
  v26 = 1LL << ((unsigned __int8)v24 - 1);
  if ( v24 <= 0x40 )
  {
    v40 = 0;
LABEL_25:
    v40 |= v26;
    goto LABEL_26;
  }
  sub_C43690((__int64)&v40, 0, 0);
  if ( v41 <= 0x40 )
    goto LABEL_25;
  *(_QWORD *)(v40 + 8LL * (v25 >> 6)) |= v26;
LABEL_26:
  v47 = v41;
  if ( v41 > 0x40 )
    sub_C43780((__int64)&v46, (const void **)&v40);
  else
    v46 = (const void *)v40;
  v43 = v41;
  if ( v41 > 0x40 )
    sub_C43780((__int64)&v42, (const void **)&v40);
  else
    v42 = (const void *)v40;
  sub_C46A40((__int64)&v42, 1);
  v27 = v43;
  v28 = v42;
  v43 = 0;
  v45 = v27;
  v44 = v42;
  if ( v27 <= 0x40 )
  {
    if ( v42 == v46 )
    {
      sub_AADB10((__int64)&v56, v27, 1);
      goto LABEL_38;
    }
  }
  else
  {
    v39 = v42;
    v29 = sub_C43C50((__int64)&v44, &v46);
    v28 = v39;
    if ( v29 )
    {
      sub_AADB10((__int64)&v56, v27, 1);
      v31 = v39;
      if ( !v39 )
        goto LABEL_38;
      goto LABEL_37;
    }
  }
  v30 = v47;
  v47 = 0;
  v49 = v27;
  v51 = v30;
  v48 = v28;
  v50 = v46;
  sub_AADC30((__int64)&v56, (__int64)&v48, (__int64 *)&v50);
  if ( v49 > 0x40 && v48 )
    j_j___libc_free_0_0(v48);
  if ( v51 > 0x40 )
  {
    v31 = v50;
    if ( v50 )
LABEL_37:
      j_j___libc_free_0_0(v31);
  }
LABEL_38:
  if ( v43 > 0x40 && v42 )
    j_j___libc_free_0_0(v42);
  if ( v47 > 0x40 && v46 )
    j_j___libc_free_0_0(v46);
  v32 = sub_AB1BB0((__int64)&v56, (__int64)&v52);
  if ( v59 > 0x40 && v58 )
    j_j___libc_free_0_0(v58);
  if ( v57 > 0x40 && v56 )
    j_j___libc_free_0_0(v56);
  if ( v41 > 0x40 && v40 )
    j_j___libc_free_0_0(v40);
  if ( v55 > 0x40 && v54 )
    j_j___libc_free_0_0(v54);
  if ( v53 > 0x40 && v52 )
    j_j___libc_free_0_0(v52);
  if ( !v32 )
    goto LABEL_2;
  if ( (*(_BYTE *)(a4 + 7) & 0x40) != 0 )
    v33 = *(_BYTE ***)(a4 - 8);
  else
    v33 = (_BYTE **)(a4 - 32LL * (*(_DWORD *)(a4 + 4) & 0x7FFFFFF));
  v34 = *v33;
  *(_BYTE *)a1 = 1;
  *(_QWORD *)(a1 + 8) = a4;
  v35 = *v34 == 82;
  *(_QWORD *)(a1 + 24) = 0;
  *(_DWORD *)(a1 + 16) = !v35 + 19;
  return a1;
}
