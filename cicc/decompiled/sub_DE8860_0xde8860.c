// Function: sub_DE8860
// Address: 0xde8860
//
_QWORD *__fastcall sub_DE8860(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // r14d
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rax
  void *v12; // r10
  __int64 v13; // r8
  __int64 v14; // r9
  char *v15; // rsi
  __int64 v16; // rcx
  unsigned int v17; // r8d
  char *v18; // rdx
  __int64 v19; // rax
  char *v20; // rdi
  __int64 v21; // rax
  _QWORD *v22; // rax
  __int64 v23; // r8
  _QWORD *v24; // r8
  char *v26; // rdi
  _QWORD *v27; // rbx
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // rax
  _QWORD *v32; // r12
  __int64 v33; // [rsp+10h] [rbp-C0h]
  int na; // [rsp+18h] [rbp-B8h]
  _QWORD *nb; // [rsp+18h] [rbp-B8h]
  _QWORD *n; // [rsp+18h] [rbp-B8h]
  size_t nc; // [rsp+18h] [rbp-B8h]
  _QWORD *src; // [rsp+20h] [rbp-B0h]
  void *srca; // [rsp+20h] [rbp-B0h]
  _QWORD *v40; // [rsp+28h] [rbp-A8h]
  unsigned __int64 v41; // [rsp+28h] [rbp-A8h]
  int v42; // [rsp+28h] [rbp-A8h]
  char *v43; // [rsp+38h] [rbp-98h]
  _QWORD v45[2]; // [rsp+50h] [rbp-80h] BYREF
  _QWORD v46[2]; // [rsp+60h] [rbp-70h] BYREF
  char *v47; // [rsp+70h] [rbp-60h] BYREF
  __int64 v48; // [rsp+78h] [rbp-58h]
  _QWORD v49[10]; // [rsp+80h] [rbp-50h] BYREF

  v5 = a4;
  v43 = *(char **)(a1 + 48);
  v8 = **(_QWORD **)(a1 + 32);
  v9 = sub_D33D80((_QWORD *)a1, (__int64)a3, (__int64)a3, a4, a5);
  if ( *(_WORD *)(v8 + 24) != 5 )
    return sub_DC2B70((__int64)a3, **(_QWORD **)(a1 + 32), a2, v5);
  v10 = v9;
  v11 = *(_QWORD *)(v8 + 40);
  v12 = *(void **)(v8 + 32);
  v13 = 8 * v11;
  v47 = (char *)v49;
  v48 = 0x400000000LL;
  v14 = (8 * v11) >> 3;
  if ( (unsigned __int64)(8 * v11) > 0x20 )
  {
    nc = 8 * v11;
    srca = v12;
    v41 = (8 * v11) >> 3;
    sub_C8D5F0((__int64)&v47, v49, v41, 8u, v13, v14);
    LODWORD(v14) = v41;
    v12 = srca;
    v13 = nc;
    v26 = &v47[8 * (unsigned int)v48];
  }
  else
  {
    v15 = (char *)v49;
    if ( !v13 )
      goto LABEL_4;
    v26 = (char *)v49;
  }
  v42 = v14;
  memcpy(v26, v12, v13);
  v15 = v47;
  LODWORD(v13) = v48;
  LODWORD(v14) = v42;
LABEL_4:
  LODWORD(v48) = v14 + v13;
  v17 = v14 + v13;
  v16 = v17;
  v18 = &v15[8 * v17];
  if ( v15 != v18 )
  {
    while ( 1 )
    {
      v19 = *(_QWORD *)v15;
      v20 = v15;
      v15 += 8;
      if ( v10 == v19 )
        break;
      if ( v18 == v15 )
        goto LABEL_11;
    }
    if ( v18 != v15 )
    {
      memmove(v20, v15, v18 - v15);
      v17 = v48;
    }
    v16 = v17 - 1;
    LODWORD(v48) = v17 - 1;
  }
LABEL_11:
  if ( *(_QWORD *)(v8 + 40) == v16 )
  {
LABEL_19:
    if ( v47 != (char *)v49 )
      _libc_free(v47, v15);
    return sub_DC2B70((__int64)a3, **(_QWORD **)(a1 + 32), a2, v5);
  }
  v40 = sub_DC7EB0(a3, (__int64)&v47, *(_WORD *)(v8 + 28) & 2, 0);
  src = sub_DC1960((__int64)a3, (__int64)v40, v10, (__int64)v43, 0);
  if ( *((_WORD *)src + 12) == 8 )
  {
    v15 = (char *)sub_DCF3A0(a3, v43, 0);
    if ( (*((_BYTE *)src + 28) & 2) != 0
      && !sub_D96A50((__int64)v15)
      && (unsigned __int8)sub_DBEDC0((__int64)a3, (__int64)v15) )
    {
      goto LABEL_27;
    }
  }
  else
  {
    sub_DCF3A0(a3, v43, 0);
    src = 0;
  }
  v21 = sub_D95540(**(_QWORD **)(a1 + 32));
  na = sub_D97050((__int64)a3, v21);
  v22 = (_QWORD *)sub_B2BE50(*a3);
  v33 = sub_BCCE00(v22, 2 * na);
  nb = sub_DC2B70((__int64)a3, v10, v33, v5);
  v46[0] = sub_DC2B70((__int64)a3, (__int64)v40, v33, v5);
  v45[0] = v46;
  v46[1] = nb;
  v45[1] = 0x200000002LL;
  n = sub_DC7EB0(a3, (__int64)v45, 0, 0);
  v23 = v33;
  if ( (_QWORD *)v45[0] != v46 )
  {
    _libc_free(v45[0], v45);
    v23 = v33;
  }
  v15 = (char *)v8;
  if ( n != sub_DC2B70((__int64)a3, v8, v23, v5) )
  {
    v15 = (char *)v45;
    v24 = sub_DBEE70(v10, v45, a3);
    if ( v24 )
    {
      v15 = v43;
      if ( (unsigned __int8)sub_DDD5B0(a3, (__int64)v43, LODWORD(v45[0]), (__int64)v40, (__int64)v24) )
        goto LABEL_27;
    }
    goto LABEL_19;
  }
  if ( src && (*(_BYTE *)(a1 + 28) & 2) != 0 )
  {
    v15 = (char *)src;
    sub_D97270((__int64)a3, (__int64)src, 2);
  }
LABEL_27:
  if ( v47 != (char *)v49 )
    _libc_free(v47, v15);
  if ( !v40 )
    return sub_DC2B70((__int64)a3, **(_QWORD **)(a1 + 32), a2, v5);
  v27 = sub_DC2B70((__int64)a3, (__int64)v40, a2, v5);
  v31 = sub_D33D80((_QWORD *)a1, (__int64)a3, v28, v29, v30);
  v49[0] = sub_DC2B70((__int64)a3, v31, a2, v5);
  v47 = (char *)v49;
  v49[1] = v27;
  v48 = 0x200000002LL;
  v32 = sub_DC7EB0(a3, (__int64)&v47, 0, 0);
  if ( v47 != (char *)v49 )
    _libc_free(v47, &v47);
  return v32;
}
