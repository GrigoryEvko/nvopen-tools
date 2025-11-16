// Function: sub_3702F90
// Address: 0x3702f90
//
__int64 __fastcall sub_3702F90(__int64 a1, char *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r12d
  size_t v8; // r13
  char *v9; // r8
  _BYTE *v10; // r15
  _BYTE *v11; // rcx
  _BYTE *v12; // rdi
  char *v13; // r14
  size_t v14; // r15
  size_t v15; // r15
  __int64 v16; // rax
  unsigned __int64 v17; // rcx
  int v18; // r12d
  __int64 v19; // rdx
  __int64 v20; // rdi
  __int64 result; // rax
  size_t v22; // rdi
  size_t v23; // rax
  bool v24; // cf
  __int64 v25; // rdi
  unsigned __int64 v26; // rdx
  char *v27; // r9
  char *v28; // r11
  size_t v29; // rdx
  size_t v30; // r13
  _BYTE *v31; // rcx
  char *v32; // r10
  size_t v33; // rax
  __int64 v34; // rax
  char *v35; // rdi
  char *v36; // rax
  char *v37; // r8
  char *v38; // rax
  char *v39; // [rsp+0h] [rbp-60h]
  char *v40; // [rsp+8h] [rbp-58h]
  void *dest; // [rsp+10h] [rbp-50h]
  char *desta; // [rsp+10h] [rbp-50h]
  char *v43; // [rsp+18h] [rbp-48h]
  _BYTE *v44; // [rsp+18h] [rbp-48h]
  void *v45; // [rsp+18h] [rbp-48h]
  void *v46; // [rsp+18h] [rbp-48h]
  _BYTE *v47; // [rsp+20h] [rbp-40h]
  char *v48; // [rsp+20h] [rbp-40h]
  __int64 v49; // [rsp+20h] [rbp-40h]
  char *v50; // [rsp+20h] [rbp-40h]
  _BYTE *v51; // [rsp+20h] [rbp-40h]
  char *src; // [rsp+28h] [rbp-38h]
  char *srcb; // [rsp+28h] [rbp-38h]
  char *srcd; // [rsp+28h] [rbp-38h]
  char *srca; // [rsp+28h] [rbp-38h]
  char *srcc; // [rsp+28h] [rbp-38h]

  v6 = (int)a2;
  v8 = *(_QWORD *)(a1 + 240);
  v9 = *(char **)(a1 + 232);
  v10 = *(_BYTE **)(a1 + 48);
  if ( !v8 )
    goto LABEL_5;
  v11 = *(_BYTE **)(a1 + 64);
  v12 = *(_BYTE **)(a1 + 56);
  v13 = &v10[(unsigned int)a2];
  if ( v8 > v11 - v12 )
  {
    v22 = v12 - v10;
    if ( v8 > 0x7FFFFFFFFFFFFFFFLL - v22 )
      sub_4262D8((__int64)"vector::_M_range_insert");
    v23 = v22;
    if ( v8 >= v22 )
      v23 = v8;
    v24 = __CFADD__(v23, v22);
    v25 = v23 + v22;
    v26 = v25;
    if ( v24 || v25 < 0 )
    {
      v26 = 0x7FFFFFFFFFFFFFFFLL;
    }
    else if ( !v25 )
    {
      src = 0;
      v27 = 0;
      goto LABEL_19;
    }
    v48 = v9;
    srcb = (char *)v26;
    v34 = sub_22077B0(v26);
    v10 = *(_BYTE **)(a1 + 48);
    v27 = (char *)v34;
    v11 = *(_BYTE **)(a1 + 64);
    v9 = v48;
    src = &srcb[v34];
LABEL_19:
    v28 = &v27[v13 - v10];
    if ( v13 == v10 )
    {
      v29 = v8;
      a2 = v9;
      dest = v27;
      v43 = &v28[v8];
      v30 = 0;
      v47 = v11;
      memcpy(&v27[v13 - v10], v9, v29);
      v31 = v47;
      v32 = v43;
      a6 = (__int64)dest;
      v33 = *(_QWORD *)(a1 + 56) - (_QWORD)v13;
      if ( !v33 )
        goto LABEL_21;
    }
    else
    {
      v40 = &v28[v8];
      v44 = v11;
      v49 = (__int64)v27;
      v39 = v9;
      desta = &v27[v13 - v10];
      memmove(v27, v10, v13 - v10);
      memcpy(desta, v39, v8);
      a6 = v49;
      v31 = v44;
      v32 = v40;
      v33 = *(_QWORD *)(a1 + 56) - (_QWORD)v13;
      if ( !v33 )
        goto LABEL_30;
    }
    a2 = v13;
    v46 = (void *)a6;
    v51 = v31;
    v30 = v33;
    v38 = (char *)memcpy(v32, v13, v33);
    a6 = (__int64)v46;
    v31 = v51;
    v32 = v38;
LABEL_21:
    v32 += v30;
    if ( !v10 )
    {
LABEL_22:
      *(_QWORD *)(a1 + 48) = a6;
      *(_QWORD *)(a1 + 56) = v32;
      *(_QWORD *)(a1 + 64) = src;
      goto LABEL_5;
    }
LABEL_30:
    v45 = (void *)a6;
    a2 = (char *)(v31 - v10);
    v50 = v32;
    j_j___libc_free_0((unsigned __int64)v10);
    a6 = (__int64)v45;
    v32 = v50;
    goto LABEL_22;
  }
  v14 = v12 - v13;
  if ( v8 >= v12 - v13 )
  {
    if ( v8 != v14 )
    {
      a2 = &v9[v14];
      srcc = v9;
      memmove(v12, &v9[v14], v8 - v14);
      v12 = *(_BYTE **)(a1 + 56);
      v9 = srcc;
    }
    v35 = &v12[v8 - v14];
    *(_QWORD *)(a1 + 56) = v35;
    if ( v14 )
    {
      srcd = v9;
      memmove(v35, v13, v14);
      *(_QWORD *)(a1 + 56) += v14;
      a2 = srcd;
      memmove(v13, srcd, v14);
    }
  }
  else
  {
    v15 = &v12[-v8] - v13;
    srca = v9;
    v36 = (char *)memmove(v12, &v12[-v8], v8);
    *(_QWORD *)(a1 + 56) += v8;
    v37 = srca;
    if ( v15 )
    {
      memmove(&v36[-v15], v13, v15);
      v37 = srca;
    }
    a2 = v37;
    memmove(v13, v37, v8);
  }
LABEL_5:
  v16 = *(unsigned int *)(a1 + 8);
  v17 = *(unsigned int *)(a1 + 12);
  v18 = v6 + 8;
  if ( v16 + 1 > v17 )
  {
    a2 = (char *)(a1 + 16);
    sub_C8D5F0(a1, (const void *)(a1 + 16), v16 + 1, 4u, (__int64)v9, a6);
    v16 = *(unsigned int *)(a1 + 8);
  }
  v19 = *(_QWORD *)a1;
  *(_DWORD *)(*(_QWORD *)a1 + 4 * v16) = v18;
  ++*(_DWORD *)(a1 + 8);
  if ( *(_BYTE *)(a1 + 128) )
  {
    result = *(_QWORD *)(a1 + 120);
  }
  else
  {
    v20 = *(_QWORD *)(a1 + 104);
    result = 0;
    if ( v20 )
      result = (*(__int64 (__fastcall **)(__int64, char *, __int64, unsigned __int64, char *))(*(_QWORD *)v20 + 40LL))(
                 v20,
                 a2,
                 v19,
                 v17,
                 v9)
             - *(_QWORD *)(a1 + 112);
  }
  *(_QWORD *)(a1 + 136) = result;
  return result;
}
