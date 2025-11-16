// Function: sub_17E4990
// Address: 0x17e4990
//
__int64 __fastcall sub_17E4990(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  unsigned int v8; // esi
  int v9; // r9d
  __int64 v10; // r10
  int v11; // r15d
  __int64 *v12; // rcx
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rdi
  int v17; // r11d
  __int64 *v18; // r15
  unsigned int v19; // edx
  __int64 *v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v23; // r14
  __int64 *v24; // rbx
  __int64 v25; // r15
  int v27; // r15d
  int v28; // edi
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // r9
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // rdi
  __int64 v34; // r9
  unsigned __int64 v35; // rdi
  unsigned __int64 v36; // rdi
  __int64 v37; // rcx
  int v38; // edx
  int v39; // eax
  __int64 v40; // rax
  __int64 v41; // r14
  unsigned __int64 v42; // rdi
  unsigned __int64 v43; // rdi
  __int64 v44; // r14
  unsigned __int64 v45; // rdi
  unsigned __int64 v46; // rdi
  char *v47; // r12
  char *v48; // rcx
  __int64 v49; // rax
  __int64 v50; // rsi
  bool v51; // cf
  unsigned __int64 v52; // rax
  __int64 v53; // r15
  __int64 v54; // rax
  __int64 *v55; // rcx
  _QWORD *v56; // r14
  __int64 *v57; // r15
  __int64 v58; // rax
  __int64 v59; // rdi
  int v60; // [rsp+8h] [rbp-68h]
  int v61; // [rsp+8h] [rbp-68h]
  int v62; // [rsp+8h] [rbp-68h]
  int v63; // [rsp+8h] [rbp-68h]
  __int64 *v64; // [rsp+10h] [rbp-60h]
  __int64 v65; // [rsp+10h] [rbp-60h]
  __int64 v66; // [rsp+10h] [rbp-60h]
  __int64 v67; // [rsp+10h] [rbp-60h]
  __int64 v68; // [rsp+10h] [rbp-60h]
  __int64 v69; // [rsp+10h] [rbp-60h]
  __int64 v71; // [rsp+18h] [rbp-58h]
  __int64 *v72; // [rsp+28h] [rbp-48h] BYREF
  __int64 v73; // [rsp+30h] [rbp-40h] BYREF
  __int64 v74; // [rsp+38h] [rbp-38h]

  v4 = a1 + 32;
  v73 = a2;
  v8 = *(_DWORD *)(a1 + 56);
  v9 = *(_DWORD *)(a1 + 48);
  v74 = 0;
  if ( !v8 )
  {
    ++*(_QWORD *)(a1 + 32);
    v27 = v9 + 1;
LABEL_86:
    v61 = v9;
    sub_17E3B60(v4, 2 * v8);
    sub_17E1E90(v4, &v73, &v72);
    v12 = v72;
    v29 = v73;
    v9 = v61;
    v28 = *(_DWORD *)(a1 + 48) + 1;
    goto LABEL_22;
  }
  v10 = *(_QWORD *)(a1 + 40);
  v11 = 1;
  v12 = 0;
  v13 = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v14 = (__int64 *)(v10 + 16LL * v13);
  v15 = *v14;
  if ( a2 == *v14 )
  {
LABEL_3:
    v73 = a3;
    v74 = 0;
    goto LABEL_4;
  }
  while ( v15 != -8 )
  {
    if ( v15 != -16 || v12 )
      v14 = v12;
    v13 = (v8 - 1) & (v11 + v13);
    v15 = *(_QWORD *)(v10 + 16LL * v13);
    if ( a2 == v15 )
      goto LABEL_3;
    ++v11;
    v12 = v14;
    v14 = (__int64 *)(v10 + 16LL * v13);
  }
  v27 = v9 + 1;
  if ( !v12 )
    v12 = v14;
  ++*(_QWORD *)(a1 + 32);
  v28 = v9 + 1;
  if ( 4 * v27 >= 3 * v8 )
    goto LABEL_86;
  v29 = a2;
  if ( v8 - *(_DWORD *)(a1 + 52) - v27 <= v8 >> 3 )
  {
    v62 = v9;
    sub_17E3B60(v4, v8);
    sub_17E1E90(v4, &v73, &v72);
    v12 = v72;
    v29 = v73;
    v9 = v62;
    v28 = *(_DWORD *)(a1 + 48) + 1;
  }
LABEL_22:
  *(_DWORD *)(a1 + 48) = v28;
  if ( *v12 != -8 )
    --*(_DWORD *)(a1 + 52);
  *v12 = v29;
  v64 = v12;
  v12[1] = v74;
  sub_17E22C0(&v73, v9);
  v30 = v73;
  v73 = 0;
  v31 = v64[1];
  v64[1] = v30;
  if ( v31 )
  {
    v32 = *(_QWORD *)(v31 + 72);
    if ( v32 != v31 + 88 )
    {
      v65 = v31;
      _libc_free(v32);
      v31 = v65;
    }
    v33 = *(_QWORD *)(v31 + 40);
    if ( v33 != v31 + 56 )
    {
      v66 = v31;
      _libc_free(v33);
      v31 = v66;
    }
    j_j___libc_free_0(v31, 104);
    v34 = v73;
    if ( v73 )
    {
      v35 = *(_QWORD *)(v73 + 72);
      if ( v35 != v73 + 88 )
      {
        v67 = v73;
        _libc_free(v35);
        v34 = v67;
      }
      v36 = *(_QWORD *)(v34 + 40);
      if ( v36 != v34 + 56 )
      {
        v68 = v34;
        _libc_free(v36);
        v34 = v68;
      }
      j_j___libc_free_0(v34, 104);
    }
  }
  v8 = *(_DWORD *)(a1 + 56);
  v73 = a3;
  v9 = v27;
  v74 = 0;
  if ( !v8 )
  {
    ++*(_QWORD *)(a1 + 32);
    goto LABEL_37;
  }
LABEL_4:
  v16 = *(_QWORD *)(a1 + 40);
  v17 = 1;
  v18 = 0;
  v19 = (v8 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v20 = (__int64 *)(v16 + 16LL * v19);
  v21 = *v20;
  if ( a3 == *v20 )
    goto LABEL_5;
  while ( v21 != -8 )
  {
    if ( v18 || v21 != -16 )
      v20 = v18;
    v19 = (v8 - 1) & (v17 + v19);
    v21 = *(_QWORD *)(v16 + 16LL * v19);
    if ( a3 == v21 )
      goto LABEL_5;
    ++v17;
    v18 = v20;
    v20 = (__int64 *)(v16 + 16LL * v19);
  }
  if ( !v18 )
    v18 = v20;
  v39 = *(_DWORD *)(a1 + 48);
  ++*(_QWORD *)(a1 + 32);
  v38 = v39 + 1;
  if ( 4 * (v39 + 1) < 3 * v8 )
  {
    v37 = a3;
    if ( v8 - *(_DWORD *)(a1 + 52) - v38 <= v8 >> 3 )
    {
      v63 = v9;
      sub_17E3B60(v4, v8);
      sub_17E1E90(v4, &v73, &v72);
      v18 = v72;
      v37 = v73;
      v9 = v63;
      v38 = *(_DWORD *)(a1 + 48) + 1;
    }
    goto LABEL_48;
  }
LABEL_37:
  v60 = v9;
  sub_17E3B60(v4, 2 * v8);
  sub_17E1E90(v4, &v73, &v72);
  v18 = v72;
  v37 = v73;
  v9 = v60;
  v38 = *(_DWORD *)(a1 + 48) + 1;
LABEL_48:
  *(_DWORD *)(a1 + 48) = v38;
  if ( *v18 != -8 )
    --*(_DWORD *)(a1 + 52);
  *v18 = v37;
  v18[1] = v74;
  sub_17E22C0(&v73, v9);
  v40 = v73;
  v73 = 0;
  v41 = v18[1];
  v18[1] = v40;
  if ( v41 )
  {
    v42 = *(_QWORD *)(v41 + 72);
    if ( v42 != v41 + 88 )
      _libc_free(v42);
    v43 = *(_QWORD *)(v41 + 40);
    if ( v43 != v41 + 56 )
      _libc_free(v43);
    j_j___libc_free_0(v41, 104);
    v44 = v73;
    if ( v73 )
    {
      v45 = *(_QWORD *)(v73 + 72);
      if ( v45 != v73 + 88 )
        _libc_free(v45);
      v46 = *(_QWORD *)(v44 + 40);
      if ( v46 != v44 + 56 )
        _libc_free(v46);
      j_j___libc_free_0(v44, 104);
    }
  }
LABEL_5:
  v22 = sub_22077B0(40);
  v23 = v22;
  if ( v22 )
  {
    *(_QWORD *)v22 = a2;
    *(_QWORD *)(v22 + 8) = a3;
    *(_DWORD *)(v22 + 24) = 0;
    *(_QWORD *)(v22 + 16) = a4;
    *(_QWORD *)(v22 + 32) = 0;
  }
  v24 = *(__int64 **)(a1 + 16);
  if ( v24 != *(__int64 **)(a1 + 24) )
  {
    if ( v24 )
    {
      *v24 = v22;
      v24 = *(__int64 **)(a1 + 16);
    }
    v25 = (__int64)(v24 + 1);
    *(_QWORD *)(a1 + 16) = v24 + 1;
    return *(_QWORD *)(v25 - 8);
  }
  v47 = *(char **)(a1 + 8);
  v48 = (char *)((char *)v24 - v47);
  v49 = ((char *)v24 - v47) >> 3;
  if ( v49 == 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v50 = 1;
  if ( v49 )
    v50 = ((char *)v24 - v47) >> 3;
  v51 = __CFADD__(v50, v49);
  v52 = v50 + v49;
  if ( v51 )
  {
    v53 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_69:
    v54 = sub_22077B0(v53);
    v48 = (char *)((char *)v24 - v47);
    v71 = v54;
    v69 = v54 + v53;
    v25 = v54 + 8;
    goto LABEL_70;
  }
  if ( v52 )
  {
    if ( v52 > 0xFFFFFFFFFFFFFFFLL )
      v52 = 0xFFFFFFFFFFFFFFFLL;
    v53 = 8 * v52;
    goto LABEL_69;
  }
  v69 = 0;
  v25 = 8;
  v71 = 0;
LABEL_70:
  v55 = (__int64 *)&v48[v71];
  if ( v55 )
    *v55 = v23;
  if ( v24 != (__int64 *)v47 )
  {
    v56 = (_QWORD *)v71;
    v57 = (__int64 *)v47;
    while ( 1 )
    {
      v59 = *v57;
      if ( v56 )
        break;
      if ( !v59 )
        goto LABEL_75;
      ++v57;
      j_j___libc_free_0(v59, 40);
      v58 = 8;
      if ( v24 == v57 )
      {
LABEL_80:
        v25 = (__int64)(v56 + 2);
        goto LABEL_81;
      }
LABEL_76:
      v56 = (_QWORD *)v58;
    }
    *v56 = v59;
    *v57 = 0;
LABEL_75:
    ++v57;
    v58 = (__int64)(v56 + 1);
    if ( v24 == v57 )
      goto LABEL_80;
    goto LABEL_76;
  }
LABEL_81:
  if ( v47 )
    j_j___libc_free_0(v47, *(_QWORD *)(a1 + 24) - (_QWORD)v47);
  *(_QWORD *)(a1 + 16) = v25;
  *(_QWORD *)(a1 + 8) = v71;
  *(_QWORD *)(a1 + 24) = v69;
  return *(_QWORD *)(v25 - 8);
}
