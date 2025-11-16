// Function: sub_24A4AF0
// Address: 0x24a4af0
//
__int64 __fastcall sub_24A4AF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v7; // esi
  int v8; // r14d
  __int64 v9; // r10
  int v10; // r15d
  __int64 *v11; // r9
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // rdi
  __int64 v15; // r9
  int v16; // r11d
  __int64 *v17; // r15
  unsigned int v18; // edx
  __int64 *v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // r14
  __int64 *v23; // rbx
  __int64 v24; // r15
  int v26; // r15d
  int v27; // r10d
  __int64 v28; // rdx
  __int64 v29; // rax
  unsigned __int64 v30; // r14
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // rdi
  __int64 v33; // rdi
  int v34; // edx
  int v35; // eax
  __int64 v36; // rax
  unsigned __int64 v37; // r14
  unsigned __int64 v38; // rdi
  unsigned __int64 v39; // rdi
  unsigned __int64 v40; // r12
  char *v41; // rcx
  __int64 v42; // rax
  __int64 v43; // rsi
  bool v44; // cf
  unsigned __int64 v45; // rax
  unsigned __int64 v46; // r15
  __int64 v47; // rax
  __int64 *v48; // rcx
  _QWORD *v49; // r14
  unsigned __int64 *v50; // r15
  __int64 v51; // rax
  unsigned __int64 v52; // rdi
  __int64 *v53; // [rsp+8h] [rbp-68h]
  __int64 v54; // [rsp+10h] [rbp-60h]
  unsigned __int64 v55; // [rsp+10h] [rbp-60h]
  __int64 v57; // [rsp+18h] [rbp-58h]
  __int64 *v58; // [rsp+28h] [rbp-48h] BYREF
  __int64 v59; // [rsp+30h] [rbp-40h] BYREF
  __int64 v60; // [rsp+38h] [rbp-38h]

  v59 = a2;
  v7 = *(_DWORD *)(a1 + 56);
  v8 = *(_DWORD *)(a1 + 48);
  v54 = a1 + 32;
  v60 = 0;
  if ( v7 )
  {
    v9 = *(_QWORD *)(a1 + 40);
    v10 = 1;
    v11 = 0;
    v12 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v13 = (__int64 *)(v9 + 16LL * v12);
    v14 = *v13;
    if ( a2 == *v13 )
    {
LABEL_3:
      v59 = a3;
      v60 = 0;
      goto LABEL_4;
    }
    while ( v14 != -4096 )
    {
      if ( v11 || v14 != -8192 )
        v13 = v11;
      v12 = (v7 - 1) & (v10 + v12);
      v14 = *(_QWORD *)(v9 + 16LL * v12);
      if ( a2 == v14 )
        goto LABEL_3;
      ++v10;
      v11 = v13;
      v13 = (__int64 *)(v9 + 16LL * v12);
    }
    v26 = v8 + 1;
    if ( !v11 )
      v11 = v13;
    ++*(_QWORD *)(a1 + 32);
    v27 = v8 + 1;
    v58 = v11;
    if ( 4 * v26 < 3 * v7 )
    {
      v28 = a2;
      if ( v7 - *(_DWORD *)(a1 + 52) - v26 > v7 >> 3 )
        goto LABEL_21;
      goto LABEL_79;
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 32);
    v26 = v8 + 1;
    v58 = 0;
  }
  v7 *= 2;
LABEL_79:
  sub_24A41B0(v54, v7);
  sub_24A2B00(v54, &v59, &v58);
  v28 = v59;
  v11 = v58;
  v27 = *(_DWORD *)(a1 + 48) + 1;
LABEL_21:
  *(_DWORD *)(a1 + 48) = v27;
  if ( *v11 != -4096 )
    --*(_DWORD *)(a1 + 52);
  *v11 = v28;
  v53 = v11;
  v11[1] = v60;
  v29 = sub_22077B0(0x68u);
  if ( v29 )
  {
    *(_QWORD *)v29 = v29;
    *(_QWORD *)(v29 + 40) = v29 + 56;
    *(_DWORD *)(v29 + 8) = v8;
    *(_DWORD *)(v29 + 12) = 0;
    *(_BYTE *)(v29 + 24) = 0;
    *(_QWORD *)(v29 + 32) = 0;
    *(_QWORD *)(v29 + 48) = 0x200000000LL;
    *(_QWORD *)(v29 + 72) = v29 + 88;
    *(_QWORD *)(v29 + 80) = 0x200000000LL;
  }
  v30 = v53[1];
  v53[1] = v29;
  if ( v30 )
  {
    v31 = *(_QWORD *)(v30 + 72);
    if ( v31 != v30 + 88 )
      _libc_free(v31);
    v32 = *(_QWORD *)(v30 + 40);
    if ( v32 != v30 + 56 )
      _libc_free(v32);
    j_j___libc_free_0(v30);
  }
  v7 = *(_DWORD *)(a1 + 56);
  v59 = a3;
  v8 = v26;
  v60 = 0;
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 32);
    v58 = 0;
    goto LABEL_33;
  }
LABEL_4:
  v15 = *(_QWORD *)(a1 + 40);
  v16 = 1;
  v17 = 0;
  v18 = (v7 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v19 = (__int64 *)(v15 + 16LL * v18);
  v20 = *v19;
  if ( a3 == *v19 )
    goto LABEL_5;
  while ( v20 != -4096 )
  {
    if ( v17 || v20 != -8192 )
      v19 = v17;
    v18 = (v7 - 1) & (v16 + v18);
    v20 = *(_QWORD *)(v15 + 16LL * v18);
    if ( a3 == v20 )
      goto LABEL_5;
    ++v16;
    v17 = v19;
    v19 = (__int64 *)(v15 + 16LL * v18);
  }
  if ( !v17 )
    v17 = v19;
  v35 = *(_DWORD *)(a1 + 48);
  ++*(_QWORD *)(a1 + 32);
  v34 = v35 + 1;
  v58 = v17;
  if ( 4 * (v35 + 1) < 3 * v7 )
  {
    v33 = a3;
    if ( v7 - *(_DWORD *)(a1 + 52) - v34 > v7 >> 3 )
      goto LABEL_44;
    goto LABEL_34;
  }
LABEL_33:
  v7 *= 2;
LABEL_34:
  sub_24A41B0(v54, v7);
  sub_24A2B00(v54, &v59, &v58);
  v33 = v59;
  v17 = v58;
  v34 = *(_DWORD *)(a1 + 48) + 1;
LABEL_44:
  *(_DWORD *)(a1 + 48) = v34;
  if ( *v17 != -4096 )
    --*(_DWORD *)(a1 + 52);
  *v17 = v33;
  v17[1] = v60;
  v36 = sub_22077B0(0x68u);
  if ( v36 )
  {
    *(_QWORD *)v36 = v36;
    *(_QWORD *)(v36 + 40) = v36 + 56;
    *(_DWORD *)(v36 + 8) = v8;
    *(_DWORD *)(v36 + 12) = 0;
    *(_BYTE *)(v36 + 24) = 0;
    *(_QWORD *)(v36 + 32) = 0;
    *(_QWORD *)(v36 + 48) = 0x200000000LL;
    *(_QWORD *)(v36 + 72) = v36 + 88;
    *(_QWORD *)(v36 + 80) = 0x200000000LL;
  }
  v37 = v17[1];
  v17[1] = v36;
  if ( v37 )
  {
    v38 = *(_QWORD *)(v37 + 72);
    if ( v38 != v37 + 88 )
      _libc_free(v38);
    v39 = *(_QWORD *)(v37 + 40);
    if ( v39 != v37 + 56 )
      _libc_free(v39);
    j_j___libc_free_0(v37);
  }
LABEL_5:
  v21 = sub_22077B0(0x30u);
  v22 = v21;
  if ( v21 )
  {
    *(_QWORD *)v21 = a2;
    *(_QWORD *)(v21 + 8) = a3;
    *(_BYTE *)(v21 + 26) = 0;
    *(_QWORD *)(v21 + 16) = a4;
    *(_WORD *)(v21 + 24) = 0;
    *(_BYTE *)(v21 + 40) = 0;
  }
  v23 = *(__int64 **)(a1 + 16);
  if ( v23 != *(__int64 **)(a1 + 24) )
  {
    if ( v23 )
    {
      *v23 = v21;
      v23 = *(__int64 **)(a1 + 16);
    }
    v24 = (__int64)(v23 + 1);
    *(_QWORD *)(a1 + 16) = v23 + 1;
    return *(_QWORD *)(v24 - 8);
  }
  v40 = *(_QWORD *)(a1 + 8);
  v41 = (char *)v23 - v40;
  v42 = (__int64)((__int64)v23 - v40) >> 3;
  if ( v42 == 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v43 = 1;
  if ( v42 )
    v43 = (__int64)((__int64)v23 - v40) >> 3;
  v44 = __CFADD__(v43, v42);
  v45 = v43 + v42;
  if ( v44 )
  {
    v46 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_62:
    v47 = sub_22077B0(v46);
    v41 = (char *)v23 - v40;
    v57 = v47;
    v55 = v47 + v46;
    v24 = v47 + 8;
    goto LABEL_63;
  }
  if ( v45 )
  {
    if ( v45 > 0xFFFFFFFFFFFFFFFLL )
      v45 = 0xFFFFFFFFFFFFFFFLL;
    v46 = 8 * v45;
    goto LABEL_62;
  }
  v55 = 0;
  v24 = 8;
  v57 = 0;
LABEL_63:
  v48 = (__int64 *)&v41[v57];
  if ( v48 )
    *v48 = v22;
  if ( v23 != (__int64 *)v40 )
  {
    v49 = (_QWORD *)v57;
    v50 = (unsigned __int64 *)v40;
    while ( 1 )
    {
      v52 = *v50;
      if ( v49 )
        break;
      if ( !v52 )
        goto LABEL_68;
      ++v50;
      j_j___libc_free_0(v52);
      v51 = 8;
      if ( v23 == (__int64 *)v50 )
      {
LABEL_73:
        v24 = (__int64)(v49 + 2);
        goto LABEL_74;
      }
LABEL_69:
      v49 = (_QWORD *)v51;
    }
    *v49 = v52;
    *v50 = 0;
LABEL_68:
    ++v50;
    v51 = (__int64)(v49 + 1);
    if ( v23 == (__int64 *)v50 )
      goto LABEL_73;
    goto LABEL_69;
  }
LABEL_74:
  if ( v40 )
    j_j___libc_free_0(v40);
  *(_QWORD *)(a1 + 16) = v24;
  *(_QWORD *)(a1 + 8) = v57;
  *(_QWORD *)(a1 + 24) = v55;
  return *(_QWORD *)(v24 - 8);
}
