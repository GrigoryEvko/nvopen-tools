// Function: sub_2E168A0
// Address: 0x2e168a0
//
__int64 __fastcall sub_2E168A0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // r12
  __int64 v8; // rbx
  char v9; // r14
  int v10; // eax
  unsigned int v11; // r14d
  __int64 v12; // rdx
  __int64 v13; // rbx
  unsigned __int64 v14; // r13
  unsigned __int64 v15; // rax
  __int64 *v16; // rdx
  __int64 v17; // rax
  __int64 v18; // r8
  __int64 *v19; // r12
  __int64 *v20; // rbx
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  unsigned int v25; // eax
  unsigned __int64 v26; // r14
  unsigned int v27; // r13d
  unsigned __int64 v28; // r12
  unsigned __int64 v29; // rdi
  __int64 v31; // r8
  unsigned __int64 i; // rax
  __int64 j; // rsi
  __int16 v34; // dx
  __int64 v35; // rsi
  __int64 v36; // rdi
  unsigned int v37; // ecx
  __int64 *v38; // rdx
  __int64 v39; // r9
  unsigned __int64 v40; // r13
  __int64 *v41; // rax
  __int64 v42; // r8
  __int64 *v43; // rdx
  __int64 v44; // rcx
  unsigned int v45; // edi
  __int64 v46; // rax
  __int64 v47; // rax
  unsigned __int64 v48; // rdx
  __int64 *v49; // rax
  int v50; // edx
  int v51; // r10d
  __int128 v52; // [rsp-20h] [rbp-220h]
  __int64 v54; // [rsp+20h] [rbp-1E0h]
  __int64 v55; // [rsp+20h] [rbp-1E0h]
  __int64 v56; // [rsp+28h] [rbp-1D8h]
  unsigned __int64 v57[2]; // [rsp+50h] [rbp-1B0h] BYREF
  _BYTE v58[48]; // [rsp+60h] [rbp-1A0h] BYREF
  _BYTE *v59; // [rsp+90h] [rbp-170h]
  __int64 v60; // [rsp+98h] [rbp-168h]
  _BYTE v61[16]; // [rsp+A0h] [rbp-160h] BYREF
  unsigned __int64 v62; // [rsp+B0h] [rbp-150h]
  _BYTE *v63; // [rsp+C0h] [rbp-140h] BYREF
  __int64 v64; // [rsp+C8h] [rbp-138h]
  _BYTE v65[304]; // [rsp+D0h] [rbp-130h] BYREF

  v6 = a2;
  v7 = (__int64)a1;
  v8 = *(_QWORD *)(a2 + 104);
  if ( v8 )
  {
    v9 = 0;
    do
    {
      a2 = v8;
      sub_2E16290(a1, v8, *(_DWORD *)(v6 + 112), a4, a5, a6);
      v10 = *(_DWORD *)(v8 + 8);
      v8 = *(_QWORD *)(v8 + 104);
      if ( !v10 )
        v9 = 1;
    }
    while ( v8 );
    if ( v9 )
      sub_2E0AF60(v6);
  }
  v11 = *(_DWORD *)(v6 + 112);
  v12 = a1[1];
  v63 = v65;
  v64 = 0x1000000000LL;
  if ( (v11 & 0x80000000) != 0 )
  {
    v13 = *(_QWORD *)(*(_QWORD *)(v12 + 56) + 16LL * (v11 & 0x7FFFFFFF) + 8);
    if ( !v13 )
      goto LABEL_15;
  }
  else
  {
    v13 = *(_QWORD *)(*(_QWORD *)(v12 + 304) + 8LL * v11);
    if ( !v13 )
      goto LABEL_15;
  }
  v14 = *(_QWORD *)(v13 + 16);
LABEL_11:
  if ( (unsigned __int16)(*(_WORD *)(v14 + 68) - 14) <= 4u )
  {
LABEL_12:
    v15 = v14;
    goto LABEL_14;
  }
  a2 = v11;
  if ( !(unsigned __int8)sub_2E89D80(v14, v11, 0) )
    goto LABEL_56;
  v31 = *(_QWORD *)(v7 + 32);
  for ( i = v14; (*(_BYTE *)(i + 44) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
    ;
  for ( ; (*(_BYTE *)(v14 + 44) & 8) != 0; v14 = *(_QWORD *)(v14 + 8) )
    ;
  for ( j = *(_QWORD *)(v14 + 8); j != i; i = *(_QWORD *)(i + 8) )
  {
    v34 = *(_WORD *)(i + 68);
    if ( (unsigned __int16)(v34 - 14) > 4u && v34 != 24 )
      break;
  }
  v35 = *(unsigned int *)(v31 + 144);
  v36 = *(_QWORD *)(v31 + 128);
  if ( !(_DWORD)v35 )
    goto LABEL_59;
  v37 = (v35 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
  v38 = (__int64 *)(v36 + 16LL * v37);
  v39 = *v38;
  if ( i != *v38 )
  {
    v50 = 1;
    while ( v39 != -4096 )
    {
      v51 = v50 + 1;
      v37 = (v35 - 1) & (v37 + v50);
      v38 = (__int64 *)(v36 + 16LL * v37);
      v39 = *v38;
      if ( i == *v38 )
        goto LABEL_43;
      v50 = v51;
    }
LABEL_59:
    v38 = (__int64 *)(v36 + 16 * v35);
  }
LABEL_43:
  v40 = v38[1] & 0xFFFFFFFFFFFFFFF8LL;
  a2 = v40;
  v41 = (__int64 *)sub_2E09D00((__int64 *)v6, v40);
  v42 = v40 | 4;
  v43 = v41;
  v44 = *(_QWORD *)v6 + 24LL * *(unsigned int *)(v6 + 8);
  if ( v41 == (__int64 *)v44 )
    goto LABEL_56;
  v45 = *(_DWORD *)(v40 + 24);
  a2 = *(unsigned int *)((*v41 & 0xFFFFFFFFFFFFFFF8LL) + 24);
  if ( (unsigned __int64)((unsigned int)a2 | (*v41 >> 1) & 3) > v45 )
    goto LABEL_56;
  a6 = v41[2];
  if ( v40 == (v41[1] & 0xFFFFFFFFFFFFFFF8LL) )
  {
    if ( (__int64 *)v44 == v41 + 3 )
    {
      if ( a6 )
        goto LABEL_51;
    }
    else
    {
      v43 = v41 + 3;
      a2 = *(unsigned int *)((v41[3] & 0xFFFFFFFFFFFFFFF8LL) + 24);
      if ( v40 != *(_QWORD *)(a6 + 8) )
        goto LABEL_47;
    }
LABEL_56:
    v14 = *(_QWORD *)(v13 + 16);
    goto LABEL_12;
  }
  if ( v40 == *(_QWORD *)(a6 + 8) )
    goto LABEL_56;
LABEL_47:
  if ( v45 >= (unsigned int)a2 )
  {
    v46 = v43[2];
    if ( a6 != v46 )
    {
      if ( v46 )
        v42 = *(_QWORD *)(v46 + 8);
    }
  }
LABEL_51:
  v47 = (unsigned int)v64;
  v48 = (unsigned int)v64 + 1LL;
  if ( v48 > HIDWORD(v64) )
  {
    a2 = (__int64)v65;
    v55 = a6;
    v56 = v42;
    sub_C8D5F0((__int64)&v63, v65, v48, 0x10u, v42, a6);
    v47 = (unsigned int)v64;
    a6 = v55;
    v42 = v56;
  }
  v49 = (__int64 *)&v63[16 * v47];
  *v49 = v42;
  v49[1] = a6;
  v15 = *(_QWORD *)(v13 + 16);
  LODWORD(v64) = v64 + 1;
LABEL_14:
  while ( 1 )
  {
    v13 = *(_QWORD *)(v13 + 32);
    if ( !v13 )
      break;
    v14 = *(_QWORD *)(v13 + 16);
    if ( v15 != v14 )
      goto LABEL_11;
  }
LABEL_15:
  v16 = *(__int64 **)(v6 + 64);
  v57[1] = 0x200000000LL;
  v60 = 0x200000000LL;
  v17 = *(unsigned int *)(v6 + 72);
  v57[0] = (unsigned __int64)v58;
  v18 = (__int64)&v16[v17];
  v59 = v61;
  v62 = 0;
  if ( v16 != (__int64 *)v18 )
  {
    v54 = v7;
    v19 = &v16[v17];
    v20 = v16;
    do
    {
      if ( (*(_QWORD *)(*v20 + 8) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        *((_QWORD *)&v52 + 1) = *(_QWORD *)(*v20 + 8) & 0xFFFFFFFFFFFFFFF8LL | 6;
        *(_QWORD *)&v52 = *(_QWORD *)(*v20 + 8);
        sub_2E0F080((__int64)v57, a2, *v20, *((__int64 *)&v52 + 1), v18, a6, v52, *v20);
      }
      ++v20;
    }
    while ( v19 != v20 );
    v7 = v54;
  }
  sub_2E123D0(v7, (__int64)v57, (__int64)&v63, v11, 0, 0);
  sub_2E16070(v6, (__int64)v57, v21, v22, v23, v24);
  v25 = sub_2E11C60(v7, v6, a3);
  v26 = v62;
  v27 = v25;
  if ( v62 )
  {
    v28 = *(_QWORD *)(v62 + 16);
    while ( v28 )
    {
      sub_2E10270(*(_QWORD *)(v28 + 24));
      v29 = v28;
      v28 = *(_QWORD *)(v28 + 16);
      j_j___libc_free_0(v29);
    }
    j_j___libc_free_0(v26);
  }
  if ( v59 != v61 )
    _libc_free((unsigned __int64)v59);
  if ( (_BYTE *)v57[0] != v58 )
    _libc_free(v57[0]);
  if ( v63 != v65 )
    _libc_free((unsigned __int64)v63);
  return v27;
}
