// Function: sub_2E16290
// Address: 0x2e16290
//
void __fastcall sub_2E16290(_QWORD *a1, __int64 a2, int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rdx
  __int64 v10; // rbx
  char v11; // al
  __int64 *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 *v15; // rbx
  __int64 *v16; // r12
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 *v21; // r15
  __int64 v22; // r12
  __int64 v23; // r14
  const __m128i *v24; // rsi
  __int64 v25; // rcx
  unsigned __int64 v26; // r14
  unsigned __int64 v27; // r13
  unsigned __int64 v28; // rdi
  __int64 v29; // rdi
  unsigned __int16 v30; // ax
  unsigned __int64 v31; // rcx
  __int64 v32; // r8
  unsigned __int64 i; // rax
  __int64 j; // rsi
  __int16 v35; // dx
  __int64 v36; // rsi
  unsigned int v37; // ecx
  __int64 *v38; // rdx
  __int64 v39; // r10
  unsigned __int64 v40; // r12
  __int64 v41; // r14
  __int64 *v42; // rdx
  __int64 v43; // rdi
  unsigned int v44; // esi
  unsigned int v45; // ecx
  __int64 v46; // r12
  __int64 v47; // rax
  __int64 v48; // rax
  unsigned __int64 v49; // rdx
  __int64 *v50; // rax
  int v51; // edx
  int v52; // r11d
  __int128 v53; // [rsp-20h] [rbp-220h]
  __int64 *k; // [rsp+28h] [rbp-1D8h]
  __int64 v56; // [rsp+28h] [rbp-1D8h]
  __m128i v57; // [rsp+30h] [rbp-1D0h]
  unsigned __int64 v58[2]; // [rsp+50h] [rbp-1B0h] BYREF
  _BYTE v59[48]; // [rsp+60h] [rbp-1A0h] BYREF
  _BYTE *v60; // [rsp+90h] [rbp-170h]
  __int64 v61; // [rsp+98h] [rbp-168h]
  _BYTE v62[16]; // [rsp+A0h] [rbp-160h] BYREF
  unsigned __int64 v63; // [rsp+B0h] [rbp-150h]
  _BYTE *v64; // [rsp+C0h] [rbp-140h] BYREF
  __int64 v65; // [rsp+C8h] [rbp-138h]
  _BYTE v66[304]; // [rsp+D0h] [rbp-130h] BYREF

  v64 = v66;
  v9 = a1[1];
  v65 = 0x1000000000LL;
  if ( a3 < 0 )
    v10 = *(_QWORD *)(*(_QWORD *)(v9 + 56) + 16LL * (a3 & 0x7FFFFFFF) + 8);
  else
    v10 = *(_QWORD *)(*(_QWORD *)(v9 + 304) + 8LL * (unsigned int)a3);
  while ( 1 )
  {
    if ( !v10 )
      goto LABEL_7;
    if ( (*(_BYTE *)(v10 + 3) & 0x10) == 0 )
    {
      v11 = *(_BYTE *)(v10 + 4);
      if ( (v11 & 8) == 0 )
        break;
    }
    v10 = *(_QWORD *)(v10 + 32);
  }
  v29 = 0;
LABEL_32:
  if ( (v11 & 1) != 0 || (v11 & 2) != 0 )
    goto LABEL_37;
  v30 = (*(_DWORD *)v10 >> 8) & 0xFFF;
  if ( (*(_BYTE *)(v10 + 3) & 0x10) != 0 )
  {
    if ( !v30 )
      goto LABEL_37;
  }
  else if ( !v30 )
  {
    goto LABEL_42;
  }
  if ( (*(_OWORD *)(a2 + 112) & *(_OWORD *)(*(_QWORD *)(a1[2] + 272LL) + 16LL * v30)) == 0 )
    goto LABEL_37;
LABEL_42:
  v31 = *(_QWORD *)(v10 + 16);
  v32 = a1[4];
  for ( i = v31; (*(_BYTE *)(i + 44) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
    ;
  for ( ; (*(_BYTE *)(v31 + 44) & 8) != 0; v31 = *(_QWORD *)(v31 + 8) )
    ;
  for ( j = *(_QWORD *)(v31 + 8); j != i; i = *(_QWORD *)(i + 8) )
  {
    v35 = *(_WORD *)(i + 68);
    if ( (unsigned __int16)(v35 - 14) > 4u && v35 != 24 )
      break;
  }
  v36 = *(_QWORD *)(v32 + 128);
  a5 = *(unsigned int *)(v32 + 144);
  if ( (_DWORD)a5 )
  {
    a6 = (unsigned int)(a5 - 1);
    v37 = a6 & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
    v38 = (__int64 *)(v36 + 16LL * v37);
    v39 = *v38;
    if ( *v38 == i )
      goto LABEL_52;
    v51 = 1;
    while ( v39 != -4096 )
    {
      v52 = v51 + 1;
      v37 = a6 & (v51 + v37);
      v38 = (__int64 *)(v36 + 16LL * v37);
      v39 = *v38;
      if ( *v38 == i )
        goto LABEL_52;
      v51 = v52;
    }
  }
  v38 = (__int64 *)(v36 + 16LL * (unsigned int)a5);
LABEL_52:
  v40 = v38[1] & 0xFFFFFFFFFFFFFFF8LL;
  v41 = v40 | 4;
  if ( (v40 | 4) == v29 )
    goto LABEL_37;
  v42 = (__int64 *)sub_2E09D00((__int64 *)a2, v38[1] & 0xFFFFFFFFFFFFFFF8LL);
  v43 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
  if ( v42 == (__int64 *)v43 )
    goto LABEL_65;
  v44 = *(_DWORD *)(v40 + 24);
  v45 = *(_DWORD *)((*v42 & 0xFFFFFFFFFFFFFFF8LL) + 24);
  if ( (unsigned __int64)(v45 | (*v42 >> 1) & 3) > v44 )
    goto LABEL_65;
  a5 = v42[2];
  if ( v40 == (v42[1] & 0xFFFFFFFFFFFFFFF8LL) )
  {
    if ( (__int64 *)v43 == v42 + 3 )
    {
      if ( a5 )
        goto LABEL_72;
LABEL_65:
      v29 = v40 | 4;
      goto LABEL_37;
    }
    v45 = *(_DWORD *)((v42[3] & 0xFFFFFFFFFFFFFFF8LL) + 24);
    v42 += 3;
  }
  if ( v40 == *(_QWORD *)(a5 + 8) )
    goto LABEL_65;
  v46 = v40 | 4;
  if ( v44 < v45 )
    goto LABEL_61;
  v47 = v42[2];
  if ( v47 && a5 != v47 )
  {
    v46 = *(_QWORD *)(v47 + 8);
    goto LABEL_61;
  }
LABEL_72:
  v46 = v41;
LABEL_61:
  v48 = (unsigned int)v65;
  v49 = (unsigned int)v65 + 1LL;
  if ( v49 > HIDWORD(v65) )
  {
    v56 = a5;
    sub_C8D5F0((__int64)&v64, v66, v49, 0x10u, a5, a6);
    v48 = (unsigned int)v65;
    a5 = v56;
  }
  v50 = (__int64 *)&v64[16 * v48];
  v29 = v41;
  *v50 = v46;
  v50[1] = a5;
  LODWORD(v65) = v65 + 1;
LABEL_37:
  while ( 1 )
  {
    v10 = *(_QWORD *)(v10 + 32);
    if ( !v10 )
      break;
    if ( (*(_BYTE *)(v10 + 3) & 0x10) == 0 )
    {
      v11 = *(_BYTE *)(v10 + 4);
      if ( (v11 & 8) == 0 )
        goto LABEL_32;
    }
  }
LABEL_7:
  v12 = *(__int64 **)(a2 + 64);
  v58[1] = 0x200000000LL;
  v61 = 0x200000000LL;
  v13 = *(unsigned int *)(a2 + 72);
  v58[0] = (unsigned __int64)v59;
  v14 = (__int64)&v12[v13];
  v60 = v62;
  v63 = 0;
  if ( v12 != (__int64 *)v14 )
  {
    v15 = v12;
    v16 = &v12[v13];
    do
    {
      if ( (*(_QWORD *)(*v15 + 8) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        *((_QWORD *)&v53 + 1) = *(_QWORD *)(*v15 + 8) & 0xFFFFFFFFFFFFFFF8LL | 6;
        *(_QWORD *)&v53 = *(_QWORD *)(*v15 + 8);
        sub_2E0F080((__int64)v58, *((__int64 *)&v53 + 1), *v15, v14, a5, a6, v53, *v15);
      }
      ++v15;
    }
    while ( v16 != v15 );
  }
  sub_2E123D0((__int64)a1, (__int64)v58, (__int64)&v64, a3, *(_QWORD *)(a2 + 112), *(_QWORD *)(a2 + 120));
  sub_2E16070(a2, (__int64)v58, v17, v18, v19, v20);
  v21 = *(__int64 **)(a2 + 64);
  for ( k = &v21[*(unsigned int *)(a2 + 72)]; k != v21; ++v21 )
  {
    v22 = *v21;
    v23 = *(_QWORD *)(*v21 + 8);
    if ( (v23 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v24 = (const __m128i *)sub_2E09D00((__int64 *)a2, *(_QWORD *)(*v21 + 8));
      if ( v24 == (const __m128i *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8))
        || (*(_DWORD *)((v24->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v24->m128i_i64[0] >> 1) & 3) > (*(_DWORD *)((v23 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v23 >> 1) & 3) )
      {
        BUG();
      }
      v25 = *(_QWORD *)(v22 + 8);
      if ( v24->m128i_i64[1] == (v25 & 0xFFFFFFFFFFFFFFF8LL | 6) && (v25 & 6) == 0 )
      {
        *(_QWORD *)(v22 + 8) = 0;
        v57 = _mm_loadu_si128(v24);
        sub_2E0C3B0(a2, v57.m128i_i64[0], v57.m128i_i64[1], 0);
      }
    }
  }
  v26 = v63;
  if ( v63 )
  {
    v27 = *(_QWORD *)(v63 + 16);
    while ( v27 )
    {
      sub_2E10270(*(_QWORD *)(v27 + 24));
      v28 = v27;
      v27 = *(_QWORD *)(v27 + 16);
      j_j___libc_free_0(v28);
    }
    j_j___libc_free_0(v26);
  }
  if ( v60 != v62 )
    _libc_free((unsigned __int64)v60);
  if ( (_BYTE *)v58[0] != v59 )
    _libc_free(v58[0]);
  if ( v64 != v66 )
    _libc_free((unsigned __int64)v64);
}
