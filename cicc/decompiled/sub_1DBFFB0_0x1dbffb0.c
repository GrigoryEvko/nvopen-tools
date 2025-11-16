// Function: sub_1DBFFB0
// Address: 0x1dbffb0
//
void __fastcall sub_1DBFFB0(_QWORD *a1, __int64 a2, int a3, __int64 a4, __int64 a5, __int64 a6)
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
  int v19; // r8d
  int v20; // r9d
  __int64 *v21; // r8
  __int64 *v22; // r13
  __int64 v23; // r15
  __int64 v24; // r14
  const __m128i *v25; // rsi
  __int64 v26; // rcx
  __int64 v27; // r14
  __int64 v28; // r12
  __int64 v29; // rdi
  __int64 v30; // rdi
  unsigned __int16 v31; // ax
  __int64 v32; // rcx
  unsigned __int64 i; // rdx
  __int64 v34; // rsi
  __int64 v35; // rcx
  __int64 *v36; // rax
  __int64 v37; // r10
  unsigned __int64 v38; // r14
  __int64 v39; // r15
  __int64 *v40; // rdx
  __int64 v41; // rdi
  unsigned int v42; // esi
  unsigned int v43; // ecx
  __int64 v44; // r14
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 *v47; // rax
  int v48; // eax
  __int64 v49; // rax
  int v50; // r11d
  __int64 v51; // rax
  __int128 v52; // [rsp-20h] [rbp-220h]
  __int64 v54; // [rsp+20h] [rbp-1E0h]
  __int64 *v55; // [rsp+28h] [rbp-1D8h]
  __int64 v56; // [rsp+28h] [rbp-1D8h]
  __m128i v57; // [rsp+30h] [rbp-1D0h]
  unsigned __int64 v58[2]; // [rsp+50h] [rbp-1B0h] BYREF
  _BYTE v59[48]; // [rsp+60h] [rbp-1A0h] BYREF
  _BYTE *v60; // [rsp+90h] [rbp-170h]
  __int64 v61; // [rsp+98h] [rbp-168h]
  _BYTE v62[16]; // [rsp+A0h] [rbp-160h] BYREF
  __int64 v63; // [rsp+B0h] [rbp-150h]
  _BYTE *v64; // [rsp+C0h] [rbp-140h] BYREF
  __int64 v65; // [rsp+C8h] [rbp-138h]
  _BYTE v66[304]; // [rsp+D0h] [rbp-130h] BYREF

  v9 = a1[30];
  v64 = v66;
  v65 = 0x1000000000LL;
  if ( a3 < 0 )
    v10 = *(_QWORD *)(*(_QWORD *)(v9 + 24) + 16LL * (a3 & 0x7FFFFFFF) + 8);
  else
    v10 = *(_QWORD *)(*(_QWORD *)(v9 + 272) + 8LL * (unsigned int)a3);
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
  v30 = 0;
LABEL_34:
  if ( (v11 & 1) != 0 || (v11 & 2) != 0 )
    goto LABEL_55;
  v31 = (*(_DWORD *)v10 >> 8) & 0xFFF;
  if ( (*(_BYTE *)(v10 + 3) & 0x10) != 0 )
  {
    if ( !v31 )
      goto LABEL_55;
  }
  else if ( !v31 )
  {
    goto LABEL_39;
  }
  if ( (*(_DWORD *)(a2 + 112) & *(_DWORD *)(*(_QWORD *)(a1[31] + 248LL) + 4LL * v31)) == 0 )
    goto LABEL_55;
LABEL_39:
  v32 = a1[34];
  for ( i = *(_QWORD *)(v10 + 16); (*(_BYTE *)(i + 46) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
    ;
  v34 = *(_QWORD *)(v32 + 368);
  v35 = *(unsigned int *)(v32 + 384);
  if ( (_DWORD)v35 )
  {
    a6 = (unsigned int)(v35 - 1);
    a5 = (unsigned int)a6 & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
    v36 = (__int64 *)(v34 + 16 * a5);
    v37 = *v36;
    if ( i == *v36 )
      goto LABEL_43;
    v48 = 1;
    while ( v37 != -8 )
    {
      v50 = v48 + 1;
      v51 = (unsigned int)a6 & ((_DWORD)a5 + v48);
      a5 = (unsigned int)v51;
      v36 = (__int64 *)(v34 + 16 * v51);
      v37 = *v36;
      if ( *v36 == i )
        goto LABEL_43;
      v48 = v50;
    }
  }
  v36 = (__int64 *)(v34 + 16 * v35);
LABEL_43:
  v38 = v36[1] & 0xFFFFFFFFFFFFFFF8LL;
  v39 = v38 | 4;
  if ( (v38 | 4) == v30 )
    goto LABEL_55;
  v40 = (__int64 *)sub_1DB3C70((__int64 *)a2, v36[1] & 0xFFFFFFFFFFFFFFF8LL);
  v41 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
  if ( v40 == (__int64 *)v41 )
    goto LABEL_61;
  v42 = *(_DWORD *)(v38 + 24);
  v43 = *(_DWORD *)((*v40 & 0xFFFFFFFFFFFFFFF8LL) + 24);
  if ( (unsigned __int64)(v43 | (*v40 >> 1) & 3) > v42 )
    goto LABEL_61;
  a5 = v40[2];
  if ( v38 == (v40[1] & 0xFFFFFFFFFFFFFFF8LL) )
  {
    a6 = (__int64)(v40 + 3);
    if ( (__int64 *)v41 == v40 + 3 )
    {
      if ( a5 )
        goto LABEL_69;
LABEL_61:
      v30 = v38 | 4;
      goto LABEL_55;
    }
    v49 = v40[3];
    v40 += 3;
    v43 = *(_DWORD *)((v49 & 0xFFFFFFFFFFFFFFF8LL) + 24);
  }
  if ( v38 == *(_QWORD *)(a5 + 8) )
    goto LABEL_61;
  v44 = v38 | 4;
  if ( v43 > v42 )
    goto LABEL_52;
  v45 = v40[2];
  if ( a5 != v45 && v45 )
  {
    v44 = *(_QWORD *)(v45 + 8);
    goto LABEL_52;
  }
LABEL_69:
  v44 = v39;
LABEL_52:
  v46 = (unsigned int)v65;
  if ( (unsigned int)v65 >= HIDWORD(v65) )
  {
    v56 = a5;
    sub_16CD150((__int64)&v64, v66, 0, 16, a5, a6);
    v46 = (unsigned int)v65;
    a5 = v56;
  }
  v47 = (__int64 *)&v64[16 * v46];
  v30 = v39;
  *v47 = v44;
  v47[1] = a5;
  LODWORD(v65) = v65 + 1;
LABEL_55:
  while ( 1 )
  {
    v10 = *(_QWORD *)(v10 + 32);
    if ( !v10 )
      break;
    if ( (*(_BYTE *)(v10 + 3) & 0x10) == 0 )
    {
      v11 = *(_BYTE *)(v10 + 4);
      if ( (v11 & 8) == 0 )
        goto LABEL_34;
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
    v54 = a2;
    v16 = &v12[v13];
    do
    {
      if ( (*(_QWORD *)(*v15 + 8) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        *((_QWORD *)&v52 + 1) = *(_QWORD *)(*v15 + 8) & 0xFFFFFFFFFFFFFFF8LL | 6;
        *(_QWORD *)&v52 = *(_QWORD *)(*v15 + 8);
        sub_1DB8610((__int64)v58, *((__int64 *)&v52 + 1), *v15, v14, a5, a6, v52, *v15);
      }
      ++v15;
    }
    while ( v16 != v15 );
    a2 = v54;
  }
  sub_1DBB5C0((__int64)a1, (__int64)v58, (__int64)&v64, a3, *(_DWORD *)(a2 + 112), a6);
  sub_1DBFD90(a2, (__int64)v58, v17, v18, v19, v20);
  v21 = *(__int64 **)(a2 + 64);
  v55 = &v21[*(unsigned int *)(a2 + 72)];
  if ( v21 != v55 )
  {
    v22 = *(__int64 **)(a2 + 64);
    do
    {
      v23 = *v22;
      v24 = *(_QWORD *)(*v22 + 8);
      if ( (v24 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v25 = (const __m128i *)sub_1DB3C70((__int64 *)a2, *(_QWORD *)(*v22 + 8));
        if ( v25 == (const __m128i *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8))
          || (*(_DWORD *)((v25->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v25->m128i_i64[0] >> 1) & 3) > (*(_DWORD *)((v24 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v24 >> 1) & 3) )
        {
          BUG();
        }
        v26 = *(_QWORD *)(v23 + 8);
        if ( v25->m128i_i64[1] == (v26 & 0xFFFFFFFFFFFFFFF8LL | 6) && (v26 & 6) == 0 )
        {
          *(_QWORD *)(v23 + 8) = 0;
          v57 = _mm_loadu_si128(v25);
          sub_1DB4410(a2, v57.m128i_i64[0], v57.m128i_i64[1], 0);
        }
      }
      ++v22;
    }
    while ( v55 != v22 );
  }
  v27 = v63;
  if ( v63 )
  {
    v28 = *(_QWORD *)(v63 + 16);
    while ( v28 )
    {
      sub_1DB97B0(*(_QWORD *)(v28 + 24));
      v29 = v28;
      v28 = *(_QWORD *)(v28 + 16);
      j_j___libc_free_0(v29, 56);
    }
    j_j___libc_free_0(v27, 48);
  }
  if ( v60 != v62 )
    _libc_free((unsigned __int64)v60);
  if ( (_BYTE *)v58[0] != v59 )
    _libc_free(v58[0]);
  if ( v64 != v66 )
    _libc_free((unsigned __int64)v64);
}
