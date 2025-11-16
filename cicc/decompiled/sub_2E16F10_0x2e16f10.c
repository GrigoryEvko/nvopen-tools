// Function: sub_2E16F10
// Address: 0x2e16f10
//
unsigned __int64 *__fastcall sub_2E16F10(unsigned __int64 *a1, __int64 a2, unsigned int a3, unsigned __int64 a4)
{
  __int64 v4; // r8
  unsigned int v5; // eax
  __int64 v9; // r14
  unsigned __int64 v10; // rdx
  __int64 v11; // r12
  __int64 v12; // r8
  unsigned __int64 v13; // r14
  unsigned __int64 i; // rax
  unsigned __int64 j; // rdx
  __int64 k; // rsi
  __int16 v17; // dx
  __int64 v18; // rsi
  __int64 v19; // rdi
  unsigned int v20; // ecx
  __int64 *v21; // rdx
  __int64 v22; // r10
  __int64 v23; // r9
  __int64 v24; // rdx
  int v25; // r10d
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // r9
  unsigned __int64 v28; // r8
  __int64 v29; // rdx
  unsigned __int64 v30; // rcx
  __int64 v31; // r9
  unsigned __int64 v32; // rdi
  __int64 v33; // rdx
  __int64 m; // rsi
  int v35; // eax
  __int64 v36; // rsi
  __int64 v37; // r9
  __int64 *v38; // rax
  __int64 v39; // r11
  __int64 v40; // rax
  unsigned __int64 v41; // rax
  unsigned int v43; // eax
  __int64 v44; // rcx
  __int64 *v45; // r14
  __int64 v46; // rax
  int v47; // eax
  int v48; // edx
  __int64 v49; // r12
  __int64 v50; // r9
  _QWORD *v51; // rax
  _QWORD *v52; // rsi
  int v53; // ebx
  int v54; // r9d
  __int128 v55; // [rsp-20h] [rbp-60h]
  __int64 v56; // [rsp-10h] [rbp-50h]
  int v57; // [rsp+0h] [rbp-40h]
  unsigned __int64 v58; // [rsp+0h] [rbp-40h]
  int v59; // [rsp+0h] [rbp-40h]
  unsigned __int64 v60; // [rsp+8h] [rbp-38h]
  unsigned __int64 v61; // [rsp+8h] [rbp-38h]
  __int64 v62; // [rsp+8h] [rbp-38h]

  v4 = a3;
  v5 = a3 & 0x7FFFFFFF;
  v9 = 8LL * (a3 & 0x7FFFFFFF);
  v10 = *(unsigned int *)(a2 + 160);
  if ( v5 >= (unsigned int)v10 || (v11 = *(_QWORD *)(*(_QWORD *)(a2 + 152) + 8LL * v5)) == 0 )
  {
    v43 = v5 + 1;
    if ( (unsigned int)v10 < v43 && v43 != v10 )
    {
      if ( v43 >= v10 )
      {
        v49 = *(_QWORD *)(a2 + 168);
        v50 = v43 - v10;
        if ( v43 > (unsigned __int64)*(unsigned int *)(a2 + 164) )
        {
          v59 = v4;
          v62 = v43 - v10;
          sub_C8D5F0(a2 + 152, (const void *)(a2 + 168), v43, 8u, v4, v50);
          v10 = *(unsigned int *)(a2 + 160);
          LODWORD(v4) = v59;
          v50 = v62;
        }
        v44 = *(_QWORD *)(a2 + 152);
        v51 = (_QWORD *)(v44 + 8 * v10);
        v52 = &v51[v50];
        if ( v51 != v52 )
        {
          do
            *v51++ = v49;
          while ( v52 != v51 );
          LODWORD(v10) = *(_DWORD *)(a2 + 160);
          v44 = *(_QWORD *)(a2 + 152);
        }
        *(_DWORD *)(a2 + 160) = v50 + v10;
        goto LABEL_32;
      }
      *(_DWORD *)(a2 + 160) = v43;
    }
    v44 = *(_QWORD *)(a2 + 152);
LABEL_32:
    v45 = (__int64 *)(v44 + v9);
    v46 = sub_2E10F30(v4);
    *v45 = v46;
    v11 = v46;
  }
  v12 = *(_QWORD *)(a2 + 32);
  v13 = a4;
  for ( i = a4; (*(_BYTE *)(i + 44) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
    ;
  for ( j = a4; (*(_BYTE *)(j + 44) & 8) != 0; j = *(_QWORD *)(j + 8) )
    ;
  for ( k = *(_QWORD *)(j + 8); k != i; i = *(_QWORD *)(i + 8) )
  {
    v17 = *(_WORD *)(i + 68);
    if ( (unsigned __int16)(v17 - 14) > 4u && v17 != 24 )
      break;
  }
  v18 = *(unsigned int *)(v12 + 144);
  v19 = *(_QWORD *)(v12 + 128);
  if ( !(_DWORD)v18 )
  {
LABEL_41:
    v21 = (__int64 *)(v19 + 16 * v18);
    goto LABEL_13;
  }
  v20 = (v18 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
  v21 = (__int64 *)(v19 + 16LL * v20);
  v22 = *v21;
  if ( i != *v21 )
  {
    v48 = 1;
    while ( v22 != -4096 )
    {
      v54 = v48 + 1;
      v20 = (v18 - 1) & (v48 + v20);
      v21 = (__int64 *)(v19 + 16LL * v20);
      v22 = *v21;
      if ( *v21 == i )
        goto LABEL_13;
      v48 = v54;
    }
    goto LABEL_41;
  }
LABEL_13:
  v23 = v21[1];
  v24 = *(_QWORD *)(a2 + 56);
  v25 = *(_DWORD *)(v11 + 72);
  *(_QWORD *)(a2 + 136) += 16LL;
  v26 = (v24 + 15) & 0xFFFFFFFFFFFFFFF0LL;
  v27 = v23 & 0xFFFFFFFFFFFFFFF8LL | 4;
  if ( *(_QWORD *)(a2 + 64) >= v26 + 16 && v24 )
  {
    *(_QWORD *)(a2 + 56) = v26 + 16;
    v28 = (v24 + 15) & 0xFFFFFFFFFFFFFFF0LL;
    if ( !v26 )
      goto LABEL_17;
  }
  else
  {
    v57 = v25;
    v60 = v27;
    v26 = sub_9D1E70(a2 + 56, 16, 16, 4);
    v25 = v57;
    v27 = v60;
    v28 = v26;
  }
  *(_DWORD *)v28 = v25;
  *(_QWORD *)(v28 + 8) = v27;
LABEL_17:
  v29 = *(unsigned int *)(v11 + 72);
  if ( v29 + 1 > (unsigned __int64)*(unsigned int *)(v11 + 76) )
  {
    v58 = v28;
    v61 = v26;
    sub_C8D5F0(v11 + 64, (const void *)(v11 + 80), v29 + 1, 8u, v28, v29 + 1);
    v29 = *(unsigned int *)(v11 + 72);
    v28 = v58;
    v26 = v61;
  }
  *(_QWORD *)(*(_QWORD *)(v11 + 64) + 8 * v29) = v26;
  v30 = a4;
  ++*(_DWORD *)(v11 + 72);
  v31 = *(_QWORD *)(a2 + 32);
  v32 = *(_QWORD *)(*(_QWORD *)(v31 + 152) + 16LL * *(unsigned int *)(*(_QWORD *)(a4 + 24) + 24LL) + 8);
  if ( (*(_DWORD *)(a4 + 44) & 4) != 0 )
  {
    do
      v30 = *(_QWORD *)v30 & 0xFFFFFFFFFFFFFFF8LL;
    while ( (*(_BYTE *)(v30 + 44) & 4) != 0 );
  }
  v33 = *(_DWORD *)(a4 + 44) & 8;
  if ( (_DWORD)v33 )
  {
    do
      v13 = *(_QWORD *)(v13 + 8);
    while ( (*(_BYTE *)(v13 + 44) & 8) != 0 );
  }
  for ( m = *(_QWORD *)(v13 + 8); m != v30; v30 = *(_QWORD *)(v30 + 8) )
  {
    v35 = *(unsigned __int16 *)(v30 + 68);
    v33 = (unsigned int)(v35 - 14);
    if ( (unsigned __int16)(v35 - 14) > 4u && (_WORD)v35 != 24 )
      break;
  }
  v36 = *(_QWORD *)(v31 + 128);
  v37 = *(unsigned int *)(v31 + 144);
  if ( !(_DWORD)v37 )
  {
LABEL_38:
    v38 = (__int64 *)(v36 + 16LL * (unsigned int)v37);
    goto LABEL_29;
  }
  v33 = ((_DWORD)v37 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
  v38 = (__int64 *)(v36 + 16 * v33);
  v39 = *v38;
  if ( *v38 != v30 )
  {
    v47 = 1;
    while ( v39 != -4096 )
    {
      v53 = v47 + 1;
      v33 = ((_DWORD)v37 - 1) & (unsigned int)(v47 + v33);
      v38 = (__int64 *)(v36 + 16LL * (unsigned int)v33);
      v39 = *v38;
      if ( *v38 == v30 )
        goto LABEL_29;
      v47 = v53;
    }
    goto LABEL_38;
  }
LABEL_29:
  v40 = v38[1];
  a1[1] = v32;
  a1[2] = v28;
  v56 = a1[2];
  v41 = v40 & 0xFFFFFFFFFFFFFFF8LL | 4;
  *((_QWORD *)&v55 + 1) = a1[1];
  *a1 = v41;
  *(_QWORD *)&v55 = v41;
  sub_2E0F080(v11, v36, v33, v30, v28, v37, v55, v56);
  return a1;
}
