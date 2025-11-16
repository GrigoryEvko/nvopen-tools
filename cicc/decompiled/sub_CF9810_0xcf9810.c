// Function: sub_CF9810
// Address: 0xcf9810
//
__int64 __fastcall sub_CF9810(__int64 a1, _BYTE **a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v5; // r13
  const void *v6; // r14
  _BYTE *v7; // r15
  unsigned __int64 v8; // rcx
  size_t v9; // rdx
  int v10; // eax
  __int64 v11; // rcx
  unsigned __int64 v12; // r13
  size_t v13; // rdx
  int v14; // eax
  unsigned int v15; // r13d
  unsigned int v16; // esi
  __int64 v17; // r8
  __int64 v18; // rdi
  unsigned int *v19; // rdx
  int v20; // r11d
  __int64 v21; // rcx
  unsigned int *v22; // rax
  __int64 v23; // r9
  unsigned int *v24; // rbx
  _BYTE *v25; // rdi
  _QWORD *v27; // r15
  __int64 v28; // rax
  __int64 v29; // r13
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // r15
  __int64 v33; // r13
  __int64 v34; // rdi
  __int64 *v35; // rdi
  int v36; // eax
  int v37; // esi
  unsigned int v38; // eax
  int v39; // ecx
  int v40; // edi
  int v41; // r10d
  __int64 v42; // rdi
  __int64 v43; // rdi
  size_t v44; // r15
  size_t v45; // rcx
  size_t v46; // rdx
  int v47; // eax
  unsigned int v48; // edi
  __int64 v49; // r15
  int v50; // eax
  int v51; // eax
  int v52; // eax
  __int64 v53; // r12
  int v54; // edi
  int v55; // esi
  __int64 v57; // [rsp+10h] [rbp-C0h]
  unsigned __int64 v58; // [rsp+18h] [rbp-B8h]
  __int64 v59; // [rsp+18h] [rbp-B8h]
  size_t v60; // [rsp+18h] [rbp-B8h]
  __int64 src; // [rsp+48h] [rbp-88h]
  char *v62[2]; // [rsp+58h] [rbp-78h] BYREF
  _BYTE v63[104]; // [rsp+68h] [rbp-68h] BYREF

  v3 = a1 + 40;
  v5 = *(_QWORD *)(a1 + 48);
  v57 = a1 + 40;
  if ( !v5 )
  {
    v3 = a1 + 40;
    goto LABEL_30;
  }
  v6 = *a2;
  v7 = a2[1];
  do
  {
    while ( 1 )
    {
      v8 = *(_QWORD *)(v5 + 40);
      v9 = (size_t)v7;
      if ( v8 <= (unsigned __int64)v7 )
        v9 = *(_QWORD *)(v5 + 40);
      if ( v9 )
      {
        v58 = *(_QWORD *)(v5 + 40);
        v10 = memcmp(*(const void **)(v5 + 32), v6, v9);
        v8 = v58;
        if ( v10 )
          break;
      }
      v11 = v8 - (_QWORD)v7;
      if ( v11 >= 0x80000000LL )
        goto LABEL_12;
      if ( v11 > (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
        v10 = v11;
        break;
      }
LABEL_3:
      v5 = *(_QWORD *)(v5 + 24);
      if ( !v5 )
        goto LABEL_13;
    }
    if ( v10 < 0 )
      goto LABEL_3;
LABEL_12:
    v3 = v5;
    v5 = *(_QWORD *)(v5 + 16);
  }
  while ( v5 );
LABEL_13:
  if ( v57 == v3 )
    goto LABEL_30;
  v12 = *(_QWORD *)(v3 + 40);
  v13 = (size_t)v7;
  if ( v12 <= (unsigned __int64)v7 )
    v13 = *(_QWORD *)(v3 + 40);
  if ( v13 )
  {
    v14 = memcmp(v6, *(const void **)(v3 + 32), v13);
    if ( v14 )
    {
LABEL_21:
      if ( v14 < 0 )
        goto LABEL_30;
LABEL_22:
      v15 = *(_DWORD *)(v3 + 64);
      if ( !v15 )
        goto LABEL_35;
LABEL_23:
      v16 = *(_DWORD *)(a1 + 24);
      if ( !v16 )
        goto LABEL_40;
      goto LABEL_24;
    }
  }
  if ( (__int64)&v7[-v12] > 0x7FFFFFFF )
    goto LABEL_22;
  if ( (__int64)&v7[-v12] >= (__int64)0xFFFFFFFF80000000LL )
  {
    v14 = (_DWORD)v7 - v12;
    goto LABEL_21;
  }
LABEL_30:
  v27 = (_QWORD *)v3;
  v28 = sub_22077B0(72);
  v29 = v28 + 32;
  v3 = v28;
  *(_QWORD *)(v28 + 32) = v28 + 48;
  v59 = v28 + 48;
  sub_CF9030((__int64 *)(v28 + 32), *a2, (__int64)&a2[1][(_QWORD)*a2]);
  *(_DWORD *)(v3 + 64) = 0;
  v30 = sub_A288A0((_QWORD *)(a1 + 32), v27, v29);
  v32 = v30;
  v33 = v31;
  if ( !v31 )
  {
    v42 = *(_QWORD *)(v3 + 32);
    if ( v59 != v42 )
      j_j___libc_free_0(v42, *(_QWORD *)(v3 + 48) + 1LL);
    v43 = v3;
    v3 = v32;
    j_j___libc_free_0(v43, 72);
    goto LABEL_22;
  }
  if ( v30 || v57 == v31 )
  {
LABEL_33:
    v34 = 1;
    goto LABEL_34;
  }
  v44 = *(_QWORD *)(v3 + 40);
  v46 = *(_QWORD *)(v31 + 40);
  v45 = v46;
  if ( v44 <= v46 )
    v46 = *(_QWORD *)(v3 + 40);
  if ( v46
    && (v60 = v45, v47 = memcmp(*(const void **)(v3 + 32), *(const void **)(v33 + 32), v46), v45 = v60, (v48 = v47) != 0) )
  {
LABEL_55:
    v34 = v48 >> 31;
  }
  else
  {
    v49 = v44 - v45;
    v34 = 0;
    if ( v49 <= 0x7FFFFFFF )
    {
      if ( v49 < (__int64)0xFFFFFFFF80000000LL )
        goto LABEL_33;
      v48 = v49;
      goto LABEL_55;
    }
  }
LABEL_34:
  sub_220F040(v34, v3, v33, v57);
  ++*(_QWORD *)(a1 + 72);
  v15 = *(_DWORD *)(v3 + 64);
  if ( v15 )
    goto LABEL_23;
LABEL_35:
  *(_DWORD *)(v3 + 64) = ((__int64)(*(_QWORD *)(a1 + 88) - *(_QWORD *)(a1 + 80)) >> 5) + 1;
  v35 = *(__int64 **)(a1 + 88);
  if ( v35 == *(__int64 **)(a1 + 96) )
  {
    sub_8FD760((__m128i **)(a1 + 80), *(const __m128i **)(a1 + 88), (__int64)a2);
  }
  else
  {
    if ( v35 )
    {
      *v35 = (__int64)(v35 + 2);
      sub_CF9030(v35, *a2, (__int64)&a2[1][(_QWORD)*a2]);
      v35 = *(__int64 **)(a1 + 88);
    }
    *(_QWORD *)(a1 + 88) = v35 + 4;
  }
  v16 = *(_DWORD *)(a1 + 24);
  v15 = *(_DWORD *)(v3 + 64);
  if ( !v16 )
  {
LABEL_40:
    ++*(_QWORD *)a1;
    goto LABEL_41;
  }
LABEL_24:
  v17 = v16 - 1;
  v18 = *(_QWORD *)(a1 + 8);
  v19 = 0;
  v20 = 1;
  v21 = (unsigned int)v17 & (37 * v15);
  v22 = (unsigned int *)(v18 + (v21 << 7));
  v23 = *v22;
  if ( v15 == (_DWORD)v23 )
  {
LABEL_25:
    v24 = v22 + 2;
    goto LABEL_26;
  }
  while ( (_DWORD)v23 != -1 )
  {
    if ( (_DWORD)v23 == -2 && !v19 )
      v19 = v22;
    v21 = (unsigned int)v17 & (v20 + (_DWORD)v21);
    v22 = (unsigned int *)(v18 + ((unsigned __int64)(unsigned int)v21 << 7));
    v23 = *v22;
    if ( v15 == (_DWORD)v23 )
      goto LABEL_25;
    ++v20;
  }
  if ( !v19 )
    v19 = v22;
  v50 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v39 = v50 + 1;
  if ( 4 * (v50 + 1) < 3 * v16 )
  {
    if ( v16 - *(_DWORD *)(a1 + 20) - v39 > v16 >> 3 )
      goto LABEL_69;
    sub_C61E30(a1, v16);
    v51 = *(_DWORD *)(a1 + 24);
    if ( v51 )
    {
      v52 = v51 - 1;
      v17 = *(_QWORD *)(a1 + 8);
      v23 = 0;
      LODWORD(v53) = v52 & (37 * v15);
      v39 = *(_DWORD *)(a1 + 16) + 1;
      v54 = 1;
      v19 = (unsigned int *)(v17 + ((unsigned __int64)(unsigned int)v53 << 7));
      v55 = *v19;
      if ( v15 == *v19 )
        goto LABEL_69;
      while ( v55 != -1 )
      {
        if ( !v23 && v55 == -2 )
          v23 = (__int64)v19;
        v53 = v52 & (unsigned int)(v53 + v54);
        v19 = (unsigned int *)(v17 + (v53 << 7));
        v55 = *v19;
        if ( v15 == *v19 )
          goto LABEL_69;
        ++v54;
      }
LABEL_45:
      if ( v23 )
        v19 = (unsigned int *)v23;
      goto LABEL_69;
    }
LABEL_86:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_41:
  sub_C61E30(a1, 2 * v16);
  v36 = *(_DWORD *)(a1 + 24);
  if ( !v36 )
    goto LABEL_86;
  v37 = v36 - 1;
  v17 = *(_QWORD *)(a1 + 8);
  v38 = (v36 - 1) & (37 * v15);
  v39 = *(_DWORD *)(a1 + 16) + 1;
  v19 = (unsigned int *)(v17 + ((unsigned __int64)v38 << 7));
  v40 = *v19;
  if ( v15 != *v19 )
  {
    v41 = 1;
    v23 = 0;
    while ( v40 != -1 )
    {
      if ( v40 == -2 && !v23 )
        v23 = (__int64)v19;
      v38 = v37 & (v41 + v38);
      v19 = (unsigned int *)(v17 + ((unsigned __int64)v38 << 7));
      v40 = *v19;
      if ( v15 == *v19 )
        goto LABEL_69;
      ++v41;
    }
    goto LABEL_45;
  }
LABEL_69:
  *(_DWORD *)(a1 + 16) = v39;
  if ( *v19 != -1 )
    --*(_DWORD *)(a1 + 20);
  *v19 = v15;
  memset(v19 + 2, 0, 0x78u);
  v21 = (__int64)(v19 + 20);
  *((_QWORD *)v19 + 9) = 0x300000000LL;
  *((_QWORD *)v19 + 4) = v19 + 12;
  v24 = v19 + 2;
  *((_QWORD *)v19 + 8) = v19 + 20;
LABEL_26:
  LOBYTE(src) = 0;
  v62[0] = v63;
  v62[1] = (char *)0x300000000LL;
  v25 = (_BYTE *)*((_QWORD *)v24 + 3);
  *(_QWORD *)v24 = 0;
  *((_QWORD *)v24 + 1) = 0;
  *((_BYTE *)v24 + 16) = 0;
  *((_QWORD *)v24 + 4) = 0;
  *v25 = 0;
  LOBYTE(src) = 0;
  sub_CF8ED0((__int64)(v24 + 14), v62, 0, v21, v17, v23);
  if ( v62[0] != v63 )
    _libc_free(v62[0], v62);
  sub_2240AE0(v24 + 6, a3);
  return v15;
}
