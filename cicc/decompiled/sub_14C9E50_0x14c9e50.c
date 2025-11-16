// Function: sub_14C9E50
// Address: 0x14c9e50
//
__int64 __fastcall sub_14C9E50(__int64 a1, _BYTE **a2, __int64 a3)
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
  __int64 v17; // rcx
  unsigned int v18; // edx
  unsigned int *v19; // r12
  unsigned int v20; // eax
  _BYTE *v21; // rdi
  unsigned int v22; // esi
  __int64 v23; // rcx
  unsigned int v24; // edx
  unsigned int *v25; // rdi
  unsigned int v26; // eax
  _QWORD *v28; // r15
  __int64 v29; // rax
  __int64 v30; // r13
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // r15
  __int64 v34; // r13
  __int64 v35; // rdi
  __int64 *v36; // rdi
  int v37; // r10d
  unsigned int *v38; // r9
  int v39; // eax
  int v40; // edx
  __int64 v41; // rdi
  __int64 v42; // rdi
  size_t v43; // r15
  size_t v44; // rcx
  size_t v45; // rdx
  int v46; // eax
  unsigned int v47; // edi
  __int64 v48; // r15
  int v49; // r10d
  unsigned int *v50; // r9
  int v51; // eax
  int v52; // edx
  int v53; // eax
  int v54; // ecx
  __int64 v55; // r8
  unsigned int v56; // eax
  unsigned int v57; // esi
  int v58; // r9d
  unsigned int *v59; // r10
  int v60; // eax
  int v61; // ecx
  __int64 v62; // rdi
  unsigned int v63; // eax
  unsigned int v64; // esi
  int v65; // r9d
  unsigned int *v66; // r8
  int v67; // eax
  int v68; // eax
  __int64 v69; // r8
  unsigned int *v70; // r9
  __int64 v71; // r12
  int v72; // esi
  unsigned int v73; // ecx
  int v74; // eax
  int v75; // eax
  __int64 v76; // rsi
  __int64 v77; // r15
  int v78; // edi
  unsigned int v79; // ecx
  unsigned int *v80; // r8
  __int64 v82; // [rsp+10h] [rbp-80h]
  unsigned __int64 v83; // [rsp+18h] [rbp-78h]
  __int64 v84; // [rsp+18h] [rbp-78h]
  size_t v85; // [rsp+18h] [rbp-78h]
  __int64 src; // [rsp+50h] [rbp-40h]

  v3 = a1 + 40;
  v5 = *(_QWORD *)(a1 + 48);
  v82 = a1 + 40;
  if ( !v5 )
  {
    v3 = a1 + 40;
    goto LABEL_31;
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
        v83 = *(_QWORD *)(v5 + 40);
        v10 = memcmp(*(const void **)(v5 + 32), v6, v9);
        v8 = v83;
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
  if ( v82 == v3 )
    goto LABEL_31;
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
        goto LABEL_31;
LABEL_22:
      v15 = *(_DWORD *)(v3 + 64);
      if ( !v15 )
        goto LABEL_36;
      goto LABEL_23;
    }
  }
  if ( (__int64)&v7[-v12] > 0x7FFFFFFF )
    goto LABEL_22;
  if ( (__int64)&v7[-v12] >= (__int64)0xFFFFFFFF80000000LL )
  {
    v14 = (_DWORD)v7 - v12;
    goto LABEL_21;
  }
LABEL_31:
  v28 = (_QWORD *)v3;
  v29 = sub_22077B0(72);
  v30 = v29 + 32;
  v3 = v29;
  *(_QWORD *)(v29 + 32) = v29 + 48;
  v84 = v29 + 48;
  sub_14C8390((__int64 *)(v29 + 32), *a2, (__int64)&a2[1][(_QWORD)*a2]);
  *(_DWORD *)(v3 + 64) = 0;
  v31 = sub_A288A0((_QWORD *)(a1 + 32), v28, v30);
  v33 = v31;
  v34 = v32;
  if ( !v32 )
  {
    v41 = *(_QWORD *)(v3 + 32);
    if ( v84 != v41 )
      j_j___libc_free_0(v41, *(_QWORD *)(v3 + 48) + 1LL);
    v42 = v3;
    v3 = v33;
    j_j___libc_free_0(v42, 72);
    goto LABEL_22;
  }
  if ( v31 || v82 == v32 )
  {
LABEL_34:
    v35 = 1;
    goto LABEL_35;
  }
  v43 = *(_QWORD *)(v3 + 40);
  v45 = *(_QWORD *)(v32 + 40);
  v44 = v45;
  if ( v43 <= v45 )
    v45 = *(_QWORD *)(v3 + 40);
  if ( v45
    && (v85 = v44, v46 = memcmp(*(const void **)(v3 + 32), *(const void **)(v34 + 32), v45), v44 = v85, (v47 = v46) != 0) )
  {
LABEL_58:
    v35 = v47 >> 31;
  }
  else
  {
    v48 = v43 - v44;
    v35 = 0;
    if ( v48 <= 0x7FFFFFFF )
    {
      if ( v48 < (__int64)0xFFFFFFFF80000000LL )
        goto LABEL_34;
      v47 = v48;
      goto LABEL_58;
    }
  }
LABEL_35:
  sub_220F040(v35, v3, v34, v82);
  ++*(_QWORD *)(a1 + 72);
  v15 = *(_DWORD *)(v3 + 64);
  if ( !v15 )
  {
LABEL_36:
    *(_DWORD *)(v3 + 64) = ((__int64)(*(_QWORD *)(a1 + 88) - *(_QWORD *)(a1 + 80)) >> 5) + 1;
    v36 = *(__int64 **)(a1 + 88);
    if ( v36 == *(__int64 **)(a1 + 96) )
    {
      sub_8FD760((__m128i **)(a1 + 80), *(const __m128i **)(a1 + 88), (__int64)a2);
    }
    else
    {
      if ( v36 )
      {
        *v36 = (__int64)(v36 + 2);
        sub_14C8390(v36, *a2, (__int64)&a2[1][(_QWORD)*a2]);
        v36 = *(__int64 **)(a1 + 88);
      }
      *(_QWORD *)(a1 + 88) = v36 + 4;
    }
    v15 = *(_DWORD *)(v3 + 64);
  }
LABEL_23:
  v16 = *(_DWORD *)(a1 + 24);
  LOBYTE(src) = 0;
  if ( !v16 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_81;
  }
  v17 = *(_QWORD *)(a1 + 8);
  v18 = (v16 - 1) & (37 * v15);
  v19 = (unsigned int *)(v17 + 72LL * v18);
  v20 = *v19;
  if ( v15 == *v19 )
  {
    v21 = (_BYTE *)*((_QWORD *)v19 + 5);
    goto LABEL_26;
  }
  v37 = 1;
  v38 = 0;
  while ( 2 )
  {
    if ( v20 == -1 )
    {
      v39 = *(_DWORD *)(a1 + 16);
      if ( v38 )
        v19 = v38;
      ++*(_QWORD *)a1;
      v40 = v39 + 1;
      if ( 4 * (v39 + 1) < 3 * v16 )
      {
        if ( v16 - *(_DWORD *)(a1 + 20) - v40 > v16 >> 3 )
        {
LABEL_47:
          *(_DWORD *)(a1 + 16) = v40;
          if ( *v19 != -1 )
            --*(_DWORD *)(a1 + 20);
          *v19 = v15;
          v21 = v19 + 14;
          *(_OWORD *)(v19 + 2) = 0;
          *((_QWORD *)v19 + 1) = 0;
          *(_OWORD *)(v19 + 6) = 0;
          *((_QWORD *)v19 + 2) = 0;
          *((_QWORD *)v19 + 5) = v19 + 14;
          *((_QWORD *)v19 + 3) = -1;
          *((_QWORD *)v19 + 6) = 0;
          *((_BYTE *)v19 + 32) = 0;
          *(_OWORD *)(v19 + 14) = 0;
          goto LABEL_50;
        }
        sub_14C9BF0(a1, v16);
        v74 = *(_DWORD *)(a1 + 24);
        if ( v74 )
        {
          v75 = v74 - 1;
          v76 = *(_QWORD *)(a1 + 8);
          v66 = 0;
          LODWORD(v77) = v75 & (37 * v15);
          v19 = (unsigned int *)(v76 + 72LL * (unsigned int)v77);
          v40 = *(_DWORD *)(a1 + 16) + 1;
          v78 = 1;
          v79 = *v19;
          if ( v15 == *v19 )
            goto LABEL_47;
          while ( v79 != -1 )
          {
            if ( !v66 && v79 == -2 )
              v66 = v19;
            v77 = v75 & (unsigned int)(v77 + v78);
            v19 = (unsigned int *)(v76 + 72 * v77);
            v79 = *v19;
            if ( v15 == *v19 )
              goto LABEL_47;
            ++v78;
          }
          goto LABEL_85;
        }
        goto LABEL_128;
      }
LABEL_81:
      sub_14C9BF0(a1, 2 * v16);
      v60 = *(_DWORD *)(a1 + 24);
      if ( v60 )
      {
        v61 = v60 - 1;
        v62 = *(_QWORD *)(a1 + 8);
        v63 = (v60 - 1) & (37 * v15);
        v19 = (unsigned int *)(v62 + 72LL * v63);
        v64 = *v19;
        v40 = *(_DWORD *)(a1 + 16) + 1;
        if ( v15 == *v19 )
          goto LABEL_47;
        v65 = 1;
        v66 = 0;
        while ( v64 != -1 )
        {
          if ( !v66 && v64 == -2 )
            v66 = v19;
          v63 = v61 & (v65 + v63);
          v19 = (unsigned int *)(v62 + 72LL * v63);
          v64 = *v19;
          if ( v15 == *v19 )
            goto LABEL_47;
          ++v65;
        }
LABEL_85:
        if ( v66 )
          v19 = v66;
        goto LABEL_47;
      }
LABEL_128:
      ++*(_DWORD *)(a1 + 16);
      BUG();
    }
    if ( v20 != -2 || v38 )
      v19 = v38;
    v18 = (v16 - 1) & (v37 + v18);
    v80 = (unsigned int *)(v17 + 72LL * v18);
    v20 = *v80;
    if ( v15 != *v80 )
    {
      ++v37;
      v38 = v19;
      v19 = (unsigned int *)(v17 + 72LL * v18);
      continue;
    }
    break;
  }
  v21 = (_BYTE *)*((_QWORD *)v80 + 5);
  v19 = (unsigned int *)(v17 + 72LL * v18);
LABEL_26:
  *((_QWORD *)v19 + 1) = 0;
  *((_QWORD *)v19 + 2) = 0;
  *((_QWORD *)v19 + 3) = -1;
  *((_BYTE *)v19 + 32) = 0;
LABEL_50:
  *((_QWORD *)v19 + 6) = 0;
  *v21 = 0;
  LOBYTE(src) = 0;
  v22 = *(_DWORD *)(a1 + 24);
  if ( !v22 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_73;
  }
  v23 = *(_QWORD *)(a1 + 8);
  v24 = (v22 - 1) & (37 * v15);
  v25 = (unsigned int *)(v23 + 72LL * v24);
  v26 = *v25;
  if ( v15 == *v25 )
    goto LABEL_29;
  v49 = 1;
  v50 = 0;
  while ( v26 != -1 )
  {
    if ( !v50 && v26 == -2 )
      v50 = v25;
    v24 = (v22 - 1) & (v49 + v24);
    v25 = (unsigned int *)(v23 + 72LL * v24);
    v26 = *v25;
    if ( v15 == *v25 )
      goto LABEL_29;
    ++v49;
  }
  v51 = *(_DWORD *)(a1 + 16);
  if ( v50 )
    v25 = v50;
  ++*(_QWORD *)a1;
  v52 = v51 + 1;
  if ( 4 * (v51 + 1) >= 3 * v22 )
  {
LABEL_73:
    sub_14C9BF0(a1, 2 * v22);
    v53 = *(_DWORD *)(a1 + 24);
    if ( v53 )
    {
      v54 = v53 - 1;
      v55 = *(_QWORD *)(a1 + 8);
      v56 = (v53 - 1) & (37 * v15);
      v25 = (unsigned int *)(v55 + 72LL * v56);
      v57 = *v25;
      v52 = *(_DWORD *)(a1 + 16) + 1;
      if ( v15 != *v25 )
      {
        v58 = 1;
        v59 = 0;
        while ( v57 != -1 )
        {
          if ( !v59 && v57 == -2 )
            v59 = v25;
          v56 = v54 & (v58 + v56);
          v25 = (unsigned int *)(v55 + 72LL * v56);
          v57 = *v25;
          if ( v15 == *v25 )
            goto LABEL_69;
          ++v58;
        }
        if ( v59 )
          v25 = v59;
      }
      goto LABEL_69;
    }
    goto LABEL_128;
  }
  if ( v22 - *(_DWORD *)(a1 + 20) - v52 <= v22 >> 3 )
  {
    sub_14C9BF0(a1, v22);
    v67 = *(_DWORD *)(a1 + 24);
    if ( v67 )
    {
      v68 = v67 - 1;
      v69 = *(_QWORD *)(a1 + 8);
      v70 = 0;
      LODWORD(v71) = v68 & (37 * v15);
      v25 = (unsigned int *)(v69 + 72LL * (unsigned int)v71);
      v52 = *(_DWORD *)(a1 + 16) + 1;
      v72 = 1;
      v73 = *v25;
      if ( v15 != *v25 )
      {
        while ( v73 != -1 )
        {
          if ( !v70 && v73 == -2 )
            v70 = v25;
          v71 = v68 & (unsigned int)(v71 + v72);
          v25 = (unsigned int *)(v69 + 72 * v71);
          v73 = *v25;
          if ( v15 == *v25 )
            goto LABEL_69;
          ++v72;
        }
        if ( v70 )
          v25 = v70;
      }
      goto LABEL_69;
    }
    goto LABEL_128;
  }
LABEL_69:
  *(_DWORD *)(a1 + 16) = v52;
  if ( *v25 != -1 )
    --*(_DWORD *)(a1 + 20);
  *v25 = v15;
  *(_OWORD *)(v25 + 6) = 0;
  *((_QWORD *)v25 + 5) = v25 + 14;
  *((_QWORD *)v25 + 3) = -1;
  *((_QWORD *)v25 + 6) = 0;
  *(_OWORD *)(v25 + 2) = 0;
  *(_OWORD *)(v25 + 14) = 0;
LABEL_29:
  sub_2240AE0(v25 + 10, a3);
  return v15;
}
