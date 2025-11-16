// Function: sub_2E6A250
// Address: 0x2e6a250
//
__int64 __fastcall sub_2E6A250(__int64 a1, unsigned __int64 *a2, __int64 a3, unsigned __int8 a4)
{
  _QWORD *v5; // rax
  _QWORD *v6; // rax
  __int64 v7; // r9
  __int64 *v8; // r14
  __int64 *v9; // r11
  __int64 v10; // r8
  int v11; // esi
  unsigned int v12; // ecx
  __int64 *v13; // rax
  __int64 *v14; // rax
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 v18; // r15
  bool v19; // r13
  char v20; // di
  __int64 v21; // r8
  int v22; // esi
  unsigned int v23; // ecx
  __int64 *v24; // rax
  __int64 *v25; // rax
  unsigned __int64 v26; // r12
  __int64 v27; // r13
  __int64 v28; // rax
  __int64 v29; // rcx
  char v30; // di
  unsigned int v31; // esi
  unsigned int v32; // esi
  unsigned int v33; // eax
  unsigned __int64 v34; // rdx
  __int64 v35; // rcx
  unsigned int v36; // edi
  __int64 v37; // rcx
  unsigned int v38; // eax
  unsigned __int64 v39; // rdx
  __int64 v40; // rcx
  unsigned int v41; // edi
  __int64 v42; // rcx
  int v44; // ecx
  __int64 v45; // rsi
  int v46; // ecx
  unsigned int v47; // eax
  __int64 v48; // rdi
  int v49; // ecx
  __int64 v50; // rsi
  int v51; // ecx
  unsigned int v52; // eax
  __int64 v53; // rdi
  int v54; // ecx
  __int64 v55; // rsi
  int v56; // ecx
  unsigned int v57; // eax
  __int64 v58; // rdi
  int v59; // ecx
  __int64 v60; // rsi
  int v61; // ecx
  unsigned int v62; // eax
  __int64 v63; // rdi
  __int64 *v64; // [rsp+0h] [rbp-60h]
  __int64 v65; // [rsp+8h] [rbp-58h]
  __int64 *v66; // [rsp+8h] [rbp-58h]
  int v67; // [rsp+8h] [rbp-58h]
  int v68; // [rsp+8h] [rbp-58h]
  __int64 *v69; // [rsp+8h] [rbp-58h]
  __int64 *v70; // [rsp+8h] [rbp-58h]
  __int64 *v71; // [rsp+8h] [rbp-58h]
  __int64 *v72; // [rsp+8h] [rbp-58h]
  __int64 v73; // [rsp+10h] [rbp-50h]

  v5 = (_QWORD *)(a1 + 16);
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 1;
  v73 = a1 + 304;
  do
  {
    if ( v5 )
      *v5 = -4096;
    v5 += 9;
  }
  while ( v5 != (_QWORD *)(a1 + 304) );
  v6 = (_QWORD *)(a1 + 320);
  *(_QWORD *)(a1 + 304) = 0;
  *(_QWORD *)(a1 + 312) = 1;
  do
  {
    if ( v6 )
      *v6 = -4096;
    v6 += 9;
  }
  while ( v6 != (_QWORD *)(a1 + 608) );
  *(_QWORD *)(a1 + 616) = a1 + 632;
  *(_QWORD *)(a1 + 624) = 0x400000000LL;
  sub_2E690F0(a2, a3, a1 + 616, 0, 0);
  v8 = *(__int64 **)(a1 + 616);
  v9 = &v8[2 * *(unsigned int *)(a1 + 624)];
  while ( v9 != v8 )
  {
    v17 = v8[1];
    v18 = *v8;
    v19 = !((v17 >> 2) & 1) == (bool)(a4 ^ 1);
    v20 = *(_BYTE *)(a1 + 8) & 1;
    if ( v20 )
    {
      v21 = a1 + 16;
      v22 = 3;
    }
    else
    {
      v32 = *(_DWORD *)(a1 + 24);
      v21 = *(_QWORD *)(a1 + 16);
      if ( !v32 )
      {
        v38 = *(_DWORD *)(a1 + 8);
        ++*(_QWORD *)a1;
        v39 = 0;
        v40 = (v38 >> 1) + 1;
LABEL_36:
        v21 = 3 * v32;
        goto LABEL_37;
      }
      v22 = v32 - 1;
    }
    v23 = v22 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
    v24 = (__int64 *)(v21 + 72LL * v23);
    v7 = *v24;
    if ( v18 == *v24 )
    {
LABEL_20:
      v25 = v24 + 1;
      goto LABEL_21;
    }
    v67 = 1;
    v39 = 0;
    while ( v7 != -4096 )
    {
      if ( v7 == -8192 && !v39 )
        v39 = (unsigned __int64)v24;
      v23 = v22 & (v67 + v23);
      v24 = (__int64 *)(v21 + 72LL * v23);
      v7 = *v24;
      if ( v18 == *v24 )
        goto LABEL_20;
      ++v67;
    }
    v21 = 12;
    v32 = 4;
    if ( !v39 )
      v39 = (unsigned __int64)v24;
    v38 = *(_DWORD *)(a1 + 8);
    ++*(_QWORD *)a1;
    v40 = (v38 >> 1) + 1;
    if ( !v20 )
    {
      v32 = *(_DWORD *)(a1 + 24);
      goto LABEL_36;
    }
LABEL_37:
    if ( 4 * (int)v40 >= (unsigned int)v21 )
    {
      v70 = v9;
      sub_2E66810(a1, 2 * v32, v39, v40, v21, v7);
      v9 = v70;
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v50 = a1 + 16;
        v51 = 3;
      }
      else
      {
        v49 = *(_DWORD *)(a1 + 24);
        v50 = *(_QWORD *)(a1 + 16);
        if ( !v49 )
          goto LABEL_119;
        v51 = v49 - 1;
      }
      v52 = v51 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v39 = v50 + 72LL * v52;
      v53 = *(_QWORD *)v39;
      if ( v18 != *(_QWORD *)v39 )
      {
        v21 = 1;
        v7 = 0;
        while ( v53 != -4096 )
        {
          if ( v53 == -8192 && !v7 )
            v7 = v39;
          v52 = v51 & (v21 + v52);
          v39 = v50 + 72LL * v52;
          v53 = *(_QWORD *)v39;
          if ( v18 == *(_QWORD *)v39 )
            goto LABEL_64;
          v21 = (unsigned int)(v21 + 1);
        }
LABEL_79:
        if ( v7 )
          v39 = v7;
      }
LABEL_64:
      v38 = *(_DWORD *)(a1 + 8);
      goto LABEL_39;
    }
    v41 = v32 - *(_DWORD *)(a1 + 12) - v40;
    v42 = v32 >> 3;
    if ( v41 <= (unsigned int)v42 )
    {
      v72 = v9;
      sub_2E66810(a1, v32, v39, v42, v21, v7);
      v9 = v72;
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v60 = a1 + 16;
        v61 = 3;
      }
      else
      {
        v59 = *(_DWORD *)(a1 + 24);
        v60 = *(_QWORD *)(a1 + 16);
        if ( !v59 )
        {
LABEL_119:
          *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
          BUG();
        }
        v61 = v59 - 1;
      }
      v62 = v61 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v39 = v60 + 72LL * v62;
      v63 = *(_QWORD *)v39;
      if ( v18 != *(_QWORD *)v39 )
      {
        v21 = 1;
        v7 = 0;
        while ( v63 != -4096 )
        {
          if ( v63 == -8192 && !v7 )
            v7 = v39;
          v62 = v61 & (v21 + v62);
          v39 = v60 + 72LL * v62;
          v63 = *(_QWORD *)v39;
          if ( v18 == *(_QWORD *)v39 )
            goto LABEL_64;
          v21 = (unsigned int)(v21 + 1);
        }
        goto LABEL_79;
      }
      goto LABEL_64;
    }
LABEL_39:
    *(_DWORD *)(a1 + 8) = (2 * (v38 >> 1) + 2) | v38 & 1;
    if ( *(_QWORD *)v39 != -4096 )
      --*(_DWORD *)(a1 + 12);
    *(_QWORD *)v39 = v18;
    v25 = (__int64 *)(v39 + 8);
    *(_QWORD *)(v39 + 8) = v39 + 24;
    *(_QWORD *)(v39 + 16) = 0x200000000LL;
    *(_QWORD *)(v39 + 40) = v39 + 56;
    *(_QWORD *)(v39 + 48) = 0x200000000LL;
    *(_OWORD *)(v39 + 24) = 0;
    *(_OWORD *)(v39 + 56) = 0;
LABEL_21:
    v26 = v17 & 0xFFFFFFFFFFFFFFF8LL;
    v27 = 4LL * v19;
    v28 = (__int64)&v25[v27];
    v29 = *(unsigned int *)(v28 + 8);
    if ( v29 + 1 > (unsigned __int64)*(unsigned int *)(v28 + 12) )
    {
      v64 = v9;
      v65 = v28;
      sub_C8D5F0(v28, (const void *)(v28 + 16), v29 + 1, 8u, v21, v7);
      v28 = v65;
      v9 = v64;
      v29 = *(unsigned int *)(v65 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v28 + 8 * v29) = v26;
    ++*(_DWORD *)(v28 + 8);
    v30 = *(_BYTE *)(a1 + 312) & 1;
    if ( v30 )
    {
      v10 = a1 + 320;
      v11 = 3;
    }
    else
    {
      v31 = *(_DWORD *)(a1 + 328);
      v10 = *(_QWORD *)(a1 + 320);
      if ( !v31 )
      {
        v33 = *(_DWORD *)(a1 + 312);
        ++*(_QWORD *)(a1 + 304);
        v34 = 0;
        v35 = (v33 >> 1) + 1;
        goto LABEL_29;
      }
      v11 = v31 - 1;
    }
    v12 = v11 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
    v13 = (__int64 *)(v10 + 72LL * v12);
    v7 = *v13;
    if ( v26 != *v13 )
    {
      v68 = 1;
      v34 = 0;
      while ( v7 != -4096 )
      {
        if ( v7 == -8192 && !v34 )
          v34 = (unsigned __int64)v13;
        v12 = v11 & (v68 + v12);
        v13 = (__int64 *)(v10 + 72LL * v12);
        v7 = *v13;
        if ( v26 == *v13 )
          goto LABEL_13;
        ++v68;
      }
      v10 = 12;
      v31 = 4;
      if ( !v34 )
        v34 = (unsigned __int64)v13;
      v33 = *(_DWORD *)(a1 + 312);
      ++*(_QWORD *)(a1 + 304);
      v35 = (v33 >> 1) + 1;
      if ( !v30 )
      {
        v31 = *(_DWORD *)(a1 + 328);
LABEL_29:
        v10 = 3 * v31;
      }
      if ( 4 * (int)v35 >= (unsigned int)v10 )
      {
        v69 = v9;
        sub_2E66810(v73, 2 * v31, v34, v35, v10, v7);
        v9 = v69;
        if ( (*(_BYTE *)(a1 + 312) & 1) != 0 )
        {
          v45 = a1 + 320;
          v46 = 3;
        }
        else
        {
          v44 = *(_DWORD *)(a1 + 328);
          v45 = *(_QWORD *)(a1 + 320);
          if ( !v44 )
            goto LABEL_120;
          v46 = v44 - 1;
        }
        v47 = v46 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v34 = v45 + 72LL * v47;
        v48 = *(_QWORD *)v34;
        if ( v26 != *(_QWORD *)v34 )
        {
          v10 = 1;
          v7 = 0;
          while ( v48 != -4096 )
          {
            if ( v48 == -8192 && !v7 )
              v7 = v34;
            v47 = v46 & (v10 + v47);
            v34 = v45 + 72LL * v47;
            v48 = *(_QWORD *)v34;
            if ( v26 == *(_QWORD *)v34 )
              goto LABEL_59;
            v10 = (unsigned int)(v10 + 1);
          }
LABEL_71:
          if ( v7 )
            v34 = v7;
        }
      }
      else
      {
        v36 = v31 - *(_DWORD *)(a1 + 316) - v35;
        v37 = v31 >> 3;
        if ( v36 > (unsigned int)v37 )
        {
LABEL_32:
          *(_DWORD *)(a1 + 312) = (2 * (v33 >> 1) + 2) | v33 & 1;
          if ( *(_QWORD *)v34 != -4096 )
            --*(_DWORD *)(a1 + 316);
          *(_QWORD *)v34 = v26;
          v14 = (__int64 *)(v34 + 8);
          *(_QWORD *)(v34 + 8) = v34 + 24;
          *(_QWORD *)(v34 + 16) = 0x200000000LL;
          *(_QWORD *)(v34 + 40) = v34 + 56;
          *(_QWORD *)(v34 + 48) = 0x200000000LL;
          *(_OWORD *)(v34 + 24) = 0;
          *(_OWORD *)(v34 + 56) = 0;
          goto LABEL_14;
        }
        v71 = v9;
        sub_2E66810(v73, v31, v34, v37, v10, v7);
        v9 = v71;
        if ( (*(_BYTE *)(a1 + 312) & 1) != 0 )
        {
          v55 = a1 + 320;
          v56 = 3;
        }
        else
        {
          v54 = *(_DWORD *)(a1 + 328);
          v55 = *(_QWORD *)(a1 + 320);
          if ( !v54 )
          {
LABEL_120:
            *(_DWORD *)(a1 + 312) = (2 * (*(_DWORD *)(a1 + 312) >> 1) + 2) | *(_DWORD *)(a1 + 312) & 1;
            BUG();
          }
          v56 = v54 - 1;
        }
        v57 = v56 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v34 = v55 + 72LL * v57;
        v58 = *(_QWORD *)v34;
        if ( v26 != *(_QWORD *)v34 )
        {
          v10 = 1;
          v7 = 0;
          while ( v58 != -4096 )
          {
            if ( v58 == -8192 && !v7 )
              v7 = v34;
            v57 = v56 & (v10 + v57);
            v34 = v55 + 72LL * v57;
            v58 = *(_QWORD *)v34;
            if ( v26 == *(_QWORD *)v34 )
              goto LABEL_59;
            v10 = (unsigned int)(v10 + 1);
          }
          goto LABEL_71;
        }
      }
LABEL_59:
      v33 = *(_DWORD *)(a1 + 312);
      goto LABEL_32;
    }
LABEL_13:
    v14 = v13 + 1;
LABEL_14:
    v15 = (__int64)&v14[v27];
    v16 = *(unsigned int *)(v15 + 8);
    if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(v15 + 12) )
    {
      v66 = v9;
      sub_C8D5F0(v15, (const void *)(v15 + 16), v16 + 1, 8u, v10, v7);
      v16 = *(unsigned int *)(v15 + 8);
      v9 = v66;
    }
    v8 += 2;
    *(_QWORD *)(*(_QWORD *)v15 + 8 * v16) = v18;
    ++*(_DWORD *)(v15 + 8);
  }
  *(_BYTE *)(a1 + 608) = a4;
  return a4;
}
