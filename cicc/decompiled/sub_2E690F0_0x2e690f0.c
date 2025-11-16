// Function: sub_2E690F0
// Address: 0x2e690f0
//
void __fastcall sub_2E690F0(unsigned __int64 *a1, __int64 a2, __int64 a3, unsigned __int8 a4, __int64 a5)
{
  __int64 *v5; // rax
  char v7; // dl
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  unsigned int v10; // eax
  unsigned __int64 *v11; // r12
  __int64 v12; // r9
  unsigned __int64 *v13; // r15
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // r14
  unsigned __int64 v16; // r13
  int v17; // ebx
  int v18; // esi
  int v19; // r11d
  __int64 *v20; // rdx
  unsigned int i; // eax
  __int64 v22; // rdi
  __int64 v23; // r10
  unsigned int v24; // esi
  int *v25; // rdx
  unsigned __int32 v26; // eax
  unsigned __int32 v27; // eax
  char v28; // dl
  __int64 v29; // rax
  __int64 *v30; // r14
  __int64 *v31; // rbx
  __int64 *v32; // r15
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 *v35; // r14
  int v36; // eax
  __int64 v37; // r8
  unsigned __int64 v38; // r12
  __int64 v39; // rax
  __int64 *v40; // rax
  __int64 v41; // r12
  __int64 *v42; // r14
  unsigned __int8 v43; // r8
  __int64 v44; // r13
  unsigned __int64 v45; // rbx
  char v46; // dl
  __int64 *v47; // rdi
  int v48; // esi
  int v49; // r11d
  __int64 *v50; // rcx
  unsigned int m; // eax
  __int64 v52; // r10
  unsigned __int32 v53; // eax
  unsigned __int32 v54; // edi
  __int64 *v55; // rdi
  int v56; // esi
  int v57; // r11d
  __int64 *v58; // rcx
  unsigned int v59; // eax
  __int64 v60; // r10
  unsigned int v61; // esi
  unsigned int v62; // esi
  _DWORD *v63; // rcx
  __m128i *v64; // r12
  __int64 v65; // rbx
  unsigned __int64 v66; // rax
  const __m128i *v67; // r14
  __m128i *v68; // r15
  __m128i v69; // xmm0
  unsigned __int32 v70; // eax
  unsigned __int32 v71; // edi
  unsigned __int32 v72; // eax
  unsigned __int32 v73; // edi
  __int64 *v74; // rsi
  int v75; // ecx
  int v76; // r10d
  unsigned int j; // eax
  __int64 v78; // rdi
  unsigned int v79; // eax
  __int64 *v80; // rsi
  int v81; // ecx
  int v82; // r10d
  unsigned int k; // eax
  __int64 v84; // rdi
  unsigned int v85; // eax
  __int64 *v86; // rbx
  __int64 v87; // rax
  unsigned int v88; // eax
  __int64 *v89; // rsi
  int v90; // edx
  int v91; // r10d
  unsigned int jj; // eax
  __int64 v93; // rdi
  unsigned int v94; // eax
  __int64 *v95; // rsi
  int v96; // edx
  int v97; // r10d
  unsigned int n; // eax
  __int64 v99; // rdi
  unsigned int v100; // eax
  __int64 *v101; // rsi
  int v102; // edx
  int v103; // r10d
  unsigned int ii; // eax
  __int64 v105; // rdi
  unsigned int v106; // eax
  __int64 *v107; // rsi
  int v108; // edx
  int v109; // r10d
  unsigned int kk; // eax
  __int64 v111; // rdi
  unsigned int v112; // eax
  unsigned int v113; // eax
  unsigned int v114; // eax
  __int64 v115; // [rsp+0h] [rbp-F0h]
  unsigned __int8 v116; // [rsp+0h] [rbp-F0h]
  unsigned __int8 v117; // [rsp+0h] [rbp-F0h]
  __int64 v118; // [rsp+8h] [rbp-E8h]
  __int64 v120; // [rsp+10h] [rbp-E0h]
  __int64 v121; // [rsp+20h] [rbp-D0h]
  __m128i *v122; // [rsp+20h] [rbp-D0h]
  unsigned __int8 v124; // [rsp+2Bh] [rbp-C5h]
  unsigned __int8 v125; // [rsp+2Bh] [rbp-C5h]
  _BYTE v126[4]; // [rsp+2Ch] [rbp-C4h] BYREF
  const __m128i *v127[2]; // [rsp+30h] [rbp-C0h] BYREF
  __m128i v128; // [rsp+40h] [rbp-B0h] BYREF
  __m128i v129; // [rsp+50h] [rbp-A0h] BYREF
  __int64 *v130; // [rsp+60h] [rbp-90h] BYREF
  unsigned int v131; // [rsp+68h] [rbp-88h]
  _BYTE v132[48]; // [rsp+C0h] [rbp-30h] BYREF

  v5 = (__int64 *)&v130;
  v121 = a2;
  v126[0] = a5;
  v129.m128i_i64[0] = 0;
  v129.m128i_i64[1] = 1;
  do
  {
    *v5 = -4096;
    v5 += 3;
    *(v5 - 2) = -4096;
  }
  while ( v5 != (__int64 *)v132 );
  v7 = v129.m128i_i8[8] & 1;
  if ( (_DWORD)a2 )
  {
    ++v129.m128i_i64[0];
    v8 = (4 * (int)a2 / 3u + 1) | ((unsigned __int64)(4 * (int)a2 / 3u + 1) >> 1);
    v9 = (((v8 >> 2) | v8) >> 4) | (v8 >> 2) | v8;
    LODWORD(a2) = ((((v9 >> 8) | v9) >> 16) | (v9 >> 8) | v9) + 1;
    v10 = 4;
    if ( v7 )
      goto LABEL_6;
    goto LABEL_5;
  }
  ++v129.m128i_i64[0];
  if ( !v7 )
  {
LABEL_5:
    v10 = v131;
LABEL_6:
    if ( v10 < (unsigned int)a2 )
      sub_2E64C00(&v129, a2);
  }
  v11 = a1;
  v12 = (__int64)&a1[2 * v121];
  if ( a1 == (unsigned __int64 *)v12 )
    goto LABEL_28;
  v118 = a3;
  v13 = &a1[2 * v121];
  v12 = a4;
  do
  {
    v14 = v11[1];
    v15 = *v11;
    v16 = v14 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !(_BYTE)v12 )
    {
      v16 = *v11;
      v15 = v11[1] & 0xFFFFFFFFFFFFFFF8LL;
    }
    v17 = (v14 & 4) == 0 ? 1 : -1;
    if ( (v129.m128i_i8[8] & 1) != 0 )
    {
      a5 = (__int64)&v130;
      v18 = 3;
    }
    else
    {
      v24 = v131;
      a5 = (__int64)v130;
      if ( !v131 )
      {
        v53 = v129.m128i_u32[2];
        ++v129.m128i_i64[0];
        v20 = 0;
        v54 = ((unsigned __int32)v129.m128i_i32[2] >> 1) + 1;
        goto LABEL_64;
      }
      v18 = v131 - 1;
    }
    v19 = 1;
    v20 = 0;
    for ( i = v18
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4)
                | ((unsigned __int64)(((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4)))); ; i = v18 & v88 )
    {
      v22 = a5 + 24LL * i;
      v23 = *(_QWORD *)v22;
      if ( v16 == *(_QWORD *)v22 && v15 == *(_QWORD *)(v22 + 8) )
      {
        v25 = (int *)(v22 + 16);
        v17 += *(_DWORD *)(v22 + 16);
        goto LABEL_26;
      }
      if ( v23 == -4096 )
        break;
      if ( v23 == -8192 && *(_QWORD *)(v22 + 8) == -8192 && !v20 )
        v20 = (__int64 *)(a5 + 24LL * i);
LABEL_160:
      v88 = v19 + i;
      ++v19;
    }
    if ( *(_QWORD *)(v22 + 8) != -4096 )
      goto LABEL_160;
    v53 = v129.m128i_u32[2];
    a5 = 12;
    v24 = 4;
    if ( !v20 )
      v20 = (__int64 *)v22;
    ++v129.m128i_i64[0];
    v54 = ((unsigned __int32)v129.m128i_i32[2] >> 1) + 1;
    if ( (v129.m128i_i8[8] & 1) == 0 )
    {
      v24 = v131;
LABEL_64:
      a5 = 3 * v24;
    }
    if ( 4 * v54 >= (unsigned int)a5 )
    {
      v116 = v12;
      sub_2E64C00(&v129, 2 * v24);
      v12 = v116;
      if ( (v129.m128i_i8[8] & 1) != 0 )
      {
        v74 = (__int64 *)&v130;
        v75 = 3;
      }
      else
      {
        v74 = v130;
        if ( !v131 )
          goto LABEL_254;
        v75 = v131 - 1;
      }
      v76 = 1;
      a5 = 0;
      for ( j = v75
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4)
                  | ((unsigned __int64)(((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4)))); ; j = v75 & v79 )
      {
        v20 = &v74[3 * j];
        v78 = *v20;
        if ( v16 == *v20 && v15 == v20[1] )
          break;
        if ( v78 == -4096 )
        {
          if ( v20[1] == -4096 )
          {
LABEL_222:
            if ( a5 )
              v20 = (__int64 *)a5;
            break;
          }
        }
        else if ( v78 == -8192 && v20[1] == -8192 && !a5 )
        {
          a5 = (__int64)&v74[3 * j];
        }
        v79 = v76 + j;
        ++v76;
      }
    }
    else
    {
      if ( v24 - v129.m128i_i32[3] - v54 > v24 >> 3 )
        goto LABEL_67;
      v117 = v12;
      sub_2E64C00(&v129, v24);
      v12 = v117;
      if ( (v129.m128i_i8[8] & 1) != 0 )
      {
        v80 = (__int64 *)&v130;
        v81 = 3;
      }
      else
      {
        v80 = v130;
        if ( !v131 )
          goto LABEL_254;
        v81 = v131 - 1;
      }
      v82 = 1;
      a5 = 0;
      for ( k = v81
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4)
                  | ((unsigned __int64)(((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4)))); ; k = v81 & v85 )
      {
        v20 = &v80[3 * k];
        v84 = *v20;
        if ( v16 == *v20 && v15 == v20[1] )
          break;
        if ( v84 == -4096 )
        {
          if ( v20[1] == -4096 )
            goto LABEL_222;
        }
        else if ( v84 == -8192 && v20[1] == -8192 && !a5 )
        {
          a5 = (__int64)&v80[3 * k];
        }
        v85 = v82 + k;
        ++v82;
      }
    }
    v53 = v129.m128i_u32[2];
LABEL_67:
    v129.m128i_i32[2] = (2 * (v53 >> 1) + 2) | v53 & 1;
    if ( *v20 != -4096 || v20[1] != -4096 )
      --v129.m128i_i32[3];
    *v20 = v16;
    v25 = (int *)(v20 + 2);
    *((_QWORD *)v25 - 1) = v15;
    *v25 = 0;
LABEL_26:
    v11 += 2;
    *v25 = v17;
  }
  while ( v13 != v11 );
  a3 = v118;
LABEL_28:
  v26 = v129.m128i_u32[2];
  *(_DWORD *)(a3 + 8) = 0;
  v27 = v26 >> 1;
  if ( *(_DWORD *)(a3 + 12) < v27 )
  {
    sub_C8D5F0(a3, (const void *)(a3 + 16), v27, 0x10u, a5, v12);
    v27 = (unsigned __int32)v129.m128i_i32[2] >> 1;
  }
  v28 = v129.m128i_i8[8] & 1;
  if ( v27 )
  {
    if ( v28 )
    {
      v33 = (__int64)v130;
      v31 = (__int64 *)&v130;
      v32 = (__int64 *)v132;
      if ( v130 != (__int64 *)-4096LL )
        goto LABEL_34;
      goto LABEL_73;
    }
    v29 = v131;
    v30 = v130;
    v31 = v130;
    v32 = &v130[3 * v131];
    if ( v130 != v32 )
    {
      while ( 1 )
      {
        v33 = *v31;
        if ( *v31 == -4096 )
        {
LABEL_73:
          if ( v31[1] != -4096 )
            goto LABEL_35;
        }
        else
        {
LABEL_34:
          if ( v33 != -8192 || v31[1] != -8192 )
            goto LABEL_35;
        }
        v31 += 3;
        if ( v31 == v32 )
          goto LABEL_35;
      }
    }
LABEL_37:
    v34 = 3 * v29;
  }
  else
  {
    if ( v28 )
    {
      v86 = (__int64 *)&v130;
      v87 = 12;
    }
    else
    {
      v86 = v130;
      v87 = 3LL * v131;
    }
    v31 = &v86[v87];
    v32 = v31;
LABEL_35:
    if ( !v28 )
    {
      v30 = v130;
      v29 = v131;
      goto LABEL_37;
    }
    v30 = (__int64 *)&v130;
    v34 = 12;
  }
  v35 = &v30[v34];
  while ( v35 != v31 )
  {
LABEL_42:
    v36 = *((_DWORD *)v31 + 4);
    if ( v36 )
    {
      v37 = *v31;
      v38 = (4LL * (v36 <= 0)) | v31[1] & 0xFFFFFFFFFFFFFFFBLL;
      v39 = *(unsigned int *)(a3 + 8);
      if ( v39 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
      {
        v115 = *v31;
        sub_C8D5F0(a3, (const void *)(a3 + 16), v39 + 1, 0x10u, v37, v12);
        v39 = *(unsigned int *)(a3 + 8);
        v37 = v115;
      }
      v40 = (__int64 *)(*(_QWORD *)a3 + 16 * v39);
      *v40 = v37;
      v40[1] = v38;
      ++*(_DWORD *)(a3 + 8);
    }
    do
    {
      while ( 1 )
      {
        v31 += 3;
        if ( v31 == v32 )
        {
LABEL_41:
          if ( v35 == v31 )
            goto LABEL_50;
          goto LABEL_42;
        }
        if ( *v31 == -4096 )
          break;
        if ( *v31 != -8192 || v31[1] != -8192 )
          goto LABEL_41;
      }
    }
    while ( v31[1] == -4096 );
  }
LABEL_50:
  v41 = 0;
  v42 = (__int64 *)a1;
  if ( !v121 )
    goto LABEL_96;
  v120 = a3;
  v43 = a4;
  while ( 2 )
  {
    v44 = *v42;
    v45 = v42[1] & 0xFFFFFFFFFFFFFFF8LL;
    v46 = v129.m128i_i8[8] & 1;
    if ( !v43 )
    {
      if ( v46 )
      {
        v47 = (__int64 *)&v130;
        v48 = 3;
      }
      else
      {
        v61 = v131;
        v47 = v130;
        if ( !v131 )
        {
          v70 = v129.m128i_u32[2];
          ++v129.m128i_i64[0];
          v50 = 0;
          v71 = ((unsigned __int32)v129.m128i_i32[2] >> 1) + 1;
          goto LABEL_109;
        }
        v48 = v131 - 1;
      }
      v49 = 1;
      v50 = 0;
      for ( m = v48
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4)
                  | ((unsigned __int64)(((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4)))); ; m = v48 & v114 )
      {
        v12 = (__int64)&v47[3 * m];
        v52 = *(_QWORD *)v12;
        if ( *(_QWORD *)v12 == v44 && *(_QWORD *)(v12 + 8) == v45 )
          goto LABEL_93;
        if ( v52 == -4096 )
        {
          if ( *(_QWORD *)(v12 + 8) == -4096 )
          {
            v70 = v129.m128i_u32[2];
            v61 = 4;
            if ( !v50 )
              v50 = (__int64 *)v12;
            ++v129.m128i_i64[0];
            v12 = 12;
            v71 = ((unsigned __int32)v129.m128i_i32[2] >> 1) + 1;
            if ( !v46 )
            {
              v61 = v131;
LABEL_109:
              v12 = 3 * v61;
            }
            if ( 4 * v71 >= (unsigned int)v12 )
            {
              sub_2E64C00(&v129, 2 * v61);
              v43 = 0;
              if ( (v129.m128i_i8[8] & 1) != 0 )
              {
                v95 = (__int64 *)&v130;
                v96 = 3;
              }
              else
              {
                v95 = v130;
                if ( !v131 )
                  goto LABEL_254;
                v96 = v131 - 1;
              }
              v97 = 1;
              v12 = 0;
              for ( n = v96
                      & (((0xBF58476D1CE4E5B9LL
                         * (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4)
                          | ((unsigned __int64)(((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4)) << 32))) >> 31)
                       ^ (484763065 * (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4)))); ; n = v96 & v100 )
              {
                v50 = &v95[3 * n];
                v99 = *v50;
                if ( *v50 == v44 && v50[1] == v45 )
                  break;
                if ( v99 == -4096 )
                {
                  if ( v50[1] == -4096 )
                  {
LABEL_227:
                    if ( v12 )
                      v50 = (__int64 *)v12;
                    goto LABEL_229;
                  }
                }
                else if ( v99 == -8192 && v50[1] == -8192 && !v12 )
                {
                  v12 = (__int64)&v95[3 * n];
                }
                v100 = v97 + n;
                ++v97;
              }
              goto LABEL_229;
            }
            if ( v61 - v129.m128i_i32[3] - v71 > v61 >> 3 )
            {
LABEL_112:
              v129.m128i_i32[2] = (2 * (v70 >> 1) + 2) | v70 & 1;
              if ( *v50 != -4096 || v50[1] != -4096 )
                --v129.m128i_i32[3];
              *v50 = v44;
              v63 = v50 + 2;
              *((_QWORD *)v63 - 1) = v45;
              *v63 = 0;
              goto LABEL_94;
            }
            sub_2E64C00(&v129, v61);
            v43 = 0;
            if ( (v129.m128i_i8[8] & 1) != 0 )
            {
              v101 = (__int64 *)&v130;
              v102 = 3;
LABEL_185:
              v103 = 1;
              v12 = 0;
              for ( ii = v102
                       & (((0xBF58476D1CE4E5B9LL
                          * (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4)
                           | ((unsigned __int64)(((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4)) << 32))) >> 31)
                        ^ (484763065 * (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4)))); ; ii = v102 & v106 )
              {
                v50 = &v101[3 * ii];
                v105 = *v50;
                if ( *v50 == v44 && v50[1] == v45 )
                  break;
                if ( v105 == -4096 )
                {
                  if ( v50[1] == -4096 )
                    goto LABEL_227;
                }
                else if ( v105 == -8192 && v50[1] == -8192 && !v12 )
                {
                  v12 = (__int64)&v101[3 * ii];
                }
                v106 = v103 + ii;
                ++v103;
              }
LABEL_229:
              v70 = v129.m128i_u32[2];
              goto LABEL_112;
            }
            v101 = v130;
            if ( v131 )
            {
              v102 = v131 - 1;
              goto LABEL_185;
            }
LABEL_254:
            v129.m128i_i32[2] = (2 * ((unsigned __int32)v129.m128i_i32[2] >> 1) + 2) | v129.m128i_i8[8] & 1;
            BUG();
          }
        }
        else if ( v52 == -8192 && *(_QWORD *)(v12 + 8) == -8192 && !v50 )
        {
          v50 = &v47[3 * m];
        }
        v114 = v49 + m;
        ++v49;
      }
    }
    if ( v46 )
    {
      v55 = (__int64 *)&v130;
      v56 = 3;
    }
    else
    {
      v62 = v131;
      v55 = v130;
      if ( !v131 )
      {
        v72 = v129.m128i_u32[2];
        ++v129.m128i_i64[0];
        v58 = 0;
        v73 = ((unsigned __int32)v129.m128i_i32[2] >> 1) + 1;
        goto LABEL_116;
      }
      v56 = v131 - 1;
    }
    v57 = 1;
    v58 = 0;
    v59 = v56
        & (((0xBF58476D1CE4E5B9LL
           * (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4)
            | ((unsigned __int64)(((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4)) << 32))) >> 31)
         ^ (484763065 * (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4))));
    while ( 2 )
    {
      v12 = (__int64)&v55[3 * v59];
      v60 = *(_QWORD *)v12;
      if ( *(_QWORD *)v12 == v45 && *(_QWORD *)(v12 + 8) == v44 )
      {
LABEL_93:
        v63 = (_DWORD *)(v12 + 16);
        goto LABEL_94;
      }
      if ( v60 != -4096 )
      {
        if ( v60 == -8192 && *(_QWORD *)(v12 + 8) == -8192 && !v58 )
          v58 = &v55[3 * v59];
        goto LABEL_212;
      }
      if ( *(_QWORD *)(v12 + 8) != -4096 )
      {
LABEL_212:
        v113 = v57 + v59;
        ++v57;
        v59 = v56 & v113;
        continue;
      }
      break;
    }
    v72 = v129.m128i_u32[2];
    v62 = 4;
    if ( !v58 )
      v58 = (__int64 *)v12;
    ++v129.m128i_i64[0];
    v12 = 12;
    v73 = ((unsigned __int32)v129.m128i_i32[2] >> 1) + 1;
    if ( !v46 )
    {
      v62 = v131;
LABEL_116:
      v12 = 3 * v62;
    }
    if ( 4 * v73 >= (unsigned int)v12 )
    {
      v124 = v43;
      sub_2E64C00(&v129, 2 * v62);
      v43 = v124;
      if ( (v129.m128i_i8[8] & 1) != 0 )
      {
        v89 = (__int64 *)&v130;
        v90 = 3;
      }
      else
      {
        v89 = v130;
        if ( !v131 )
          goto LABEL_254;
        v90 = v131 - 1;
      }
      v91 = 1;
      v12 = 0;
      for ( jj = v90
               & (((0xBF58476D1CE4E5B9LL
                  * (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4)
                   | ((unsigned __int64)(((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4)) << 32))) >> 31)
                ^ (484763065 * (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4)))); ; jj = v90 & v94 )
      {
        v58 = &v89[3 * jj];
        v93 = *v58;
        if ( v45 == *v58 && v58[1] == v44 )
          break;
        if ( v93 == -4096 )
        {
          if ( v58[1] == -4096 )
          {
LABEL_231:
            if ( v12 )
              v58 = (__int64 *)v12;
            break;
          }
        }
        else if ( v93 == -8192 && v58[1] == -8192 && !v12 )
        {
          v12 = (__int64)&v89[3 * jj];
        }
        v94 = v91 + jj;
        ++v91;
      }
    }
    else
    {
      if ( v62 - v129.m128i_i32[3] - v73 > v62 >> 3 )
        goto LABEL_119;
      v125 = v43;
      sub_2E64C00(&v129, v62);
      v43 = v125;
      if ( (v129.m128i_i8[8] & 1) != 0 )
      {
        v107 = (__int64 *)&v130;
        v108 = 3;
      }
      else
      {
        v107 = v130;
        if ( !v131 )
          goto LABEL_254;
        v108 = v131 - 1;
      }
      v109 = 1;
      v12 = 0;
      for ( kk = v108
               & (((0xBF58476D1CE4E5B9LL
                  * (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4)
                   | ((unsigned __int64)(((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4)) << 32))) >> 31)
                ^ (484763065 * (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4)))); ; kk = v108 & v112 )
      {
        v58 = &v107[3 * kk];
        v111 = *v58;
        if ( v45 == *v58 && v58[1] == v44 )
          break;
        if ( v111 == -4096 )
        {
          if ( v58[1] == -4096 )
            goto LABEL_231;
        }
        else if ( v111 == -8192 && v58[1] == -8192 && !v12 )
        {
          v12 = (__int64)&v107[3 * kk];
        }
        v112 = v109 + kk;
        ++v109;
      }
    }
    v72 = v129.m128i_u32[2];
LABEL_119:
    v129.m128i_i32[2] = (2 * (v72 >> 1) + 2) | v72 & 1;
    if ( *v58 != -4096 || v58[1] != -4096 )
      --v129.m128i_i32[3];
    *v58 = v45;
    v63 = v58 + 2;
    *((_QWORD *)v63 - 1) = v44;
    *v63 = 0;
LABEL_94:
    *v63 = v41++;
    v42 += 2;
    if ( v41 != v121 )
      continue;
    break;
  }
  a3 = v120;
LABEL_96:
  v64 = *(__m128i **)a3;
  v65 = 16LL * *(unsigned int *)(a3 + 8);
  v122 = (__m128i *)(*(_QWORD *)a3 + v65);
  if ( *(__m128i **)a3 != v122 )
  {
    _BitScanReverse64(&v66, v65 >> 4);
    sub_2E67E00(v64, v122->m128i_i64, 2LL * (int)(63 - (v66 ^ 0x3F)), &v129, (const __m128i *)v126, v12);
    if ( (unsigned __int64)v65 <= 0x100 )
    {
      sub_2E672B0(v64, v122, &v129, (__int64)v126);
    }
    else
    {
      v67 = v64 + 16;
      sub_2E672B0(v64, v64 + 16, &v129, (__int64)v126);
      if ( v122 != &v64[16] )
      {
        do
        {
          v68 = (__m128i *)&v67[-1];
          v127[0] = &v129;
          v127[1] = (const __m128i *)v126;
          v128 = _mm_loadu_si128(v67);
          while ( sub_2E651A0(v127, v128.m128i_i64, v68->m128i_i64) )
          {
            v69 = _mm_loadu_si128(v68--);
            v68[2] = v69;
          }
          ++v67;
          v68[1] = _mm_load_si128(&v128);
        }
        while ( v122 != v67 );
      }
    }
  }
  if ( (v129.m128i_i8[8] & 1) == 0 )
    sub_C7D6A0((__int64)v130, 24LL * v131, 8);
}
