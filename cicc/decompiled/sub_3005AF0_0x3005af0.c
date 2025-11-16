// Function: sub_3005AF0
// Address: 0x3005af0
//
__int64 __fastcall sub_3005AF0(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rax
  __int64 v11; // r12
  unsigned __int64 *v12; // rdi
  unsigned __int64 v13; // rdx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rsi
  unsigned __int64 v17; // rcx
  unsigned __int64 v18; // rbx
  __int64 v19; // rax
  __m128i *v20; // rdx
  const __m128i *v21; // rax
  unsigned __int64 v22; // rbx
  __int64 v23; // rcx
  __int64 v24; // rax
  unsigned __int64 v25; // r13
  __m128i *v26; // rdx
  const __m128i *v27; // rax
  __int64 v28; // r15
  __int64 *v29; // rax
  __int64 *v30; // rdx
  _QWORD *v31; // rdi
  __int64 v32; // r14
  __int64 *v33; // rax
  char v34; // dl
  unsigned __int64 v35; // rdx
  unsigned __int64 *v36; // rax
  __int64 v37; // rbx
  char *v38; // rax
  __int64 v39; // rdx
  _BYTE *v40; // r13
  _BYTE *v41; // r14
  _QWORD *v42; // r15
  unsigned __int64 v43; // rbx
  __int64 m; // r12
  int v45; // r12d
  char v46; // di
  __m128i **v47; // rbx
  __int64 v48; // rax
  __m128i **v49; // r13
  __m128i **v50; // rax
  __int64 v51; // rax
  __int64 v52; // rbx
  __int64 v53; // rax
  __int64 v54; // r13
  int v55; // eax
  unsigned int v56; // r14d
  __m128i **v57; // rax
  __int64 v58; // r12
  __int64 v59; // rdi
  __int64 (*v60)(); // rax
  __int64 v61; // rax
  unsigned __int8 *v62; // rsi
  __int64 v63; // rax
  __int64 v64; // rax
  _QWORD *v65; // r14
  __int64 v66; // r10
  _QWORD *v67; // rax
  __int64 *v68; // r10
  __int64 v69; // r9
  __int64 v70; // r9
  __int64 v71; // rax
  __int64 v72; // rdx
  unsigned int v73; // eax
  __int64 v74; // r9
  unsigned int v75; // edx
  bool v76; // cl
  char v77; // al
  unsigned int v78; // ecx
  unsigned int v79; // ecx
  __int64 v80; // rdx
  __int64 v81; // rcx
  unsigned __int64 v82; // r8
  __int64 v83; // r9
  int v84; // eax
  unsigned int v85; // r12d
  _QWORD *v87; // r14
  char v88; // al
  char v89; // r10
  unsigned int v90; // eax
  _BYTE *v91; // rsi
  unsigned int v92; // edi
  __int64 v93; // rsi
  unsigned int v94; // eax
  __int64 *v95; // r14
  __int64 i; // r13
  char *v97; // rdx
  char *v98; // rdi
  __int64 v99; // rax
  __int64 v100; // rcx
  char *v101; // rax
  bool v102; // zf
  __int64 *v103; // rsi
  __int64 *v104; // rax
  __int64 v105; // rdx
  __int64 v106; // rdi
  unsigned int v107; // eax
  __int64 v108; // rdi
  __int64 *v109; // rax
  __int64 v110; // rsi
  __int64 *v111; // r10
  __int64 v112; // rdx
  __int64 *v113; // rax
  __int64 *v114; // rcx
  unsigned __int64 v115; // r13
  unsigned __int64 v116; // rdi
  __int64 v117; // r13
  __int64 v118; // rax
  __int64 v119; // r14
  __int64 j; // r15
  unsigned int k; // r13d
  __int64 v122; // rax
  unsigned int v123; // esi
  __int64 *v124; // rax
  _QWORD *v125; // [rsp+18h] [rbp-1C8h]
  __int64 v126; // [rsp+20h] [rbp-1C0h]
  __int64 v127; // [rsp+28h] [rbp-1B8h]
  _QWORD *v128; // [rsp+28h] [rbp-1B8h]
  __int64 *v129; // [rsp+28h] [rbp-1B8h]
  __int64 v130; // [rsp+30h] [rbp-1B0h]
  __int64 *v131; // [rsp+30h] [rbp-1B0h]
  __int64 v132; // [rsp+30h] [rbp-1B0h]
  __int64 v133; // [rsp+30h] [rbp-1B0h]
  __int64 v134; // [rsp+30h] [rbp-1B0h]
  __int16 v135; // [rsp+38h] [rbp-1A8h]
  __m128i *v137; // [rsp+48h] [rbp-198h]
  unsigned __int32 v138; // [rsp+48h] [rbp-198h]
  unsigned int v139; // [rsp+50h] [rbp-190h]
  __int64 v140; // [rsp+50h] [rbp-190h]
  unsigned int v141; // [rsp+50h] [rbp-190h]
  __int64 v142; // [rsp+58h] [rbp-188h]
  unsigned __int64 v143; // [rsp+68h] [rbp-178h]
  __int64 v144; // [rsp+68h] [rbp-178h]
  __m128i *v145; // [rsp+78h] [rbp-168h] BYREF
  _BYTE *v146; // [rsp+80h] [rbp-160h] BYREF
  _BYTE *v147; // [rsp+88h] [rbp-158h]
  _BYTE *v148; // [rsp+90h] [rbp-150h]
  __m128i v149; // [rsp+A0h] [rbp-140h] BYREF
  __int64 v150; // [rsp+B0h] [rbp-130h]
  __m128i v151; // [rsp+C0h] [rbp-120h] BYREF
  unsigned __int64 v152; // [rsp+D0h] [rbp-110h]
  char *v153; // [rsp+D8h] [rbp-108h]
  __int64 v154; // [rsp+E0h] [rbp-100h]
  __int64 v155; // [rsp+F0h] [rbp-F0h] BYREF
  char *v156; // [rsp+F8h] [rbp-E8h]
  __int64 v157; // [rsp+100h] [rbp-E0h]
  int v158; // [rsp+108h] [rbp-D8h]
  char v159; // [rsp+10Ch] [rbp-D4h]
  char v160; // [rsp+110h] [rbp-D0h] BYREF
  unsigned __int64 v161; // [rsp+150h] [rbp-90h] BYREF
  __m128i **n; // [rsp+158h] [rbp-88h]
  __int64 v163; // [rsp+160h] [rbp-80h]
  __int64 v164; // [rsp+168h] [rbp-78h]
  char v165; // [rsp+170h] [rbp-70h] BYREF
  const __m128i *v166; // [rsp+178h] [rbp-68h]
  __m128i *v167; // [rsp+180h] [rbp-60h]
  __int64 v168; // [rsp+188h] [rbp-58h]

  v3 = *(_QWORD *)(a1 + 8);
  v155 = 0;
  v156 = &v160;
  v157 = 8;
  v158 = 0;
  v159 = 1;
  v4 = sub_B82360(v3, (__int64)&unk_501FE44);
  if ( v4 && (v5 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v4 + 104LL))(v4, &unk_501FE44)) != 0 )
    v126 = v5 + 200;
  else
    v126 = 0;
  v6 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_50208AC);
  if ( v6 && (v10 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v6 + 104LL))(v6, &unk_50208AC)) != 0 )
    v11 = v10 + 200;
  else
    v11 = 0;
  v12 = &v161;
  v151.m128i_i64[0] = (__int64)a2;
  sub_3005920((__int64 *)&v161, &v151, (__int64)&v155, v7, v8, v9);
  v16 = v163;
  v17 = (unsigned __int64)n;
  v152 = 0;
  v151 = (__m128i)v161;
  v153 = 0;
  v18 = v163 - (_QWORD)n;
  if ( (__m128i **)v163 == n )
  {
    v18 = 0;
    v12 = 0;
  }
  else
  {
    if ( v18 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_244;
    v19 = sub_22077B0(v163 - (_QWORD)n);
    v16 = v163;
    v17 = (unsigned __int64)n;
    v12 = (unsigned __int64 *)v19;
  }
  v151.m128i_i64[1] = (__int64)v12;
  v152 = (unsigned __int64)v12;
  v153 = (char *)v12 + v18;
  if ( v17 == v16 )
  {
    v22 = (unsigned __int64)v12;
  }
  else
  {
    v20 = (__m128i *)v12;
    v21 = (const __m128i *)v17;
    do
    {
      if ( v20 )
      {
        *v20 = _mm_loadu_si128(v21);
        v14 = v21[1].m128i_i64[0];
        v20[1].m128i_i64[0] = v14;
      }
      v21 = (const __m128i *)((char *)v21 + 24);
      v20 = (__m128i *)((char *)v20 + 24);
    }
    while ( v21 != (const __m128i *)v16 );
    v22 = (unsigned __int64)&v12[(((unsigned __int64)&v21[-2].m128i_u64[1] - v17) >> 3) + 3];
  }
  v16 = (__int64)v167;
  v23 = (__int64)v166;
  v152 = v22;
  v13 = (char *)v167 - (char *)v166;
  v137 = (__m128i *)((char *)v167 - (char *)v166);
  if ( v167 == v166 )
  {
    v25 = 0;
    goto LABEL_95;
  }
  if ( v13 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_244:
    sub_4261EA(v12, v16, v13);
  v24 = sub_22077B0((char *)v167 - (char *)v166);
  v16 = (__int64)v167;
  v23 = (__int64)v166;
  v22 = v152;
  v12 = (unsigned __int64 *)v151.m128i_i64[1];
  v25 = v24;
  if ( v166 == v167 )
  {
LABEL_95:
    v143 = 0;
    goto LABEL_24;
  }
  v26 = (__m128i *)v24;
  v27 = v166;
  do
  {
    if ( v26 )
    {
      *v26 = _mm_loadu_si128(v27);
      v14 = v27[1].m128i_i64[0];
      v26[1].m128i_i64[0] = v14;
    }
    v27 = (const __m128i *)((char *)v27 + 24);
    v26 = (__m128i *)((char *)v26 + 24);
  }
  while ( v27 != (const __m128i *)v16 );
  v143 = 8 * (((unsigned __int64)&v27[-2].m128i_u64[1] - v23) >> 3) + 24;
  while ( 1 )
  {
LABEL_24:
    if ( v143 != v22 - (_QWORD)v12 )
      goto LABEL_25;
LABEL_36:
    if ( v12 == (unsigned __int64 *)v22 )
      break;
    v35 = v25;
    v36 = v12;
    while ( *v36 == *(_QWORD *)v35 )
    {
      v23 = *((unsigned __int8 *)v36 + 16);
      if ( (_BYTE)v23 != *(_BYTE *)(v35 + 16) )
        break;
      if ( (_BYTE)v23 )
      {
        v23 = *(_QWORD *)(v35 + 8);
        if ( v36[1] != v23 )
          break;
      }
      v36 += 3;
      v35 += 24LL;
      if ( v36 == (unsigned __int64 *)v22 )
        goto LABEL_43;
    }
LABEL_25:
    while ( 2 )
    {
      v28 = *(_QWORD *)(v22 - 24);
      if ( !*(_BYTE *)(v22 - 8) )
      {
        v29 = *(__int64 **)(v28 + 112);
        *(_BYTE *)(v22 - 8) = 1;
        *(_QWORD *)(v22 - 16) = v29;
        goto LABEL_27;
      }
      while ( 1 )
      {
        v29 = *(__int64 **)(v22 - 16);
LABEL_27:
        v23 = *(unsigned int *)(v28 + 120);
        if ( v29 == (__int64 *)(*(_QWORD *)(v28 + 112) + 8 * v23) )
          break;
        v30 = v29 + 1;
        *(_QWORD *)(v22 - 16) = v29 + 1;
        v31 = (_QWORD *)v151.m128i_i64[0];
        v32 = *v29;
        if ( !*(_BYTE *)(v151.m128i_i64[0] + 28) )
          goto LABEL_34;
        v33 = *(__int64 **)(v151.m128i_i64[0] + 8);
        v16 = *(unsigned int *)(v151.m128i_i64[0] + 20);
        v30 = &v33[v16];
        if ( v33 == v30 )
        {
LABEL_90:
          if ( (unsigned int)v16 < *(_DWORD *)(v151.m128i_i64[0] + 16) )
          {
            *(_DWORD *)(v151.m128i_i64[0] + 20) = v16 + 1;
            *v30 = v32;
            ++*v31;
LABEL_35:
            v16 = (__int64)&v149;
            v149.m128i_i64[0] = v32;
            LOBYTE(v150) = 0;
            sub_30058E0(&v151.m128i_u64[1], &v149);
            v22 = v152;
            v12 = (unsigned __int64 *)v151.m128i_i64[1];
            if ( v143 != v152 - v151.m128i_i64[1] )
              goto LABEL_25;
            goto LABEL_36;
          }
LABEL_34:
          v16 = v32;
          sub_C8CC70(v151.m128i_i64[0], v32, (__int64)v30, v23, v14, v15);
          if ( v34 )
            goto LABEL_35;
        }
        else
        {
          while ( v32 != *v33 )
          {
            if ( v30 == ++v33 )
              goto LABEL_90;
          }
        }
      }
      v152 -= 24LL;
      v12 = (unsigned __int64 *)v151.m128i_i64[1];
      v22 = v152;
      if ( v152 != v151.m128i_i64[1] )
        continue;
      break;
    }
  }
LABEL_43:
  if ( v25 )
  {
    v16 = (__int64)v137;
    j_j___libc_free_0(v25);
    v12 = (unsigned __int64 *)v151.m128i_i64[1];
  }
  if ( v12 )
  {
    v16 = v153 - (char *)v12;
    j_j___libc_free_0((unsigned __int64)v12);
  }
  if ( v166 )
  {
    v16 = v168 - (_QWORD)v166;
    j_j___libc_free_0((unsigned __int64)v166);
  }
  if ( n )
  {
    v16 = v164 - (_QWORD)n;
    j_j___libc_free_0((unsigned __int64)n);
  }
  v146 = 0;
  v147 = 0;
  v37 = a2[41];
  v148 = 0;
  v125 = a2 + 40;
  if ( (_QWORD *)v37 == a2 + 40 )
  {
    v45 = 0;
    goto LABEL_138;
  }
  while ( 2 )
  {
    if ( v159 )
    {
      v38 = v156;
      v39 = (__int64)&v156[8 * HIDWORD(v157)];
      if ( v156 == (char *)v39 )
      {
LABEL_154:
        v161 = v37;
        v91 = v147;
        if ( v147 == v148 )
        {
          sub_3005160((__int64)&v146, v147, &v161);
        }
        else
        {
          if ( v147 )
          {
            *(_QWORD *)v147 = v37;
            v91 = v147;
          }
          v147 = v91 + 8;
        }
        if ( v11 )
        {
          v92 = *(_DWORD *)(v11 + 24);
          v93 = *(_QWORD *)(v11 + 8);
          if ( v92 )
          {
            v14 = v92 - 1;
            v39 = ((unsigned int)v37 >> 4) ^ ((unsigned int)v37 >> 9);
            v94 = v14 & (((unsigned int)v37 >> 4) ^ ((unsigned int)v37 >> 9));
            v95 = (__int64 *)(v93 + 16LL * v94);
            v23 = *v95;
            if ( v37 == *v95 )
            {
LABEL_161:
              if ( v95 != (__int64 *)(v93 + 16LL * v92) )
              {
                for ( i = v95[1]; i; i = *(_QWORD *)i )
                {
                  v97 = *(char **)(i + 40);
                  v98 = *(char **)(i + 32);
                  v99 = (v97 - v98) >> 5;
                  v100 = (v97 - v98) >> 3;
                  if ( v99 > 0 )
                  {
                    v101 = &v98[32 * v99];
                    while ( *(_QWORD *)v98 != v37 )
                    {
                      if ( *((_QWORD *)v98 + 1) == v37 )
                      {
                        v98 += 8;
                        goto LABEL_170;
                      }
                      if ( *((_QWORD *)v98 + 2) == v37 )
                      {
                        v98 += 16;
                        goto LABEL_170;
                      }
                      if ( *((_QWORD *)v98 + 3) == v37 )
                      {
                        v98 += 24;
                        goto LABEL_170;
                      }
                      v98 += 32;
                      if ( v98 == v101 )
                      {
                        v100 = (v97 - v98) >> 3;
                        goto LABEL_217;
                      }
                    }
                    goto LABEL_170;
                  }
LABEL_217:
                  if ( v100 != 2 )
                  {
                    if ( v100 != 3 )
                    {
                      if ( v100 == 1 )
                        goto LABEL_228;
                      v98 = *(char **)(i + 40);
                      goto LABEL_170;
                    }
                    if ( *(_QWORD *)v98 == v37 )
                      goto LABEL_170;
                    v98 += 8;
                  }
                  if ( *(_QWORD *)v98 != v37 )
                  {
                    v98 += 8;
LABEL_228:
                    if ( *(_QWORD *)v98 != v37 )
                      v98 = *(char **)(i + 40);
                  }
LABEL_170:
                  if ( v98 + 8 != v97 )
                  {
                    memmove(v98, v98 + 8, v97 - (v98 + 8));
                    v97 = *(char **)(i + 40);
                  }
                  v102 = *(_BYTE *)(i + 84) == 0;
                  *(_QWORD *)(i + 40) = v97 - 8;
                  if ( v102 )
                  {
                    v124 = sub_C8CA60(i + 56, v37);
                    if ( v124 )
                    {
                      *v124 = -2;
                      ++*(_DWORD *)(i + 80);
                      ++*(_QWORD *)(i + 56);
                    }
                  }
                  else
                  {
                    v103 = *(__int64 **)(i + 64);
                    v23 = (__int64)&v103[*(unsigned int *)(i + 76)];
                    v39 = *(unsigned int *)(i + 76);
                    v104 = v103;
                    if ( v103 != (__int64 *)v23 )
                    {
                      while ( *v104 != v37 )
                      {
                        if ( (__int64 *)v23 == ++v104 )
                          goto LABEL_178;
                      }
                      v105 = (unsigned int)(v39 - 1);
                      *(_DWORD *)(i + 76) = v105;
                      v39 = v103[v105];
                      *v104 = v39;
                      ++*(_QWORD *)(i + 56);
                    }
                  }
LABEL_178:
                  ;
                }
                *v95 = -8192;
                --*(_DWORD *)(v11 + 16);
                ++*(_DWORD *)(v11 + 20);
              }
            }
            else
            {
              v39 = 1;
              while ( v23 != -4096 )
              {
                v15 = (unsigned int)(v39 + 1);
                v94 = v14 & (v39 + v94);
                v95 = (__int64 *)(v93 + 16LL * v94);
                v23 = *v95;
                if ( *v95 == v37 )
                  goto LABEL_161;
                v39 = (unsigned int)v15;
              }
            }
          }
        }
        if ( !v126 )
          goto LABEL_198;
        if ( v37 )
        {
          v106 = (unsigned int)(*(_DWORD *)(v37 + 24) + 1);
          v107 = *(_DWORD *)(v37 + 24) + 1;
        }
        else
        {
          v106 = 0;
          v107 = 0;
        }
        if ( *(_DWORD *)(v126 + 32) <= v107 )
          goto LABEL_198;
        v108 = 8 * v106;
        v109 = (__int64 *)(v108 + *(_QWORD *)(v126 + 24));
        v23 = *v109;
        if ( !*v109 )
          goto LABEL_198;
        *(_BYTE *)(v126 + 112) = 0;
        v14 = *(_QWORD *)(v23 + 8);
        if ( !v14 )
        {
LABEL_194:
          v115 = *v109;
          *v109 = 0;
          if ( v115 )
          {
            v116 = *(_QWORD *)(v115 + 24);
            if ( v116 != v115 + 40 )
              _libc_free(v116);
            j_j___libc_free_0(v115);
          }
LABEL_198:
          v16 = *(unsigned int *)(v37 + 120);
          if ( (_DWORD)v16 )
          {
            do
            {
              v117 = **(_QWORD **)(v37 + 112);
              v118 = sub_2E311E0(v117);
              v119 = *(_QWORD *)(v117 + 56);
              for ( j = v118; j != v119; v119 = *(_QWORD *)(v119 + 8) )
              {
                for ( k = (*(_DWORD *)(v119 + 40) & 0xFFFFFF) - 1; k > 1; k -= 2 )
                {
                  while ( 1 )
                  {
                    v122 = *(_QWORD *)(v119 + 32) + 40LL * k;
                    if ( *(_BYTE *)v122 == 4 && *(_QWORD *)(v122 + 24) == v37 )
                      break;
                    k -= 2;
                    if ( k <= 1 )
                      goto LABEL_206;
                  }
                  sub_2E8A650(v119, k);
                  v123 = k - 1;
                  sub_2E8A650(v119, v123);
                }
LABEL_206:
                if ( (*(_BYTE *)v119 & 4) == 0 )
                {
                  while ( (*(_BYTE *)(v119 + 44) & 8) != 0 )
                    v119 = *(_QWORD *)(v119 + 8);
                }
              }
              v16 = *(_QWORD *)(v37 + 112);
              sub_2E33590(v37, (__int64 *)v16, 0);
              v23 = *(unsigned int *)(v37 + 120);
            }
            while ( (_DWORD)v23 );
          }
          goto LABEL_57;
        }
        v15 = *(_QWORD *)(v14 + 24);
        v110 = 8LL * *(unsigned int *)(v14 + 32);
        v111 = (__int64 *)(v15 + v110);
        v112 = v110 >> 3;
        if ( v110 >> 5 )
        {
          v113 = *(__int64 **)(v14 + 24);
          while ( 1 )
          {
            v39 = *v113;
            if ( v23 == *v113 )
              goto LABEL_193;
            v39 = v113[1];
            if ( v23 == v39 )
            {
              ++v113;
              goto LABEL_193;
            }
            v39 = v113[2];
            if ( v23 == v39 )
            {
              v113 += 2;
              goto LABEL_193;
            }
            v39 = v113[3];
            if ( v23 == v39 )
            {
              v113 += 3;
              goto LABEL_193;
            }
            v113 += 4;
            if ( (__int64 *)(v15 + 32 * (v110 >> 5)) == v113 )
            {
              v112 = v111 - v113;
              goto LABEL_240;
            }
          }
        }
        v113 = *(__int64 **)(v14 + 24);
LABEL_240:
        if ( v112 != 2 )
        {
          if ( v112 != 3 )
          {
            if ( v112 != 1 )
            {
LABEL_243:
              v39 = *v111;
              v113 = (__int64 *)(v15 + v110);
LABEL_193:
              v114 = (__int64 *)(v15 + v110 - 8);
              *v113 = *v114;
              *v114 = v39;
              v23 = v126;
              --*(_DWORD *)(v14 + 32);
              v109 = (__int64 *)(v108 + *(_QWORD *)(v126 + 24));
              goto LABEL_194;
            }
LABEL_253:
            v39 = *v113;
            if ( v23 == *v113 )
              goto LABEL_193;
            goto LABEL_243;
          }
          v39 = *v113;
          if ( v23 == *v113 )
            goto LABEL_193;
          ++v113;
        }
        v39 = *v113;
        if ( v23 == *v113 )
          goto LABEL_193;
        ++v113;
        goto LABEL_253;
      }
      while ( *(_QWORD *)v38 != v37 )
      {
        v38 += 8;
        if ( (char *)v39 == v38 )
          goto LABEL_154;
      }
    }
    else
    {
      v16 = v37;
      if ( !sub_C8CA60((__int64)&v155, v37) )
        goto LABEL_154;
    }
LABEL_57:
    v37 = *(_QWORD *)(v37 + 8);
    if ( v125 != (_QWORD *)v37 )
      continue;
    break;
  }
  v40 = v146;
  v41 = v147;
  while ( v41 != v40 )
  {
    v42 = *(_QWORD **)v40;
    v43 = *(_QWORD *)(*(_QWORD *)v40 + 56LL);
    for ( m = *(_QWORD *)v40 + 48LL; m != v43; v43 = *(_QWORD *)(v43 + 8) )
    {
      while ( !sub_2E88F60(v43) )
      {
        v43 = *(_QWORD *)(v43 + 8);
        if ( m == v43 )
          goto LABEL_64;
      }
      v16 = v43;
      sub_2E79700(v42[4], v43);
    }
LABEL_64:
    v40 += 8;
    sub_2E32710(v42);
  }
  v45 = 0;
  if ( v125 != (_QWORD *)a2[41] )
  {
    v142 = a2[41];
    while ( 1 )
    {
      v46 = 1;
      v47 = *(__m128i ***)(v142 + 64);
      v48 = *(unsigned int *)(v142 + 72);
      v163 = 8;
      v161 = 0;
      v49 = &v47[v48];
      LODWORD(v164) = 0;
      BYTE4(v164) = 1;
      for ( n = (__m128i **)&v165; v49 != v47; ++v47 )
      {
        v16 = (__int64)*v47;
        if ( !v46 )
          goto LABEL_101;
        v50 = n;
        v23 = HIDWORD(v163);
        v39 = (__int64)&n[HIDWORD(v163)];
        if ( n != (__m128i **)v39 )
        {
          while ( (__m128i *)v16 != *v50 )
          {
            if ( (__m128i **)v39 == ++v50 )
              goto LABEL_102;
          }
          continue;
        }
LABEL_102:
        if ( HIDWORD(v163) < (unsigned int)v163 )
        {
          v23 = (unsigned int)++HIDWORD(v163);
          *(_QWORD *)v39 = v16;
          v46 = BYTE4(v164);
          ++v161;
        }
        else
        {
LABEL_101:
          sub_C8CC70((__int64)&v161, v16, v39, v23, v14, v15);
          v46 = BYTE4(v164);
        }
      }
      v51 = sub_2E311E0(v142);
      v52 = *(_QWORD *)(v142 + 56);
      v144 = v51;
      if ( v52 != v51 )
        break;
LABEL_135:
      if ( !BYTE4(v164) )
        _libc_free((unsigned __int64)n);
      v142 = *(_QWORD *)(v142 + 8);
      if ( v125 == (_QWORD *)v142 )
        goto LABEL_138;
    }
    while ( 1 )
    {
      if ( !v52 )
        BUG();
      v53 = v52;
      if ( (*(_BYTE *)v52 & 4) == 0 && (*(_BYTE *)(v52 + 44) & 8) != 0 )
      {
        do
          v53 = *(_QWORD *)(v53 + 8);
        while ( (*(_BYTE *)(v53 + 44) & 8) != 0 );
      }
      v54 = *(_QWORD *)(v53 + 8);
      v55 = *(_DWORD *)(v52 + 40) & 0xFFFFFF;
      v56 = v55 - 1;
      if ( (unsigned int)(v55 - 1) > 1 )
      {
        while ( 1 )
        {
          v16 = *(_QWORD *)(*(_QWORD *)(v52 + 32) + 40LL * v56 + 24);
          if ( BYTE4(v164) )
            break;
          if ( !sub_C8CA60((__int64)&v161, v16) )
            goto LABEL_100;
LABEL_85:
          v56 -= 2;
          if ( v56 <= 1 )
          {
            v55 = *(_DWORD *)(v52 + 40) & 0xFFFFFF;
            goto LABEL_87;
          }
        }
        v57 = n;
        v39 = (__int64)&n[HIDWORD(v163)];
        if ( n != (__m128i **)v39 )
        {
          while ( (__m128i *)v16 != *v57 )
          {
            if ( (__m128i **)v39 == ++v57 )
              goto LABEL_100;
          }
          goto LABEL_85;
        }
LABEL_100:
        v45 = 1;
        sub_2E8A650(v52, v56);
        v16 = v56 - 1;
        sub_2E8A650(v52, v16);
        goto LABEL_85;
      }
LABEL_87:
      if ( v55 != 3 )
        goto LABEL_88;
      v58 = *(_QWORD *)(v52 + 32);
      v23 = *(unsigned int *)(v58 + 8);
      v138 = *(_DWORD *)(v58 + 48);
      v139 = v23;
      if ( (_DWORD)v23 == v138 )
      {
        v45 = 1;
LABEL_88:
        if ( v54 == v144 )
          goto LABEL_135;
        goto LABEL_89;
      }
      v135 = (*(_DWORD *)(v58 + 40) >> 8) & 0xFFF;
      if ( !v135 )
      {
        v16 = v138;
        v87 = (_QWORD *)a2[4];
        if ( sub_2EBE590((__int64)v87, v138, *(_QWORD *)(v87[7] + 16 * (v23 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL, 0) )
        {
          if ( (*(_BYTE *)(v58 + 44) & 1) == 0 )
          {
            v16 = v139;
            sub_2EBECB0(v87, v139, v138);
            goto LABEL_134;
          }
        }
      }
      v59 = a2[2];
      v60 = *(__int64 (**)())(*(_QWORD *)v59 + 128LL);
      if ( v60 == sub_2DAC790 )
        BUG();
      v61 = ((__int64 (__fastcall *)(__int64, __int64))v60)(v59, v16);
      v62 = *(unsigned __int8 **)(v52 + 56);
      v63 = *(_QWORD *)(v61 + 8);
      v145 = (__m128i *)v62;
      v127 = v63 - 800;
      if ( v62 )
      {
        sub_B96E90((__int64)&v145, (__int64)v62, 1);
        v149.m128i_i64[0] = (__int64)v145;
        if ( v145 )
        {
          sub_B976B0((__int64)&v145, (unsigned __int8 *)v145, (__int64)&v149);
          v145 = 0;
        }
      }
      else
      {
        v149.m128i_i64[0] = 0;
      }
      v149.m128i_i64[1] = 0;
      v150 = 0;
      v64 = sub_2E311E0(v142);
      v65 = *(_QWORD **)(v142 + 32);
      v66 = v64;
      v151.m128i_i64[0] = v149.m128i_i64[0];
      if ( v149.m128i_i64[0] )
      {
        v130 = v64;
        sub_B96E90((__int64)&v151, v149.m128i_i64[0], 1);
        v66 = v130;
      }
      v131 = (__int64 *)v66;
      v67 = sub_2E7B380(v65, v127, (unsigned __int8 **)&v151, 0);
      v68 = v131;
      v69 = (__int64)v67;
      if ( v151.m128i_i64[0] )
      {
        v128 = v67;
        sub_B91220((__int64)&v151, v151.m128i_i64[0]);
        v69 = (__int64)v128;
        v68 = v131;
      }
      v129 = v68;
      v132 = v69;
      sub_2E31040((__int64 *)(v142 + 40), v69);
      v70 = v132;
      v71 = *(_QWORD *)v132;
      v72 = *v129;
      *(_QWORD *)(v132 + 8) = v129;
      v72 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v132 = v72 | v71 & 7;
      *(_QWORD *)(v72 + 8) = v132;
      *v129 = v132 | *v129 & 7;
      if ( v149.m128i_i64[1] )
      {
        sub_2E882B0(v132, (__int64)v65, v149.m128i_i64[1]);
        v70 = v132;
      }
      if ( v150 )
      {
        v133 = v70;
        sub_2E88680(v70, (__int64)v65, v150);
        v70 = v133;
      }
      v151.m128i_i64[0] = 0x10000000;
      v151.m128i_i32[2] = v139;
      v140 = v70;
      v152 = 0;
      v153 = 0;
      v154 = 0;
      sub_2E8EAD0(v70, (__int64)v65, &v151);
      v73 = *(unsigned __int8 *)(v58 + 43);
      v74 = v140;
      if ( (v73 & 0x10) != 0 )
        break;
      v75 = (v73 >> 3) & 4;
      v76 = (v73 & 0x40) != 0;
      if ( (v73 & 0x40) != 0 )
      {
        v75 |= 8u;
        goto LABEL_121;
      }
LABEL_123:
      v77 = *(_BYTE *)(v58 + 44);
      if ( (v77 & 1) != 0 )
        v75 |= 0x20u;
      v78 = v75;
      if ( (v77 & 2) != 0 )
      {
        BYTE1(v78) = BYTE1(v75) | 1;
        v75 = v78;
      }
      v79 = v75;
      if ( (v77 & 8) != 0 )
      {
        LOBYTE(v79) = v75 | 0x80;
        v75 = v79;
      }
      if ( (unsigned int)(*(_DWORD *)(v58 + 48) - 1) <= 0x3FFFFFFE )
      {
        v134 = v140;
        v141 = v75;
        v88 = sub_2EAB300(v58 + 40);
        v75 = v141;
        v74 = v134;
        v89 = v88;
        v90 = v141;
        if ( v89 )
        {
          BYTE1(v90) = BYTE1(v141) | 2;
          v75 = v90;
        }
      }
      v151.m128i_i8[0] = 0;
      v152 = 0;
      v153 = 0;
      v154 = 0;
      v151.m128i_i8[3] = ((unsigned __int8)(v75 >> 9) << 7)
                       | (((v75 & 0x18) != 0) << 6)
                       | (32 * ((v75 & 4) != 0)) & 0x3F
                       | (16 * ((v75 & 2) != 0)) & 0x3F
                       | v151.m128i_i8[3] & 0xF;
      v151.m128i_i16[1] &= 0xF00Fu;
      v151.m128i_i8[4] = (8 * ((unsigned __int8)v75 >> 7))
                       | (2 * (BYTE1(v75) & 1)) & 0xF3
                       | ((v75 & 0x20) != 0)
                       | v151.m128i_i8[4] & 0xF0;
      v151.m128i_i32[2] = v138;
      v151.m128i_i32[0] = ((v135 & 0xFFF) << 8) | v151.m128i_i32[0] & 0xFFF000FF;
      sub_2E8EAD0(v74, (__int64)v65, &v151);
      if ( v149.m128i_i64[0] )
        sub_B91220((__int64)&v149, v149.m128i_i64[0]);
      v16 = (__int64)v145;
      if ( v145 )
        sub_B91220((__int64)&v145, (__int64)v145);
LABEL_134:
      v45 = 1;
      sub_2E88E20(v52);
      if ( v54 == v144 )
        goto LABEL_135;
LABEL_89:
      v52 = v54;
    }
    v76 = (v73 & 0x40) != 0;
    v75 = (v73 & 0x20) == 0 ? 2 : 6;
LABEL_121:
    if ( v76 && (v73 & 0x10) != 0 )
      v75 |= 0x10u;
    goto LABEL_123;
  }
LABEL_138:
  sub_2E7A760((__int64)a2, 0);
  v84 = v126;
  if ( v126 )
    sub_30052F0(v126, 0, v80, v81, v82, v83);
  LOBYTE(v84) = v147 != v146;
  v85 = v84 | v45;
  if ( v146 )
    j_j___libc_free_0((unsigned __int64)v146);
  if ( !v159 )
    _libc_free((unsigned __int64)v156);
  return v85;
}
