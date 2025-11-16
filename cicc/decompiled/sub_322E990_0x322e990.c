// Function: sub_322E990
// Address: 0x322e990
//
__int64 __fastcall sub_322E990(_QWORD *a1, __int64 a2, _UNKNOWN **a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  void ***v7; // rbx
  void ***v8; // r13
  __int64 v9; // rax
  __m128i v10; // xmm0
  __int64 v11; // r12
  __int64 v12; // rax
  void **v13; // r12
  void *v14; // rax
  void *v15; // rax
  __int64 v16; // rdx
  unsigned __int64 v17; // rcx
  __int64 v18; // r13
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // r9
  __int64 v22; // rbx
  __int64 v23; // rax
  __int64 *v24; // rax
  __int64 v25; // r15
  __int64 *v26; // rax
  __int64 v27; // r12
  int v28; // r11d
  __int64 *v29; // rdi
  unsigned int v30; // edx
  __int64 *v31; // rax
  __m128i *v32; // rdx
  __m128i *v33; // rsi
  _QWORD *v34; // rdi
  __int64 v35; // r14
  __int64 *v36; // rbx
  __int64 v37; // rsi
  unsigned int v38; // edx
  int v39; // eax
  __int64 v40; // r8
  __int64 v41; // rax
  __int64 v42; // rsi
  int v43; // r11d
  __int64 *v44; // r10
  unsigned int v45; // edx
  __int64 v46; // r8
  __int64 v47; // rax
  unsigned __int64 *v48; // rbx
  __int64 v49; // r12
  __int64 *v50; // rdi
  int v51; // r11d
  __int64 *v52; // rax
  __m128i *v53; // rsi
  _QWORD *v54; // rdi
  unsigned __int64 v55; // rsi
  unsigned int v56; // edx
  int v57; // eax
  __int64 v58; // r8
  unsigned __int64 v59; // rax
  unsigned __int64 v60; // rsi
  __int64 *v61; // r10
  int v62; // r11d
  unsigned int v63; // edx
  __int64 v64; // r8
  __int64 v65; // rdi
  __int64 v66; // r12
  void (__fastcall *v67)(__int64, _QWORD, _QWORD); // rbx
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rdi
  unsigned int v71; // r14d
  unsigned int v72; // eax
  _QWORD *v73; // rbx
  _QWORD *v74; // r12
  unsigned __int64 v75; // rdi
  __int64 *v76; // rbx
  unsigned __int64 v77; // r12
  unsigned __int64 v78; // rdi
  __int64 *v80; // r12
  __int64 *v81; // rdx
  __int64 *v82; // rbx
  char *i; // rsi
  __int64 v84; // rax
  char *v85; // r12
  char *v86; // r13
  __int64 v87; // rbx
  unsigned __int64 v88; // rax
  char *v89; // rbx
  __int64 v90; // rdx
  __int64 v91; // rcx
  char *v92; // rax
  char *v93; // rsi
  _QWORD *v94; // r12
  int v95; // r10d
  __int64 *v96; // rax
  unsigned int v97; // ecx
  __int64 *v98; // rbx
  _QWORD *v99; // rdx
  _QWORD *v100; // rbx
  _QWORD *v101; // r13
  int v102; // edx
  unsigned __int64 v103; // rcx
  unsigned __int64 v104; // r12
  int v105; // eax
  __int64 v106; // rdi
  unsigned int v107; // r12d
  __int64 v108; // rdi
  __int64 v109; // r8
  void (*v110)(); // rax
  __int64 v111; // rdi
  void (*v112)(); // rax
  __int64 v113; // rdi
  __int64 v114; // r8
  void (*v115)(); // rax
  __int64 v116; // rdi
  __int64 v117; // r8
  void (*v118)(); // rax
  _QWORD *v119; // r12
  _QWORD *v120; // rbx
  __int64 v121; // rax
  __int64 v122; // r8
  __int64 v123; // rsi
  unsigned int v124; // ecx
  __int64 *v125; // rdx
  __int64 v126; // r9
  __int64 v127; // rsi
  __int64 v128; // rdi
  void (*v129)(); // rax
  int v130; // edx
  int v131; // edx
  int v132; // r10d
  unsigned int v133; // ecx
  __int64 v134; // rdi
  int v135; // ebx
  __int64 *v136; // r8
  int v137; // r11d
  unsigned int v138; // r13d
  __int64 *v139; // rdi
  __int64 v140; // rsi
  char *v141; // rsi
  int v142; // r11d
  int v143; // r11d
  void **v144; // r12
  char *v145; // [rsp+0h] [rbp-100h]
  __int64 v146; // [rsp+8h] [rbp-F8h]
  int v147; // [rsp+10h] [rbp-F0h]
  __m128i v149; // [rsp+20h] [rbp-E0h] BYREF
  void *src; // [rsp+30h] [rbp-D0h] BYREF
  char *v151; // [rsp+38h] [rbp-C8h]
  char *v152; // [rsp+40h] [rbp-C0h]
  __int64 v153; // [rsp+50h] [rbp-B0h] BYREF
  __int64 *v154; // [rsp+58h] [rbp-A8h]
  __int64 v155; // [rsp+60h] [rbp-A0h]
  unsigned int v156; // [rsp+68h] [rbp-98h]
  __m128i v157; // [rsp+70h] [rbp-90h] BYREF
  char v158; // [rsp+90h] [rbp-70h]
  char v159; // [rsp+91h] [rbp-6Fh]
  __int64 v160; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v161; // [rsp+A8h] [rbp-58h]
  __int64 v162; // [rsp+B0h] [rbp-50h]
  unsigned int v163; // [rsp+B8h] [rbp-48h]
  __int64 *v164; // [rsp+C0h] [rbp-40h]
  __int64 v165; // [rsp+C8h] [rbp-38h]
  _BYTE v166[48]; // [rsp+D0h] [rbp-30h] BYREF

  v6 = (__int64)a1;
  v7 = (void ***)a1[88];
  v8 = (void ***)a1[89];
  v160 = 0;
  v161 = 0;
  v162 = 0;
  v163 = 0;
  v164 = (__int64 *)v166;
  v165 = 0;
  if ( v7 == v8 )
  {
    v153 = 0;
    v154 = 0;
    v155 = 0;
    v156 = 0;
    goto LABEL_87;
  }
  do
  {
    v13 = *v7;
    v14 = **v7;
    if ( v14 )
    {
      a3 = &off_4C5D170;
      if ( v14 != off_4C5D170 )
        goto LABEL_4;
    }
    else if ( (*((_BYTE *)v13 + 9) & 0x70) == 0x20 && *((char *)v13 + 8) >= 0 )
    {
      *((_BYTE *)v13 + 8) |= 8u;
      v15 = sub_E807D0((__int64)v13[3]);
      *v13 = v15;
      if ( v15 )
      {
        a3 = &off_4C5D170;
        if ( v15 != off_4C5D170 )
        {
          v144 = *v7;
          v14 = **v7;
          if ( !v14 )
          {
            if ( (*((_BYTE *)v144 + 9) & 0x70) != 0x20 || *((char *)v144 + 8) < 0 )
              BUG();
            *((_BYTE *)v144 + 8) |= 8u;
            v14 = sub_E807D0((__int64)v144[3]);
            *v144 = v14;
          }
LABEL_4:
          v157.m128i_i64[0] = *((_QWORD *)v14 + 1);
          goto LABEL_5;
        }
      }
    }
    v157.m128i_i64[0] = 0;
LABEL_5:
    v9 = sub_322E5F0((__int64)&v160, v157.m128i_i64, (__int64)a3, a4, a5, a6);
    v10 = _mm_loadu_si128((const __m128i *)v7);
    v11 = v9;
    v12 = *(unsigned int *)(v9 + 8);
    a4 = *(unsigned int *)(v11 + 12);
    a3 = (_UNKNOWN **)(v12 + 1);
    if ( v12 + 1 > a4 )
    {
      v149 = v10;
      sub_C8D5F0(v11, (const void *)(v11 + 16), (unsigned __int64)a3, 0x10u, a5, a6);
      v12 = *(unsigned int *)(v11 + 8);
      v10 = _mm_load_si128(&v149);
    }
    v7 += 2;
    *(__m128i *)(*(_QWORD *)v11 + 16 * v12) = v10;
    ++*(_DWORD *)(v11 + 8);
  }
  while ( v8 != v7 );
  v153 = 0;
  v154 = 0;
  v155 = 0;
  v156 = 0;
  v16 = 19LL * (unsigned int)v165;
  v17 = (unsigned __int64)&v164[19 * (unsigned int)v165];
  v149.m128i_i64[0] = v17;
  if ( (__int64 *)v17 != v164 )
  {
    v18 = (__int64)(v164 + 1);
    v19 = *v164;
    if ( !*v164 )
      goto LABEL_54;
    while ( 2 )
    {
      v20 = sub_E9A820(*(__int64 **)(a1[1] + 224LL), v19, v16, v17, a5);
      v17 = *(unsigned int *)(v18 + 12);
      v22 = v20;
      v23 = *(unsigned int *)(v18 + 8);
      if ( v23 + 1 > v17 )
      {
        sub_C8D5F0(v18, (const void *)(v18 + 16), v23 + 1, 0x10u, a5, v21);
        v23 = *(unsigned int *)(v18 + 8);
      }
      v24 = (__int64 *)(*(_QWORD *)v18 + 16 * v23);
      *v24 = v22;
      v24[1] = 0;
      v25 = (unsigned int)(*(_DWORD *)(v18 + 8) + 1);
      v26 = *(__int64 **)v18;
      *(_DWORD *)(v18 + 8) = v25;
      v16 = *v26;
      if ( (unsigned int)v25 > 1 )
      {
        v27 = 1;
        while ( 1 )
        {
          v35 = (__int64)&v26[2 * v27 - 2];
          v36 = &v26[2 * v27];
          if ( v36[1] != *(_QWORD *)(v35 + 8) )
            break;
LABEL_25:
          if ( v25 == ++v27 )
            goto LABEL_52;
          v26 = *(__int64 **)v18;
        }
        v157.m128i_i64[0] = v16;
        v157.m128i_i64[1] = *v36;
        if ( !v156 )
        {
          ++v153;
          goto LABEL_30;
        }
        v17 = *(_QWORD *)(v35 + 8);
        v28 = 1;
        v29 = 0;
        v30 = (v156 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v31 = &v154[4 * v30];
        a5 = *v31;
        if ( *v31 == v17 )
        {
LABEL_20:
          v32 = (__m128i *)v31[2];
          v33 = (__m128i *)v31[3];
          v34 = v31 + 1;
          if ( v32 != v33 )
          {
            if ( v32 )
            {
              *v32 = _mm_load_si128(&v157);
              v32 = (__m128i *)v31[2];
            }
            v31[2] = (__int64)v32[1].m128i_i64;
            goto LABEL_24;
          }
LABEL_35:
          sub_3224BB0((__int64)v34, v33, &v157);
LABEL_24:
          v16 = *v36;
          goto LABEL_25;
        }
        while ( a5 != -4096 )
        {
          if ( a5 == -8192 && !v29 )
            v29 = v31;
          v30 = (v156 - 1) & (v28 + v30);
          v31 = &v154[4 * v30];
          a5 = *v31;
          if ( v17 == *v31 )
            goto LABEL_20;
          ++v28;
        }
        if ( !v29 )
          v29 = v31;
        ++v153;
        v39 = v155 + 1;
        if ( 4 * ((int)v155 + 1) >= 3 * v156 )
        {
LABEL_30:
          sub_3228980((__int64)&v153, 2 * v156);
          if ( !v156 )
            goto LABEL_243;
          v37 = *(_QWORD *)(v35 + 8);
          v38 = (v156 - 1) & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
          v39 = v155 + 1;
          v29 = &v154[4 * v38];
          v40 = *v29;
          if ( v37 != *v29 )
          {
            v142 = 1;
            v44 = 0;
            while ( v40 != -4096 )
            {
              if ( !v44 && v40 == -8192 )
                v44 = v29;
              v38 = (v156 - 1) & (v142 + v38);
              v29 = &v154[4 * v38];
              v40 = *v29;
              if ( v37 == *v29 )
                goto LABEL_32;
              ++v142;
            }
            goto LABEL_49;
          }
        }
        else if ( v156 - HIDWORD(v155) - v39 <= v156 >> 3 )
        {
          sub_3228980((__int64)&v153, v156);
          if ( !v156 )
          {
LABEL_243:
            LODWORD(v155) = v155 + 1;
            BUG();
          }
          v42 = *(_QWORD *)(v35 + 8);
          v43 = 1;
          v44 = 0;
          v45 = (v156 - 1) & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
          v39 = v155 + 1;
          v29 = &v154[4 * v45];
          v46 = *v29;
          if ( v42 != *v29 )
          {
            while ( v46 != -4096 )
            {
              if ( !v44 && v46 == -8192 )
                v44 = v29;
              v45 = (v156 - 1) & (v43 + v45);
              v29 = &v154[4 * v45];
              v46 = *v29;
              if ( v42 == *v29 )
                goto LABEL_32;
              ++v43;
            }
LABEL_49:
            if ( v44 )
              v29 = v44;
          }
        }
LABEL_32:
        LODWORD(v155) = v39;
        if ( *v29 != -4096 )
          --HIDWORD(v155);
        v41 = *(_QWORD *)(v35 + 8);
        v34 = v29 + 1;
        v33 = 0;
        *v34 = 0;
        v34[1] = 0;
        *(v34 - 1) = v41;
        v34[2] = 0;
        goto LABEL_35;
      }
LABEL_52:
      v47 = v18 + 152;
      if ( v149.m128i_i64[0] != v18 + 144 )
      {
        v18 += 152;
        v19 = *(_QWORD *)(v47 - 8);
        if ( v19 )
          continue;
LABEL_54:
        v48 = *(unsigned __int64 **)v18;
        v49 = *(_QWORD *)v18 + 16LL * *(unsigned int *)(v18 + 8);
        if ( *(_QWORD *)v18 == v49 )
          goto LABEL_52;
        while ( 2 )
        {
          v157 = (__m128i)*v48;
          if ( !v156 )
          {
            ++v153;
            goto LABEL_64;
          }
          v17 = v48[1];
          v50 = 0;
          v51 = 1;
          v16 = (v156 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
          v52 = &v154[4 * v16];
          a5 = *v52;
          if ( v17 == *v52 )
          {
LABEL_57:
            v53 = (__m128i *)v52[2];
            v54 = v52 + 1;
            if ( (__m128i *)v52[3] != v53 )
            {
              if ( v53 )
              {
                *v53 = _mm_load_si128(&v157);
                v53 = (__m128i *)v52[2];
              }
              v52[2] = (__int64)v53[1].m128i_i64;
LABEL_61:
              v48 += 2;
              if ( (unsigned __int64 *)v49 == v48 )
                goto LABEL_52;
              continue;
            }
LABEL_69:
            sub_3224BB0((__int64)v54, v53, &v157);
            goto LABEL_61;
          }
          break;
        }
        while ( a5 != -4096 )
        {
          if ( !v50 && a5 == -8192 )
            v50 = v52;
          v16 = (v156 - 1) & (v51 + (_DWORD)v16);
          v52 = &v154[4 * (unsigned int)v16];
          a5 = *v52;
          if ( v17 == *v52 )
            goto LABEL_57;
          ++v51;
        }
        if ( !v50 )
          v50 = v52;
        ++v153;
        v57 = v155 + 1;
        if ( 4 * ((int)v155 + 1) >= 3 * v156 )
        {
LABEL_64:
          sub_3228980((__int64)&v153, 2 * v156);
          if ( !v156 )
            goto LABEL_242;
          v55 = v48[1];
          v56 = (v156 - 1) & (((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4));
          v57 = v155 + 1;
          v50 = &v154[4 * v56];
          v58 = *v50;
          if ( v55 != *v50 )
          {
            v61 = 0;
            v143 = 1;
            while ( v58 != -4096 )
            {
              if ( !v61 && v58 == -8192 )
                v61 = v50;
              v56 = (v156 - 1) & (v143 + v56);
              v50 = &v154[4 * v56];
              v58 = *v50;
              if ( v55 == *v50 )
                goto LABEL_66;
              ++v143;
            }
            goto LABEL_83;
          }
        }
        else if ( v156 - HIDWORD(v155) - v57 <= v156 >> 3 )
        {
          sub_3228980((__int64)&v153, v156);
          if ( !v156 )
          {
LABEL_242:
            LODWORD(v155) = v155 + 1;
            BUG();
          }
          v60 = v48[1];
          v61 = 0;
          v62 = 1;
          v63 = (v156 - 1) & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
          v57 = v155 + 1;
          v50 = &v154[4 * v63];
          v64 = *v50;
          if ( v60 != *v50 )
          {
            while ( v64 != -4096 )
            {
              if ( !v61 && v64 == -8192 )
                v61 = v50;
              v63 = (v156 - 1) & (v62 + v63);
              v50 = &v154[4 * v63];
              v64 = *v50;
              if ( v60 == *v50 )
                goto LABEL_66;
              ++v62;
            }
LABEL_83:
            if ( v61 )
              v50 = v61;
          }
        }
LABEL_66:
        LODWORD(v155) = v57;
        if ( *v50 != -4096 )
          --HIDWORD(v155);
        v59 = v48[1];
        v54 = v50 + 1;
        v53 = 0;
        *v54 = 0;
        v54[1] = 0;
        *(v54 - 1) = v59;
        v54[2] = 0;
        goto LABEL_69;
      }
      break;
    }
    v6 = (__int64)a1;
  }
LABEL_87:
  v65 = *(_QWORD *)(v6 + 8);
  v66 = *(_QWORD *)(v65 + 224);
  v67 = *(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v66 + 176LL);
  v68 = sub_31DA6B0(v65);
  v67(v66, *(_QWORD *)(v68 + 152), 0);
  v69 = *(_QWORD *)(v6 + 8);
  v70 = (__int64)v154;
  src = 0;
  v71 = *(_DWORD *)(*(_QWORD *)(v69 + 208) + 8LL);
  v151 = 0;
  v152 = 0;
  if ( (_DWORD)v155 )
  {
    v80 = &v154[4 * v156];
    if ( v154 != v80 )
    {
      v81 = v154;
      while ( 1 )
      {
        v82 = v81;
        if ( *v81 != -8192 && *v81 != -4096 )
          break;
        v81 += 4;
        if ( v80 == v81 )
          goto LABEL_88;
      }
      if ( v81 != v80 )
      {
        v157.m128i_i64[0] = *v81;
        i = 0;
LABEL_174:
        sub_3224D40((__int64)&src, i, &v157);
        for ( i = v151; ; v151 = i )
        {
          v82 += 4;
          if ( v82 == v80 )
            break;
          while ( 1 )
          {
            v84 = *v82;
            if ( *v82 != -4096 && v84 != -8192 )
              break;
            v82 += 4;
            if ( v80 == v82 )
              goto LABEL_118;
          }
          if ( v82 == v80 )
            break;
          v157.m128i_i64[0] = *v82;
          if ( i == v152 )
            goto LABEL_174;
          if ( i )
          {
            *(_QWORD *)i = v84;
            i = v151;
          }
          i += 8;
        }
LABEL_118:
        v85 = (char *)src;
        v86 = i;
        if ( i == src )
          goto LABEL_156;
        v87 = i - (_BYTE *)src;
        _BitScanReverse64(&v88, (i - (_BYTE *)src) >> 3);
        sub_3218DE0((char *)src, i, 2LL * (int)(63 - (v88 ^ 0x3F)));
        if ( v87 <= 128 )
        {
          sub_3218510(v85, i);
        }
        else
        {
          v89 = v85 + 128;
          sub_3218510(v85, v85 + 128);
          if ( i != v85 + 128 )
          {
            do
            {
              while ( 1 )
              {
                v90 = *((_QWORD *)v89 - 1);
                v91 = *(_QWORD *)v89;
                v92 = v89 - 8;
                if ( *(_DWORD *)(*(_QWORD *)v89 + 72LL) < *(_DWORD *)(v90 + 72) )
                  break;
                v141 = v89;
                v89 += 8;
                *(_QWORD *)v141 = v91;
                if ( v86 == v89 )
                  goto LABEL_124;
              }
              do
              {
                *((_QWORD *)v92 + 1) = v90;
                v93 = v92;
                v90 = *((_QWORD *)v92 - 1);
                v92 -= 8;
              }
              while ( *(_DWORD *)(v91 + 72) < *(_DWORD *)(v90 + 72) );
              v89 += 8;
              *(_QWORD *)v93 = v91;
            }
            while ( v86 != v89 );
          }
        }
LABEL_124:
        v85 = (char *)src;
        v145 = v151;
        if ( src == v151 )
          goto LABEL_156;
        v149.m128i_i64[0] = (__int64)src;
LABEL_126:
        v94 = *(_QWORD **)v149.m128i_i64[0];
        if ( v156 )
        {
          v95 = 1;
          v96 = 0;
          v97 = (v156 - 1) & (((unsigned int)v94 >> 9) ^ ((unsigned int)v94 >> 4));
          v98 = &v154[4 * v97];
          v99 = (_QWORD *)*v98;
          if ( v94 == (_QWORD *)*v98 )
            goto LABEL_128;
          while ( v99 != (_QWORD *)-4096LL )
          {
            if ( v96 || v99 != (_QWORD *)-8192LL )
              v98 = v96;
            v97 = (v156 - 1) & (v95 + v97);
            v99 = (_QWORD *)v154[4 * v97];
            if ( v94 == v99 )
            {
              v98 = &v154[4 * v97];
LABEL_128:
              v100 = v98 + 1;
LABEL_129:
              v101 = (_QWORD *)v94[51];
              if ( !v101 )
                v101 = v94;
              v102 = sub_31DF6B0(*(_QWORD *)(v6 + 8)) + 4;
              if ( 2 * v71 )
              {
                _BitScanReverse64(&v103, 2 * v71);
                v104 = 0x8000000000000000LL >> ((unsigned __int8)v103 ^ 0x3Fu);
                v146 = -(__int64)v104;
              }
              else
              {
                LODWORD(v146) = 0;
                LODWORD(v104) = 0;
              }
              v147 = v102;
              v105 = sub_31DF740(*(_QWORD *)(v6 + 8));
              v106 = *(_QWORD *)(v6 + 8);
              v107 = (v146 & (v147 + v105 + v104 - 1)) - (v147 + v105);
              v159 = 1;
              v157.m128i_i64[0] = (__int64)"Length of ARange Set";
              v158 = 3;
              sub_31F0F40(v106);
              v108 = *(_QWORD *)(v6 + 8);
              v109 = *(_QWORD *)(v108 + 224);
              v110 = *(void (**)())(*(_QWORD *)v109 + 120LL);
              v159 = 1;
              v157.m128i_i64[0] = (__int64)"DWARF Arange version number";
              v158 = 3;
              if ( v110 != nullsub_98 )
              {
                ((void (__fastcall *)(__int64, __m128i *, __int64))v110)(v109, &v157, 1);
                v108 = *(_QWORD *)(v6 + 8);
              }
              sub_31DC9F0(v108, 2);
              v111 = *(_QWORD *)(*(_QWORD *)(v6 + 8) + 224LL);
              v112 = *(void (**)())(*(_QWORD *)v111 + 120LL);
              v159 = 1;
              v157.m128i_i64[0] = (__int64)"Offset Into Debug Info Section";
              v158 = 3;
              if ( v112 != nullsub_98 )
                ((void (__fastcall *)(__int64, __m128i *, __int64))v112)(v111, &v157, 1);
              sub_321F970(v6, v101);
              v113 = *(_QWORD *)(v6 + 8);
              v114 = *(_QWORD *)(v113 + 224);
              v115 = *(void (**)())(*(_QWORD *)v114 + 120LL);
              v159 = 1;
              v157.m128i_i64[0] = (__int64)"Address Size (in bytes)";
              v158 = 3;
              if ( v115 != nullsub_98 )
              {
                ((void (__fastcall *)(__int64, __m128i *, __int64))v115)(v114, &v157, 1);
                v113 = *(_QWORD *)(v6 + 8);
              }
              sub_31DC9D0(v113, v71);
              v116 = *(_QWORD *)(v6 + 8);
              v117 = *(_QWORD *)(v116 + 224);
              v118 = *(void (**)())(*(_QWORD *)v117 + 120LL);
              v159 = 1;
              v157.m128i_i64[0] = (__int64)"Segment Size (in bytes)";
              v158 = 3;
              if ( v118 != nullsub_98 )
              {
                ((void (__fastcall *)(__int64, __m128i *, __int64))v118)(v117, &v157, 1);
                v116 = *(_QWORD *)(v6 + 8);
              }
              sub_31DC9D0(v116, 0);
              sub_E99280(*(_QWORD ***)(*(_QWORD *)(v6 + 8) + 224LL), v107, 0xFFu);
              v119 = (_QWORD *)v100[1];
              v120 = (_QWORD *)*v100;
              if ( v120 == v119 )
              {
LABEL_152:
                v128 = *(_QWORD *)(*(_QWORD *)(v6 + 8) + 224LL);
                v129 = *(void (**)())(*(_QWORD *)v128 + 120LL);
                v159 = 1;
                v157.m128i_i64[0] = (__int64)"ARange terminator";
                v158 = 3;
                if ( v129 != nullsub_98 )
                {
                  ((void (__fastcall *)(__int64, __m128i *, __int64))v129)(v128, &v157, 1);
                  v128 = *(_QWORD *)(*(_QWORD *)(v6 + 8) + 224LL);
                }
                (*(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v128 + 536LL))(v128, 0, v71);
                (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(*(_QWORD *)(v6 + 8) + 224LL) + 536LL))(
                  *(_QWORD *)(*(_QWORD *)(v6 + 8) + 224LL),
                  0,
                  v71);
                v149.m128i_i64[0] += 8;
                if ( v145 == (char *)v149.m128i_i64[0] )
                {
                  v85 = (char *)src;
LABEL_156:
                  if ( v85 )
                    j_j___libc_free_0((unsigned __int64)v85);
                  v70 = (__int64)v154;
                  goto LABEL_88;
                }
                goto LABEL_126;
              }
              while ( 2 )
              {
                while ( 2 )
                {
                  (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD))(**(_QWORD **)(v6 + 8) + 432LL))(
                    *(_QWORD *)(v6 + 8),
                    *v120,
                    0,
                    v71,
                    0);
                  v121 = *(unsigned int *)(v6 + 752);
                  v122 = *v120;
                  v123 = *(_QWORD *)(v6 + 736);
                  if ( (_DWORD)v121 )
                  {
                    v124 = (v121 - 1) & (((unsigned int)v122 >> 9) ^ ((unsigned int)v122 >> 4));
                    v125 = (__int64 *)(v123 + 16LL * v124);
                    v126 = *v125;
                    if ( v122 == *v125 )
                    {
LABEL_147:
                      if ( v125 != (__int64 *)(v123 + 16 * v121) )
                      {
                        v127 = v125[1];
                        if ( v127 && v120[1] )
                        {
LABEL_144:
                          v120 += 2;
                          sub_31DCA50(*(_QWORD *)(v6 + 8));
                          if ( v119 == v120 )
                            goto LABEL_152;
                          continue;
                        }
                        if ( !v127 )
                          v127 = 1;
                        goto LABEL_151;
                      }
                    }
                    else
                    {
                      v131 = 1;
                      while ( v126 != -4096 )
                      {
                        v132 = v131 + 1;
                        v124 = (v121 - 1) & (v131 + v124);
                        v125 = (__int64 *)(v123 + 16LL * v124);
                        v126 = *v125;
                        if ( v122 == *v125 )
                          goto LABEL_147;
                        v131 = v132;
                      }
                    }
                  }
                  break;
                }
                v127 = 1;
                if ( v120[1] )
                  goto LABEL_144;
LABEL_151:
                v120 += 2;
                (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)(v6 + 8) + 224LL) + 536LL))(
                  *(_QWORD *)(*(_QWORD *)(v6 + 8) + 224LL),
                  v127,
                  v71);
                if ( v119 == v120 )
                  goto LABEL_152;
                continue;
              }
            }
            ++v95;
            v96 = v98;
            v98 = &v154[4 * v97];
          }
          if ( !v96 )
            v96 = v98;
          ++v153;
          v130 = v155 + 1;
          if ( 4 * ((int)v155 + 1) < 3 * v156 )
          {
            if ( v156 - HIDWORD(v155) - v130 > v156 >> 3 )
            {
LABEL_165:
              LODWORD(v155) = v130;
              if ( *v96 != -4096 )
                --HIDWORD(v155);
              *v96 = (__int64)v94;
              v100 = v96 + 1;
              v96[1] = 0;
              v96[2] = 0;
              v96[3] = 0;
              goto LABEL_129;
            }
            sub_3228980((__int64)&v153, v156);
            if ( v156 )
            {
              v137 = 1;
              v138 = (v156 - 1) & (((unsigned int)v94 >> 9) ^ ((unsigned int)v94 >> 4));
              v130 = v155 + 1;
              v139 = 0;
              v96 = &v154[4 * v138];
              v140 = *v96;
              if ( v94 != (_QWORD *)*v96 )
              {
                while ( v140 != -4096 )
                {
                  if ( !v139 && v140 == -8192 )
                    v139 = v96;
                  v138 = (v156 - 1) & (v137 + v138);
                  v96 = &v154[4 * v138];
                  v140 = *v96;
                  if ( v94 == (_QWORD *)*v96 )
                    goto LABEL_165;
                  ++v137;
                }
                if ( v139 )
                  v96 = v139;
              }
              goto LABEL_165;
            }
LABEL_240:
            LODWORD(v155) = v155 + 1;
            BUG();
          }
        }
        else
        {
          ++v153;
        }
        sub_3228980((__int64)&v153, 2 * v156);
        if ( v156 )
        {
          v133 = (v156 - 1) & (((unsigned int)v94 >> 9) ^ ((unsigned int)v94 >> 4));
          v130 = v155 + 1;
          v96 = &v154[4 * v133];
          v134 = *v96;
          if ( v94 != (_QWORD *)*v96 )
          {
            v135 = 1;
            v136 = 0;
            while ( v134 != -4096 )
            {
              if ( v134 == -8192 && !v136 )
                v136 = v96;
              v133 = (v156 - 1) & (v135 + v133);
              v96 = &v154[4 * v133];
              v134 = *v96;
              if ( v94 == (_QWORD *)*v96 )
                goto LABEL_165;
              ++v135;
            }
            if ( v136 )
              v96 = v136;
          }
          goto LABEL_165;
        }
        goto LABEL_240;
      }
    }
  }
LABEL_88:
  v72 = v156;
  if ( v156 )
  {
    v73 = (_QWORD *)v70;
    v74 = (_QWORD *)(v70 + 32LL * v156);
    do
    {
      if ( *v73 != -4096 && *v73 != -8192 )
      {
        v75 = v73[1];
        if ( v75 )
          j_j___libc_free_0(v75);
      }
      v73 += 4;
    }
    while ( v74 != v73 );
    v72 = v156;
    v70 = (__int64)v154;
  }
  sub_C7D6A0(v70, 32LL * v72, 8);
  v76 = v164;
  v77 = (unsigned __int64)&v164[19 * (unsigned int)v165];
  if ( v164 != (__int64 *)v77 )
  {
    do
    {
      v77 -= 152LL;
      v78 = *(_QWORD *)(v77 + 8);
      if ( v78 != v77 + 24 )
        _libc_free(v78);
    }
    while ( v76 != (__int64 *)v77 );
    v77 = (unsigned __int64)v164;
  }
  if ( (_BYTE *)v77 != v166 )
    _libc_free(v77);
  return sub_C7D6A0(v161, 16LL * v163, 8);
}
