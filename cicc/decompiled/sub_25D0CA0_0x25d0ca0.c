// Function: sub_25D0CA0
// Address: 0x25d0ca0
//
void __fastcall sub_25D0CA0(
        __int64 a1,
        __int64 a2,
        int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  unsigned __int32 i; // eax
  __int64 v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 *v15; // r10
  __int64 *v16; // rbx
  int v17; // ecx
  unsigned int v18; // edx
  __int64 v19; // rdi
  __int64 v20; // rax
  int v21; // ecx
  __int64 v22; // rsi
  __int64 v23; // r15
  float v24; // xmm0_4
  char v25; // r14
  float v26; // xmm1_4
  int v27; // r11d
  unsigned int v28; // esi
  __int64 v29; // r9
  __int64 v30; // r8
  __int64 v31; // rdi
  __int64 v32; // r13
  __int64 v33; // rcx
  __int64 v34; // r15
  float v35; // xmm0_4
  __int64 v36; // rax
  int v37; // edx
  __int64 v38; // rax
  int v39; // r8d
  unsigned __int64 v40; // rax
  __int64 v41; // r8
  __int64 v42; // rdi
  __int64 v43; // rsi
  __int64 v44; // r8
  __int64 v45; // r11
  __m128i v46; // xmm3
  __m128i v47; // xmm4
  __m128i v48; // xmm5
  __int64 v49; // rsi
  __int64 *v50; // r15
  __int64 v51; // r8
  __int64 *v52; // r12
  char v53; // bl
  char v54; // r14
  __int64 v55; // rdi
  int v56; // r9d
  __int64 v57; // rdx
  __int64 v58; // rsi
  char *v59; // r12
  char *v60; // rsi
  __int64 v61; // rdx
  unsigned __int64 v62; // rax
  unsigned __int64 v63; // rdi
  __m128i *v64; // rax
  __m128i *v65; // rax
  unsigned __int64 v66; // rdx
  __m128i *v67; // rax
  unsigned __int64 v68; // rcx
  __int64 (__fastcall **v69)(); // rax
  __int64 v70; // r15
  __int64 v71; // rax
  __int64 v72; // rbx
  __int64 *v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // r8
  __int64 v77; // r9
  int v78; // eax
  char v79; // al
  int v80; // ecx
  __int64 v81; // rax
  __int64 v82; // r14
  __int64 v83; // rcx
  __int64 v84; // rax
  unsigned __int64 v85; // rdx
  unsigned __int64 v86; // rdi
  __int64 v87; // rcx
  __int64 v88; // rax
  __int32 v89; // r15d
  __int64 v90; // rdx
  unsigned __int8 v91; // al
  int v92; // eax
  __int64 v93; // rax
  __int64 v94; // rdx
  char v95; // al
  __int64 *v96; // r10
  bool v97; // zf
  char *v98; // rax
  char *v99; // rsi
  int v100; // edx
  unsigned int v101; // esi
  __m128i v102; // xmm6
  int v103; // r8d
  __int64 v104; // rdi
  int v105; // r8d
  unsigned int v106; // eax
  __int64 v107; // rdx
  int v108; // r14d
  int v109; // r8d
  int v110; // r8d
  __int64 v111; // rdi
  int v112; // r14d
  unsigned int v113; // edx
  __int64 v114; // rax
  char v115; // r14
  __int64 v116; // r15
  __int64 v117; // rax
  char v118; // r14
  unsigned __int64 v119; // rdi
  __int64 v120; // [rsp+10h] [rbp-590h]
  __int64 v121; // [rsp+18h] [rbp-588h]
  __int64 *v122; // [rsp+20h] [rbp-580h]
  __int64 *v123; // [rsp+28h] [rbp-578h]
  __int64 *v124; // [rsp+30h] [rbp-570h]
  __int64 v125; // [rsp+30h] [rbp-570h]
  int v126; // [rsp+30h] [rbp-570h]
  __int64 *v127; // [rsp+30h] [rbp-570h]
  unsigned __int64 v128; // [rsp+30h] [rbp-570h]
  __int64 *v130; // [rsp+40h] [rbp-560h]
  __int64 *v131; // [rsp+40h] [rbp-560h]
  int v132; // [rsp+40h] [rbp-560h]
  char v133; // [rsp+48h] [rbp-558h]
  __int64 v134; // [rsp+48h] [rbp-558h]
  __int64 *v135; // [rsp+48h] [rbp-558h]
  int v136; // [rsp+48h] [rbp-558h]
  int v139; // [rsp+70h] [rbp-530h]
  __int64 *v140; // [rsp+70h] [rbp-530h]
  __int64 *v141; // [rsp+70h] [rbp-530h]
  __int64 v143; // [rsp+88h] [rbp-518h] BYREF
  unsigned __int64 v144[2]; // [rsp+90h] [rbp-510h] BYREF
  __m128i v145; // [rsp+A0h] [rbp-500h] BYREF
  _BYTE *v146; // [rsp+B0h] [rbp-4F0h] BYREF
  size_t v147; // [rsp+B8h] [rbp-4E8h]
  _QWORD v148[2]; // [rsp+C0h] [rbp-4E0h] BYREF
  char *v149; // [rsp+D0h] [rbp-4D0h] BYREF
  size_t v150; // [rsp+D8h] [rbp-4C8h]
  _QWORD v151[5]; // [rsp+E0h] [rbp-4C0h] BYREF
  __int64 v152; // [rsp+108h] [rbp-498h]
  __int64 v153; // [rsp+110h] [rbp-490h]
  __m128i v154; // [rsp+120h] [rbp-480h] BYREF
  __m128i v155; // [rsp+130h] [rbp-470h] BYREF
  __m128i v156; // [rsp+140h] [rbp-460h]
  __int64 v157; // [rsp+150h] [rbp-450h]
  __m128i v158; // [rsp+160h] [rbp-440h] BYREF
  __m128i v159; // [rsp+170h] [rbp-430h] BYREF
  __m128i v160; // [rsp+180h] [rbp-420h] BYREF
  __int64 v161; // [rsp+190h] [rbp-410h]
  __int64 *v162; // [rsp+198h] [rbp-408h]

  v158.m128i_i64[0] = (__int64)&v159;
  v158.m128i_i64[1] = 0x8000000000LL;
  sub_25D06E0(a6, a1, (__int64)&v158);
  for ( i = v158.m128i_u32[2]; v158.m128i_i32[2]; i = v158.m128i_u32[2] )
  {
    v12 = *(_QWORD *)(v158.m128i_i64[0] + 8LL * i - 8);
    v158.m128i_i32[2] = i - 1;
    sub_25D06E0(a6, v12, (__int64)&v158);
  }
  if ( (__m128i *)v158.m128i_i64[0] != &v159 )
    _libc_free(v158.m128i_u64[0]);
  v13 = *(_QWORD *)(a1 + 64);
  v14 = 16LL * *(unsigned int *)(a1 + 72);
  if ( v13 == v13 + v14 )
    return;
  v15 = (__int64 *)(v13 + v14);
  v16 = *(__int64 **)(a1 + 64);
LABEL_9:
  v143 = *v16;
  if ( (int)qword_4FF13C8 >= 0 && (int)qword_4FF13C8 <= dword_4FF06A8 )
    goto LABEL_8;
  v20 = v143;
  v21 = *(_DWORD *)(a4 + 24);
  v22 = *(_QWORD *)(a4 + 8);
  v23 = *(_QWORD *)(v143 & 0xFFFFFFFFFFFFFFF8LL);
  if ( v21 )
  {
    v17 = v21 - 1;
    v18 = v17 & (((0xBF58476D1CE4E5B9LL * v23) >> 31) ^ (484763065 * v23));
    v19 = *(_QWORD *)(v22 + 16LL * v18);
    if ( v23 == v19 )
      goto LABEL_8;
    v39 = 1;
    while ( v19 != -1 )
    {
      v18 = v17 & (v39 + v18);
      v19 = *(_QWORD *)(v22 + 16LL * v18);
      if ( v23 == v19 )
        goto LABEL_8;
      ++v39;
    }
  }
  v24 = (float)a3;
  v25 = v16[1] & 7;
  v26 = (float)a3;
  switch ( v25 )
  {
    case 3:
      v26 = v24 * *(float *)&qword_4FF1048;
      break;
    case 1:
      v26 = v24 * *(float *)&qword_4FF0E88;
      break;
    case 4:
      v26 = v24 * *(float *)&qword_4FF0F68;
      break;
  }
  v27 = (int)v26;
  v28 = *(_DWORD *)(a9 + 24);
  if ( v28 )
  {
    v29 = v28 - 1;
    v30 = *(_QWORD *)(a9 + 8);
    v31 = ((unsigned int)((0xBF58476D1CE4E5B9LL * v23) >> 31) ^ (484763065 * (_DWORD)v23)) & (v28 - 1);
    v32 = v30 + 32 * v31;
    v33 = *(_QWORD *)v32;
    if ( v23 == *(_QWORD *)v32 )
      goto LABEL_18;
    v132 = 1;
    v134 = 0;
    while ( v33 != -1 )
    {
      if ( !v134 )
      {
        if ( v33 != -2 )
          v32 = 0;
        v134 = v32;
      }
      LODWORD(v31) = v29 & (v132 + v31);
      v32 = v30 + 32LL * (unsigned int)v31;
      v33 = *(_QWORD *)v32;
      if ( v23 == *(_QWORD *)v32 )
      {
        v20 = v143;
LABEL_18:
        v34 = *(_QWORD *)(v32 + 16);
        if ( v34 )
        {
          if ( (float)*(int *)(v32 + 24) < v26 )
          {
            *(_DWORD *)(v32 + 24) = v27;
            goto LABEL_21;
          }
          goto LABEL_8;
        }
        if ( (float)*(int *)(v32 + 24) < v26 )
        {
          v133 = 1;
          goto LABEL_37;
        }
        if ( (_BYTE)qword_4FF0CC8 )
          ++*(_DWORD *)(*(_QWORD *)(v32 + 8) + 16LL);
LABEL_8:
        v16 += 2;
        if ( v15 == v16 )
          return;
        goto LABEL_9;
      }
      ++v132;
    }
    if ( v134 )
      v32 = v134;
    ++*(_QWORD *)a9;
    v80 = *(_DWORD *)(a9 + 16) + 1;
    if ( 4 * v80 >= 3 * v28 )
      goto LABEL_126;
    if ( v28 - *(_DWORD *)(a9 + 20) - v80 <= v28 >> 3 )
    {
      v123 = v15;
      v128 = ((0xBF58476D1CE4E5B9LL * v23) >> 31) ^ (0xBF58476D1CE4E5B9LL * v23);
      sub_25CFC20(a9, v28);
      v109 = *(_DWORD *)(a9 + 24);
      if ( !v109 )
      {
LABEL_167:
        ++*(_DWORD *)(a9 + 16);
        BUG();
      }
      v110 = v109 - 1;
      v111 = *(_QWORD *)(a9 + 8);
      v29 = 0;
      v112 = 1;
      v27 = (int)v26;
      v113 = v110 & v128;
      v15 = v123;
      v80 = *(_DWORD *)(a9 + 16) + 1;
      v32 = v111 + 32LL * (v110 & (unsigned int)v128);
      v114 = *(_QWORD *)v32;
      if ( v23 != *(_QWORD *)v32 )
      {
        while ( v114 != -1 )
        {
          if ( v114 == -2 && !v29 )
            v29 = v32;
          v113 = v110 & (v112 + v113);
          v32 = v111 + 32LL * v113;
          v114 = *(_QWORD *)v32;
          if ( v23 == *(_QWORD *)v32 )
            goto LABEL_89;
          ++v112;
        }
        goto LABEL_130;
      }
    }
  }
  else
  {
    ++*(_QWORD *)a9;
LABEL_126:
    v127 = v15;
    sub_25CFC20(a9, 2 * v28);
    v103 = *(_DWORD *)(a9 + 24);
    if ( !v103 )
      goto LABEL_167;
    v104 = *(_QWORD *)(a9 + 8);
    v105 = v103 - 1;
    v27 = (int)v26;
    v15 = v127;
    v80 = *(_DWORD *)(a9 + 16) + 1;
    v106 = v105 & (((0xBF58476D1CE4E5B9LL * v23) >> 31) ^ (484763065 * v23));
    v32 = v104 + 32LL * v106;
    v107 = *(_QWORD *)v32;
    if ( v23 != *(_QWORD *)v32 )
    {
      v108 = 1;
      v29 = 0;
      while ( v107 != -1 )
      {
        if ( !v29 && v107 == -2 )
          v29 = v32;
        v106 = v105 & (v108 + v106);
        v32 = v104 + 32LL * v106;
        v107 = *(_QWORD *)v32;
        if ( v23 == *(_QWORD *)v32 )
          goto LABEL_89;
        ++v108;
      }
LABEL_130:
      if ( v29 )
        v32 = v29;
    }
  }
LABEL_89:
  *(_DWORD *)(a9 + 16) = v80;
  if ( *(_QWORD *)v32 != -1 )
    --*(_DWORD *)(a9 + 20);
  *(_QWORD *)v32 = v23;
  v20 = v143;
  *(_QWORD *)(v32 + 8) = 0;
  *(_QWORD *)(v32 + 16) = 0;
  *(_DWORD *)(v32 + 24) = v27;
  v133 = 0;
  v25 = v16[1] & 7;
LABEL_37:
  v40 = v20 & 0xFFFFFFFFFFFFFFF8LL;
  v124 = v15;
  v41 = *(_QWORD *)(v40 + 32);
  v42 = *(_QWORD *)(a1 + 32);
  v43 = *(_QWORD *)(v40 + 24);
  v152 = *(_QWORD *)(a1 + 24);
  v44 = v41 - v43;
  v151[3] = v43;
  v153 = v42;
  v151[4] = v44 >> 3;
  v151[2] = a2;
  sub_25CD050((__int64)&v158, v43, v44 >> 3, v152, v44, v29, a2, v43, v44 >> 3, v152, v42);
  v46 = _mm_load_si128(&v158);
  v47 = _mm_load_si128(&v159);
  v48 = _mm_load_si128(&v160);
  v130 = v162;
  v157 = v161;
  v49 = v159.m128i_i64[1];
  v15 = v124;
  v154 = v46;
  v155 = v47;
  v156 = v48;
  if ( (__int64 *)v158.m128i_i64[0] == v162 )
  {
    *(_QWORD *)(v32 + 16) = 0;
    v56 = 0;
  }
  else
  {
    v50 = (__int64 *)v158.m128i_i64[0];
    v51 = a4;
    v52 = v16;
    v125 = 0;
    v53 = v25;
    v54 = *(_BYTE *)(v158.m128i_i64[1] + 336);
    do
    {
      v55 = *v50;
      if ( !v54 || (v56 = 2, *(char *)(v55 + 12) < 0) )
      {
        switch ( *(_BYTE *)(v55 + 12) & 0xF )
        {
          case 0:
          case 1:
          case 3:
          case 5:
          case 6:
          case 7:
          case 8:
            v78 = *(_DWORD *)(v55 + 8);
            if ( !v78 )
            {
              v55 = *(_QWORD *)(v55 + 64);
              v78 = *(_DWORD *)(v55 + 8);
            }
            if ( v78 == 1 )
            {
              v120 = v45;
              v121 = v51;
              v122 = v15;
              v79 = sub_25CD160(v55, v49, (const void *)v156.m128i_i64[0], v156.m128i_i64[1]);
              v15 = v122;
              v51 = v121;
              v45 = v120;
              if ( v79 )
              {
                v56 = 5;
              }
              else if ( (*(_BYTE *)(v55 + 12) & 0x40) != 0 )
              {
                v56 = 6;
              }
              else
              {
                v92 = *(_DWORD *)(v55 + 60);
                if ( (unsigned int)(int)v26 >= *(_DWORD *)(v55 + 56) || (v92 & 0x20) != 0 || (_BYTE)qword_4FF12E8 )
                {
                  if ( (v92 & 0x10) == 0 || (_BYTE)qword_4FF12E8 )
                  {
                    *(_QWORD *)(v32 + 16) = v55;
                    v25 = v53;
                    v34 = v55;
                    v16 = v52;
                    a4 = v121;
                    if ( !*(_DWORD *)(v55 + 8) )
                      v34 = *(_QWORD *)(v55 + 64);
                    v93 = v143;
                    *(_QWORD *)(v32 + 16) = v34;
                    v94 = *(_QWORD *)(v34 + 32);
                    v154.m128i_i64[0] = *(_QWORD *)(v34 + 24);
                    v154.m128i_i64[1] = v94;
                    sub_25D0260(a7, v154.m128i_i64[0], v94, *(_QWORD *)(v93 & 0xFFFFFFFFFFFFFFF8LL));
                    v15 = v122;
                    if ( a8 )
                    {
                      v95 = sub_25CE0A0(a8, (__int64)&v154, &v149);
                      v96 = v122;
                      v97 = v95 == 0;
                      v98 = v149;
                      v99 = v149 + 16;
                      if ( v97 )
                      {
                        v158.m128i_i64[0] = (__int64)v149;
                        ++*(_QWORD *)a8;
                        v100 = *(_DWORD *)(a8 + 16) + 1;
                        v101 = *(_DWORD *)(a8 + 24);
                        if ( 4 * v100 >= 3 * v101 )
                        {
                          sub_25CE770(a8, 2 * v101);
                          sub_25CE0A0(a8, (__int64)&v154, &v158);
                          v96 = v122;
                          v100 = *(_DWORD *)(a8 + 16) + 1;
                          v98 = (char *)v158.m128i_i64[0];
                        }
                        else if ( v101 - *(_DWORD *)(a8 + 20) - v100 <= v101 >> 3 )
                        {
                          sub_25CE770(a8, v101);
                          sub_25CE0A0(a8, (__int64)&v154, &v158);
                          v96 = v122;
                          v100 = *(_DWORD *)(a8 + 16) + 1;
                          v98 = (char *)v158.m128i_i64[0];
                        }
                        *(_DWORD *)(a8 + 16) = v100;
                        if ( *(_QWORD *)v98 != -1 )
                          --*(_DWORD *)(a8 + 20);
                        v102 = _mm_load_si128(&v154);
                        *((_QWORD *)v98 + 2) = 0;
                        v99 = v98 + 16;
                        *((_QWORD *)v98 + 3) = 0;
                        *((_QWORD *)v98 + 4) = 0;
                        *((_DWORD *)v98 + 10) = 0;
                        *(__m128i *)v98 = v102;
                      }
                      v135 = v96;
                      sub_25CF280((__int64)&v158, (__int64)v99, &v143);
                      v15 = v135;
                    }
LABEL_21:
                    if ( v25 == 3 )
                      v35 = v24 * *(float *)&qword_4FF1128;
                    else
                      v35 = v24 * *(float *)&qword_4FF1208;
                    ++dword_4FF06A8;
                    v36 = *(unsigned int *)(a5 + 8);
                    v37 = v36;
                    if ( *(_DWORD *)(a5 + 12) <= (unsigned int)v36 )
                    {
                      v140 = v15;
                      v81 = sub_C8D7D0(a5, a5 + 16, 0, 0x10u, (unsigned __int64 *)&v158, v29);
                      v15 = v140;
                      v82 = v81;
                      v83 = 16LL * *(unsigned int *)(a5 + 8);
                      v84 = v83 + v81;
                      if ( v84 )
                      {
                        *(_QWORD *)(v84 + 8) = v34;
                        *(_DWORD *)v84 = (int)v35;
                        v83 = 16LL * *(unsigned int *)(a5 + 8);
                      }
                      v85 = *(_QWORD *)a5;
                      v86 = *(_QWORD *)a5 + v83;
                      if ( *(_QWORD *)a5 != v86 )
                      {
                        v87 = v82 + v83;
                        v88 = v82;
                        do
                        {
                          if ( v88 )
                          {
                            *(_DWORD *)v88 = *(_DWORD *)v85;
                            *(_QWORD *)(v88 + 8) = *(_QWORD *)(v85 + 8);
                          }
                          v88 += 16;
                          v85 += 16LL;
                        }
                        while ( v88 != v87 );
                        v86 = *(_QWORD *)a5;
                      }
                      v89 = v158.m128i_i32[0];
                      if ( v86 != a5 + 16 )
                      {
                        _libc_free(v86);
                        v15 = v140;
                      }
                      ++*(_DWORD *)(a5 + 8);
                      *(_QWORD *)a5 = v82;
                      *(_DWORD *)(a5 + 12) = v89;
                      goto LABEL_8;
                    }
                    v38 = *(_QWORD *)a5 + 16 * v36;
                    if ( v38 )
                    {
                      *(_QWORD *)(v38 + 8) = v34;
                      *(_DWORD *)v38 = (int)v35;
                      v37 = *(_DWORD *)(a5 + 8);
                    }
                    v16 += 2;
                    *(_DWORD *)(a5 + 8) = v37 + 1;
                    if ( v15 == v16 )
                      return;
                    goto LABEL_9;
                  }
                  v125 = v55;
                  v56 = 7;
                }
                else
                {
                  v125 = v55;
                  v56 = 3;
                }
              }
            }
            else
            {
              v56 = 1;
            }
            break;
          case 2:
          case 4:
          case 9:
          case 0xA:
            v56 = 4;
            break;
          default:
            BUG();
        }
      }
      ++v50;
    }
    while ( v130 != v50 );
    *(_QWORD *)(v32 + 16) = 0;
    v16 = v52;
    a4 = v51;
    if ( (_BYTE)qword_4FF0848 && v125 )
    {
      v57 = *(_QWORD *)(v125 + 32);
      v58 = *(_QWORD *)(v125 + 24);
      v126 = v56;
      v131 = v15;
      v139 = v45;
      sub_25D0520(a7, v58, v57, *(_QWORD *)(v143 & 0xFFFFFFFFFFFFFFF8LL));
      v56 = v126;
      v15 = v131;
      LODWORD(v45) = v139;
    }
  }
  if ( v133 )
  {
    *(_DWORD *)(v32 + 24) = v45;
    if ( (_BYTE)qword_4FF0CC8 )
    {
      *(_DWORD *)(*(_QWORD *)(v32 + 8) + 12LL) = v56;
      ++*(_DWORD *)(*(_QWORD *)(v32 + 8) + 16LL);
      v90 = *(_QWORD *)(v32 + 8);
      v91 = v16[1] & 7;
      if ( v91 < *(_BYTE *)(v90 + 8) )
        v91 = *(_BYTE *)(v90 + 8);
      *(_BYTE *)(v90 + 8) = v91;
    }
  }
  else if ( (_BYTE)qword_4FF0CC8 )
  {
    v115 = *((_BYTE *)v16 + 8);
    v136 = v56;
    v141 = v15;
    v116 = v143;
    v117 = sub_22077B0(0x18u);
    v118 = v115 & 7;
    v15 = v141;
    v56 = v136;
    if ( v117 )
    {
      *(_QWORD *)v117 = v116;
      *(_BYTE *)(v117 + 8) = v118;
      *(_DWORD *)(v117 + 12) = v136;
      *(_DWORD *)(v117 + 16) = 1;
    }
    v119 = *(_QWORD *)(v32 + 8);
    *(_QWORD *)(v32 + 8) = v117;
    if ( v119 )
    {
      j_j___libc_free_0(v119);
      v56 = v136;
      v15 = v141;
    }
  }
  if ( !(_BYTE)qword_4FF12E8 )
    goto LABEL_8;
  v59 = sub_25CCE70(v56);
  v60 = *(char **)((v143 & 0xFFFFFFFFFFFFFFF8LL) + 8);
  if ( (v143 & 1) != 0 )
    v60 = (char *)sub_BD5D20(*(_QWORD *)((v143 & 0xFFFFFFFFFFFFFFF8LL) + 8));
  else
    v61 = *(_QWORD *)((v143 & 0xFFFFFFFFFFFFFFF8LL) + 16);
  if ( v60 )
  {
    v149 = (char *)v151;
    sub_25CCF50((__int64 *)&v149, v60, (__int64)&v60[v61]);
  }
  else
  {
    LOBYTE(v151[0]) = 0;
    v149 = (char *)v151;
    v150 = 0;
  }
  v146 = v148;
  sub_25CCF50((__int64 *)&v146, "Failed to import function ", (__int64)"");
  v62 = 15;
  v63 = 15;
  if ( v146 != (_BYTE *)v148 )
    v63 = v148[0];
  if ( v147 + v150 <= v63 )
    goto LABEL_58;
  if ( v149 != (char *)v151 )
    v62 = v151[0];
  if ( v147 + v150 <= v62 )
    v64 = (__m128i *)sub_2241130((unsigned __int64 *)&v149, 0, 0, v146, v147);
  else
LABEL_58:
    v64 = (__m128i *)sub_2241490((unsigned __int64 *)&v146, v149, v150);
  v154.m128i_i64[0] = (__int64)&v155;
  if ( (__m128i *)v64->m128i_i64[0] == &v64[1] )
  {
    v155 = _mm_loadu_si128(v64 + 1);
  }
  else
  {
    v154.m128i_i64[0] = v64->m128i_i64[0];
    v155.m128i_i64[0] = v64[1].m128i_i64[0];
  }
  v154.m128i_i64[1] = v64->m128i_i64[1];
  v64->m128i_i64[0] = (__int64)v64[1].m128i_i64;
  v64->m128i_i64[1] = 0;
  v64[1].m128i_i8[0] = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v154.m128i_i64[1]) <= 7 )
    goto LABEL_166;
  v65 = (__m128i *)sub_2241490((unsigned __int64 *)&v154, " due to ", 8u);
  v158.m128i_i64[0] = (__int64)&v159;
  if ( (__m128i *)v65->m128i_i64[0] == &v65[1] )
  {
    v159 = _mm_loadu_si128(v65 + 1);
  }
  else
  {
    v158.m128i_i64[0] = v65->m128i_i64[0];
    v159.m128i_i64[0] = v65[1].m128i_i64[0];
  }
  v158.m128i_i64[1] = v65->m128i_i64[1];
  v65->m128i_i64[0] = (__int64)v65[1].m128i_i64;
  v65->m128i_i64[1] = 0;
  v65[1].m128i_i8[0] = 0;
  v66 = strlen(v59);
  if ( v66 > 0x3FFFFFFFFFFFFFFFLL - v158.m128i_i64[1] )
LABEL_166:
    sub_4262D8((__int64)"basic_string::append");
  v67 = (__m128i *)sub_2241490((unsigned __int64 *)&v158, v59, v66);
  v144[0] = (unsigned __int64)&v145;
  if ( (__m128i *)v67->m128i_i64[0] == &v67[1] )
  {
    v145 = _mm_loadu_si128(v67 + 1);
  }
  else
  {
    v144[0] = v67->m128i_i64[0];
    v145.m128i_i64[0] = v67[1].m128i_i64[0];
  }
  v68 = v67->m128i_u64[1];
  v67[1].m128i_i8[0] = 0;
  v144[1] = v68;
  v67->m128i_i64[0] = (__int64)v67[1].m128i_i64;
  v67->m128i_i64[1] = 0;
  sub_2240A30((unsigned __int64 *)&v158);
  if ( (__m128i *)v154.m128i_i64[0] != &v155 )
    j_j___libc_free_0(v154.m128i_u64[0]);
  sub_2240A30((unsigned __int64 *)&v146);
  sub_2240A30((unsigned __int64 *)&v149);
  v69 = sub_2241E50();
  v158.m128i_i64[0] = (__int64)v144;
  v160.m128i_i16[0] = 260;
  v70 = (__int64)v69;
  v71 = sub_22077B0(0x40u);
  v72 = v71;
  if ( v71 )
    sub_C63EB0(v71, (__int64)&v158, 95, v70);
  v158.m128i_i64[0] = (__int64)"Error importing module: ";
  v160.m128i_i16[0] = 259;
  v73 = (__int64 *)sub_CB72A0();
  v154.m128i_i64[0] = v72 | 1;
  v149 = 0;
  sub_C63F70((unsigned __int64 *)&v154, v73, v74, v75, v76, v77, v158.m128i_i8[0]);
  sub_9C66B0(v154.m128i_i64);
  sub_9C66B0((__int64 *)&v149);
  sub_2240A30(v144);
}
