// Function: sub_1AA7EA0
// Address: 0x1aa7ea0
//
__int64 __fastcall sub_1AA7EA0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __m128 a6,
        __m128i a7,
        __m128i a8,
        double a9,
        double a10,
        double a11,
        double a12,
        __m128 a13)
{
  unsigned int v13; // r12d
  __int64 v15; // r14
  __int64 v18; // rbx
  int v19; // eax
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // rcx
  _QWORD *v25; // rax
  _QWORD *v26; // rcx
  __int64 v27; // rax
  __int64 v28; // rax
  unsigned __int64 v29; // rsi
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // rax
  int v32; // eax
  unsigned int v33; // r12d
  __int64 v34; // r13
  int v35; // r15d
  const __m128i *v36; // rsi
  __int64 v37; // rax
  const __m128i *v38; // rsi
  __int64 v39; // rax
  __int64 v40; // r12
  unsigned __int64 *v41; // rcx
  unsigned __int64 v42; // rdx
  double v43; // xmm4_8
  double v44; // xmm5_8
  double v45; // xmm4_8
  double v46; // xmm5_8
  int v47; // r9d
  __int64 v48; // rcx
  unsigned __int64 v49; // rdx
  __int64 v50; // rcx
  unsigned __int64 v51; // r12
  __int64 v52; // rsi
  _QWORD **v53; // r12
  _QWORD **v54; // rbx
  __int64 v55; // rdx
  __int64 v56; // rax
  __int64 v57; // rax
  int v58; // r10d
  __int64 v59; // rdi
  unsigned int v60; // ecx
  __int64 *v61; // rdx
  __int64 v62; // rsi
  __int64 *v63; // rax
  __int64 v64; // rsi
  int v65; // r11d
  unsigned int i; // edx
  __int64 *v67; // rcx
  __int64 v68; // r8
  _BYTE *v69; // rbx
  _BYTE *v70; // r12
  size_t v71; // r10
  __int64 v72; // r14
  __int64 *v73; // r15
  _BYTE *v74; // rsi
  __int64 v75; // rbx
  __int64 v76; // rax
  _BYTE *v77; // rax
  __int64 v78; // rdx
  __int64 v79; // rcx
  int v80; // r8d
  int v81; // r9d
  unsigned int v82; // eax
  __int64 v83; // rcx
  int v84; // edi
  unsigned int j; // edx
  __int64 *v86; // rsi
  __int64 v87; // r12
  __int64 v88; // rdx
  __int64 v89; // r8
  _BYTE *v90; // rax
  __int64 v91; // r8
  __int64 v92; // rax
  __int64 v93; // rcx
  unsigned int v94; // edx
  __int64 *v95; // rbx
  int v96; // esi
  __int64 v97; // rdi
  __int64 *v98; // r12
  char *v99; // rdx
  char *v100; // rdi
  __int64 v101; // rax
  char *v102; // rax
  _QWORD *v103; // rax
  __int64 v104; // rax
  _QWORD *v105; // rdx
  __int64 v106; // rdx
  __int64 v107; // rcx
  signed __int64 v108; // rcx
  __int64 v109; // rax
  __int64 v110; // rdx
  int v111; // r8d
  int v112; // r9d
  double v113; // xmm4_8
  double v114; // xmm5_8
  __int64 v115; // r12
  __int64 v116; // r13
  __int64 v117; // r15
  __int64 v118; // rax
  __int64 v119; // rbx
  __int64 v120; // rax
  __int64 v121; // rax
  unsigned int v122; // eax
  int v123; // edx
  unsigned int k; // edi
  _QWORD *v125; // rbx
  __int64 v126; // r12
  __int64 v127; // rdi
  int v128; // edx
  int v129; // r8d
  unsigned int v130; // edx
  unsigned int v131; // edi
  unsigned int v132; // edx
  __int64 v133; // [rsp+10h] [rbp-170h]
  __int64 v134; // [rsp+10h] [rbp-170h]
  unsigned __int64 v135; // [rsp+18h] [rbp-168h]
  __int64 v136; // [rsp+18h] [rbp-168h]
  __int64 v137; // [rsp+18h] [rbp-168h]
  __int64 v138; // [rsp+18h] [rbp-168h]
  __int64 v139; // [rsp+18h] [rbp-168h]
  __int64 v140; // [rsp+20h] [rbp-160h]
  unsigned __int64 v141; // [rsp+20h] [rbp-160h]
  _QWORD *v142; // [rsp+30h] [rbp-150h]
  __int64 *v143; // [rsp+30h] [rbp-150h]
  unsigned int v144; // [rsp+30h] [rbp-150h]
  __int64 v145; // [rsp+30h] [rbp-150h]
  _BYTE *v146; // [rsp+38h] [rbp-148h]
  __int64 v147; // [rsp+38h] [rbp-148h]
  __int64 v148; // [rsp+38h] [rbp-148h]
  __int64 v151[2]; // [rsp+50h] [rbp-130h] BYREF
  const __m128i *v152; // [rsp+60h] [rbp-120h] BYREF
  __m128i *v153; // [rsp+68h] [rbp-118h]
  const __m128i *v154; // [rsp+70h] [rbp-110h]
  _BYTE *v155; // [rsp+80h] [rbp-100h] BYREF
  __int64 v156; // [rsp+88h] [rbp-F8h]
  _BYTE v157[16]; // [rsp+90h] [rbp-F0h] BYREF
  __int64 v158[4]; // [rsp+A0h] [rbp-E0h] BYREF
  char v159; // [rsp+C0h] [rbp-C0h]
  _BYTE *v160; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 v161; // [rsp+D8h] [rbp-A8h]
  _BYTE v162[32]; // [rsp+E0h] [rbp-A0h] BYREF
  __m128i v163; // [rsp+100h] [rbp-80h] BYREF
  _QWORD v164[14]; // [rsp+110h] [rbp-70h] BYREF

  if ( *(_WORD *)(a1 + 18) )
    return 0;
  v15 = a1;
  v18 = sub_157F120(a1);
  LOBYTE(v13) = v18 == 0 || a1 == v18;
  if ( (_BYTE)v13 )
    return 0;
  v19 = *(unsigned __int8 *)(sub_157EBA0(v18) + 16);
  if ( (unsigned int)(v19 - 24) > 6 )
  {
    if ( (unsigned int)(v19 - 32) <= 2 )
      return v13;
  }
  else if ( (unsigned int)(v19 - 24) > 4 )
  {
    return v13;
  }
  if ( a1 != sub_157F210(v18) )
    return 0;
  v20 = sub_157F280(a1);
  v22 = v21;
  v23 = v20;
  while ( 1 )
  {
    if ( v22 == v23 )
    {
      v160 = v162;
      v161 = 0x400000000LL;
      v28 = *(_QWORD *)(a1 + 48);
      if ( !v28 )
        BUG();
      if ( *(_BYTE *)(v28 - 8) == 77 )
      {
        v109 = sub_157F280(a1);
        if ( v109 != v110 )
        {
          v148 = a2;
          v115 = v109;
          v116 = v110;
          v145 = a5;
          v117 = v18;
          do
          {
            if ( (*(_BYTE *)(v115 + 23) & 0x40) != 0 )
              v118 = *(_QWORD *)(v115 - 8);
            else
              v118 = v115 - 24LL * (*(_DWORD *)(v115 + 20) & 0xFFFFFFF);
            v119 = *(_QWORD *)v118;
            if ( *(_BYTE *)(*(_QWORD *)v118 + 16LL) != 77 || a1 != *(_QWORD *)(v119 + 40) )
            {
              v120 = (unsigned int)v161;
              if ( (unsigned int)v161 >= HIDWORD(v161) )
              {
                sub_16CD150((__int64)&v160, v162, 0, 8, v111, v112);
                v120 = (unsigned int)v161;
              }
              *(_QWORD *)&v160[8 * v120] = v119;
              LODWORD(v161) = v161 + 1;
            }
            v121 = *(_QWORD *)(v115 + 32);
            if ( !v121 )
              BUG();
            v115 = 0;
            if ( *(_BYTE *)(v121 - 8) == 77 )
              v115 = v121 - 24;
          }
          while ( v116 != v115 );
          v18 = v117;
          a2 = v148;
          a5 = v145;
        }
        sub_1AA62D0(a1, a4, a6, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9, v113, v114, a12, a13);
      }
      v152 = 0;
      v153 = 0;
      v154 = 0;
      if ( a5 )
      {
        v29 = 1;
        v30 = sub_157EBA0(a1);
        if ( v30 )
          v29 = 2 * (unsigned int)sub_15F4D60(v30) + 1;
        sub_1953AE0(&v152, v29);
        v163.m128i_i64[0] = v18;
        v163.m128i_i64[1] = v15 | 4;
        if ( v153 == v154 )
        {
          sub_17F2860(&v152, v153, &v163);
        }
        else
        {
          if ( v153 )
          {
            a8 = _mm_loadu_si128(&v163);
            *v153 = a8;
          }
          ++v153;
        }
        v31 = sub_157EBA0(v15);
        if ( v31 )
        {
          v135 = v31;
          v32 = sub_15F4D60(v31);
          if ( v32 )
          {
            v140 = a2;
            v33 = 0;
            v34 = v135;
            v136 = a5;
            v35 = v32;
            do
            {
              v39 = sub_15F4DF0(v34, v33);
              v163.m128i_i64[0] = v15;
              v36 = v153;
              v163.m128i_i64[1] = v39 | 4;
              if ( v153 == v154 )
              {
                sub_17F2860(&v152, v153, &v163);
              }
              else
              {
                if ( v153 )
                {
                  a6 = (__m128)_mm_loadu_si128(&v163);
                  *v153 = (__m128i)a6;
                  v36 = v153;
                }
                v153 = (__m128i *)&v36[1];
              }
              v37 = sub_15F4DF0(v34, v33);
              v163.m128i_i64[0] = v18;
              v38 = v153;
              v163.m128i_i64[1] = v37 & 0xFFFFFFFFFFFFFFFBLL;
              if ( v153 == v154 )
              {
                sub_17F2860(&v152, v153, &v163);
              }
              else
              {
                if ( v153 )
                {
                  a7 = _mm_loadu_si128(&v163);
                  *v153 = a7;
                  v38 = v153;
                }
                v153 = (__m128i *)&v38[1];
              }
              ++v33;
            }
            while ( v35 != v33 );
            a2 = v140;
            a5 = v136;
          }
        }
      }
      v40 = v18 + 40;
      v142 = (_QWORD *)(*(_QWORD *)(v18 + 40) & 0xFFFFFFFFFFFFFFF8LL);
      sub_157EA20(v18 + 40, (__int64)(v142 - 3));
      v41 = (unsigned __int64 *)v142[1];
      v42 = *v142 & 0xFFFFFFFFFFFFFFF8LL;
      *v41 = v42 | *v41 & 7;
      *(_QWORD *)(v42 + 8) = v41;
      *v142 &= 7uLL;
      v142[1] = 0;
      sub_164BEC0(
        (__int64)(v142 - 3),
        (__int64)(v142 - 3),
        v42,
        (__int64)v41,
        a6,
        *(double *)a7.m128i_i64,
        *(double *)a8.m128i_i64,
        a9,
        v43,
        v44,
        a12,
        a13);
      sub_164D160(v15, v18, a6, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9, v45, v46, a12, a13);
      v48 = v15 + 40;
      if ( v15 + 40 != (*(_QWORD *)(v15 + 40) & 0xFFFFFFFFFFFFFFF8LL) && v40 != v48 )
      {
        v143 = *(__int64 **)(v15 + 48);
        sub_157EA80(v18 + 40, v15 + 40, (__int64)v143, v48);
        if ( (__int64 *)(v15 + 40) != v143 )
        {
          v49 = *(_QWORD *)(v15 + 40) & 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)((*v143 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v15 + 40;
          *(_QWORD *)(v15 + 40) = *(_QWORD *)(v15 + 40) & 7LL | *v143 & 0xFFFFFFFFFFFFFFF8LL;
          v50 = *(_QWORD *)(v18 + 40);
          *(_QWORD *)(v49 + 8) = v40;
          v50 &= 0xFFFFFFFFFFFFFFF8LL;
          *v143 = v50 | *v143 & 7;
          *(_QWORD *)(v50 + 8) = v143;
          *(_QWORD *)(v18 + 40) = v49 | *(_QWORD *)(v18 + 40) & 7LL;
        }
      }
      v51 = (unsigned __int64)v160;
      v146 = &v160[8 * (unsigned int)v161];
      if ( v160 != v146 )
      {
        v137 = v18;
        do
        {
          v52 = *(_QWORD *)v51;
          if ( *(_BYTE *)(*(_QWORD *)v51 + 16LL) > 0x17u )
          {
            v163.m128i_i64[0] = 0;
            v155 = v157;
            v156 = 0x200000000LL;
            v163.m128i_i64[1] = 1;
            v164[0] = -8;
            v164[1] = -8;
            v164[2] = -8;
            v164[3] = -8;
            sub_1AEA1F0(&v155, v52);
            if ( v155 != &v155[8 * (unsigned int)v156] )
            {
              v141 = v51;
              v53 = (_QWORD **)v155;
              v54 = (_QWORD **)&v155[8 * (unsigned int)v156];
              do
              {
                v55 = *((_DWORD *)*v53 + 5) & 0xFFFFFFF;
                v56 = (*v53)[3 * (2 - v55)];
                v151[0] = *(_QWORD *)((*v53)[3 * (1 - v55)] + 24LL);
                v151[1] = *(_QWORD *)(v56 + 24);
                sub_1AA79E0((__int64)v158, &v163, v151);
                if ( !v159 )
                  sub_15F20C0(*v53);
                ++v53;
              }
              while ( v54 != v53 );
              v51 = v141;
            }
            if ( (v163.m128i_i8[8] & 1) == 0 )
              j___libc_free_0(v164[0]);
            if ( v155 != v157 )
              _libc_free((unsigned __int64)v155);
          }
          v51 += 8LL;
        }
        while ( v146 != (_BYTE *)v51 );
        v18 = v137;
      }
      if ( (*(_BYTE *)(v18 + 23) & 0x20) == 0 )
        sub_164B7C0(v18, v15);
      if ( a2 )
      {
        v57 = *(unsigned int *)(a2 + 48);
        if ( (_DWORD)v57 )
        {
          v58 = v57 - 1;
          v59 = *(_QWORD *)(a2 + 32);
          v144 = ((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4);
          v60 = (v57 - 1) & v144;
          v61 = (__int64 *)(v59 + 16LL * v60);
          v62 = *v61;
          if ( v15 == *v61 )
          {
LABEL_68:
            v63 = (__int64 *)(v59 + 16 * v57);
            if ( v63 == v61 )
              goto LABEL_98;
            v64 = v61[1];
            if ( !v64 )
              goto LABEL_98;
            v65 = 1;
            for ( i = v58 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4)); ; i = v58 & v130 )
            {
              v67 = (__int64 *)(v59 + 16LL * i);
              if ( v18 == *v67 )
                break;
              if ( *v67 == -8 )
                goto LABEL_187;
              v130 = v65 + i;
              ++v65;
            }
            if ( v63 != v67 )
            {
              v68 = v67[1];
              goto LABEL_74;
            }
LABEL_187:
            v68 = 0;
LABEL_74:
            v69 = *(_BYTE **)(v64 + 32);
            v70 = *(_BYTE **)(v64 + 24);
            v163.m128i_i64[0] = (__int64)v164;
            v71 = v69 - v70;
            v163.m128i_i64[1] = 0x800000000LL;
            if ( (unsigned __int64)(v69 - v70) > 0x40 )
            {
              v134 = v68;
              sub_16CD150((__int64)&v163, v164, (v69 - v70) >> 3, 8, v68, v47);
              v68 = v134;
              v71 = v69 - v70;
            }
            if ( v70 != v69 )
            {
              v138 = v68;
              memmove((void *)(v163.m128i_i64[0] + 8LL * v163.m128i_u32[2]), v70, v71);
              v68 = v138;
            }
            v163.m128i_i32[2] += (v69 - v70) >> 3;
            v147 = v163.m128i_i64[0] + 8LL * v163.m128i_u32[2];
            if ( v163.m128i_i64[0] != v147 )
            {
              v139 = v15;
              v72 = v68;
              v133 = a5;
              v73 = (__int64 *)v163.m128i_i64[0];
              do
              {
                v75 = *v73;
                *(_BYTE *)(a2 + 72) = 0;
                v76 = *(_QWORD *)(v75 + 8);
                if ( v72 != v76 )
                {
                  v158[0] = v75;
                  v77 = sub_1AA5610(*(_QWORD **)(v76 + 24), *(_QWORD *)(v76 + 32), v158);
                  sub_15CDF70(*(_QWORD *)(v75 + 8) + 24LL, v77);
                  *(_QWORD *)(v75 + 8) = v72;
                  v158[0] = v75;
                  v74 = *(_BYTE **)(v72 + 32);
                  if ( v74 == *(_BYTE **)(v72 + 40) )
                  {
                    sub_15CE310(v72 + 24, v74, v158);
                  }
                  else
                  {
                    if ( v74 )
                    {
                      *(_QWORD *)v74 = v75;
                      v74 = *(_BYTE **)(v72 + 32);
                    }
                    v74 += 8;
                    *(_QWORD *)(v72 + 32) = v74;
                  }
                  if ( *(_DWORD *)(v75 + 16) != *(_DWORD *)(*(_QWORD *)(v75 + 8) + 16LL) + 1 )
                    sub_1AA5500(v75, (__int64)v74, v78, v79, v80, v81);
                }
                ++v73;
              }
              while ( (__int64 *)v147 != v73 );
              v15 = v139;
              a5 = v133;
            }
            v82 = *(_DWORD *)(a2 + 48);
            if ( !v82 )
              goto LABEL_198;
            v83 = *(_QWORD *)(a2 + 32);
            v84 = 1;
            for ( j = (v82 - 1) & v144; ; j = (v82 - 1) & v132 )
            {
              v86 = (__int64 *)(v83 + 16LL * j);
              v87 = *v86;
              if ( v15 == *v86 )
                break;
              if ( v87 == -8 )
                goto LABEL_198;
              v132 = v84 + j;
              ++v84;
            }
            if ( v86 == (__int64 *)(v83 + 16LL * v82) )
            {
LABEL_198:
              v158[0] = 0;
              *(_BYTE *)(a2 + 72) = 0;
              BUG();
            }
            v88 = v86[1];
            *(_BYTE *)(a2 + 72) = 0;
            v158[0] = v88;
            v89 = *(_QWORD *)(v88 + 8);
            if ( v89 )
            {
              v90 = sub_1AA5610(*(_QWORD **)(v89 + 24), *(_QWORD *)(v89 + 32), v158);
              sub_15CDF70(v91 + 24, v90);
              v82 = *(_DWORD *)(a2 + 48);
              if ( !v82 )
              {
LABEL_96:
                if ( (_QWORD *)v163.m128i_i64[0] != v164 )
                  _libc_free(v163.m128i_u64[0]);
                goto LABEL_98;
              }
              v83 = *(_QWORD *)(a2 + 32);
            }
            v122 = v82 - 1;
            v123 = 1;
            for ( k = v122 & v144; ; k = v122 & v131 )
            {
              v125 = (_QWORD *)(v83 + 16LL * k);
              if ( v87 == *v125 )
                break;
              if ( *v125 == -8 )
                goto LABEL_96;
              v131 = v123 + k;
              ++v123;
            }
            v126 = v125[1];
            if ( v126 )
            {
              v127 = *(_QWORD *)(v126 + 24);
              if ( v127 )
                j_j___libc_free_0(v127, *(_QWORD *)(v126 + 40) - v127);
              j_j___libc_free_0(v126, 56);
            }
            *v125 = -16;
            --*(_DWORD *)(a2 + 40);
            ++*(_DWORD *)(a2 + 44);
            goto LABEL_96;
          }
          v128 = 1;
          while ( v62 != -8 )
          {
            v129 = v128 + 1;
            v60 = v58 & (v128 + v60);
            v61 = (__int64 *)(v59 + 16LL * v60);
            v62 = *v61;
            if ( v15 == *v61 )
              goto LABEL_68;
            v128 = v129;
          }
        }
      }
LABEL_98:
      if ( a3 )
      {
        v92 = *(unsigned int *)(a3 + 24);
        if ( (_DWORD)v92 )
        {
          v93 = *(_QWORD *)(a3 + 8);
          v94 = (v92 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
          v95 = (__int64 *)(v93 + 16LL * v94);
          v96 = 1;
          v97 = *v95;
          if ( v15 == *v95 )
          {
LABEL_101:
            if ( v95 != (__int64 *)(v93 + 16 * v92) )
            {
              v98 = (__int64 *)v95[1];
              if ( v98 )
              {
                while ( 1 )
                {
                  v99 = (char *)v98[5];
                  v100 = (char *)v98[4];
                  v101 = (v99 - v100) >> 5;
                  if ( v101 <= 0 )
                    break;
                  v102 = &v100[32 * v101];
                  while ( v15 != *(_QWORD *)v100 )
                  {
                    if ( v15 == *((_QWORD *)v100 + 1) )
                    {
                      v100 += 8;
                      break;
                    }
                    if ( v15 == *((_QWORD *)v100 + 2) )
                    {
                      v100 += 16;
                      break;
                    }
                    if ( v15 == *((_QWORD *)v100 + 3) )
                    {
                      v100 += 24;
                      break;
                    }
                    v100 += 32;
                    if ( v100 == v102 )
                      goto LABEL_138;
                  }
LABEL_110:
                  if ( v100 + 8 != v99 )
                  {
                    memmove(v100, v100 + 8, v99 - (v100 + 8));
                    v99 = (char *)v98[5];
                  }
                  v103 = (_QWORD *)v98[8];
                  v98[5] = (__int64)(v99 - 8);
                  if ( (_QWORD *)v98[9] == v103 )
                  {
                    v105 = &v103[*((unsigned int *)v98 + 21)];
                    if ( v103 == v105 )
                    {
LABEL_136:
                      v103 = v105;
                    }
                    else
                    {
                      while ( v15 != *v103 )
                      {
                        if ( v105 == ++v103 )
                          goto LABEL_136;
                      }
                    }
                    goto LABEL_130;
                  }
                  v103 = sub_16CC9F0((__int64)(v98 + 7), v15);
                  if ( v15 == *v103 )
                  {
                    v106 = v98[9];
                    if ( v106 == v98[8] )
                      v107 = *((unsigned int *)v98 + 21);
                    else
                      v107 = *((unsigned int *)v98 + 20);
                    v105 = (_QWORD *)(v106 + 8 * v107);
LABEL_130:
                    if ( v105 != v103 )
                    {
                      *v103 = -2;
                      ++*((_DWORD *)v98 + 22);
                    }
                    goto LABEL_115;
                  }
                  v104 = v98[9];
                  if ( v104 == v98[8] )
                  {
                    v103 = (_QWORD *)(v104 + 8LL * *((unsigned int *)v98 + 21));
                    v105 = v103;
                    goto LABEL_130;
                  }
LABEL_115:
                  v98 = (__int64 *)*v98;
                  if ( !v98 )
                    goto LABEL_116;
                }
                v102 = (char *)v98[4];
LABEL_138:
                v108 = v99 - v102;
                if ( v99 - v102 != 16 )
                {
                  if ( v108 != 24 )
                  {
                    v100 = (char *)v98[5];
                    if ( v108 != 8 )
                      goto LABEL_110;
LABEL_141:
                    if ( v15 != *(_QWORD *)v102 )
                      v102 = (char *)v98[5];
                    v100 = v102;
                    goto LABEL_110;
                  }
                  if ( v15 == *(_QWORD *)v102 )
                  {
                    v100 = v102;
                    goto LABEL_110;
                  }
                  v102 += 8;
                }
                v100 = v102;
                if ( v15 == *(_QWORD *)v102 )
                  goto LABEL_110;
                v102 += 8;
                goto LABEL_141;
              }
LABEL_116:
              *v95 = -16;
              --*(_DWORD *)(a3 + 16);
              ++*(_DWORD *)(a3 + 20);
            }
          }
          else
          {
            while ( v97 != -8 )
            {
              v94 = (v92 - 1) & (v96 + v94);
              v95 = (__int64 *)(v93 + 16LL * v94);
              v97 = *v95;
              if ( v15 == *v95 )
                goto LABEL_101;
              ++v96;
            }
          }
        }
      }
      if ( a4 )
        sub_1413520(a4);
      if ( a5 )
      {
        sub_15CD5A0(a5, v15);
        sub_15CD9D0(a5, v152->m128i_i64, v153 - v152);
      }
      else
      {
        sub_157F980(v15);
      }
      if ( v152 )
        j_j___libc_free_0(v152, (char *)v154 - (char *)v152);
      if ( v160 != v162 )
        _libc_free((unsigned __int64)v160);
      return 1;
    }
    v24 = 3LL * (*(_DWORD *)(v23 + 20) & 0xFFFFFFF);
    if ( (*(_BYTE *)(v23 + 23) & 0x40) != 0 )
    {
      v25 = *(_QWORD **)(v23 - 8);
      v26 = &v25[v24];
    }
    else
    {
      v25 = (_QWORD *)(v23 - v24 * 8);
      v26 = (_QWORD *)v23;
    }
    if ( v25 != v26 )
      break;
LABEL_15:
    v27 = *(_QWORD *)(v23 + 32);
    if ( !v27 )
      BUG();
    v23 = 0;
    if ( *(_BYTE *)(v27 - 8) == 77 )
      v23 = v27 - 24;
  }
  while ( v23 != *v25 )
  {
    v25 += 3;
    if ( v26 == v25 )
      goto LABEL_15;
  }
  return v13;
}
