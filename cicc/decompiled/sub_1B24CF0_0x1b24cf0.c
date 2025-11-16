// Function: sub_1B24CF0
// Address: 0x1b24cf0
//
__int64 __fastcall sub_1B24CF0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128 a4,
        __m128 a5,
        __m128 a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 *v10; // rax
  __int64 *v11; // rdx
  unsigned int v12; // r12d
  __int64 v13; // r15
  __int64 *v14; // rbx
  __int64 v15; // rcx
  __int64 v16; // r13
  unsigned __int64 v17; // r13
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rbx
  __m128 *v21; // r9
  int v22; // eax
  unsigned int v23; // eax
  __m128i *v24; // rsi
  __m128i *v25; // r12
  __int64 v26; // rbx
  __int64 v27; // r14
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 *v32; // rax
  __int64 v33; // rdi
  __int64 *v34; // rbx
  __m128i *v36; // r8
  signed __int64 v37; // rsi
  unsigned __int64 v38; // rax
  unsigned __int64 v39; // rdi
  bool v40; // cf
  unsigned __int64 v41; // rax
  __int64 v42; // r10
  __int64 v43; // rax
  __int64 v44; // r10
  __int64 v45; // rax
  __int64 *v46; // rsi
  __m128 *v47; // rdx
  const __m128i *v48; // rax
  __m128i *v49; // r14
  __m128i *v50; // r8
  signed __int64 v51; // rbx
  unsigned __int64 v52; // rax
  __m128i *v53; // rbx
  __int64 *v54; // rdi
  unsigned int v55; // eax
  unsigned __int64 v56; // rdi
  int v57; // r12d
  unsigned int v58; // eax
  __int64 v59; // rsi
  unsigned int v60; // ecx
  __int64 v61; // rax
  __int64 v62; // rdx
  __int64 v63; // rbx
  _QWORD *v64; // rax
  __int64 v65; // r14
  __int64 v66; // rcx
  _QWORD *v67; // rdi
  _QWORD *v68; // rbx
  _QWORD *v69; // rdi
  unsigned __int64 v70; // rax
  __int64 v71; // r12
  unsigned __int64 *v72; // rcx
  unsigned __int64 v73; // rdx
  double v74; // xmm4_8
  double v75; // xmm5_8
  __int64 v76; // rbx
  __int64 *v77; // rdi
  __int64 *v78; // r13
  __int64 *v79; // rax
  __int64 *v80; // rax
  __m128i *v81; // rdi
  const __m128i *v82; // rax
  __int64 v83; // rsi
  __int64 v84; // rdx
  unsigned int v85; // r10d
  __int64 *v86; // rdx
  __int64 v87; // rdx
  unsigned int v88; // esi
  __int64 *v89; // rdx
  __int64 *v90; // r8
  __int64 *v91; // rcx
  __int64 *v92; // rdi
  char *v93; // rsi
  _QWORD *v94; // rdi
  const __m128i *v95; // r12
  _BYTE *v96; // rsi
  const __m128i *v97; // rcx
  __int64 v98; // rax
  const __m128i *v99; // r14
  __int64 v100; // r11
  __int64 v101; // r15
  unsigned int v102; // r13d
  __int64 v103; // rax
  __int64 v104; // rdx
  unsigned int v105; // esi
  __int64 *v106; // rbx
  _BYTE *v107; // rdx
  __int64 v108; // rbx
  _BYTE *v109; // rsi
  int v110; // ebx
  __int64 v111; // rcx
  unsigned int v112; // edx
  __int64 *v113; // rax
  __int64 v114; // rdi
  int v115; // edx
  unsigned int v116; // ebx
  unsigned int v117; // edx
  __int64 *v118; // rax
  unsigned int v119; // ebx
  __m128i *v120; // rdx
  __m128i *v121; // rsi
  signed __int64 v122; // rax
  const __m128i *v123; // rax
  __int64 *v124; // rdx
  __int64 *v125; // rsi
  __int64 *v126; // rcx
  __int64 *v127; // r11
  int v128; // edx
  __int64 v129; // rdx
  __int64 v130; // rdi
  unsigned int v131; // r9d
  __int64 v132; // rsi
  int v133; // r11d
  __int64 *v134; // rcx
  __int64 v135; // rdi
  int v136; // r11d
  unsigned int v137; // r9d
  __int64 v138; // rsi
  _QWORD *v139; // rax
  __int64 v140; // r11
  int v141; // r12d
  unsigned int v142; // ebx
  signed __int64 v143; // [rsp+8h] [rbp-198h]
  int v144; // [rsp+8h] [rbp-198h]
  int v145; // [rsp+8h] [rbp-198h]
  __m128i *v146; // [rsp+10h] [rbp-190h]
  unsigned __int64 v147; // [rsp+10h] [rbp-190h]
  __int64 v148; // [rsp+18h] [rbp-188h]
  __int64 v149; // [rsp+18h] [rbp-188h]
  __int64 v150; // [rsp+18h] [rbp-188h]
  __int64 v151; // [rsp+20h] [rbp-180h]
  __int64 v152; // [rsp+20h] [rbp-180h]
  __int64 v153; // [rsp+20h] [rbp-180h]
  __int64 v154; // [rsp+28h] [rbp-178h]
  __m128 *v155; // [rsp+28h] [rbp-178h]
  __int64 v156; // [rsp+28h] [rbp-178h]
  __int64 *v157; // [rsp+30h] [rbp-170h]
  __int64 v159; // [rsp+40h] [rbp-160h]
  __int64 v160; // [rsp+48h] [rbp-158h]
  __int64 v161; // [rsp+48h] [rbp-158h]
  __int64 v162; // [rsp+50h] [rbp-150h]
  __int64 v163; // [rsp+58h] [rbp-148h]
  char v164; // [rsp+6Fh] [rbp-131h] BYREF
  __int64 v165; // [rsp+70h] [rbp-130h] BYREF
  __int64 v166; // [rsp+78h] [rbp-128h] BYREF
  __m128i v167; // [rsp+80h] [rbp-120h] BYREF
  __m128i v168; // [rsp+90h] [rbp-110h] BYREF
  void *src; // [rsp+A0h] [rbp-100h] BYREF
  __m128i *v170; // [rsp+A8h] [rbp-F8h]
  __m128i *v171; // [rsp+B0h] [rbp-F0h]
  __int64 *v172; // [rsp+C0h] [rbp-E0h] BYREF
  _BYTE *v173; // [rsp+C8h] [rbp-D8h]
  _BYTE *v174; // [rsp+D0h] [rbp-D0h]
  const char *v175; // [rsp+E0h] [rbp-C0h] BYREF
  __int64 v176; // [rsp+E8h] [rbp-B8h]
  __int64 v177; // [rsp+F0h] [rbp-B0h]
  unsigned int v178; // [rsp+F8h] [rbp-A8h]
  __int64 v179; // [rsp+100h] [rbp-A0h] BYREF
  __int64 *v180; // [rsp+108h] [rbp-98h]
  __int64 *v181; // [rsp+110h] [rbp-90h]
  __int64 v182; // [rsp+118h] [rbp-88h]
  int v183; // [rsp+120h] [rbp-80h]
  _BYTE v184[120]; // [rsp+128h] [rbp-78h] BYREF

  v10 = (__int64 *)v184;
  v11 = (__int64 *)v184;
  v12 = 0;
  v13 = *(_QWORD *)(a2 + 80);
  v179 = 0;
  v180 = (__int64 *)v184;
  v181 = (__int64 *)v184;
  v182 = 8;
  v183 = 0;
  v162 = a2 + 72;
  if ( v13 == a2 + 72 )
    goto LABEL_45;
  do
  {
    while ( 1 )
    {
      v15 = v13;
      v13 = *(_QWORD *)(v13 + 8);
      v16 = v15 - 24;
      if ( v11 == v10 )
      {
        v14 = &v10[HIDWORD(v182)];
        if ( v14 == v10 )
        {
          v124 = v10;
        }
        else
        {
          do
          {
            if ( v16 == *v10 )
              break;
            ++v10;
          }
          while ( v14 != v10 );
          v124 = v14;
        }
LABEL_16:
        while ( v124 != v10 )
        {
          if ( (unsigned __int64)*v10 < 0xFFFFFFFFFFFFFFFELL )
            goto LABEL_6;
          ++v10;
        }
        if ( v10 != v14 )
          goto LABEL_7;
      }
      else
      {
        v14 = &v11[(unsigned int)v182];
        v10 = sub_16CC9F0((__int64)&v179, v15 - 24);
        if ( v16 == *v10 )
        {
          if ( v181 == v180 )
            v124 = &v181[HIDWORD(v182)];
          else
            v124 = &v181[(unsigned int)v182];
          goto LABEL_16;
        }
        if ( v181 == v180 )
        {
          v10 = &v181[HIDWORD(v182)];
          v124 = v10;
          goto LABEL_16;
        }
        v10 = &v181[(unsigned int)v182];
LABEL_6:
        if ( v10 != v14 )
          goto LABEL_7;
      }
      v17 = sub_157EBA0(v16);
      if ( *(_BYTE *)(v17 + 16) == 27
        && (!*(_BYTE *)(a1 + 153) || !(unsigned __int8)sub_1B24B00(v17, &v165, &v166, &v164)) )
      {
        break;
      }
LABEL_7:
      v11 = v181;
      v10 = v180;
      if ( v162 == v13 )
        goto LABEL_105;
    }
    v163 = *(_QWORD *)(v17 + 40);
    v159 = *(_QWORD *)(v163 + 56);
    if ( (*(_BYTE *)(v17 + 23) & 0x40) != 0 )
      v18 = *(_QWORD *)(v17 - 8);
    else
      v18 = v17 - 24LL * (*(_DWORD *)(v17 + 20) & 0xFFFFFFF);
    v160 = *(_QWORD *)(v18 + 24);
    v157 = *(__int64 **)v18;
    v19 = *(_QWORD *)(v159 + 80);
    if ( !v19 || v163 != v19 - 24 )
    {
      v20 = *(_QWORD *)(v163 + 8);
      if ( !v20 )
        goto LABEL_49;
      while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v20) + 16) - 25) > 9u )
      {
        v20 = *(_QWORD *)(v20 + 8);
        if ( !v20 )
          goto LABEL_49;
      }
    }
    if ( v163 == sub_157F0B0(v163) )
    {
LABEL_49:
      v11 = v181;
      v10 = v180;
      if ( v181 != v180 )
        goto LABEL_50;
      v90 = &v181[HIDWORD(v182)];
      if ( v181 == v90 )
      {
LABEL_232:
        if ( HIDWORD(v182) >= (unsigned int)v182 )
        {
LABEL_50:
          sub_16CCBA0((__int64)&v179, v163);
          v11 = v181;
          v10 = v180;
          goto LABEL_104;
        }
        ++HIDWORD(v182);
        *v90 = v163;
        v10 = v180;
        ++v179;
        v11 = v181;
      }
      else
      {
        v91 = v181;
        v92 = 0;
        while ( v163 != *v91 )
        {
          if ( *v91 == -2 )
            v92 = v91;
          if ( v90 == ++v91 )
          {
            if ( !v92 )
              goto LABEL_232;
            *v92 = v163;
            v11 = v181;
            --v183;
            v10 = v180;
            ++v179;
            goto LABEL_104;
          }
        }
      }
      goto LABEL_104;
    }
    if ( (*(_DWORD *)(v17 + 20) & 0xFFFFFFFu) >> 1 != 1 )
    {
      v171 = 0;
      v22 = *(_DWORD *)(v17 + 20);
      src = 0;
      v170 = 0;
      v23 = (v22 & 0xFFFFFFFu) >> 1;
      if ( v23 == 1 )
        goto LABEL_81;
      v24 = 0;
      v25 = 0;
      v26 = 0;
      v27 = v23 - 1;
      while ( 1 )
      {
        v31 = 24;
        if ( (_DWORD)v26 != -2 )
          v31 = 24LL * (unsigned int)(2 * v26 + 3);
        v28 = (*(_BYTE *)(v17 + 23) & 0x40) != 0
            ? *(_QWORD *)(v17 - 8)
            : v17 - 24LL * (*(_DWORD *)(v17 + 20) & 0xFFFFFFF);
        ++v26;
        v29 = *(_QWORD *)(v28 + v31);
        v30 = *(_QWORD *)(v28 + 24LL * (unsigned int)(2 * v26));
        if ( v25 == v24 )
          break;
        if ( v25 )
        {
          v25->m128i_i64[0] = v30;
          v25->m128i_i64[1] = v30;
          v25[1].m128i_i64[0] = v29;
          v25 = v170;
        }
        v25 = (__m128i *)((char *)v25 + 24);
        v170 = v25;
        if ( v27 == v26 )
          goto LABEL_73;
LABEL_36:
        v24 = v171;
      }
      v36 = (__m128i *)src;
      v37 = (char *)v25 - (_BYTE *)src;
      v38 = 0xAAAAAAAAAAAAAAABLL * (((char *)v25 - (_BYTE *)src) >> 3);
      if ( v38 == 0x555555555555555LL )
        sub_4262D8((__int64)"vector::_M_realloc_insert");
      v39 = 1;
      if ( v38 )
        v39 = 0xAAAAAAAAAAAAAAABLL * (((char *)v25 - (_BYTE *)src) >> 3);
      v40 = __CFADD__(v39, v38);
      v41 = v39 - 0x5555555555555555LL * (((char *)v25 - (_BYTE *)src) >> 3);
      if ( v40 )
      {
        v42 = 0x7FFFFFFFFFFFFFF8LL;
      }
      else
      {
        if ( !v41 )
        {
          v45 = 24;
          v44 = 0;
          v21 = 0;
LABEL_62:
          v46 = (__int64 *)((char *)v21 + v37);
          if ( v46 )
          {
            *v46 = v30;
            v46[1] = v30;
            v46[2] = v29;
          }
          if ( v36 == v25 )
          {
            v25 = (__m128i *)v45;
          }
          else
          {
            v47 = v21;
            v48 = v36;
            do
            {
              if ( v47 )
              {
                a3 = (__m128)_mm_loadu_si128(v48);
                *v47 = a3;
                v29 = v48[1].m128i_i64[0];
                v47[1].m128_u64[0] = v29;
              }
              v48 = (const __m128i *)((char *)v48 + 24);
              v47 = (__m128 *)((char *)v47 + 24);
            }
            while ( v48 != v25 );
            v25 = (__m128i *)((char *)&v21[3]
                            + 24
                            * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)((char *)v25 - (char *)v36 - 24) >> 3))
                             & 0x1FFFFFFFFFFFFFFFLL));
          }
          if ( v36 )
          {
            v152 = v44;
            v155 = v21;
            j_j___libc_free_0(v36, (char *)v171 - (char *)v36);
            v44 = v152;
            v21 = v155;
          }
          src = v21;
          v170 = v25;
          v171 = (__m128i *)v44;
          if ( v27 != v26 )
            goto LABEL_36;
LABEL_73:
          v49 = (__m128i *)src;
          v50 = v25;
          v51 = (char *)v25 - (_BYTE *)src;
          if ( src != v25 )
          {
            _BitScanReverse64(&v52, 0xAAAAAAAAAAAAAAABLL * (v51 >> 3));
            sub_1B24480((__int64)src, v25, 2LL * (int)(63 - (v52 ^ 0x3F)), v29, (__int64)v25, (__int64)v21);
            if ( v51 <= 384 )
            {
              sub_1B235B0(v49, v25);
            }
            else
            {
              v53 = v49 + 24;
              sub_1B235B0(v49, (__m128i *)v49[24].m128i_i64);
              if ( &v49[24] != v25 )
              {
                do
                {
                  v54 = (__int64 *)v53;
                  v53 = (__m128i *)((char *)v53 + 24);
                  sub_1B233A0(v54);
                }
                while ( v25 != v53 );
              }
            }
            v50 = v170;
            v25 = (__m128i *)src;
            v51 = (char *)v170 - (_BYTE *)src;
          }
          if ( (unsigned __int64)v51 <= 0x18 )
            goto LABEL_81;
          v81 = (__m128i *)((char *)v25 + 24);
          if ( &v25[1].m128i_u64[1] != (unsigned __int64 *)v50 )
          {
            v82 = (__m128i *)((char *)v25 + 24);
            while ( 1 )
            {
              v88 = *(_DWORD *)(v82->m128i_i64[0] + 32);
              v89 = *(__int64 **)(v82->m128i_i64[0] + 24);
              v83 = v88 <= 0x40
                  ? (__int64)((_QWORD)v89 << (64 - (unsigned __int8)v88)) >> (64 - (unsigned __int8)v88)
                  : *v89;
              v84 = v25->m128i_i64[1];
              v85 = *(_DWORD *)(v84 + 32);
              v86 = *(__int64 **)(v84 + 24);
              v87 = v85 > 0x40
                  ? *v86
                  : (__int64)((_QWORD)v86 << (64 - (unsigned __int8)v85)) >> (64 - (unsigned __int8)v85);
              if ( v82[1].m128i_i64[0] == v25[1].m128i_i64[0] && v87 + 1 == v83 )
                break;
              if ( v81 == v82 )
              {
                v82 = (const __m128i *)((char *)v82 + 24);
                v25 = v81;
                v81 = (__m128i *)((char *)v81 + 24);
                if ( v50 == v82 )
                {
LABEL_142:
                  v50 = v170;
                  v93 = (char *)v81;
                  goto LABEL_143;
                }
              }
              else
              {
                a4 = (__m128)_mm_loadu_si128(v82);
                *(__m128 *)((char *)v25 + 24) = a4;
                v25[2].m128i_i64[1] = v82[1].m128i_i64[0];
                v25 = v81;
                v81 = (__m128i *)((char *)v81 + 24);
LABEL_129:
                v82 = (const __m128i *)((char *)v82 + 24);
                if ( v50 == v82 )
                  goto LABEL_142;
              }
            }
            v25->m128i_i64[1] = v82->m128i_i64[1];
            goto LABEL_129;
          }
          v93 = (char *)v50;
LABEL_143:
          sub_1B230F0((__int64)&src, v93, v50->m128i_i8);
LABEL_81:
          v172 = 0;
          v173 = 0;
          v174 = 0;
          v153 = 0;
          v156 = 0;
          if ( *(_BYTE *)(sub_157ED60(v160) + 16) != 31 )
            goto LABEL_82;
          v95 = (const __m128i *)src;
          v96 = v173;
          v156 = *(_QWORD *)src;
          v97 = v170;
          v98 = v170[-1].m128i_i64[0];
          v175 = 0;
          v176 = 0;
          v153 = v98;
          v167.m128i_i64[0] = 0x8000000000000000LL;
          v177 = 0;
          v178 = 0;
          v167.m128i_i64[1] = 0x7FFFFFFFFFFFFFFFLL;
          if ( v173 == v174 )
          {
            sub_1B23420((__int64)&v172, v173, &v167);
            v95 = (const __m128i *)src;
            v97 = v170;
          }
          else
          {
            if ( v173 )
            {
              *(__m128i *)v173 = _mm_loadu_si128(&v167);
              v96 = v173;
              v95 = (const __m128i *)src;
              v97 = v170;
            }
            v173 = v96 + 16;
          }
          v99 = v97;
          v100 = 0;
          if ( v97 != v95 )
          {
            v149 = v13;
            v101 = 0;
            v147 = v17;
            v102 = 0;
            while ( 1 )
            {
              v117 = *(_DWORD *)(v95->m128i_i64[0] + 32);
              v118 = *(__int64 **)(v95->m128i_i64[0] + 24);
              if ( v117 <= 0x40 )
                v103 = (__int64)((_QWORD)v118 << (64 - (unsigned __int8)v117)) >> (64 - (unsigned __int8)v117);
              else
                v103 = *v118;
              v104 = v95->m128i_i64[1];
              v105 = *(_DWORD *)(v104 + 32);
              v106 = *(__int64 **)(v104 + 24);
              if ( v105 > 0x40 )
              {
                v107 = v173;
                v108 = *v106;
                if ( *((_QWORD *)v173 - 2) != v103 )
                {
LABEL_157:
                  *((_QWORD *)v107 - 1) = v103 - 1;
                  goto LABEL_158;
                }
              }
              else
              {
                v107 = v173;
                v108 = (__int64)((_QWORD)v106 << (64 - (unsigned __int8)v105)) >> (64 - (unsigned __int8)v105);
                if ( *((_QWORD *)v173 - 2) != v103 )
                  goto LABEL_157;
              }
              v173 = v107 - 16;
LABEL_158:
              if ( v108 != 0x7FFFFFFFFFFFFFFFLL )
              {
                v168.m128i_i64[1] = 0x7FFFFFFFFFFFFFFFLL;
                v109 = v173;
                v168.m128i_i64[0] = v108 + 1;
                if ( v173 == v174 )
                {
                  v144 = v103;
                  sub_1B23420((__int64)&v172, v173, &v168);
                  LODWORD(v103) = v144;
                }
                else
                {
                  if ( v173 )
                  {
                    a6 = (__m128)_mm_loadu_si128(&v168);
                    *(__m128 *)v173 = a6;
                    v109 = v173;
                  }
                  v173 = v109 + 16;
                }
              }
              v110 = v108 - v103 + 1;
              if ( !v178 )
              {
                ++v175;
                goto LABEL_209;
              }
              v111 = v95[1].m128i_i64[0];
              v112 = (v178 - 1) & (((unsigned int)v111 >> 9) ^ ((unsigned int)v111 >> 4));
              v113 = (__int64 *)(v176 + 16LL * v112);
              v114 = *v113;
              if ( v111 != *v113 )
              {
                v145 = 1;
                v127 = 0;
                while ( v114 != -8 )
                {
                  if ( !v127 && v114 == -16 )
                    v127 = v113;
                  v112 = (v178 - 1) & (v145 + v112);
                  v113 = (__int64 *)(v176 + 16LL * v112);
                  v114 = *v113;
                  if ( v111 == *v113 )
                    goto LABEL_165;
                  ++v145;
                }
                if ( v127 )
                  v113 = v127;
                ++v175;
                v128 = v177 + 1;
                if ( 4 * ((int)v177 + 1) >= 3 * v178 )
                {
LABEL_209:
                  sub_13FEAC0((__int64)&v175, 2 * v178);
                  if ( !v178 )
                    goto LABEL_264;
                  v130 = v95[1].m128i_i64[0];
                  v128 = v177 + 1;
                  v131 = (v178 - 1) & (((unsigned int)v130 >> 9) ^ ((unsigned int)v130 >> 4));
                  v113 = (__int64 *)(v176 + 16LL * v131);
                  v132 = *v113;
                  if ( v130 != *v113 )
                  {
                    v133 = 1;
                    v134 = 0;
                    while ( v132 != -8 )
                    {
                      if ( v132 == -16 && !v134 )
                        v134 = v113;
                      v131 = (v178 - 1) & (v133 + v131);
                      v113 = (__int64 *)(v176 + 16LL * v131);
                      v132 = *v113;
                      if ( v130 == *v113 )
                        goto LABEL_205;
                      ++v133;
                    }
LABEL_221:
                    if ( v134 )
                      v113 = v134;
                  }
                }
                else if ( v178 - HIDWORD(v177) - v128 <= v178 >> 3 )
                {
                  sub_13FEAC0((__int64)&v175, v178);
                  if ( !v178 )
                  {
LABEL_264:
                    LODWORD(v177) = v177 + 1;
                    BUG();
                  }
                  v135 = v95[1].m128i_i64[0];
                  v136 = 1;
                  v128 = v177 + 1;
                  v134 = 0;
                  v137 = (v178 - 1) & (((unsigned int)v135 >> 9) ^ ((unsigned int)v135 >> 4));
                  v113 = (__int64 *)(v176 + 16LL * v137);
                  v138 = *v113;
                  if ( v135 != *v113 )
                  {
                    while ( v138 != -8 )
                    {
                      if ( v138 == -16 && !v134 )
                        v134 = v113;
                      v137 = (v178 - 1) & (v136 + v137);
                      v113 = (__int64 *)(v176 + 16LL * v137);
                      v138 = *v113;
                      if ( v135 == *v113 )
                        goto LABEL_205;
                      ++v136;
                    }
                    goto LABEL_221;
                  }
                }
LABEL_205:
                LODWORD(v177) = v128;
                if ( *v113 != -8 )
                  --HIDWORD(v177);
                v129 = v95[1].m128i_i64[0];
                *((_DWORD *)v113 + 2) = 0;
                *v113 = v129;
                v115 = 0;
                goto LABEL_166;
              }
LABEL_165:
              v115 = *((_DWORD *)v113 + 2);
LABEL_166:
              v116 = v115 + v110;
              *((_DWORD *)v113 + 2) = v116;
              if ( v116 > v102 )
              {
                v101 = v95[1].m128i_i64[0];
                v102 = v116;
              }
              v95 = (const __m128i *)((char *)v95 + 24);
              if ( v99 == v95 )
              {
                v119 = v102;
                v100 = v101;
                v17 = v147;
                v13 = v149;
                goto LABEL_174;
              }
            }
          }
          v119 = 0;
LABEL_174:
          v150 = v100;
          sub_157F2D0(v160, v163, 0);
          v120 = v170;
          v121 = (__m128i *)src;
          v122 = 0xAAAAAAAAAAAAAAABLL * (((char *)v170 - (_BYTE *)src) >> 3);
          if ( v122 >> 2 > 0 )
          {
            while ( v121[1].m128i_i64[0] != v150 )
            {
              if ( v121[2].m128i_i64[1] == v150 )
              {
                v121 = (__m128i *)((char *)v121 + 24);
                goto LABEL_181;
              }
              if ( v121[4].m128i_i64[0] == v150 )
              {
                v121 += 3;
                goto LABEL_181;
              }
              if ( v121[5].m128i_i64[1] == v150 )
              {
                v121 = (__m128i *)((char *)v121 + 72);
                goto LABEL_181;
              }
              v121 += 6;
              if ( v121 == (__m128i *)((char *)src + 96 * (v122 >> 2)) )
              {
                v122 = 0xAAAAAAAAAAAAAAABLL * (((char *)v170 - (char *)v121) >> 3);
                goto LABEL_225;
              }
            }
            goto LABEL_181;
          }
LABEL_225:
          if ( v122 != 2 )
          {
            if ( v122 != 3 )
            {
              if ( v122 != 1 )
              {
LABEL_228:
                v121 = v170;
LABEL_186:
                v160 = v150;
                sub_1B230F0((__int64)&src, v121->m128i_i8, v120->m128i_i8);
                if ( v170 == src )
                {
                  v139 = sub_1648A60(56, 1u);
                  v140 = v150;
                  if ( v139 )
                  {
                    sub_15F8590((__int64)v139, v150, v163);
                    v140 = v150;
                  }
                  v161 = v140;
                  v141 = 0;
                  sub_15F20C0((_QWORD *)v17);
                  v142 = v119 - 1;
                  if ( v142 )
                  {
                    do
                    {
                      ++v141;
                      sub_157F2D0(v161, v163, 0);
                    }
                    while ( v141 != v142 );
                  }
                  j___libc_free_0(v176);
                  v77 = v172;
                  if ( !v172 )
                    goto LABEL_101;
                  goto LABEL_100;
                }
                j___libc_free_0(v176);
LABEL_82:
                v55 = *(_DWORD *)(v17 + 20) & 0xFFFFFFF;
                if ( (*(_BYTE *)(v17 + 23) & 0x40) != 0 )
                  v56 = *(_QWORD *)(v17 - 8);
                else
                  v56 = v17 - 24LL * v55;
                v57 = *(_QWORD *)(v56 + 24) == v160;
                v58 = v55 >> 1;
                v59 = v58 - 1;
                if ( v58 != 1 )
                {
                  v60 = 3;
                  v61 = 0;
                  do
                  {
                    v62 = 24;
                    if ( v61 != 4294967294LL )
                      v62 = 24LL * v60;
                    ++v61;
                    v60 += 2;
                    v57 += v160 == *(_QWORD *)(v56 + v62);
                  }
                  while ( v59 != v61 );
                }
                v175 = "NewDefault";
                LOWORD(v177) = 259;
                v63 = sub_16498A0(v17);
                v64 = (_QWORD *)sub_22077B0(64);
                v65 = (__int64)v64;
                if ( v64 )
                  sub_157FB60(v64, v63, (__int64)&v175, 0, 0);
                sub_15E01D0(v159 + 72, v65);
                v66 = *(_QWORD *)(v160 + 24);
                *(_QWORD *)(v65 + 32) = v160 + 24;
                v66 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)(v65 + 24) = v66 | *(_QWORD *)(v65 + 24) & 7LL;
                *(_QWORD *)(v66 + 8) = v65 + 24;
                *(_QWORD *)(v160 + 24) = *(_QWORD *)(v160 + 24) & 7LL | (v65 + 24);
                v67 = sub_1648A60(56, 1u);
                if ( v67 )
                  sub_15F8590((__int64)v67, v160, v65);
                v68 = sub_1B23880(
                        (__int64 *)src,
                        v170,
                        v156,
                        v153,
                        v157,
                        v163,
                        *(double *)a3.m128_u64,
                        *(double *)a4.m128_u64,
                        *(double *)a5.m128_u64,
                        v163,
                        v65,
                        &v172);
                sub_1B23670(v160, v163, v65, v57);
                v69 = sub_1648A60(56, 1u);
                if ( v69 )
                  sub_15F8590((__int64)v69, (__int64)v68, v163);
                if ( (*(_BYTE *)(v17 + 23) & 0x40) != 0 )
                  v70 = *(_QWORD *)(v17 - 8);
                else
                  v70 = v17 - 24LL * (*(_DWORD *)(v17 + 20) & 0xFFFFFFF);
                v71 = *(_QWORD *)(v70 + 24);
                sub_157EA20(v163 + 40, v17);
                v72 = *(unsigned __int64 **)(v17 + 32);
                v73 = *(_QWORD *)(v17 + 24) & 0xFFFFFFFFFFFFFFF8LL;
                *v72 = v73 | *v72 & 7;
                *(_QWORD *)(v73 + 8) = v72;
                *(_QWORD *)(v17 + 24) &= 7uLL;
                *(_QWORD *)(v17 + 32) = 0;
                sub_164BEC0(
                  v17,
                  v17,
                  v73,
                  (__int64)v72,
                  a3,
                  *(double *)a4.m128_u64,
                  *(double *)a5.m128_u64,
                  *(double *)a6.m128_u64,
                  v74,
                  v75,
                  a9,
                  a10);
                v76 = *(_QWORD *)(v71 + 8);
                if ( v76 )
                {
                  while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v76) + 16) - 25) > 9u )
                  {
                    v76 = *(_QWORD *)(v76 + 8);
                    if ( !v76 )
                      goto LABEL_113;
                  }
LABEL_99:
                  v77 = v172;
                  if ( !v172 )
                  {
LABEL_101:
                    if ( src )
                      j_j___libc_free_0(src, (char *)v171 - (_BYTE *)src);
                    v11 = v181;
                    v10 = v180;
                    goto LABEL_104;
                  }
LABEL_100:
                  j_j___libc_free_0(v77, v174 - (_BYTE *)v77);
                  goto LABEL_101;
                }
LABEL_113:
                v80 = v180;
                if ( v181 == v180 )
                {
                  v125 = &v180[HIDWORD(v182)];
                  if ( v180 != v125 )
                  {
                    v126 = 0;
                    while ( v71 != *v80 )
                    {
                      if ( *v80 == -2 )
                        v126 = v80;
                      if ( v125 == ++v80 )
                      {
                        if ( !v126 )
                          goto LABEL_248;
                        *v126 = v71;
                        --v183;
                        ++v179;
                        goto LABEL_99;
                      }
                    }
                    goto LABEL_99;
                  }
LABEL_248:
                  if ( HIDWORD(v182) < (unsigned int)v182 )
                  {
                    ++HIDWORD(v182);
                    *v125 = v71;
                    ++v179;
                    goto LABEL_99;
                  }
                }
                sub_16CCBA0((__int64)&v179, v71);
                goto LABEL_99;
              }
LABEL_246:
              if ( v121[1].m128i_i64[0] != v150 )
                goto LABEL_228;
LABEL_181:
              if ( v170 != v121 )
              {
                v123 = (__m128i *)((char *)v121 + 24);
                if ( v170 != (__m128i *)&v121[1].m128i_u64[1] )
                {
                  do
                  {
                    if ( v123[1].m128i_i64[0] != v150 )
                    {
                      a5 = (__m128)_mm_loadu_si128(v123);
                      v121 = (__m128i *)((char *)v121 + 24);
                      *(__m128 *)((char *)v121 - 24) = a5;
                      v121[-1].m128i_i64[1] = v123[1].m128i_i64[0];
                    }
                    v123 = (const __m128i *)((char *)v123 + 24);
                  }
                  while ( v120 != v123 );
                }
              }
              goto LABEL_186;
            }
            if ( v121[1].m128i_i64[0] == v150 )
              goto LABEL_181;
            v121 = (__m128i *)((char *)v121 + 24);
          }
          if ( v121[1].m128i_i64[0] == v150 )
            goto LABEL_181;
          v121 = (__m128i *)((char *)v121 + 24);
          goto LABEL_246;
        }
        if ( v41 > 0x555555555555555LL )
          v41 = 0x555555555555555LL;
        v42 = 24 * v41;
      }
      v143 = (char *)v25 - (_BYTE *)src;
      v146 = (__m128i *)src;
      v148 = v30;
      v151 = v29;
      v154 = v42;
      v43 = sub_22077B0(v42);
      v29 = v151;
      v30 = v148;
      v36 = v146;
      v21 = (__m128 *)v43;
      v37 = v143;
      v44 = v43 + v154;
      v45 = v43 + 24;
      goto LABEL_62;
    }
    v94 = sub_1648A60(56, 1u);
    if ( v94 )
      sub_15F8590((__int64)v94, v160, v163);
    sub_15F20C0((_QWORD *)v17);
    v11 = v181;
    v10 = v180;
LABEL_104:
    v12 = 1;
  }
  while ( v162 != v13 );
LABEL_105:
  if ( v11 == v10 )
    v78 = &v11[HIDWORD(v182)];
  else
    v78 = &v11[(unsigned int)v182];
  if ( v78 != v11 )
  {
    v79 = v11;
    while ( 1 )
    {
      v33 = *v79;
      v34 = v79;
      if ( (unsigned __int64)*v79 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v78 == ++v79 )
        goto LABEL_45;
    }
    while ( v34 != v78 )
    {
      sub_1AA7270(v33, 0, a3, *(double *)a4.m128_u64, *(double *)a5.m128_u64, *(double *)a6.m128_u64, a7, a8, a9, a10);
      v32 = v34 + 1;
      if ( v34 + 1 == v78 )
        break;
      while ( 1 )
      {
        v33 = *v32;
        v34 = v32;
        if ( (unsigned __int64)*v32 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v78 == ++v32 )
          goto LABEL_45;
      }
    }
  }
LABEL_45:
  if ( v181 != v180 )
    _libc_free((unsigned __int64)v181);
  return v12;
}
