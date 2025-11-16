// Function: sub_3997B50
// Address: 0x3997b50
//
void __fastcall sub_3997B50(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // rbx
  const __m128i *v7; // r14
  const __m128i *v8; // r12
  __int64 v9; // r15
  __int64 v10; // rax
  _QWORD *v11; // r15
  unsigned __int64 v12; // rax
  __int64 v13; // r15
  __int64 v14; // rax
  __m128i v15; // xmm6
  __int64 i; // r13
  __int64 v17; // r14
  unsigned __int64 *v18; // r15
  __int64 v19; // r12
  unsigned __int64 *v20; // rbx
  __int64 v21; // r15
  __m128i *v22; // r13
  __int64 v23; // rbx
  __int64 v24; // r12
  __int64 v25; // rax
  __m128i *v26; // r9
  __int64 v27; // rcx
  __m128i *v28; // rbx
  __int64 v29; // rdx
  __m128i *v30; // r15
  __int64 v31; // rax
  __m128i v32; // xmm0
  const __m128i *v33; // rax
  unsigned __int64 v34; // r9
  __int64 v35; // rbx
  int v36; // r8d
  int v37; // r9d
  __int64 v38; // rax
  __int64 *v39; // rax
  __int64 v40; // r15
  __m128i *v41; // rax
  __int64 v42; // rdx
  __int64 v43; // r12
  __int64 v44; // rcx
  unsigned int v45; // edx
  __int64 *v46; // rax
  __int64 v47; // r8
  __m128i *v48; // rsi
  __int64 m128i_i64; // r14
  __int64 *v50; // rbx
  __int64 v51; // rsi
  unsigned int v52; // edx
  int v53; // eax
  __int64 *v54; // rdi
  __int64 v55; // r8
  __int64 v56; // rax
  _QWORD *v57; // rdi
  __int64 v58; // rax
  unsigned __int64 v59; // rax
  __int64 v60; // r15
  unsigned __int64 *v61; // r12
  unsigned __int64 v62; // rcx
  unsigned int v63; // edx
  __int64 *v64; // rax
  __int64 v65; // r10
  __m128i *v66; // rsi
  unsigned __int64 v67; // rsi
  unsigned int v68; // eax
  int v69; // edx
  __int64 *v70; // rdi
  __int64 v71; // r9
  unsigned __int64 v72; // rax
  _QWORD *v73; // rdi
  int v74; // r11d
  __int64 v75; // rsi
  int v76; // r11d
  __int64 *v77; // r10
  unsigned int v78; // edx
  __int64 v79; // r8
  __int64 v80; // rdi
  __int64 v81; // r12
  void (__fastcall *v82)(__int64, _QWORD, _QWORD); // r13
  __int64 v83; // rax
  __int64 v84; // rax
  unsigned __int64 v85; // rdi
  unsigned int v86; // r15d
  _QWORD *v87; // rbx
  _QWORD *v88; // r12
  unsigned __int64 v89; // rdi
  __int64 v90; // rbx
  unsigned __int64 v91; // r12
  unsigned __int64 v92; // rdi
  int v93; // r11d
  unsigned __int64 v94; // rcx
  __int64 *v95; // r10
  int v96; // r11d
  unsigned int v97; // eax
  __int64 v98; // r9
  __int64 *v99; // r13
  __int64 *v100; // rdx
  __int64 *v101; // r12
  char *j; // rsi
  __int64 v103; // rax
  char *v104; // r14
  unsigned __int64 v105; // rax
  char *v106; // rdi
  __int64 v107; // rcx
  __int64 v108; // rdx
  char *v109; // rsi
  char *v110; // rax
  __int64 v111; // r13
  unsigned int v112; // edx
  __int64 *v113; // r12
  __int64 v114; // rax
  int v115; // ecx
  __int64 v116; // r14
  __int64 v117; // rdi
  __int64 v118; // r8
  unsigned __int64 v119; // rax
  int v120; // r9d
  unsigned int v121; // r13d
  void (*v122)(); // rax
  __int64 v123; // rdi
  __int64 v124; // r8
  void (*v125)(); // rax
  __int64 v126; // rdi
  void (*v127)(); // rax
  __int64 v128; // rdi
  __int64 v129; // r8
  void (*v130)(); // rax
  __int64 v131; // rdi
  __int64 v132; // r8
  void (*v133)(); // rax
  __int64 *v134; // r14
  __int64 *v135; // r12
  __int64 *v136; // r13
  unsigned int v137; // esi
  __int64 v138; // r9
  unsigned int v139; // edx
  __int64 *v140; // rax
  __int64 v141; // r8
  __int64 v142; // rsi
  __int64 v143; // rdi
  void (*v144)(); // rax
  __int64 *v145; // rdi
  int v146; // r11d
  int v147; // eax
  int v148; // edx
  __int64 v149; // rax
  int v150; // ecx
  int v151; // ecx
  __int64 v152; // r8
  unsigned int v153; // eax
  __int64 v154; // r9
  int v155; // r10d
  int v156; // ecx
  int v157; // ecx
  __int64 v158; // r8
  int v159; // r10d
  unsigned int v160; // eax
  __int64 v161; // r9
  int v162; // r10d
  __int64 *v163; // r8
  int v164; // eax
  unsigned int v165; // edx
  __int64 v166; // rsi
  int v167; // r10d
  __int64 *v168; // rdi
  __int64 *v169; // rsi
  unsigned int v170; // r14d
  int v171; // r9d
  __int64 v172; // rcx
  __int64 *v173; // rbx
  int v174; // r11d
  int v175; // edi
  __int64 *v176; // r11
  __int64 *v177; // r9
  int v178; // r11d
  __int64 v179; // [rsp+8h] [rbp-108h]
  int v180; // [rsp+10h] [rbp-100h]
  __m128i *src; // [rsp+28h] [rbp-E8h]
  __m128i *srcb; // [rsp+28h] [rbp-E8h]
  char *srca; // [rsp+28h] [rbp-E8h]
  __int64 v184; // [rsp+30h] [rbp-E0h]
  unsigned __int64 v185; // [rsp+30h] [rbp-E0h]
  char *v187; // [rsp+38h] [rbp-D8h]
  char *v188; // [rsp+40h] [rbp-D0h] BYREF
  char *v189; // [rsp+48h] [rbp-C8h]
  char *v190; // [rsp+50h] [rbp-C0h]
  __m128i v191; // [rsp+60h] [rbp-B0h] BYREF
  char v192; // [rsp+70h] [rbp-A0h]
  char v193; // [rsp+71h] [rbp-9Fh]
  __int64 v194; // [rsp+80h] [rbp-90h] BYREF
  __int64 *v195; // [rsp+88h] [rbp-88h]
  __int64 v196; // [rsp+90h] [rbp-80h]
  unsigned int v197; // [rsp+98h] [rbp-78h]
  __int64 v198; // [rsp+A0h] [rbp-70h] BYREF
  unsigned __int64 v199; // [rsp+A8h] [rbp-68h]
  __int64 v200; // [rsp+B0h] [rbp-60h]
  int v201; // [rsp+B8h] [rbp-58h]
  unsigned __int64 v202; // [rsp+C0h] [rbp-50h]
  __int64 v203; // [rsp+C8h] [rbp-48h]
  __int64 v204; // [rsp+D0h] [rbp-40h]

  v6 = (__int64)a1;
  v7 = (const __m128i *)a1[76];
  v8 = (const __m128i *)a1[77];
  v198 = 0;
  v199 = 0;
  v200 = 0;
  v201 = 0;
  v202 = 0;
  v203 = 0;
  v204 = 0;
  if ( v8 == v7 )
  {
    v194 = 0;
    v195 = 0;
    v196 = 0;
    v197 = 0;
    goto LABEL_86;
  }
  do
  {
    while ( 1 )
    {
      v11 = (_QWORD *)v7->m128i_i64[0];
      v12 = *(_QWORD *)v7->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
      if ( v12 )
      {
        a3 = (__int64)&off_4CF6DB8;
        if ( off_4CF6DB8 == (_UNKNOWN *)v12 )
          goto LABEL_4;
      }
      else
      {
        if ( (*((_BYTE *)v11 + 9) & 0xC) != 8 )
          goto LABEL_4;
        *((_BYTE *)v11 + 8) |= 4u;
        a3 = (__int64)sub_38CE440(v11[3]);
        v58 = a3 | *v11 & 7LL;
        *v11 = v58;
        if ( !a3 )
          goto LABEL_4;
        v59 = v58 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v59 )
        {
          v59 = 0;
          if ( (*((_BYTE *)v11 + 9) & 0xC) == 8 )
          {
            *((_BYTE *)v11 + 8) |= 4u;
            v59 = (unsigned __int64)sub_38CE440(v11[3]);
            *v11 = v59 | *v11 & 7LL;
          }
        }
        a3 = (__int64)&off_4CF6DB8;
        if ( off_4CF6DB8 == (_UNKNOWN *)v59 )
        {
LABEL_4:
          v194 = 0;
          v9 = sub_3997820((__int64)&v198, &v194, a3, a4, a5, a6);
          v10 = *(unsigned int *)(v9 + 8);
          if ( (unsigned int)v10 >= *(_DWORD *)(v9 + 12) )
          {
            sub_16CD150(v9, (const void *)(v9 + 16), 0, 16, a5, a6);
            v10 = *(unsigned int *)(v9 + 8);
          }
          *(__m128i *)(*(_QWORD *)v9 + 16 * v10) = _mm_loadu_si128(v7);
          ++*(_DWORD *)(v9 + 8);
          goto LABEL_7;
        }
        v60 = v7->m128i_i64[0];
        v12 = *(_QWORD *)v7->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v12 )
        {
          if ( (*(_BYTE *)(v60 + 9) & 0xC) != 8
            || (*(_BYTE *)(v60 + 8) |= 4u,
                v12 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v60 + 24)),
                a3 = v12 | *(_QWORD *)v60 & 7LL,
                *(_QWORD *)v60 = a3,
                !v12) )
          {
            v194 = 0;
            BUG();
          }
        }
      }
      v194 = *(_QWORD *)(v12 + 24);
      if ( *(_BYTE *)(v194 + 148) )
        break;
LABEL_7:
      if ( v8 == ++v7 )
        goto LABEL_14;
    }
    v13 = sub_3997820((__int64)&v198, &v194, a3, a4, a5, a6);
    v14 = *(unsigned int *)(v13 + 8);
    if ( (unsigned int)v14 >= *(_DWORD *)(v13 + 12) )
    {
      sub_16CD150(v13, (const void *)(v13 + 16), 0, 16, a5, a6);
      v14 = *(unsigned int *)(v13 + 8);
    }
    v15 = _mm_loadu_si128(v7++);
    *(__m128i *)(*(_QWORD *)v13 + 16 * v14) = v15;
    ++*(_DWORD *)(v13 + 8);
  }
  while ( v8 != v7 );
LABEL_14:
  v196 = 0;
  v194 = 0;
  v184 = v203;
  v195 = 0;
  v197 = 0;
  if ( v203 != v202 )
  {
    for ( i = v202 + 8; ; i += 152 )
    {
      if ( !*(_DWORD *)(i + 8) )
        goto LABEL_54;
      v17 = *(_QWORD *)(i - 8);
      v18 = *(unsigned __int64 **)i;
      v19 = 2LL * *(unsigned int *)(i + 8);
      if ( !v17 )
        break;
      v20 = &v18[v19];
      src = *(__m128i **)i;
      v21 = i;
      v22 = (__m128i *)v20;
      v23 = (v19 * 8) >> 4;
      while ( 1 )
      {
        v24 = v23;
        v25 = sub_2207800(16 * v23);
        v26 = (__m128i *)v25;
        if ( v25 )
          break;
        v23 >>= 1;
        if ( !v23 )
        {
          v173 = (__int64 *)v22;
          i = v21;
          sub_3986980(src->m128i_i64, v173, (__int64)a1);
          v34 = 0;
          goto LABEL_24;
        }
      }
      v27 = v23;
      v28 = v22;
      v29 = v25 + v24 * 16;
      i = v21;
      v30 = src;
      v31 = v25 + 16;
      *(__m128i *)(v31 - 16) = _mm_loadu_si128(src);
      if ( v29 == v31 )
      {
        v33 = v26;
      }
      else
      {
        do
        {
          v32 = _mm_loadu_si128((const __m128i *)(v31 - 16));
          v31 += 16;
          *(__m128i *)(v31 - 16) = v32;
        }
        while ( v29 != v31 );
        v33 = &v26[v24 - 1];
      }
      srcb = v26;
      *v30 = _mm_loadu_si128(v33);
      sub_3987BF0(v30, v28, v26, v27, a1);
      v34 = (unsigned __int64)srcb;
LABEL_24:
      j_j___libc_free_0(v34);
      v35 = sub_38DDE50(*(__int64 **)(a1[1] + 256LL), v17);
      v38 = *(unsigned int *)(i + 8);
      if ( (unsigned int)v38 >= *(_DWORD *)(i + 12) )
      {
        sub_16CD150(i, (const void *)(i + 16), 0, 16, v36, v37);
        v38 = *(unsigned int *)(i + 8);
      }
      v39 = (__int64 *)(*(_QWORD *)i + 16 * v38);
      *v39 = v35;
      v39[1] = 0;
      v40 = (unsigned int)(*(_DWORD *)(i + 8) + 1);
      v41 = *(__m128i **)i;
      *(_DWORD *)(i + 8) = v40;
      v42 = v41->m128i_i64[0];
      if ( (unsigned int)v40 > 1 )
      {
        v43 = 1;
        while ( 1 )
        {
          m128i_i64 = (__int64)v41[v43 - 1].m128i_i64;
          v50 = v41[v43].m128i_i64;
          if ( v50[1] != *(_QWORD *)(m128i_i64 + 8) )
            break;
LABEL_34:
          if ( ++v43 == v40 )
            goto LABEL_54;
          v41 = *(__m128i **)i;
        }
        v191.m128i_i64[0] = v42;
        v191.m128i_i64[1] = *v50;
        if ( !v197 )
        {
          ++v194;
          goto LABEL_39;
        }
        v44 = *(_QWORD *)(m128i_i64 + 8);
        v45 = (v197 - 1) & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
        v46 = &v195[4 * v45];
        v47 = *v46;
        if ( v44 == *v46 )
        {
LABEL_29:
          v48 = (__m128i *)v46[2];
          if ( v48 != (__m128i *)v46[3] )
          {
            if ( v48 )
            {
              *v48 = _mm_load_si128(&v191);
              v48 = (__m128i *)v46[2];
            }
            v46[2] = (__int64)v48[1].m128i_i64;
            goto LABEL_33;
          }
          v57 = v46 + 1;
LABEL_44:
          sub_3224BB0((__int64)v57, v48, &v191);
LABEL_33:
          v42 = *v50;
          goto LABEL_34;
        }
        v74 = 1;
        v54 = 0;
        while ( v47 != -8 )
        {
          if ( !v54 && v47 == -16 )
            v54 = v46;
          v45 = (v197 - 1) & (v74 + v45);
          v46 = &v195[4 * v45];
          v47 = *v46;
          if ( v44 == *v46 )
            goto LABEL_29;
          ++v74;
        }
        if ( !v54 )
          v54 = v46;
        ++v194;
        v53 = v196 + 1;
        if ( 4 * ((int)v196 + 1) >= 3 * v197 )
        {
LABEL_39:
          sub_3996BC0((__int64)&v194, 2 * v197);
          if ( !v197 )
            goto LABEL_291;
          v51 = *(_QWORD *)(m128i_i64 + 8);
          v52 = (v197 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
          v53 = v196 + 1;
          v54 = &v195[4 * v52];
          v55 = *v54;
          if ( v51 != *v54 )
          {
            v174 = 1;
            v77 = 0;
            while ( v55 != -8 )
            {
              if ( v55 == -16 && !v77 )
                v77 = v54;
              v52 = (v197 - 1) & (v174 + v52);
              v54 = &v195[4 * v52];
              v55 = *v54;
              if ( v51 == *v54 )
                goto LABEL_41;
              ++v174;
            }
            goto LABEL_227;
          }
        }
        else if ( v197 - HIDWORD(v196) - v53 <= v197 >> 3 )
        {
          sub_3996BC0((__int64)&v194, v197);
          if ( !v197 )
          {
LABEL_291:
            LODWORD(v196) = v196 + 1;
            BUG();
          }
          v75 = *(_QWORD *)(m128i_i64 + 8);
          v76 = 1;
          v77 = 0;
          v78 = (v197 - 1) & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
          v53 = v196 + 1;
          v54 = &v195[4 * v78];
          v79 = *v54;
          if ( v75 != *v54 )
          {
            while ( v79 != -8 )
            {
              if ( !v77 && v79 == -16 )
                v77 = v54;
              v78 = (v197 - 1) & (v76 + v78);
              v54 = &v195[4 * v78];
              v79 = *v54;
              if ( v75 == *v54 )
                goto LABEL_41;
              ++v76;
            }
LABEL_227:
            if ( v77 )
              v54 = v77;
          }
        }
LABEL_41:
        LODWORD(v196) = v53;
        if ( *v54 != -8 )
          --HIDWORD(v196);
        v56 = *(_QWORD *)(m128i_i64 + 8);
        v57 = v54 + 1;
        v48 = 0;
        *v57 = 0;
        v57[1] = 0;
        *(v57 - 1) = v56;
        v57[2] = 0;
        goto LABEL_44;
      }
LABEL_54:
      if ( v184 == i + 144 )
      {
        v6 = (__int64)a1;
        goto LABEL_86;
      }
    }
    v61 = &v18[v19];
    while ( 2 )
    {
      v191 = (__m128i)*v18;
      if ( !v197 )
      {
        ++v194;
        goto LABEL_65;
      }
      v62 = v18[1];
      v63 = (v197 - 1) & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
      v64 = &v195[4 * v63];
      v65 = *v64;
      if ( v62 == *v64 )
      {
LABEL_58:
        v66 = (__m128i *)v64[2];
        if ( v66 != (__m128i *)v64[3] )
        {
          if ( v66 )
          {
            *v66 = _mm_load_si128(&v191);
            v66 = (__m128i *)v64[2];
          }
          v64[2] = (__int64)v66[1].m128i_i64;
LABEL_62:
          v18 += 2;
          if ( v61 == v18 )
            goto LABEL_54;
          continue;
        }
        v73 = v64 + 1;
LABEL_70:
        sub_3224BB0((__int64)v73, v66, &v191);
        goto LABEL_62;
      }
      break;
    }
    v70 = 0;
    v93 = 1;
    while ( v65 != -8 )
    {
      if ( !v70 && v65 == -16 )
        v70 = v64;
      v63 = (v197 - 1) & (v93 + v63);
      v64 = &v195[4 * v63];
      v65 = *v64;
      if ( v62 == *v64 )
        goto LABEL_58;
      ++v93;
    }
    if ( !v70 )
      v70 = v64;
    ++v194;
    v69 = v196 + 1;
    if ( 4 * ((int)v196 + 1) >= 3 * v197 )
    {
LABEL_65:
      sub_3996BC0((__int64)&v194, 2 * v197);
      if ( !v197 )
        goto LABEL_290;
      v67 = v18[1];
      v68 = (v197 - 1) & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
      v69 = v196 + 1;
      v70 = &v195[4 * v68];
      v71 = *v70;
      if ( v67 != *v70 )
      {
        v95 = 0;
        v178 = 1;
        while ( v71 != -8 )
        {
          if ( v71 == -16 && !v95 )
            v95 = v70;
          v68 = (v197 - 1) & (v178 + v68);
          v70 = &v195[4 * v68];
          v71 = *v70;
          if ( v67 == *v70 )
            goto LABEL_67;
          ++v178;
        }
        goto LABEL_253;
      }
    }
    else if ( v197 - HIDWORD(v196) - v69 <= v197 >> 3 )
    {
      sub_3996BC0((__int64)&v194, v197);
      if ( !v197 )
      {
LABEL_290:
        LODWORD(v196) = v196 + 1;
        BUG();
      }
      v94 = v18[1];
      v95 = 0;
      v96 = 1;
      v97 = (v197 - 1) & (((unsigned int)v94 >> 9) ^ ((unsigned int)v94 >> 4));
      v69 = v196 + 1;
      v70 = &v195[4 * v97];
      v98 = *v70;
      if ( v94 != *v70 )
      {
        while ( v98 != -8 )
        {
          if ( !v95 && v98 == -16 )
            v95 = v70;
          v97 = (v197 - 1) & (v96 + v97);
          v70 = &v195[4 * v97];
          v98 = *v70;
          if ( v94 == *v70 )
            goto LABEL_67;
          ++v96;
        }
LABEL_253:
        if ( v95 )
          v70 = v95;
      }
    }
LABEL_67:
    LODWORD(v196) = v69;
    if ( *v70 != -8 )
      --HIDWORD(v196);
    v72 = v18[1];
    v73 = v70 + 1;
    v66 = 0;
    *v73 = 0;
    v73[1] = 0;
    *(v73 - 1) = v72;
    v73[2] = 0;
    goto LABEL_70;
  }
LABEL_86:
  v80 = *(_QWORD *)(v6 + 8);
  v81 = *(_QWORD *)(v80 + 256);
  v82 = *(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v81 + 160LL);
  v83 = sub_396DD80(v80);
  v82(v81, *(_QWORD *)(v83 + 152), 0);
  v84 = *(_QWORD *)(v6 + 8);
  v85 = (unsigned __int64)v195;
  v188 = 0;
  v86 = *(_DWORD *)(*(_QWORD *)(v84 + 240) + 8LL);
  v189 = 0;
  v190 = 0;
  if ( (_DWORD)v196 )
  {
    v99 = &v195[4 * v197];
    if ( v195 != v99 )
    {
      v100 = v195;
      while ( 1 )
      {
        v101 = v100;
        if ( *v100 != -8 && *v100 != -16 )
          break;
        v100 += 4;
        if ( v99 == v100 )
          goto LABEL_87;
      }
      if ( v99 != v100 )
      {
        v191.m128i_i64[0] = *v100;
        j = 0;
LABEL_173:
        sub_398ECE0((__int64)&v188, j, &v191);
        for ( j = v189; ; v189 = j )
        {
          v101 += 4;
          if ( v101 == v99 )
            break;
          while ( 1 )
          {
            v103 = *v101;
            if ( *v101 != -8 && v103 != -16 )
              break;
            v101 += 4;
            if ( v99 == v101 )
              goto LABEL_132;
          }
          if ( v101 == v99 )
            break;
          v191.m128i_i64[0] = *v101;
          if ( v190 == j )
            goto LABEL_173;
          if ( j )
          {
            *(_QWORD *)j = v103;
            j = v189;
          }
          j += 8;
        }
LABEL_132:
        v104 = j;
        srca = v188;
        if ( j != v188 )
        {
          _BitScanReverse64(&v105, (j - v188) >> 3);
          sub_3984E90(v188, j, 2LL * (int)(63 - (v105 ^ 0x3F)));
          if ( j - srca <= 128 )
          {
            sub_39845B0(srca, j);
          }
          else
          {
            sub_39845B0(srca, srca + 128);
            v106 = srca + 128;
            if ( srca + 128 != j )
            {
              do
              {
                v107 = *(_QWORD *)v106;
                v108 = *((_QWORD *)v106 - 1);
                v109 = v106;
                v110 = v106 - 8;
                if ( *(_DWORD *)(v108 + 600) > *(_DWORD *)(*(_QWORD *)v106 + 600LL) )
                {
                  do
                  {
                    *((_QWORD *)v110 + 1) = v108;
                    v109 = v110;
                    v108 = *((_QWORD *)v110 - 1);
                    v110 -= 8;
                  }
                  while ( *(_DWORD *)(v107 + 600) < *(_DWORD *)(v108 + 600) );
                }
                v106 += 8;
                *(_QWORD *)v109 = v107;
              }
              while ( v104 != v106 );
            }
          }
          srca = v189;
          if ( v189 != v188 )
          {
            v187 = v188;
            v185 = 2 * v86;
            v179 = v6 + 632;
            while ( 1 )
            {
              v111 = *(_QWORD *)v187;
              if ( !v197 )
                break;
              v112 = (v197 - 1) & (((unsigned int)v111 >> 9) ^ ((unsigned int)v111 >> 4));
              v113 = &v195[4 * v112];
              v114 = *v113;
              if ( v111 != *v113 )
              {
                v162 = 1;
                v163 = 0;
                while ( v114 != -8 )
                {
                  if ( v163 || v114 != -16 )
                    v113 = v163;
                  v112 = (v197 - 1) & (v162 + v112);
                  v177 = &v195[4 * v112];
                  v114 = *v177;
                  if ( v111 == *v177 )
                  {
                    v113 = &v195[4 * v112];
                    v115 = v185 * (((v177[2] - v177[1]) >> 4) + 1);
                    goto LABEL_143;
                  }
                  ++v162;
                  v163 = v113;
                  v113 = &v195[4 * v112];
                }
                if ( v163 )
                  v113 = v163;
                ++v194;
                v164 = v196 + 1;
                if ( 4 * ((int)v196 + 1) < 3 * v197 )
                {
                  if ( v197 - HIDWORD(v196) - v164 > v197 >> 3 )
                  {
LABEL_205:
                    LODWORD(v196) = v164;
                    if ( *v113 != -8 )
                      --HIDWORD(v196);
                    *v113 = v111;
                    v115 = 2 * v86;
                    v113[1] = 0;
                    v113[2] = 0;
                    v113[3] = 0;
                    goto LABEL_143;
                  }
                  sub_3996BC0((__int64)&v194, v197);
                  if ( v197 )
                  {
                    v169 = 0;
                    v170 = (v197 - 1) & (((unsigned int)v111 >> 9) ^ ((unsigned int)v111 >> 4));
                    v171 = 1;
                    v164 = v196 + 1;
                    v113 = &v195[4 * v170];
                    v172 = *v113;
                    if ( v111 != *v113 )
                    {
                      while ( v172 != -8 )
                      {
                        if ( v172 == -16 && !v169 )
                          v169 = v113;
                        v170 = (v197 - 1) & (v171 + v170);
                        v113 = &v195[4 * v170];
                        v172 = *v113;
                        if ( v111 == *v113 )
                          goto LABEL_205;
                        ++v171;
                      }
                      if ( v169 )
                        v113 = v169;
                    }
                    goto LABEL_205;
                  }
LABEL_288:
                  LODWORD(v196) = v196 + 1;
                  BUG();
                }
LABEL_209:
                sub_3996BC0((__int64)&v194, 2 * v197);
                if ( v197 )
                {
                  v165 = (v197 - 1) & (((unsigned int)v111 >> 9) ^ ((unsigned int)v111 >> 4));
                  v164 = v196 + 1;
                  v113 = &v195[4 * v165];
                  v166 = *v113;
                  if ( v111 != *v113 )
                  {
                    v167 = 1;
                    v168 = 0;
                    while ( v166 != -8 )
                    {
                      if ( v166 == -16 && !v168 )
                        v168 = v113;
                      v165 = (v197 - 1) & (v167 + v165);
                      v113 = &v195[4 * v165];
                      v166 = *v113;
                      if ( v111 == *v113 )
                        goto LABEL_205;
                      ++v167;
                    }
                    if ( v168 )
                      v113 = v168;
                  }
                  goto LABEL_205;
                }
                goto LABEL_288;
              }
              v115 = v185 * (((v113[2] - v113[1]) >> 4) + 1);
LABEL_143:
              v116 = *(_QWORD *)(v111 + 616);
              v117 = *(_QWORD *)(v6 + 8);
              if ( !v116 )
                v116 = v111;
              v118 = *(_QWORD *)(v117 + 256);
              v119 = v185 * ((v185 + 11) / v185);
              v120 = v115 + v119 - 4;
              v121 = v119 - 12;
              v122 = *(void (**)())(*(_QWORD *)v118 + 104LL);
              v193 = 1;
              v191.m128i_i64[0] = (__int64)"Length of ARange Set";
              v192 = 3;
              if ( v122 != nullsub_580 )
              {
                v180 = v120;
                ((void (__fastcall *)(__int64, __m128i *, __int64))v122)(v118, &v191, 1);
                v117 = *(_QWORD *)(v6 + 8);
                v120 = v180;
              }
              sub_396F340(v117, v120);
              v123 = *(_QWORD *)(v6 + 8);
              v124 = *(_QWORD *)(v123 + 256);
              v125 = *(void (**)())(*(_QWORD *)v124 + 104LL);
              v193 = 1;
              v191.m128i_i64[0] = (__int64)"DWARF Arange version number";
              v192 = 3;
              if ( v125 != nullsub_580 )
              {
                ((void (__fastcall *)(__int64, __m128i *, __int64))v125)(v124, &v191, 1);
                v123 = *(_QWORD *)(v6 + 8);
              }
              sub_396F320(v123, 2);
              v126 = *(_QWORD *)(*(_QWORD *)(v6 + 8) + 256LL);
              v127 = *(void (**)())(*(_QWORD *)v126 + 104LL);
              v193 = 1;
              v191.m128i_i64[0] = (__int64)"Offset Into Debug Info Section";
              v192 = 3;
              if ( v127 != nullsub_580 )
                ((void (__fastcall *)(__int64, __m128i *, __int64))v127)(v126, &v191, 1);
              sub_398A7D0(v6, v116);
              v128 = *(_QWORD *)(v6 + 8);
              v129 = *(_QWORD *)(v128 + 256);
              v130 = *(void (**)())(*(_QWORD *)v129 + 104LL);
              v193 = 1;
              v191.m128i_i64[0] = (__int64)"Address Size (in bytes)";
              v192 = 3;
              if ( v130 != nullsub_580 )
              {
                ((void (__fastcall *)(__int64, __m128i *, __int64))v130)(v129, &v191, 1);
                v128 = *(_QWORD *)(v6 + 8);
              }
              sub_396F300(v128, v86);
              v131 = *(_QWORD *)(v6 + 8);
              v132 = *(_QWORD *)(v131 + 256);
              v133 = *(void (**)())(*(_QWORD *)v132 + 104LL);
              v193 = 1;
              v191.m128i_i64[0] = (__int64)"Segment Size (in bytes)";
              v192 = 3;
              if ( v133 != nullsub_580 )
              {
                ((void (__fastcall *)(__int64, __m128i *, __int64))v133)(v132, &v191, 1);
                v131 = *(_QWORD *)(v6 + 8);
              }
              sub_396F300(v131, 0);
              sub_38DD0A0(*(__int64 **)(*(_QWORD *)(v6 + 8) + 256LL), v121, 0xFFu);
              v134 = (__int64 *)v113[2];
              v135 = (__int64 *)v113[1];
              if ( v134 == v135 )
                goto LABEL_164;
              do
              {
                while ( 1 )
                {
                  sub_396F390(*(_QWORD *)(v6 + 8), *v135, 0, v86, 0);
                  v136 = (__int64 *)v135[1];
                  if ( !v136 )
                    break;
                  v135 += 2;
                  sub_396F380(*(_QWORD *)(v6 + 8));
                  if ( v134 == v135 )
                    goto LABEL_164;
                }
                v137 = *(_DWORD *)(v6 + 656);
                if ( !v137 )
                {
                  ++*(_QWORD *)(v6 + 632);
                  goto LABEL_184;
                }
                v138 = *(_QWORD *)(v6 + 640);
                v139 = (v137 - 1) & (((unsigned int)*v135 >> 9) ^ ((unsigned int)*v135 >> 4));
                v140 = (__int64 *)(v138 + 16LL * v139);
                v141 = *v140;
                if ( *v140 == *v135 )
                {
                  v142 = 1;
                  if ( v140[1] )
                    v142 = v140[1];
                  goto LABEL_163;
                }
                v145 = 0;
                v146 = 1;
                while ( 1 )
                {
                  if ( v141 == -8 )
                  {
                    if ( !v145 )
                      v145 = v140;
                    v147 = *(_DWORD *)(v6 + 648);
                    ++*(_QWORD *)(v6 + 632);
                    v148 = v147 + 1;
                    if ( 4 * (v147 + 1) >= 3 * v137 )
                    {
LABEL_184:
                      sub_215C650(v179, 2 * v137);
                      v150 = *(_DWORD *)(v6 + 656);
                      if ( v150 )
                      {
                        v151 = v150 - 1;
                        v152 = *(_QWORD *)(v6 + 640);
                        v153 = v151 & (((unsigned int)*v135 >> 9) ^ ((unsigned int)*v135 >> 4));
                        v148 = *(_DWORD *)(v6 + 648) + 1;
                        v145 = (__int64 *)(v152 + 16LL * v153);
                        v154 = *v145;
                        if ( *v135 == *v145 )
                          goto LABEL_180;
                        v155 = 1;
                        while ( v154 != -8 )
                        {
                          if ( v154 == -16 && !v136 )
                            v136 = v145;
                          v153 = v151 & (v155 + v153);
                          v145 = (__int64 *)(v152 + 16LL * v153);
                          v154 = *v145;
                          if ( *v135 == *v145 )
                            goto LABEL_180;
                          ++v155;
                        }
LABEL_188:
                        if ( v136 )
                          v145 = v136;
                        goto LABEL_180;
                      }
                    }
                    else
                    {
                      if ( v137 - *(_DWORD *)(v6 + 652) - v148 > v137 >> 3 )
                      {
LABEL_180:
                        *(_DWORD *)(v6 + 648) = v148;
                        if ( *v145 != -8 )
                          --*(_DWORD *)(v6 + 652);
                        v149 = *v135;
                        v142 = 1;
                        v145[1] = 0;
                        *v145 = v149;
                        goto LABEL_163;
                      }
                      sub_215C650(v179, v137);
                      v156 = *(_DWORD *)(v6 + 656);
                      if ( v156 )
                      {
                        v157 = v156 - 1;
                        v158 = *(_QWORD *)(v6 + 640);
                        v159 = 1;
                        v160 = v157 & (((unsigned int)*v135 >> 9) ^ ((unsigned int)*v135 >> 4));
                        v148 = *(_DWORD *)(v6 + 648) + 1;
                        v145 = (__int64 *)(v158 + 16LL * v160);
                        v161 = *v145;
                        if ( *v135 == *v145 )
                          goto LABEL_180;
                        while ( v161 != -8 )
                        {
                          if ( !v136 && v161 == -16 )
                            v136 = v145;
                          v160 = v157 & (v159 + v160);
                          v145 = (__int64 *)(v158 + 16LL * v160);
                          v161 = *v145;
                          if ( *v135 == *v145 )
                            goto LABEL_180;
                          ++v159;
                        }
                        goto LABEL_188;
                      }
                    }
                    ++*(_DWORD *)(v6 + 648);
                    BUG();
                  }
                  if ( v145 || v141 != -16 )
                    v140 = v145;
                  v175 = v146 + 1;
                  v139 = (v137 - 1) & (v146 + v139);
                  v176 = (__int64 *)(v138 + 16LL * v139);
                  v141 = *v176;
                  if ( *v135 == *v176 )
                    break;
                  v146 = v175;
                  v145 = v140;
                  v140 = (__int64 *)(v138 + 16LL * v139);
                }
                v142 = 1;
                if ( v176[1] )
                  v142 = v176[1];
LABEL_163:
                v135 += 2;
                (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)(v6 + 8) + 256LL) + 424LL))(
                  *(_QWORD *)(*(_QWORD *)(v6 + 8) + 256LL),
                  v142,
                  v86);
              }
              while ( v134 != v135 );
LABEL_164:
              v143 = *(_QWORD *)(*(_QWORD *)(v6 + 8) + 256LL);
              v144 = *(void (**)())(*(_QWORD *)v143 + 104LL);
              v193 = 1;
              v191.m128i_i64[0] = (__int64)"ARange terminator";
              v192 = 3;
              if ( v144 != nullsub_580 )
              {
                ((void (__fastcall *)(__int64, __m128i *, __int64))v144)(v143, &v191, 1);
                v143 = *(_QWORD *)(*(_QWORD *)(v6 + 8) + 256LL);
              }
              (*(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v143 + 424LL))(v143, 0, v86);
              (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(*(_QWORD *)(v6 + 8) + 256LL) + 424LL))(
                *(_QWORD *)(*(_QWORD *)(v6 + 8) + 256LL),
                0,
                v86);
              v187 += 8;
              if ( srca == v187 )
              {
                srca = v188;
                goto LABEL_168;
              }
            }
            ++v194;
            goto LABEL_209;
          }
        }
LABEL_168:
        if ( srca )
          j_j___libc_free_0((unsigned __int64)srca);
        v85 = (unsigned __int64)v195;
      }
    }
  }
LABEL_87:
  if ( v197 )
  {
    v87 = (_QWORD *)v85;
    v88 = (_QWORD *)(v85 + 32LL * v197);
    do
    {
      if ( *v87 != -8 && *v87 != -16 )
      {
        v89 = v87[1];
        if ( v89 )
          j_j___libc_free_0(v89);
      }
      v87 += 4;
    }
    while ( v88 != v87 );
    v85 = (unsigned __int64)v195;
  }
  j___libc_free_0(v85);
  v90 = v203;
  v91 = v202;
  if ( v203 != v202 )
  {
    do
    {
      v92 = *(_QWORD *)(v91 + 8);
      if ( v92 != v91 + 24 )
        _libc_free(v92);
      v91 += 152LL;
    }
    while ( v90 != v91 );
    v91 = v202;
  }
  if ( v91 )
    j_j___libc_free_0(v91);
  j___libc_free_0(v199);
}
