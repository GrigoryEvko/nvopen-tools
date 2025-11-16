// Function: sub_39483B0
// Address: 0x39483b0
//
__int64 __fastcall sub_39483B0(unsigned __int64 *a1, __int64 a2, __int64 a3, __m128i **a4)
{
  __int64 *v5; // r13
  char *v6; // rdx
  size_t v7; // rax
  __m128i *v8; // r12
  size_t v9; // rbx
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rax
  size_t v13; // rdx
  char *v14; // rsi
  size_t v15; // rax
  bool v16; // cc
  size_t v17; // rcx
  size_t v18; // rax
  __int64 v19; // rdx
  __m128i *v20; // rdi
  char *v21; // rax
  __int64 v22; // rdx
  size_t v23; // rcx
  __int64 v24; // rsi
  unsigned int v25; // r12d
  unsigned __int64 v27; // rdx
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // rcx
  __int64 v30; // rax
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // rdx
  _BYTE *v33; // rsi
  __int64 v34; // rbx
  int v35; // eax
  void *v36; // r14
  size_t v37; // rbx
  unsigned int v38; // r8d
  _QWORD *v39; // r9
  __int64 v40; // rax
  __int64 v41; // rax
  unsigned int v42; // r8d
  _QWORD *v43; // r9
  _QWORD *v44; // rcx
  _BYTE *v45; // rdi
  __int64 *v46; // rdx
  __int64 v47; // r14
  unsigned int v48; // edx
  _QWORD *v49; // r9
  __int64 v50; // rdi
  __int64 v51; // rax
  unsigned int v52; // r8d
  _QWORD *v53; // r9
  _QWORD *v54; // rbx
  _BYTE *v55; // rdi
  __int64 *v56; // rax
  __int64 v57; // rdi
  __int64 v58; // rdi
  size_t v59; // rax
  char v60; // bl
  unsigned __int64 v61; // rdi
  void *v62; // rsi
  __m128i *v63; // rdi
  size_t v64; // rsi
  __int64 v65; // rcx
  __int64 v66; // rdx
  unsigned __int64 v67; // rdi
  __int64 v68; // rax
  _BYTE *v69; // rax
  __int64 v70; // rax
  _BYTE *v71; // rax
  __m128i v72; // xmm0
  __int64 v73; // r14
  char v74; // bl
  void *v75; // r14
  size_t v76; // rbx
  unsigned int v77; // r9d
  _QWORD *v78; // r10
  __int64 v79; // rdx
  unsigned __int64 v80; // rsi
  __int64 v81; // r8
  unsigned __int64 v82; // r14
  unsigned __int64 *v83; // rbx
  unsigned __int64 v84; // r8
  unsigned __int64 *v85; // r12
  unsigned __int64 v86; // r13
  unsigned __int64 v87; // rdi
  unsigned __int64 v88; // rdi
  unsigned __int64 v89; // r8
  __int64 v90; // rax
  __int64 v91; // rbx
  __int64 v92; // r12
  unsigned __int64 v93; // rdi
  size_t v94; // rdx
  size_t v95; // rdx
  __int64 v96; // rax
  unsigned int v97; // r9d
  _QWORD *v98; // r10
  _QWORD *v99; // rcx
  _BYTE *v100; // rdi
  __int64 *v101; // rax
  __int64 *v102; // rax
  __int64 v103; // rax
  _BYTE *v104; // rax
  unsigned __int64 *v105; // rbx
  unsigned __int64 v106; // r14
  unsigned __int64 *v107; // r12
  __int64 v108; // r14
  unsigned __int64 v109; // r13
  unsigned __int64 *v110; // r8
  unsigned __int64 v111; // rdi
  unsigned __int64 v112; // rdi
  unsigned __int64 v113; // r9
  __int64 v114; // rax
  __int64 v115; // rbx
  __int64 v116; // r12
  unsigned __int64 *v117; // r14
  unsigned __int64 v118; // rdi
  __m128i *v119; // rdi
  size_t v120; // rcx
  __int64 v121; // rdx
  __int64 v122; // rsi
  size_t v123; // rdx
  __m128i *v124; // rdi
  __m128i *v125; // rax
  size_t v126; // rsi
  __int64 v127; // rcx
  __int64 v128; // rdi
  unsigned __int64 v129; // r12
  unsigned __int64 *v130; // r14
  unsigned __int64 *i; // rbx
  unsigned __int64 v132; // r13
  unsigned __int64 v133; // rdi
  unsigned __int64 v134; // rdi
  unsigned __int64 v135; // rdi
  unsigned __int64 v136; // rdi
  __int64 v137; // rax
  __int64 v138; // r13
  __int64 v139; // rbx
  __int64 v140; // r8
  size_t v141; // rdx
  _QWORD *v142; // [rsp+8h] [rbp-328h]
  _QWORD *v143; // [rsp+8h] [rbp-328h]
  _QWORD *v144; // [rsp+10h] [rbp-320h]
  _QWORD *v145; // [rsp+10h] [rbp-320h]
  unsigned int v146; // [rsp+10h] [rbp-320h]
  _QWORD *v147; // [rsp+10h] [rbp-320h]
  _QWORD *v148; // [rsp+10h] [rbp-320h]
  _QWORD *v149; // [rsp+10h] [rbp-320h]
  unsigned int v150; // [rsp+18h] [rbp-318h]
  unsigned int v151; // [rsp+18h] [rbp-318h]
  _QWORD *v152; // [rsp+18h] [rbp-318h]
  _QWORD *v153; // [rsp+18h] [rbp-318h]
  _QWORD *v154; // [rsp+18h] [rbp-318h]
  unsigned int v155; // [rsp+18h] [rbp-318h]
  _QWORD *v156; // [rsp+18h] [rbp-318h]
  _QWORD *v157; // [rsp+18h] [rbp-318h]
  _QWORD *v158; // [rsp+20h] [rbp-310h]
  unsigned int v159; // [rsp+20h] [rbp-310h]
  __int64 *v160; // [rsp+20h] [rbp-310h]
  unsigned int v161; // [rsp+20h] [rbp-310h]
  __int64 *v162; // [rsp+20h] [rbp-310h]
  unsigned int v163; // [rsp+20h] [rbp-310h]
  unsigned int v164; // [rsp+20h] [rbp-310h]
  unsigned int v165; // [rsp+28h] [rbp-308h]
  __m128i *v166; // [rsp+28h] [rbp-308h]
  __m128i *v167; // [rsp+28h] [rbp-308h]
  __m128i *v168; // [rsp+28h] [rbp-308h]
  __int64 v169; // [rsp+28h] [rbp-308h]
  __int64 v170; // [rsp+28h] [rbp-308h]
  unsigned __int64 *v171; // [rsp+28h] [rbp-308h]
  __m128i *v172; // [rsp+28h] [rbp-308h]
  unsigned __int64 v173; // [rsp+28h] [rbp-308h]
  unsigned __int8 *v174; // [rsp+40h] [rbp-2F0h]
  size_t v175; // [rsp+48h] [rbp-2E8h]
  unsigned __int8 *v176; // [rsp+58h] [rbp-2D8h]
  unsigned __int64 v179; // [rsp+88h] [rbp-2A8h]
  int v180; // [rsp+94h] [rbp-29Ch]
  __m128i *v181; // [rsp+98h] [rbp-298h]
  void *src; // [rsp+A0h] [rbp-290h] BYREF
  size_t n; // [rsp+A8h] [rbp-288h]
  __int64 v184; // [rsp+B0h] [rbp-280h]
  _QWORD v185[2]; // [rsp+D0h] [rbp-260h] BYREF
  __int16 v186; // [rsp+E0h] [rbp-250h]
  __int64 v187[2]; // [rsp+F0h] [rbp-240h] BYREF
  __int16 v188; // [rsp+100h] [rbp-230h]
  __int64 *v189; // [rsp+110h] [rbp-220h] BYREF
  void **v190; // [rsp+118h] [rbp-218h]
  __int16 v191; // [rsp+120h] [rbp-210h]
  const char *v192; // [rsp+130h] [rbp-200h] BYREF
  char *v193; // [rsp+138h] [rbp-1F8h]
  __int16 v194; // [rsp+140h] [rbp-1F0h]
  const char *v195; // [rsp+150h] [rbp-1E0h] BYREF
  void **p_src; // [rsp+158h] [rbp-1D8h]
  __int16 v197; // [rsp+160h] [rbp-1D0h]
  __m128i v198; // [rsp+170h] [rbp-1C0h] BYREF
  _BYTE *v199; // [rsp+180h] [rbp-1B0h] BYREF
  unsigned __int64 v200; // [rsp+188h] [rbp-1A8h]
  __m128i *v201; // [rsp+190h] [rbp-1A0h] BYREF
  __int64 **v202; // [rsp+198h] [rbp-198h]
  __m128i v203; // [rsp+1A0h] [rbp-190h] BYREF
  __int64 *v204; // [rsp+1B0h] [rbp-180h] BYREF
  char *v205; // [rsp+1B8h] [rbp-178h]
  _WORD v206[8]; // [rsp+1C0h] [rbp-170h] BYREF
  __m128i *v207; // [rsp+1D0h] [rbp-160h] BYREF
  size_t v208; // [rsp+1D8h] [rbp-158h]
  __m128i v209; // [rsp+1E0h] [rbp-150h] BYREF
  __m128i *v210; // [rsp+1F0h] [rbp-140h] BYREF
  __int64 v211; // [rsp+1F8h] [rbp-138h]
  _BYTE v212[304]; // [rsp+200h] [rbp-130h] BYREF

  v5 = (__int64 *)&v207;
  v6 = *(char **)(a2 + 8);
  v210 = (__m128i *)v212;
  v211 = 0x1000000000LL;
  v7 = *(_QWORD *)(a2 + 16) - (_QWORD)v6;
  v207 = (__m128i *)v6;
  v208 = v7;
  sub_16D2880((char **)&v207, (__int64)&v210, 10, -1, 1, (int)&v210);
  v8 = v210;
  n = 1;
  src = "*";
  v181 = &v210[(unsigned int)v211];
  if ( v210 == v181 )
  {
    v25 = 1;
    goto LABEL_20;
  }
  v180 = 1;
  while ( 1 )
  {
    v9 = 0;
    v10 = sub_16D24E0(v8, byte_3F15413, 6, 0);
    v11 = v8->m128i_u64[1];
    if ( v10 < v11 )
    {
      v9 = v11 - v10;
      v11 = v10;
    }
    v207 = (__m128i *)(v8->m128i_i64[0] + v11);
    v208 = v9;
    v12 = sub_16D2680(v5, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL);
    v13 = v208;
    v14 = (char *)v207;
    v15 = v12 + 1;
    v16 = v15 <= v208;
    v17 = v208;
    v8->m128i_i64[0] = (__int64)v207;
    if ( !v16 )
      v15 = v13;
    v18 = v17 - v9 + v15;
    if ( v18 > v13 )
      v18 = v13;
    v8->m128i_i64[1] = v18;
    if ( !v18 || *v14 == 35 )
      goto LABEL_61;
    if ( *v14 != 91 )
      break;
    v19 = v18 - 1;
    if ( v14[v18 - 1] != 93 )
    {
      v203.m128i_i16[0] = 770;
      LODWORD(v195) = v180;
      v198.m128i_i64[0] = (__int64)"malformed section header on line ";
      v206[0] = 1282;
      v198.m128i_i64[1] = (__int64)v195;
      v201 = &v198;
      v202 = (__int64 **)": ";
      v204 = (__int64 *)&v201;
      LOWORD(v199) = 2307;
      v205 = (char *)v8;
      sub_16E2FC0(v5, (__int64)&v204);
      v20 = *a4;
      v21 = (char *)v207;
      if ( v207 != &v209 )
      {
        v22 = v209.m128i_i64[0];
        v23 = v208;
        if ( v20 == (__m128i *)(a4 + 2) )
        {
          *a4 = v207;
          a4[1] = (__m128i *)v23;
          a4[2] = (__m128i *)v22;
          goto LABEL_95;
        }
        goto LABEL_15;
      }
      goto LABEL_134;
    }
    v62 = v14 + 1;
    v205 = 0;
    if ( v18 == 1 )
      v19 = 1;
    src = v62;
    n = v19 - 1;
    v204 = (__int64 *)v206;
    LOBYTE(v206[0]) = 0;
    sub_16C9340((__int64)&v192, (__int64)v62, v19 - 1, 0);
    if ( !(unsigned __int8)sub_16C9430(&v192, &v204) )
    {
      v202 = &v204;
      v195 = "malformed regex for section ";
      p_src = &src;
      v197 = 1283;
      v198.m128i_i64[0] = (__int64)&v195;
      v198.m128i_i64[1] = (__int64)": '";
      LOWORD(v199) = 770;
      v201 = &v198;
      v203.m128i_i16[0] = 1026;
      sub_16E2FC0(v5, (__int64)&v201);
      v63 = *a4;
      if ( v207 == &v209 )
      {
        v95 = v208;
        if ( v208 )
        {
          if ( v208 == 1 )
            v63->m128i_i8[0] = v209.m128i_i8[0];
          else
            memcpy(v63, &v209, v208);
          v95 = v208;
          v63 = *a4;
        }
        a4[1] = (__m128i *)v95;
        v63->m128i_i8[v95] = 0;
        v63 = v207;
      }
      else
      {
        v64 = v208;
        v65 = v209.m128i_i64[0];
        if ( v63 == (__m128i *)(a4 + 2) )
        {
          *a4 = v207;
          a4[1] = (__m128i *)v64;
          a4[2] = (__m128i *)v65;
        }
        else
        {
          v66 = (__int64)a4[2];
          *a4 = v207;
          a4[1] = (__m128i *)v64;
          a4[2] = (__m128i *)v65;
          if ( v63 )
          {
            v207 = v63;
            v209.m128i_i64[0] = v66;
            goto LABEL_70;
          }
        }
        v207 = &v209;
        v63 = &v209;
      }
LABEL_70:
      v208 = 0;
      v63->m128i_i8[0] = 0;
      if ( v207 != &v209 )
        j_j___libc_free_0((unsigned __int64)v207);
      sub_16C93F0(&v192);
      v67 = (unsigned __int64)v204;
      if ( v204 != (__int64 *)v206 )
        goto LABEL_73;
      goto LABEL_19;
    }
    sub_16C93F0(&v192);
    v61 = (unsigned __int64)v204;
    if ( v204 != (__int64 *)v206 )
      goto LABEL_60;
LABEL_61:
    ++v180;
    if ( v181 == ++v8 )
    {
      v25 = 1;
      v181 = v210;
      goto LABEL_20;
    }
  }
  v179 = sub_16D20C0(v8->m128i_i64, ":", 1u, 0);
  if ( v179 == -1 )
  {
    v72 = _mm_loadu_si128(v8);
    v199 = 0;
    v200 = 0;
    v198 = v72;
LABEL_92:
    v194 = 2307;
    LODWORD(v189) = v180;
    v192 = "malformed line ";
    v193 = (char *)v189;
    v195 = (const char *)&v192;
    p_src = (void **)": '";
    v197 = 770;
    v201 = (__m128i *)&v195;
    v202 = (__int64 **)&v198;
    v203.m128i_i16[0] = 1282;
    v204 = (__int64 *)&v201;
    v205 = "'";
    v206[0] = 770;
    sub_16E2FC0(v5, (__int64)&v204);
    v20 = *a4;
    v21 = (char *)v207;
    if ( v207 != &v209 )
    {
      v23 = v208;
      v22 = v209.m128i_i64[0];
      if ( v20 == (__m128i *)(a4 + 2) )
      {
        *a4 = v207;
        a4[1] = (__m128i *)v23;
        a4[2] = (__m128i *)v22;
        goto LABEL_95;
      }
LABEL_15:
      v24 = (__int64)a4[2];
      *a4 = (__m128i *)v21;
      a4[1] = (__m128i *)v23;
      a4[2] = (__m128i *)v22;
      if ( v20 )
      {
        v207 = v20;
        v209.m128i_i64[0] = v24;
        goto LABEL_17;
      }
LABEL_95:
      v207 = &v209;
      v20 = &v209;
LABEL_17:
      v208 = 0;
      v20->m128i_i8[0] = 0;
      if ( v207 != &v209 )
        j_j___libc_free_0((unsigned __int64)v207);
      goto LABEL_19;
    }
LABEL_134:
    v94 = v208;
    if ( v208 )
    {
      if ( v208 == 1 )
        v20->m128i_i8[0] = v209.m128i_i8[0];
      else
        memcpy(v20, &v209, v208);
      v94 = v208;
      v20 = *a4;
    }
    a4[1] = (__m128i *)v94;
    v20->m128i_i8[v94] = 0;
    v20 = v207;
    goto LABEL_17;
  }
  v27 = v8->m128i_u64[1];
  v28 = v179 + 1;
  v176 = (unsigned __int8 *)v8->m128i_i64[0];
  if ( v179 + 1 > v27 )
    v28 = v8->m128i_u64[1];
  v29 = v27 - v28;
  v30 = v8->m128i_i64[0] + v28;
  if ( v179 )
  {
    if ( v179 <= v27 )
      v27 = v179;
    v179 = v27;
  }
  v199 = (_BYTE *)v30;
  v200 = v29;
  v198.m128i_i64[0] = (__int64)v176;
  v198.m128i_i64[1] = v179;
  if ( !v29 )
    goto LABEL_92;
  v31 = sub_16D20C0((__int64 *)&v199, "=", 1u, 0);
  if ( v31 == -1 )
  {
    v33 = v199;
    v31 = v200;
    v175 = 0;
    v174 = 0;
  }
  else
  {
    v32 = v31 + 1;
    v33 = v199;
    if ( v31 + 1 > v200 )
      v32 = v200;
    v175 = v200 - v32;
    v174 = &v199[v32];
    if ( v31 )
    {
      if ( v31 > v200 )
        v31 = v200;
      if ( v199 )
        goto LABEL_33;
      goto LABEL_90;
    }
  }
  if ( v33 )
  {
LABEL_33:
    v201 = &v203;
    sub_3946B50((__int64 *)&v201, v33, (__int64)&v33[v31]);
    goto LABEL_34;
  }
LABEL_90:
  v203.m128i_i8[0] = 0;
  v201 = &v203;
  v202 = 0;
LABEL_34:
  v34 = *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8);
  v35 = sub_16D1B30((__int64 *)a3, (unsigned __int8 *)src, n);
  if ( v35 == -1 )
  {
    if ( v34 != *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8) )
      goto LABEL_36;
  }
  else if ( v34 != *(_QWORD *)a3 + 8LL * v35 )
  {
    goto LABEL_36;
  }
  sub_3946E60(v187);
  v205 = 0;
  v73 = v187[0];
  v204 = (__int64 *)v206;
  LOBYTE(v206[0]) = 0;
  v207 = &v209;
  if ( src )
  {
    sub_3946B50(v5, src, (__int64)src + n);
  }
  else
  {
    v208 = 0;
    v209.m128i_i8[0] = 0;
  }
  v74 = sub_3947130(v73, v5, v180, (unsigned __int64 *)&v204);
  if ( v207 != &v209 )
    j_j___libc_free_0((unsigned __int64)v207);
  if ( v74 )
  {
    v75 = src;
    v76 = n;
    v77 = sub_16D19C0(a3, (unsigned __int8 *)src, n);
    v78 = (_QWORD *)(*(_QWORD *)a3 + 8LL * v77);
    v79 = *v78;
    if ( *v78 )
    {
      if ( v79 != -8 )
      {
LABEL_105:
        *(_QWORD *)(v79 + 8) = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(a1[1] - *a1) >> 3);
        v80 = a1[1];
        if ( v80 == a1[2] )
        {
          sub_3948010(a1, (char *)v80, (unsigned __int64 *)v187);
        }
        else
        {
          v81 = v187[0];
          v187[0] = 0;
          if ( v80 )
          {
            *(_QWORD *)v80 = v81;
            *(_QWORD *)(v80 + 8) = 0;
            *(_QWORD *)(v80 + 16) = 0;
            *(_QWORD *)(v80 + 24) = 0x2800000000LL;
            v81 = a1[1];
          }
          else if ( v81 )
          {
            v105 = *(unsigned __int64 **)(v81 + 128);
            v106 = *(_QWORD *)(v81 + 120);
            if ( v105 != (unsigned __int64 *)v106 )
            {
              v162 = v5;
              v168 = v8;
              v107 = *(unsigned __int64 **)(v81 + 120);
              v108 = v81;
              do
              {
                v109 = *v107;
                if ( *v107 )
                {
                  sub_16C93F0((_QWORD *)*v107);
                  j_j___libc_free_0(v109);
                }
                v107 += 2;
              }
              while ( v105 != v107 );
              v81 = v108;
              v8 = v168;
              v106 = *(_QWORD *)(v108 + 120);
              v5 = v162;
            }
            if ( v106 )
            {
              v169 = v81;
              j_j___libc_free_0(v106);
              v81 = v169;
            }
            v170 = v81;
            sub_3947930(v81 + 64);
            v110 = (unsigned __int64 *)v170;
            v111 = *(_QWORD *)(v170 + 64);
            if ( v111 != v170 + 112 )
            {
              j_j___libc_free_0(v111);
              v110 = (unsigned __int64 *)v170;
            }
            v112 = v110[5];
            if ( v112 )
            {
              v171 = v110;
              j_j___libc_free_0(v112);
              v110 = v171;
            }
            v113 = *v110;
            if ( *((_DWORD *)v110 + 3) )
            {
              v114 = *((unsigned int *)v110 + 2);
              if ( (_DWORD)v114 )
              {
                v172 = v8;
                v115 = 0;
                v116 = 8 * v114;
                v117 = v110;
                do
                {
                  v118 = *(_QWORD *)(v113 + v115);
                  if ( v118 != -8 && v118 )
                  {
                    _libc_free(v118);
                    v113 = *v117;
                  }
                  v115 += 8;
                }
                while ( v116 != v115 );
                v8 = v172;
                v110 = v117;
              }
            }
            v173 = (unsigned __int64)v110;
            _libc_free(v113);
            j_j___libc_free_0(v173);
            v81 = a1[1];
          }
          a1[1] = v81 + 40;
        }
        if ( v204 != (__int64 *)v206 )
          j_j___libc_free_0((unsigned __int64)v204);
        v82 = v187[0];
        if ( v187[0] )
        {
          v83 = *(unsigned __int64 **)(v187[0] + 128);
          v84 = *(_QWORD *)(v187[0] + 120);
          if ( v83 != (unsigned __int64 *)v84 )
          {
            v160 = v5;
            v166 = v8;
            v85 = *(unsigned __int64 **)(v187[0] + 120);
            do
            {
              v86 = *v85;
              if ( *v85 )
              {
                sub_16C93F0((_QWORD *)*v85);
                j_j___libc_free_0(v86);
              }
              v85 += 2;
            }
            while ( v83 != v85 );
            v8 = v166;
            v5 = v160;
            v84 = *(_QWORD *)(v82 + 120);
          }
          if ( v84 )
            j_j___libc_free_0(v84);
          sub_3947930(v82 + 64);
          v87 = *(_QWORD *)(v82 + 64);
          if ( v87 != v82 + 112 )
            j_j___libc_free_0(v87);
          v88 = *(_QWORD *)(v82 + 40);
          if ( v88 )
            j_j___libc_free_0(v88);
          v89 = *(_QWORD *)v82;
          if ( *(_DWORD *)(v82 + 12) )
          {
            v90 = *(unsigned int *)(v82 + 8);
            if ( (_DWORD)v90 )
            {
              v167 = v8;
              v91 = 0;
              v92 = 8 * v90;
              do
              {
                v93 = *(_QWORD *)(v89 + v91);
                if ( v93 != -8 && v93 )
                {
                  _libc_free(v93);
                  v89 = *(_QWORD *)v82;
                }
                v91 += 8;
              }
              while ( v92 != v91 );
              v8 = v167;
            }
          }
          _libc_free(v89);
          j_j___libc_free_0(v82);
        }
LABEL_36:
        v36 = src;
        v37 = n;
        v38 = sub_16D19C0(a3, (unsigned __int8 *)src, n);
        v39 = (_QWORD *)(*(_QWORD *)a3 + 8LL * v38);
        v40 = *v39;
        if ( *v39 )
        {
          if ( v40 != -8 )
            goto LABEL_44;
          --*(_DWORD *)(a3 + 16);
        }
        v144 = v39;
        v150 = v38;
        v41 = malloc(v37 + 17);
        v42 = v150;
        v43 = v144;
        v44 = (_QWORD *)v41;
        if ( !v41 )
        {
          if ( v37 == -17 )
          {
            v68 = malloc(1u);
            v42 = v150;
            v43 = v144;
            v44 = 0;
            if ( v68 )
            {
              v45 = (_BYTE *)(v68 + 16);
              v44 = (_QWORD *)v68;
              goto LABEL_78;
            }
          }
          v149 = v44;
          v156 = v43;
          v163 = v42;
          sub_16BD1C0("Allocation failed", 1u);
          v42 = v163;
          v43 = v156;
          v44 = v149;
        }
        v45 = v44 + 2;
        if ( v37 + 1 <= 1 )
        {
LABEL_41:
          v45[v37] = 0;
          *v44 = v37;
          v44[1] = 0;
          *v43 = v44;
          ++*(_DWORD *)(a3 + 12);
          v46 = (__int64 *)(*(_QWORD *)a3 + 8LL * (unsigned int)sub_16D1CD0(a3, v42));
          v40 = *v46;
          if ( *v46 != -8 )
            goto LABEL_43;
          do
          {
            do
            {
              v40 = v46[1];
              ++v46;
            }
            while ( v40 == -8 );
LABEL_43:
            ;
          }
          while ( !v40 );
LABEL_44:
          v47 = *a1 + 40LL * *(_QWORD *)(v40 + 8);
          v48 = sub_16D19C0(v47 + 8, v176, v179);
          v49 = (_QWORD *)(*(_QWORD *)(v47 + 8) + 8LL * v48);
          v50 = *v49;
          if ( *v49 )
          {
            if ( v50 != -8 )
              goto LABEL_52;
            --*(_DWORD *)(v47 + 24);
          }
          v145 = v49;
          v151 = v48;
          v51 = malloc(v179 + 41);
          v52 = v151;
          v53 = v145;
          v54 = (_QWORD *)v51;
          if ( !v51 )
          {
            if ( v179 == -41 )
            {
              v70 = malloc(1u);
              v52 = v151;
              v53 = v145;
              if ( v70 )
              {
                v55 = (_BYTE *)(v70 + 40);
                v54 = (_QWORD *)v70;
                goto LABEL_81;
              }
            }
            v157 = v53;
            v164 = v52;
            sub_16BD1C0("Allocation failed", 1u);
            v52 = v164;
            v53 = v157;
          }
          v55 = v54 + 5;
          if ( !v179 )
          {
LABEL_49:
            v55[v179] = 0;
            *v54 = v179;
            v54[1] = 0;
            v54[2] = 0;
            v54[3] = 0x9800000000LL;
            *v53 = v54;
            ++*(_DWORD *)(v47 + 20);
            v56 = (__int64 *)(*(_QWORD *)(v47 + 8) + 8LL * (unsigned int)sub_16D1CD0(v47 + 8, v52));
            v50 = *v56;
            if ( !*v56 || v50 == -8 )
            {
              do
              {
                do
                {
                  v50 = v56[1];
                  ++v56;
                }
                while ( v50 == -8 );
              }
              while ( !v50 );
            }
LABEL_52:
            v57 = *(_QWORD *)sub_3947710(v50 + 8, v174, v175);
            v205 = 0;
            LOBYTE(v206[0]) = 0;
            v204 = (__int64 *)v206;
            v58 = v57 + 8;
            v207 = &v209;
            if ( v201 == &v203 )
            {
              v209 = _mm_load_si128(&v203);
            }
            else
            {
              v207 = v201;
              v209.m128i_i64[0] = v203.m128i_i64[0];
            }
            v59 = (size_t)v202;
            v202 = 0;
            v203.m128i_i8[0] = 0;
            v208 = v59;
            v201 = &v203;
            v60 = sub_3947130(v58, v5, v180, (unsigned __int64 *)&v204);
            if ( v207 != &v209 )
              j_j___libc_free_0((unsigned __int64)v207);
            if ( v60 )
            {
              if ( v204 != (__int64 *)v206 )
                j_j___libc_free_0((unsigned __int64)v204);
              v61 = (unsigned __int64)v201;
              if ( v201 == &v203 )
                goto LABEL_61;
LABEL_60:
              j_j___libc_free_0(v61);
              goto LABEL_61;
            }
            v194 = 770;
            LODWORD(v184) = v180;
            v185[0] = "malformed regex in line ";
            v197 = 1026;
            v185[1] = v184;
            v186 = 2307;
            v187[0] = (__int64)v185;
            v187[1] = (__int64)": '";
            v189 = v187;
            v188 = 770;
            v190 = (void **)&v199;
            v192 = (const char *)&v189;
            v193 = "': ";
            v195 = (const char *)&v192;
            v191 = 1282;
            p_src = (void **)&v204;
            sub_16E2FC0(v5, (__int64)&v195);
            v119 = *a4;
            if ( v207 == &v209 )
            {
              v123 = v208;
              if ( v208 )
              {
                if ( v208 == 1 )
                  v119->m128i_i8[0] = v209.m128i_i8[0];
                else
                  memcpy(v119, &v209, v208);
                v123 = v208;
                v119 = *a4;
              }
              a4[1] = (__m128i *)v123;
              v119->m128i_i8[v123] = 0;
              v119 = v207;
              goto LABEL_188;
            }
            v120 = v208;
            v121 = v209.m128i_i64[0];
            if ( v119 == (__m128i *)(a4 + 2) )
            {
              *a4 = v207;
              a4[1] = (__m128i *)v120;
              a4[2] = (__m128i *)v121;
            }
            else
            {
              v122 = (__int64)a4[2];
              *a4 = v207;
              a4[1] = (__m128i *)v120;
              a4[2] = (__m128i *)v121;
              if ( v119 )
              {
                v207 = v119;
                v209.m128i_i64[0] = v122;
LABEL_188:
                v208 = 0;
                v119->m128i_i8[0] = 0;
                if ( v207 != &v209 )
                  j_j___libc_free_0((unsigned __int64)v207);
                if ( v204 != (__int64 *)v206 )
                  j_j___libc_free_0((unsigned __int64)v204);
                goto LABEL_192;
              }
            }
            v207 = &v209;
            v119 = &v209;
            goto LABEL_188;
          }
LABEL_81:
          v153 = v53;
          v159 = v52;
          v71 = memcpy(v55, v176, v179);
          v53 = v153;
          v52 = v159;
          v55 = v71;
          goto LABEL_49;
        }
LABEL_78:
        v152 = v44;
        v158 = v43;
        v165 = v42;
        v69 = memcpy(v45, v36, v37);
        v44 = v152;
        v43 = v158;
        v42 = v165;
        v45 = v69;
        goto LABEL_41;
      }
      --*(_DWORD *)(a3 + 16);
    }
    v142 = v78;
    v146 = v77;
    v96 = malloc(v76 + 17);
    v97 = v146;
    v98 = v142;
    v99 = (_QWORD *)v96;
    if ( !v96 )
    {
      if ( v76 == -17 )
      {
        v103 = malloc(1u);
        v97 = v146;
        v98 = v142;
        v99 = 0;
        if ( v103 )
        {
          v100 = (_BYTE *)(v103 + 16);
          v99 = (_QWORD *)v103;
          goto LABEL_158;
        }
      }
      v143 = v99;
      v148 = v98;
      v155 = v97;
      sub_16BD1C0("Allocation failed", 1u);
      v97 = v155;
      v98 = v148;
      v99 = v143;
    }
    v100 = v99 + 2;
    if ( v76 + 1 <= 1 )
    {
LABEL_148:
      v100[v76] = 0;
      *v99 = v76;
      v99[1] = 0;
      *v98 = v99;
      ++*(_DWORD *)(a3 + 12);
      v101 = (__int64 *)(*(_QWORD *)a3 + 8LL * (unsigned int)sub_16D1CD0(a3, v97));
      v79 = *v101;
      if ( !*v101 || v79 == -8 )
      {
        v102 = v101 + 1;
        do
        {
          do
            v79 = *v102++;
          while ( !v79 );
        }
        while ( v79 == -8 );
      }
      goto LABEL_105;
    }
LABEL_158:
    v147 = v99;
    v154 = v98;
    v161 = v97;
    v104 = memcpy(v100, v75, v76);
    v99 = v147;
    v98 = v154;
    v97 = v161;
    v100 = v104;
    goto LABEL_148;
  }
  v189 = (__int64 *)"malformed section ";
  v190 = &src;
  v192 = (const char *)&v189;
  v193 = ": '";
  v195 = (const char *)&v192;
  v191 = 1283;
  p_src = (void **)&v204;
  v194 = 770;
  v197 = 1026;
  sub_16E2FC0(v5, (__int64)&v195);
  v124 = *a4;
  v125 = *a4;
  if ( v207 == &v209 )
  {
    v141 = v208;
    if ( v208 )
    {
      if ( v208 == 1 )
        v124->m128i_i8[0] = v209.m128i_i8[0];
      else
        memcpy(v124, &v209, v208);
      v141 = v208;
      v124 = *a4;
    }
    a4[1] = (__m128i *)v141;
    v124->m128i_i8[v141] = 0;
    v125 = v207;
  }
  else
  {
    v126 = v208;
    v127 = v209.m128i_i64[0];
    if ( v125 == (__m128i *)(a4 + 2) )
    {
      *a4 = v207;
      a4[1] = (__m128i *)v126;
      a4[2] = (__m128i *)v127;
    }
    else
    {
      v128 = (__int64)a4[2];
      *a4 = v207;
      a4[1] = (__m128i *)v126;
      a4[2] = (__m128i *)v127;
      if ( v125 )
      {
        v207 = v125;
        v209.m128i_i64[0] = v128;
        goto LABEL_205;
      }
    }
    v207 = &v209;
    v125 = &v209;
  }
LABEL_205:
  v208 = 0;
  v125->m128i_i8[0] = 0;
  if ( v207 != &v209 )
    j_j___libc_free_0((unsigned __int64)v207);
  if ( v204 != (__int64 *)v206 )
    j_j___libc_free_0((unsigned __int64)v204);
  v129 = v187[0];
  if ( v187[0] )
  {
    v130 = *(unsigned __int64 **)(v187[0] + 128);
    for ( i = *(unsigned __int64 **)(v187[0] + 120); v130 != i; i += 2 )
    {
      v132 = *i;
      if ( *i )
      {
        sub_16C93F0((_QWORD *)*i);
        j_j___libc_free_0(v132);
      }
    }
    v133 = *(_QWORD *)(v129 + 120);
    if ( v133 )
      j_j___libc_free_0(v133);
    sub_3947930(v129 + 64);
    v134 = *(_QWORD *)(v129 + 64);
    if ( v134 != v129 + 112 )
      j_j___libc_free_0(v134);
    v135 = *(_QWORD *)(v129 + 40);
    if ( v135 )
      j_j___libc_free_0(v135);
    v136 = *(_QWORD *)v129;
    if ( *(_DWORD *)(v129 + 12) )
    {
      v137 = *(unsigned int *)(v129 + 8);
      if ( (_DWORD)v137 )
      {
        v138 = 8 * v137;
        v139 = 0;
        do
        {
          v140 = *(_QWORD *)(v136 + v139);
          if ( v140 && v140 != -8 )
          {
            _libc_free(*(_QWORD *)(v136 + v139));
            v136 = *(_QWORD *)v129;
          }
          v139 += 8;
        }
        while ( v139 != v138 );
      }
    }
    _libc_free(v136);
    j_j___libc_free_0(v129);
  }
LABEL_192:
  v67 = (unsigned __int64)v201;
  if ( v201 != &v203 )
LABEL_73:
    j_j___libc_free_0(v67);
LABEL_19:
  v25 = 0;
  v181 = v210;
LABEL_20:
  if ( v181 != (__m128i *)v212 )
    _libc_free((unsigned __int64)v181);
  return v25;
}
