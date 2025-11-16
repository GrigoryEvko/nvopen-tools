// Function: sub_2A33580
// Address: 0x2a33580
//
__int64 __fastcall sub_2A33580(__int64 a1, __int64 *a2, __int64 a3, char a4, __m128i *a5, __m128i *a6)
{
  unsigned int v6; // r12d
  __int64 v7; // rbx
  char v8; // cl
  __int64 v9; // rbx
  __int64 v10; // rsi
  __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 *v13; // rax
  __int64 *v14; // r14
  __int64 v15; // r13
  __int64 *v16; // rbx
  unsigned __int64 v18; // rcx
  __int64 *v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned int v23; // eax
  __m128i *v24; // r14
  int v25; // r12d
  __int64 v26; // rbx
  __int64 v27; // r15
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // rdx
  signed __int64 v32; // rsi
  unsigned __int64 v33; // rax
  unsigned __int64 v34; // rdi
  bool v35; // cf
  unsigned __int64 v36; // rax
  __int64 v37; // rax
  __m128i *v38; // r10
  __int64 *v39; // rsi
  __m128i *v40; // rdx
  const __m128i *v41; // rax
  __int64 *v42; // rax
  __m128i *v43; // r13
  __m128i *v44; // r12
  signed __int64 v45; // rbx
  unsigned __int64 v46; // rax
  __m128i *v47; // rbx
  __int64 *v48; // rdi
  unsigned int v49; // ebx
  unsigned __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  unsigned int v53; // r14d
  __int64 v54; // r12
  __int64 v55; // r13
  unsigned __int64 v56; // rax
  __int64 v57; // r12
  unsigned int v58; // eax
  unsigned int v59; // eax
  const __m128i *i; // rbx
  unsigned __int64 v61; // rax
  unsigned int v62; // eax
  unsigned int v63; // r14d
  unsigned __int64 v64; // r8
  __int64 v65; // r15
  int v66; // r10d
  __int64 *v67; // r11
  unsigned int v68; // ecx
  __int64 *v69; // r13
  __int64 v70; // rdx
  __int64 v71; // r15
  unsigned int v72; // eax
  __int64 v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // r13
  const void **v77; // r14
  _DWORD *v78; // r15
  bool v79; // al
  __int64 v80; // rdi
  unsigned __int64 v81; // rdi
  unsigned __int64 v82; // rdi
  unsigned int v83; // edx
  unsigned __int64 v84; // rax
  unsigned __int64 *v85; // r14
  unsigned __int64 *v86; // r13
  __int64 *v87; // rax
  __int64 *v88; // rax
  __int64 v89; // r8
  unsigned int v90; // eax
  unsigned __int64 v91; // r13
  bool v92; // r12
  _QWORD *v93; // r12
  __int64 v94; // r13
  unsigned __int16 v95; // bx
  _QWORD *v96; // rdi
  __int64 v97; // r12
  __int64 v98; // rdx
  __int64 v99; // rcx
  __int64 v100; // rax
  __int64 v101; // rbx
  unsigned __int64 v102; // r12
  unsigned __int64 v103; // rdi
  __int64 v104; // r12
  unsigned __int16 v105; // bx
  _QWORD *v106; // rdi
  __m128i *v107; // rbx
  const __m128i *v108; // r12
  __int64 v109; // rax
  __int64 v110; // r13
  __int64 v111; // rdi
  unsigned int v112; // r9d
  unsigned __int64 v113; // r8
  bool v114; // al
  __m128i v115; // xmm2
  __m128i *v116; // rdx
  __int64 *v117; // rax
  __int64 *v118; // rax
  __int64 v119; // rdx
  unsigned int v120; // eax
  const void *v121; // rcx
  unsigned __int64 v122; // rdi
  const void *v123; // rax
  __int64 v124; // r13
  unsigned int v125; // eax
  unsigned int v126; // eax
  int v127; // r12d
  int v128; // ebx
  __m128i *v129; // rdx
  __m128i *v130; // rsi
  signed __int64 v131; // rax
  const __m128i *v132; // rax
  __int64 v133; // rax
  __int64 v134; // rbx
  __int64 v135; // r12
  unsigned __int64 v136; // rdi
  unsigned __int64 v137; // rdx
  int v138; // edx
  __int64 v139; // rax
  __int64 v140; // rsi
  int v141; // r9d
  __int64 *v142; // rcx
  int v143; // r9d
  unsigned int v144; // eax
  __int64 v145; // rsi
  __int64 v146; // r12
  unsigned __int16 v147; // bx
  _QWORD *v148; // rdi
  unsigned int v149; // r13d
  bool v150; // al
  __int64 v151; // rax
  __int64 v152; // rbx
  __int64 v153; // r12
  unsigned __int64 v154; // rdi
  __int64 v155; // rbx
  unsigned __int64 v156; // rdi
  unsigned int v157; // r14d
  unsigned __int64 v158; // r15
  int v159; // r13d
  unsigned __int64 v160; // r10
  __int64 v161; // rax
  __int64 v163; // [rsp+10h] [rbp-260h]
  __int64 v164; // [rsp+18h] [rbp-258h]
  int v166; // [rsp+28h] [rbp-248h]
  __int64 v168; // [rsp+30h] [rbp-240h]
  __int64 v169; // [rsp+30h] [rbp-240h]
  const __m128i *v170; // [rsp+38h] [rbp-238h]
  unsigned int v171; // [rsp+40h] [rbp-230h]
  __m128i *v172; // [rsp+40h] [rbp-230h]
  signed __int64 v173; // [rsp+40h] [rbp-230h]
  __int64 v174; // [rsp+48h] [rbp-228h]
  __int64 v175; // [rsp+50h] [rbp-220h]
  __int64 v176; // [rsp+50h] [rbp-220h]
  const void *v177; // [rsp+50h] [rbp-220h]
  __int64 v178; // [rsp+58h] [rbp-218h]
  unsigned __int64 v179; // [rsp+58h] [rbp-218h]
  unsigned int v180; // [rsp+58h] [rbp-218h]
  __m128i *v181; // [rsp+58h] [rbp-218h]
  __int64 v182; // [rsp+60h] [rbp-210h]
  __int64 v183; // [rsp+60h] [rbp-210h]
  unsigned int v184; // [rsp+60h] [rbp-210h]
  unsigned int v185; // [rsp+60h] [rbp-210h]
  __int64 v186; // [rsp+60h] [rbp-210h]
  __int64 v187; // [rsp+60h] [rbp-210h]
  unsigned __int64 v188; // [rsp+60h] [rbp-210h]
  unsigned __int64 v189; // [rsp+60h] [rbp-210h]
  __int64 v190; // [rsp+60h] [rbp-210h]
  _QWORD *v191; // [rsp+68h] [rbp-208h]
  __int64 v192; // [rsp+70h] [rbp-200h]
  __m128i *v193; // [rsp+78h] [rbp-1F8h]
  __int64 v194; // [rsp+78h] [rbp-1F8h]
  __int64 v195; // [rsp+80h] [rbp-1F0h]
  unsigned __int64 v196; // [rsp+88h] [rbp-1E8h]
  __m128i *v197; // [rsp+90h] [rbp-1E0h]
  __int64 v198; // [rsp+90h] [rbp-1E0h]
  bool v199; // [rsp+90h] [rbp-1E0h]
  unsigned __int64 v200; // [rsp+90h] [rbp-1E0h]
  __int64 v201; // [rsp+98h] [rbp-1D8h]
  char v202; // [rsp+AFh] [rbp-1C1h] BYREF
  __int64 v203; // [rsp+B0h] [rbp-1C0h] BYREF
  __int64 v204; // [rsp+B8h] [rbp-1B8h] BYREF
  const void *v205; // [rsp+C0h] [rbp-1B0h] BYREF
  unsigned int v206; // [rsp+C8h] [rbp-1A8h]
  unsigned __int64 v207; // [rsp+D0h] [rbp-1A0h] BYREF
  unsigned int v208; // [rsp+D8h] [rbp-198h]
  unsigned __int64 v209; // [rsp+E0h] [rbp-190h] BYREF
  unsigned int v210; // [rsp+E8h] [rbp-188h]
  unsigned __int64 v211; // [rsp+F0h] [rbp-180h] BYREF
  unsigned int v212; // [rsp+F8h] [rbp-178h]
  unsigned __int64 v213; // [rsp+100h] [rbp-170h] BYREF
  unsigned int v214; // [rsp+108h] [rbp-168h]
  const void *v215; // [rsp+110h] [rbp-160h] BYREF
  unsigned int v216; // [rsp+118h] [rbp-158h]
  const void *v217; // [rsp+120h] [rbp-150h] BYREF
  unsigned int v218; // [rsp+128h] [rbp-148h]
  unsigned __int64 v219; // [rsp+130h] [rbp-140h] BYREF
  unsigned int v220; // [rsp+138h] [rbp-138h]
  void *src; // [rsp+140h] [rbp-130h] BYREF
  __m128i *v222; // [rsp+148h] [rbp-128h]
  __m128i *v223; // [rsp+150h] [rbp-120h]
  unsigned __int64 v224; // [rsp+160h] [rbp-110h] BYREF
  __int64 v225; // [rsp+168h] [rbp-108h]
  unsigned __int64 v226; // [rsp+170h] [rbp-100h]
  unsigned int v227; // [rsp+178h] [rbp-F8h]
  unsigned __int64 v228; // [rsp+180h] [rbp-F0h] BYREF
  __int64 v229; // [rsp+188h] [rbp-E8h]
  unsigned __int64 v230; // [rsp+190h] [rbp-E0h]
  unsigned int v231; // [rsp+198h] [rbp-D8h]
  unsigned __int64 v232; // [rsp+1A0h] [rbp-D0h] BYREF
  unsigned int v233; // [rsp+1A8h] [rbp-C8h]
  unsigned __int64 v234; // [rsp+1B0h] [rbp-C0h] BYREF
  unsigned int v235; // [rsp+1B8h] [rbp-B8h]
  const void *v236; // [rsp+1C0h] [rbp-B0h] BYREF
  unsigned int v237; // [rsp+1C8h] [rbp-A8h]
  unsigned __int64 v238; // [rsp+1D0h] [rbp-A0h] BYREF
  unsigned int v239; // [rsp+1D8h] [rbp-98h]
  __int64 v240; // [rsp+1E0h] [rbp-90h] BYREF
  __int64 *v241; // [rsp+1E8h] [rbp-88h]
  __int64 v242; // [rsp+1F0h] [rbp-80h]
  int v243; // [rsp+1F8h] [rbp-78h]
  unsigned __int8 v244; // [rsp+1FCh] [rbp-74h]
  char v245; // [rsp+200h] [rbp-70h] BYREF

  v6 = 0;
  v7 = *(_QWORD *)(a1 + 80);
  v241 = (__int64 *)&v245;
  v8 = 1;
  v240 = 0;
  v242 = 8;
  v243 = 0;
  v244 = 1;
  v192 = a1 + 72;
  v201 = v7;
  if ( v7 == a1 + 72 )
    goto LABEL_13;
  do
  {
    while ( 1 )
    {
      v9 = v201;
      v10 = v201 - 24;
      v201 = *(_QWORD *)(v201 + 8);
      if ( !v8 )
      {
        if ( sub_C8CA60((__int64)&v240, v10) )
          goto LABEL_65;
        goto LABEL_17;
      }
      v11 = v241;
      v12 = (__int64)&v241[HIDWORD(v242)];
      if ( v241 != (__int64 *)v12 )
        break;
LABEL_17:
      v18 = *(_QWORD *)(v9 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      v196 = v18;
      if ( v18 == v9 + 24
        || !v18
        || (v191 = (_QWORD *)(v18 - 24), (unsigned int)*(unsigned __int8 *)(v18 - 24) - 30 > 0xA) )
      {
LABEL_469:
        BUG();
      }
      if ( *(_BYTE *)(v18 - 24) != 32 || a4 && (unsigned __int8)sub_2A33170((__int64)v191, &v203, &v204, &v202) )
      {
LABEL_65:
        v8 = v244;
        if ( v192 == v201 )
          goto LABEL_8;
      }
      else
      {
        v19 = *(__int64 **)(v196 - 32);
        v20 = *v19;
        v195 = *(_QWORD *)(v196 + 16);
        v168 = *(_QWORD *)(v195 + 72);
        v174 = v19[4];
        v21 = *(_QWORD *)(v168 + 80);
        v175 = v20;
        if ( !v21 || v195 != v21 - 24 )
        {
          v22 = *(_QWORD *)(v195 + 16);
          if ( !v22 )
            goto LABEL_57;
          while ( 1 )
          {
            v12 = (unsigned int)**(unsigned __int8 **)(v22 + 24) - 30;
            if ( (unsigned __int8)(**(_BYTE **)(v22 + 24) - 30) <= 0xAu )
              break;
            v22 = *(_QWORD *)(v22 + 8);
            if ( !v22 )
              goto LABEL_57;
          }
        }
        if ( v195 != sub_AA54C0(v195) )
        {
          src = 0;
          v222 = 0;
          v223 = 0;
          v23 = (*(_DWORD *)(v196 - 20) & 0x7FFFFFFu) >> 1;
          if ( v23 == 1 )
          {
            v166 = 0;
            goto LABEL_75;
          }
          v24 = 0;
          v25 = 0;
          v26 = 0;
          v27 = v23 - 1;
          while ( 1 )
          {
            v28 = 32;
            if ( (_DWORD)v26 != -2 )
              v28 = 32LL * (unsigned int)(2 * v26 + 3);
            v29 = *(_QWORD *)(v196 - 32);
            ++v26;
            v30 = *(_QWORD *)(v29 + v28);
            if ( *(_QWORD *)(v29 + 32) == v30 )
              goto LABEL_34;
            v31 = *(_QWORD *)(v29 + 32LL * (unsigned int)(2 * v26));
            if ( v223 != v24 )
            {
              if ( v24 )
              {
                v24->m128i_i64[0] = v31;
                v24->m128i_i64[1] = v31;
                v24[1].m128i_i64[0] = v30;
                v24 = v222;
              }
              v24 = (__m128i *)((char *)v24 + 24);
              v222 = v24;
              goto LABEL_33;
            }
            a6 = (__m128i *)src;
            v32 = (char *)v24 - (_BYTE *)src;
            v33 = 0xAAAAAAAAAAAAAAABLL * (((char *)v24 - (_BYTE *)src) >> 3);
            if ( v33 == 0x555555555555555LL )
              sub_4262D8((__int64)"vector::_M_realloc_insert");
            v34 = 1;
            if ( v33 )
              v34 = 0xAAAAAAAAAAAAAAABLL * (((char *)v24 - (_BYTE *)src) >> 3);
            v35 = __CFADD__(v34, v33);
            v36 = v34 - 0x5555555555555555LL * (((char *)v24 - (_BYTE *)src) >> 3);
            if ( v35 )
              break;
            if ( v36 )
            {
              if ( v36 > 0x555555555555555LL )
                v36 = 0x555555555555555LL;
              v160 = 24 * v36;
              goto LABEL_457;
            }
            v37 = 24;
            v38 = 0;
            a5 = 0;
LABEL_45:
            v39 = (__int64 *)((char *)a5->m128i_i64 + v32);
            if ( v39 )
            {
              *v39 = v31;
              v39[1] = v31;
              v39[2] = v30;
            }
            if ( a6 == v24 )
            {
              v24 = (__m128i *)v37;
            }
            else
            {
              v40 = a5;
              v41 = a6;
              do
              {
                if ( v40 )
                {
                  *v40 = _mm_loadu_si128(v41);
                  v40[1].m128i_i64[0] = v41[1].m128i_i64[0];
                }
                v41 = (const __m128i *)((char *)v41 + 24);
                v40 = (__m128i *)((char *)v40 + 24);
              }
              while ( v41 != v24 );
              v24 = (__m128i *)((char *)a5
                              + 24
                              * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)((char *)v24 - (char *)a6 - 24) >> 3))
                               & 0x1FFFFFFFFFFFFFFFLL)
                              + 48);
            }
            if ( a6 )
            {
              v193 = v38;
              v197 = a5;
              j_j___libc_free_0((unsigned __int64)a6);
              v38 = v193;
              a5 = v197;
            }
            src = a5;
            v222 = v24;
            v223 = v38;
LABEL_33:
            ++v25;
LABEL_34:
            if ( v27 == v26 )
            {
              v43 = (__m128i *)src;
              v166 = v25;
              v44 = v24;
              v45 = (char *)v24 - (_BYTE *)src;
              if ( src != v24 )
              {
                _BitScanReverse64(&v46, 0xAAAAAAAAAAAAAAABLL * (v45 >> 3));
                sub_2A315E0(
                  (__int64)src,
                  v24,
                  2LL * (int)(63 - (v46 ^ 0x3F)),
                  0xAAAAAAAAAAAAAAABLL,
                  (__int64)a5,
                  (__int64)a6);
                if ( v45 <= 384 )
                {
                  sub_2A30F60(v43, v24);
                }
                else
                {
                  v47 = v43 + 24;
                  sub_2A30F60(v43, (__m128i *)v43[24].m128i_i64);
                  if ( &v43[24] != v24 )
                  {
                    do
                    {
                      v48 = (__int64 *)v47;
                      v47 = (__m128i *)((char *)v47 + 24);
                      sub_2A30EE0(v48);
                    }
                    while ( v24 != v47 );
                  }
                }
                v44 = v222;
                v24 = (__m128i *)src;
                v45 = (char *)v222 - (_BYTE *)src;
              }
              if ( (unsigned __int64)v45 <= 0x18 )
                goto LABEL_75;
              v107 = (__m128i *)((char *)v24 + 24);
              if ( &v24[1].m128i_u64[1] != (unsigned __int64 *)v44 )
              {
                v172 = v44;
                v108 = (__m128i *)((char *)v24 + 24);
                while ( 1 )
                {
                  v109 = v24->m128i_i64[1];
                  v110 = v108->m128i_i64[0];
                  v198 = v108[1].m128i_i64[0];
                  v111 = v24[1].m128i_i64[0];
                  v237 = *(_DWORD *)(v109 + 32);
                  if ( v237 > 0x40 )
                    sub_C43780((__int64)&v236, (const void **)(v109 + 24));
                  else
                    v236 = *(const void **)(v109 + 24);
                  sub_C46A40((__int64)&v236, 1);
                  v112 = v237;
                  v113 = (unsigned __int64)v236;
                  v237 = 0;
                  v233 = v112;
                  v232 = (unsigned __int64)v236;
                  if ( *(_DWORD *)(v110 + 32) > 0x40u )
                    break;
                  v114 = 0;
                  if ( v236 == *(const void **)(v110 + 24) )
                    goto LABEL_228;
LABEL_229:
                  if ( v112 > 0x40 )
                  {
                    if ( v113 )
                    {
                      v199 = v114;
                      j_j___libc_free_0_0(v113);
                      v114 = v199;
                      if ( v237 > 0x40 )
                      {
                        if ( v236 )
                        {
                          j_j___libc_free_0_0((unsigned __int64)v236);
                          v114 = v199;
                        }
                      }
                    }
                  }
                  if ( v114 )
                  {
                    v24->m128i_i64[1] = v108->m128i_i64[1];
LABEL_223:
                    v108 = (const __m128i *)((char *)v108 + 24);
                    if ( v172 == v108 )
                      goto LABEL_237;
                  }
                  else
                  {
                    if ( v108 == v107 )
                    {
                      v24 = (__m128i *)v108;
                      v107 = (__m128i *)&v108[1].m128i_u64[1];
                      goto LABEL_223;
                    }
                    v115 = _mm_loadu_si128(v108);
                    v108 = (const __m128i *)((char *)v108 + 24);
                    *(__m128i *)((char *)v24 + 24) = v115;
                    v24[2].m128i_i64[1] = v108[-1].m128i_i64[1];
                    v24 = v107;
                    v107 = (__m128i *)((char *)v107 + 24);
                    if ( v172 == v108 )
                    {
LABEL_237:
                      v116 = v222;
                      v44 = v107;
                      goto LABEL_238;
                    }
                  }
                }
                v179 = (unsigned __int64)v236;
                v185 = v112;
                v114 = sub_C43C50(v110 + 24, (const void **)&v232);
                v112 = v185;
                v113 = v179;
                if ( !v114 )
                  goto LABEL_229;
LABEL_228:
                v114 = v198 == v111;
                goto LABEL_229;
              }
              v116 = v44;
LABEL_238:
              sub_2A30C20((__int64)&src, v44->m128i_i8, v116->m128i_i8);
LABEL_75:
              v49 = *(_DWORD *)(*(_QWORD *)(**(_QWORD **)(v196 - 32) + 8LL) + 8LL) >> 8;
              v171 = v49 + 1;
              v206 = v49 + 1;
              if ( v49 + 1 <= 0x40 )
              {
                v205 = 0;
                v208 = v49;
                v50 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v49;
                if ( !v49 )
                  v50 = 0;
LABEL_78:
                v207 = v50;
                if ( v222 != src )
                  goto LABEL_79;
LABEL_217:
                sub_B43C20((__int64)&v236, v195);
                v104 = (__int64)v236;
                v105 = v237;
                v106 = sub_BD2C40(72, 1u);
                if ( v106 )
                  sub_B4C8F0((__int64)v106, v174, 1u, v104, v105);
                sub_2A31020(v174, v195, v195, (__int64)&v207);
                sub_B43D60(v191);
LABEL_206:
                if ( v208 > 0x40 && v207 )
                  j_j___libc_free_0_0(v207);
                if ( v206 > 0x40 && v205 )
                  j_j___libc_free_0_0((unsigned __int64)v205);
                if ( src )
                  j_j___libc_free_0((unsigned __int64)src);
                v8 = v244;
                v6 = 1;
                goto LABEL_63;
              }
              sub_C43690((__int64)&v205, 0, 0);
              v208 = v49;
              if ( v49 == 64 )
              {
                v50 = -1;
                goto LABEL_78;
              }
              sub_C43690((__int64)&v207, -1, 1);
              if ( v222 == src )
                goto LABEL_217;
LABEL_79:
              v51 = sub_AA5030(v174, 1);
              if ( !v51 )
                goto LABEL_469;
              if ( *(_BYTE *)(v51 - 24) == 36 )
              {
                v164 = *(_QWORD *)src;
                v52 = v222[-1].m128i_i64[0];
                v224 = 0;
                v225 = 0;
                v163 = v52;
                v226 = 0;
                goto LABEL_82;
              }
              v84 = sub_B2BEC0(v168);
              sub_9AC3E0((__int64)&v224, v175, v84, 0, a3, (__int64)v191, 0, 1);
              sub_AAF050((__int64)&v228, (__int64)&v224, 0);
              sub_22CE1E0((__int64)&v232, a2, v175, (__int64)v191, 0);
              sub_AB2160((__int64)&v236, (__int64)&v228, (__int64)&v232, 0);
              v85 = (unsigned __int64 *)(*(_QWORD *)src + 24LL);
              v86 = (unsigned __int64 *)(v222[-1].m128i_i64[0] + 24);
              sub_AB14C0((__int64)&v219, (__int64)&v236);
              if ( (int)sub_C4C880((__int64)&v219, (__int64)v85) < 0 )
                v85 = &v219;
              v214 = *((_DWORD *)v85 + 2);
              if ( v214 > 0x40 )
                sub_C43780((__int64)&v213, (const void **)v85);
              else
                v213 = *v85;
              if ( v220 > 0x40 && v219 )
                j_j___libc_free_0_0(v219);
              sub_AB13A0((__int64)&v219, (__int64)&v236);
              if ( (int)sub_C4C880((__int64)&v219, (__int64)v86) > 0 )
                v86 = &v219;
              v216 = *((_DWORD *)v86 + 2);
              if ( v216 > 0x40 )
                sub_C43780((__int64)&v215, (const void **)v86);
              else
                v215 = (const void *)*v86;
              if ( v220 > 0x40 && v219 )
                j_j___libc_free_0_0(v219);
              v87 = (__int64 *)sub_BD5C60((__int64)v191);
              v164 = sub_ACCFD0(v87, (__int64)&v213);
              v88 = (__int64 *)sub_BD5C60((__int64)v191);
              v163 = sub_ACCFD0(v88, (__int64)&v215);
              v89 = (unsigned int)(v166 - 1);
              v218 = v214;
              if ( v214 > 0x40 )
              {
                sub_C43780((__int64)&v217, (const void **)&v213);
                v89 = (unsigned int)(v166 - 1);
              }
              else
              {
                v217 = (const void *)v213;
              }
              sub_C46A40((__int64)&v217, v89);
              v90 = v218;
              v91 = (unsigned __int64)v217;
              v218 = 0;
              v220 = v90;
              v219 = (unsigned __int64)v217;
              if ( v90 <= 0x40 )
              {
                v92 = v215 == v217;
              }
              else
              {
                v92 = sub_C43C50((__int64)&v219, &v215);
                if ( v91 )
                {
                  j_j___libc_free_0_0(v91);
                  if ( v218 > 0x40 )
                  {
                    if ( v217 )
                      j_j___libc_free_0_0((unsigned __int64)v217);
                  }
                }
              }
              if ( v216 > 0x40 && v215 )
                j_j___libc_free_0_0((unsigned __int64)v215);
              if ( v214 > 0x40 && v213 )
                j_j___libc_free_0_0(v213);
              if ( v239 > 0x40 && v238 )
                j_j___libc_free_0_0(v238);
              if ( v237 > 0x40 && v236 )
                j_j___libc_free_0_0((unsigned __int64)v236);
              if ( v235 > 0x40 && v234 )
                j_j___libc_free_0_0(v234);
              if ( v233 > 0x40 && v232 )
                j_j___libc_free_0_0(v232);
              if ( v231 > 0x40 && v230 )
                j_j___libc_free_0_0(v230);
              if ( (unsigned int)v229 > 0x40 && v228 )
                j_j___libc_free_0_0(v228);
              if ( v227 > 0x40 && v226 )
                j_j___libc_free_0_0(v226);
              if ( (unsigned int)v225 > 0x40 && v224 )
                j_j___libc_free_0_0(v224);
              v224 = 0;
              v225 = 0;
              v226 = 0;
              if ( !v92 )
              {
LABEL_189:
                v93 = sub_2A319A0((const __m128i *)src, v222, v164, v163, v175, v195, v195, v174, (__int64 *)&v224);
                if ( (_QWORD *)v174 != v93 )
                  sub_2A31020(v174, v195, 0, (__int64)&v207);
                sub_B43C20((__int64)&v236, v195);
                v94 = (__int64)v236;
                v95 = v237;
                v96 = sub_BD2C40(72, 1u);
                if ( v96 )
                  sub_B4C8F0((__int64)v96, (__int64)v93, 1u, v94, v95);
                v97 = *(_QWORD *)(*(_QWORD *)(v196 - 32) + 32LL);
                sub_B43D60(v191);
                v100 = *(_QWORD *)(v97 + 16);
                if ( !v100 )
                {
LABEL_245:
                  if ( v244 )
                  {
                    v117 = v241;
                    v99 = HIDWORD(v242);
                    v98 = (__int64)&v241[HIDWORD(v242)];
                    if ( v241 != (__int64 *)v98 )
                    {
                      while ( v97 != *v117 )
                      {
                        if ( (__int64 *)v98 == ++v117 )
                          goto LABEL_249;
                      }
                      goto LABEL_195;
                    }
LABEL_249:
                    if ( HIDWORD(v242) < (unsigned int)v242 )
                    {
                      ++HIDWORD(v242);
                      *(_QWORD *)v98 = v97;
                      ++v240;
                      goto LABEL_195;
                    }
                  }
                  sub_C8CC70((__int64)&v240, v97, v98, v99, (__int64)a5, (__int64)a6);
                  goto LABEL_195;
                }
                while ( 1 )
                {
                  v98 = (unsigned int)**(unsigned __int8 **)(v100 + 24) - 30;
                  if ( (unsigned __int8)(**(_BYTE **)(v100 + 24) - 30) <= 0xAu )
                    break;
                  v100 = *(_QWORD *)(v100 + 8);
                  if ( !v100 )
                    goto LABEL_245;
                }
LABEL_195:
                v101 = v225;
                v102 = v224;
                if ( v225 != v224 )
                {
                  do
                  {
                    if ( *(_DWORD *)(v102 + 24) > 0x40u )
                    {
                      v103 = *(_QWORD *)(v102 + 16);
                      if ( v103 )
                        j_j___libc_free_0_0(v103);
                    }
                    if ( *(_DWORD *)(v102 + 8) > 0x40u && *(_QWORD *)v102 )
                      j_j___libc_free_0_0(*(_QWORD *)v102);
                    v102 += 32LL;
                  }
                  while ( v101 != v102 );
                  goto LABEL_203;
                }
                goto LABEL_204;
              }
LABEL_82:
              v228 = 0;
              v229 = 0;
              v230 = 0;
              v231 = 0;
              v210 = v206;
              if ( v206 > 0x40 )
                sub_C43780((__int64)&v209, &v205);
              else
                v209 = (unsigned __int64)v205;
              v53 = v49 - 1;
              v212 = v49;
              v54 = 1LL << ((unsigned __int8)v49 - 1);
              v55 = ~v54;
              if ( v49 <= 0x40 )
              {
                v214 = v49;
                v213 = 0;
                v56 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v49;
                if ( !v49 )
                  v56 = 0;
                v211 = v55 & v56;
LABEL_88:
                v213 |= v54;
                v233 = v49;
                goto LABEL_89;
              }
              sub_C43690((__int64)&v211, -1, 1);
              if ( v212 <= 0x40 )
                v211 &= v55;
              else
                *(_QWORD *)(v211 + 8LL * (v53 >> 6)) &= v55;
              v214 = v49;
              sub_C43690((__int64)&v213, 0, 0);
              v49 = v214;
              if ( v214 <= 0x40 )
                goto LABEL_88;
              *(_QWORD *)(v213 + 8LL * (v53 >> 6)) |= v54;
              v233 = v214;
              if ( v214 <= 0x40 )
LABEL_89:
                v232 = v213;
              else
                sub_C43780((__int64)&v232, (const void **)&v213);
              v235 = v212;
              if ( v212 > 0x40 )
              {
                sub_C43780((__int64)&v234, (const void **)&v211);
                v57 = v225;
                if ( v225 == v226 )
                  goto LABEL_359;
LABEL_92:
                if ( v57 )
                {
                  v58 = v233;
                  *(_DWORD *)(v57 + 8) = v233;
                  if ( v58 > 0x40 )
                    sub_C43780(v57, (const void **)&v232);
                  else
                    *(_QWORD *)v57 = v232;
                  v59 = v235;
                  *(_DWORD *)(v57 + 24) = v235;
                  if ( v59 > 0x40 )
                    sub_C43780(v57 + 16, (const void **)&v234);
                  else
                    *(_QWORD *)(v57 + 16) = v234;
                  v57 = v225;
                }
                v225 = v57 + 32;
              }
              else
              {
                v57 = v225;
                v234 = v211;
                if ( v225 != v226 )
                  goto LABEL_92;
LABEL_359:
                sub_2A312E0(&v224, v57, (__int64)&v232);
              }
              v169 = 0;
              v170 = v222;
              if ( src != v222 )
              {
                for ( i = (const __m128i *)src; v170 != i; i = (const __m128i *)((char *)i + 24) )
                {
                  v74 = v225;
                  v75 = i->m128i_i64[0];
                  v76 = i->m128i_i64[1];
                  v77 = (const void **)(i->m128i_i64[0] + 24);
                  v78 = (_DWORD *)(v76 + 24);
                  if ( *(_DWORD *)(v225 - 24) <= 0x40u )
                  {
                    v80 = v225 - 32;
                    if ( *(_QWORD *)(v225 - 32) == *(_QWORD *)(v75 + 24) )
                    {
LABEL_127:
                      v225 = v80;
                      if ( *(_DWORD *)(v74 - 8) > 0x40u )
                      {
                        v81 = *(_QWORD *)(v74 - 16);
                        if ( v81 )
                        {
                          v183 = v74;
                          j_j___libc_free_0_0(v81);
                          v74 = v183;
                        }
                      }
                      if ( *(_DWORD *)(v74 - 24) > 0x40u )
                      {
                        v82 = *(_QWORD *)(v74 - 32);
                        if ( v82 )
LABEL_132:
                          j_j___libc_free_0_0(v82);
                      }
LABEL_133:
                      v83 = *(_DWORD *)(v76 + 32);
                      if ( v83 > 0x40 )
                        goto LABEL_134;
                      goto LABEL_271;
                    }
                  }
                  else
                  {
                    v176 = i->m128i_i64[0];
                    v178 = v225;
                    v182 = v225 - 32;
                    v79 = sub_C43C50(v225 - 32, (const void **)(i->m128i_i64[0] + 24));
                    v80 = v182;
                    v74 = v178;
                    v75 = v176;
                    if ( v79 )
                      goto LABEL_127;
                  }
                  v237 = *(_DWORD *)(v75 + 32);
                  if ( v237 > 0x40 )
                  {
                    v187 = v74;
                    sub_C43780((__int64)&v236, v77);
                    v74 = v187;
                  }
                  else
                  {
                    v236 = *(const void **)(v75 + 24);
                  }
                  v186 = v74;
                  sub_C46F20((__int64)&v236, 1u);
                  v119 = v186;
                  v120 = v237;
                  v237 = 0;
                  v121 = v236;
                  if ( *(_DWORD *)(v186 - 8) > 0x40u )
                  {
                    v122 = *(_QWORD *)(v186 - 16);
                    if ( v122 )
                    {
                      v177 = v236;
                      v180 = v120;
                      j_j___libc_free_0_0(v122);
                      v121 = v177;
                      v120 = v180;
                      v119 = v186;
                    }
                  }
                  *(_QWORD *)(v119 - 16) = v121;
                  *(_DWORD *)(v119 - 8) = v120;
                  if ( v237 <= 0x40 )
                    goto LABEL_133;
                  v82 = (unsigned __int64)v236;
                  if ( v236 )
                    goto LABEL_132;
                  v83 = *(_DWORD *)(v76 + 32);
                  if ( v83 > 0x40 )
                  {
LABEL_134:
                    v184 = v83;
                    if ( sub_C43C50(v76 + 24, (const void **)&v211) )
                      goto LABEL_135;
                    v220 = v184;
                    sub_C43780((__int64)&v219, (const void **)(v76 + 24));
                    goto LABEL_273;
                  }
LABEL_271:
                  v123 = *(const void **)(v76 + 24);
                  if ( v123 == (const void *)v211 )
                    goto LABEL_135;
                  v220 = v83;
                  v219 = (unsigned __int64)v123;
LABEL_273:
                  sub_C46A40((__int64)&v219, 1);
                  v237 = v220;
                  v236 = (const void *)v219;
                  v239 = v212;
                  if ( v212 > 0x40 )
                  {
                    sub_C43780((__int64)&v238, (const void **)&v211);
                    v124 = v225;
                    if ( v225 != v226 )
                    {
LABEL_275:
                      if ( v124 )
                      {
                        v125 = v237;
                        *(_DWORD *)(v124 + 8) = v237;
                        if ( v125 > 0x40 )
                          sub_C43780(v124, &v236);
                        else
                          *(_QWORD *)v124 = v236;
                        v126 = v239;
                        *(_DWORD *)(v124 + 24) = v239;
                        if ( v126 > 0x40 )
                          sub_C43780(v124 + 16, (const void **)&v238);
                        else
                          *(_QWORD *)(v124 + 16) = v238;
                        v124 = v225;
                      }
                      v225 = v124 + 32;
                      goto LABEL_282;
                    }
                  }
                  else
                  {
                    v124 = v225;
                    v238 = v211;
                    if ( v225 != v226 )
                      goto LABEL_275;
                  }
                  sub_2A312E0(&v224, v124, (__int64)&v236);
LABEL_282:
                  if ( v239 > 0x40 && v238 )
                    j_j___libc_free_0_0(v238);
                  if ( v237 > 0x40 && v236 )
                    j_j___libc_free_0_0((unsigned __int64)v236);
LABEL_135:
                  sub_C44830((__int64)&v219, v77, v171);
                  sub_C44830((__int64)&v217, v78, v171);
                  if ( v220 <= 0x40 )
                  {
                    v61 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v220) & ~v219;
                    if ( !v220 )
                      v61 = 0;
                    v219 = v61;
                  }
                  else
                  {
                    sub_C43D10((__int64)&v219);
                  }
                  sub_C46250((__int64)&v219);
                  sub_C45EE0((__int64)&v219, (__int64 *)&v217);
                  v62 = v220;
                  v220 = 0;
                  v237 = v62;
                  v236 = (const void *)v219;
                  sub_C46A40((__int64)&v236, 1);
                  v216 = v237;
                  v215 = v236;
                  if ( v218 > 0x40 && v217 )
                    j_j___libc_free_0_0((unsigned __int64)v217);
                  if ( v220 > 0x40 && v219 )
                    j_j___libc_free_0_0(v219);
                  v63 = v206;
                  v237 = v206;
                  if ( v206 > 0x40 )
                  {
                    sub_C43780((__int64)&v236, &v205);
                    v63 = v237;
                    v64 = (unsigned __int64)v236;
                  }
                  else
                  {
                    v64 = (unsigned __int64)v205;
                    v236 = v205;
                  }
                  v65 = i[1].m128i_i64[0];
                  v237 = 0;
                  if ( !v231 )
                  {
                    ++v228;
                    goto LABEL_370;
                  }
                  v66 = 1;
                  v67 = 0;
                  v68 = (v231 - 1) & (((unsigned int)v65 >> 4) ^ ((unsigned int)v65 >> 9));
                  v69 = (__int64 *)(v229 + 24LL * v68);
                  v70 = *v69;
                  if ( v65 != *v69 )
                  {
                    while ( v70 != -4096 )
                    {
                      if ( v70 == -8192 && !v67 )
                        v67 = v69;
                      v68 = (v231 - 1) & (v66 + v68);
                      v69 = (__int64 *)(v229 + 24LL * v68);
                      v70 = *v69;
                      if ( v65 == *v69 )
                        goto LABEL_114;
                      ++v66;
                    }
                    if ( v67 )
                      v69 = v67;
                    ++v228;
                    v138 = v230 + 1;
                    if ( 4 * ((int)v230 + 1) >= 3 * v231 )
                    {
LABEL_370:
                      v188 = v64;
                      sub_2A33350((__int64)&v228, 2 * v231);
                      if ( !v231 )
                        goto LABEL_468;
                      v64 = v188;
                      LODWORD(v139) = (v231 - 1) & (((unsigned int)v65 >> 9) ^ ((unsigned int)v65 >> 4));
                      v69 = (__int64 *)(v229 + 24LL * (unsigned int)v139);
                      v138 = v230 + 1;
                      v140 = *v69;
                      if ( v65 != *v69 )
                      {
                        v141 = 1;
                        v142 = 0;
                        while ( v140 != -4096 )
                        {
                          if ( v140 == -8192 && !v142 )
                            v142 = v69;
                          v139 = (v231 - 1) & ((_DWORD)v139 + v141);
                          v69 = (__int64 *)(v229 + 24 * v139);
                          v140 = *v69;
                          if ( v65 == *v69 )
                            goto LABEL_353;
                          ++v141;
                        }
LABEL_382:
                        if ( v142 )
                          v69 = v142;
                      }
                    }
                    else if ( v231 - HIDWORD(v230) - v138 <= v231 >> 3 )
                    {
                      v189 = v64;
                      sub_2A33350((__int64)&v228, v231);
                      if ( !v231 )
                      {
LABEL_468:
                        LODWORD(v230) = v230 + 1;
                        BUG();
                      }
                      v142 = 0;
                      v64 = v189;
                      v143 = 1;
                      v144 = (v231 - 1) & (((unsigned int)v65 >> 4) ^ ((unsigned int)v65 >> 9));
                      v69 = (__int64 *)(v229 + 24LL * v144);
                      v138 = v230 + 1;
                      v145 = *v69;
                      if ( v65 != *v69 )
                      {
                        while ( v145 != -4096 )
                        {
                          if ( v145 == -8192 && !v142 )
                            v142 = v69;
                          v144 = (v231 - 1) & (v143 + v144);
                          v69 = (__int64 *)(v229 + 24LL * v144);
                          v145 = *v69;
                          if ( v65 == *v69 )
                            goto LABEL_353;
                          ++v143;
                        }
                        goto LABEL_382;
                      }
                    }
LABEL_353:
                    LODWORD(v230) = v138;
                    if ( *v69 != -4096 )
                      --HIDWORD(v230);
                    *v69 = v65;
                    v71 = (__int64)(v69 + 1);
                    *((_DWORD *)v69 + 4) = v63;
                    v69[1] = v64;
                    v72 = v237;
LABEL_117:
                    if ( v72 > 0x40 && v236 )
                      j_j___libc_free_0_0((unsigned __int64)v236);
                    goto LABEL_120;
                  }
LABEL_114:
                  v71 = (__int64)(v69 + 1);
                  if ( v63 > 0x40 && v64 )
                  {
                    j_j___libc_free_0_0(v64);
                    v72 = v237;
                    goto LABEL_117;
                  }
LABEL_120:
                  v73 = sub_C45EE0(v71, (__int64 *)&v215);
                  if ( (int)sub_C49970(v73, &v209) > 0 )
                  {
                    if ( v210 <= 0x40 && *((_DWORD *)v69 + 4) <= 0x40u )
                    {
                      v137 = v69[1];
                      v210 = *((_DWORD *)v69 + 4);
                      v209 = v137;
                    }
                    else
                    {
                      sub_C43990((__int64)&v209, v71);
                    }
                    v169 = i[1].m128i_i64[0];
                  }
                  if ( v216 > 0x40 && v215 )
                    j_j___libc_free_0_0((unsigned __int64)v215);
                }
              }
              v127 = 0;
              v128 = ((*(_DWORD *)(v196 - 20) & 0x7FFFFFFu) >> 1) - v166;
              if ( v128 )
              {
                do
                {
                  ++v127;
                  sub_AA5980(v174, v195, 0);
                }
                while ( v128 != v127 );
              }
              v129 = v222;
              v130 = (__m128i *)src;
              v131 = 0xAAAAAAAAAAAAAAABLL * (((char *)v222 - (_BYTE *)src) >> 3);
              if ( v131 >> 2 > 0 )
              {
                while ( v130[1].m128i_i64[0] != v169 )
                {
                  if ( v130[2].m128i_i64[1] == v169 )
                  {
                    v130 = (__m128i *)((char *)v130 + 24);
                    break;
                  }
                  if ( v130[4].m128i_i64[0] == v169 )
                  {
                    v130 += 3;
                    break;
                  }
                  if ( v130[5].m128i_i64[1] == v169 )
                  {
                    v130 = (__m128i *)((char *)v130 + 72);
                    break;
                  }
                  v130 += 6;
                  if ( v130 == (__m128i *)((char *)src + 96 * (v131 >> 2)) )
                  {
                    v131 = 0xAAAAAAAAAAAAAAABLL * (((char *)v222 - (char *)v130) >> 3);
                    goto LABEL_386;
                  }
                }
LABEL_303:
                if ( v222 != v130 )
                {
                  v132 = (__m128i *)((char *)v130 + 24);
                  if ( v222 != (__m128i *)&v130[1].m128i_u64[1] )
                  {
                    do
                    {
                      if ( v132[1].m128i_i64[0] != v169 )
                      {
                        v130 = (__m128i *)((char *)v130 + 24);
                        *(__m128i *)((char *)v130 - 24) = _mm_loadu_si128(v132);
                        v130[-1].m128i_i64[1] = v132[1].m128i_i64[0];
                      }
                      v132 = (const __m128i *)((char *)v132 + 24);
                    }
                    while ( v129 != v132 );
                  }
                }
LABEL_308:
                sub_2A30C20((__int64)&src, v130->m128i_i8, v129->m128i_i8);
                if ( v222 != src )
                {
                  v175 = **(_QWORD **)(v196 - 32);
                  if ( v235 > 0x40 && v234 )
                    j_j___libc_free_0_0(v234);
                  if ( v233 > 0x40 && v232 )
                    j_j___libc_free_0_0(v232);
                  if ( v214 > 0x40 && v213 )
                    j_j___libc_free_0_0(v213);
                  if ( v212 > 0x40 && v211 )
                    j_j___libc_free_0_0(v211);
                  if ( v210 > 0x40 && v209 )
                    j_j___libc_free_0_0(v209);
                  v133 = v231;
                  if ( v231 )
                  {
                    v134 = v229;
                    v135 = v229 + 24LL * v231;
                    do
                    {
                      if ( *(_QWORD *)v134 != -4096 && *(_QWORD *)v134 != -8192 && *(_DWORD *)(v134 + 16) > 0x40u )
                      {
                        v136 = *(_QWORD *)(v134 + 8);
                        if ( v136 )
                          j_j___libc_free_0_0(v136);
                      }
                      v134 += 24;
                    }
                    while ( v135 != v134 );
                    v133 = v231;
                  }
                  sub_C7D6A0(v229, 24 * v133, 8);
                  v174 = v169;
                  goto LABEL_189;
                }
                sub_B43C20((__int64)&v236, v195);
                v146 = (__int64)v236;
                v147 = v237;
                v148 = sub_BD2C40(72, 1u);
                if ( v148 )
                  sub_B4C8F0((__int64)v148, v169, 1u, v146, v147);
                sub_B43D60(v191);
                v149 = v210;
                if ( v210 <= 0x40 )
                  v150 = v209 == 0;
                else
                  v150 = v149 == (unsigned int)sub_C444A0((__int64)&v209);
                if ( !v150 )
                {
                  v218 = v206;
                  if ( v206 > 0x40 )
                  {
                    sub_C43780((__int64)&v217, &v205);
                    v149 = v210;
                  }
                  else
                  {
                    v217 = v205;
                  }
                  while ( 1 )
                  {
                    v220 = v149;
                    if ( v149 <= 0x40 )
                      v219 = v209;
                    else
                      sub_C43780((__int64)&v219, (const void **)&v209);
                    sub_C46F20((__int64)&v219, 1u);
                    v157 = v220;
                    v158 = v219;
                    v220 = 0;
                    v237 = v157;
                    v236 = (const void *)v219;
                    v159 = sub_C49970((__int64)&v217, (unsigned __int64 *)&v236);
                    if ( v157 > 0x40 )
                    {
                      if ( v158 )
                      {
                        j_j___libc_free_0_0(v158);
                        if ( v220 > 0x40 )
                        {
                          if ( v219 )
                            j_j___libc_free_0_0(v219);
                        }
                      }
                    }
                    if ( v159 >= 0 )
                      break;
                    sub_AA5980(v169, v195, 0);
                    sub_C46250((__int64)&v217);
                    v149 = v210;
                  }
                  if ( v218 > 0x40 && v217 )
                    j_j___libc_free_0_0((unsigned __int64)v217);
                }
                if ( v235 > 0x40 && v234 )
                  j_j___libc_free_0_0(v234);
                if ( v233 > 0x40 && v232 )
                  j_j___libc_free_0_0(v232);
                if ( v214 > 0x40 && v213 )
                  j_j___libc_free_0_0(v213);
                if ( v212 > 0x40 && v211 )
                  j_j___libc_free_0_0(v211);
                if ( v210 > 0x40 && v209 )
                  j_j___libc_free_0_0(v209);
                v151 = v231;
                if ( v231 )
                {
                  v152 = v229;
                  v153 = v229 + 24LL * v231;
                  do
                  {
                    if ( *(_QWORD *)v152 != -8192 && *(_QWORD *)v152 != -4096 && *(_DWORD *)(v152 + 16) > 0x40u )
                    {
                      v154 = *(_QWORD *)(v152 + 8);
                      if ( v154 )
                        j_j___libc_free_0_0(v154);
                    }
                    v152 += 24;
                  }
                  while ( v153 != v152 );
                  v151 = v231;
                }
                sub_C7D6A0(v229, 24 * v151, 8);
                v155 = v225;
                v102 = v224;
                if ( v225 != v224 )
                {
                  do
                  {
                    if ( *(_DWORD *)(v102 + 24) > 0x40u )
                    {
                      v156 = *(_QWORD *)(v102 + 16);
                      if ( v156 )
                        j_j___libc_free_0_0(v156);
                    }
                    if ( *(_DWORD *)(v102 + 8) > 0x40u && *(_QWORD *)v102 )
                      j_j___libc_free_0_0(*(_QWORD *)v102);
                    v102 += 32LL;
                  }
                  while ( v155 != v102 );
LABEL_203:
                  v102 = v224;
                }
LABEL_204:
                if ( v102 )
                  j_j___libc_free_0(v102);
                goto LABEL_206;
              }
LABEL_386:
              if ( v131 != 2 )
              {
                if ( v131 != 3 )
                {
                  if ( v131 != 1 )
                  {
LABEL_389:
                    v130 = v222;
                    goto LABEL_308;
                  }
LABEL_398:
                  if ( v130[1].m128i_i64[0] == v169 )
                    goto LABEL_303;
                  goto LABEL_389;
                }
                if ( v130[1].m128i_i64[0] == v169 )
                  goto LABEL_303;
                v130 = (__m128i *)((char *)v130 + 24);
              }
              if ( v130[1].m128i_i64[0] == v169 )
                goto LABEL_303;
              v130 = (__m128i *)((char *)v130 + 24);
              goto LABEL_398;
            }
          }
          v160 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_457:
          v173 = (char *)v24 - (_BYTE *)src;
          v181 = (__m128i *)src;
          v190 = v30;
          v194 = v31;
          v200 = v160;
          v161 = sub_22077B0(v160);
          v31 = v194;
          a5 = (__m128i *)v161;
          v30 = v190;
          a6 = v181;
          v38 = (__m128i *)(v161 + v200);
          v32 = v173;
          v37 = v161 + 24;
          goto LABEL_45;
        }
LABEL_57:
        v8 = v244;
        if ( !v244 )
          goto LABEL_240;
        v42 = v241;
        v12 = (__int64)&v241[HIDWORD(v242)];
        if ( v241 != (__int64 *)v12 )
        {
          while ( v195 != *v42 )
          {
            if ( (__int64 *)v12 == ++v42 )
              goto LABEL_239;
          }
          v6 = 1;
          goto LABEL_63;
        }
LABEL_239:
        if ( HIDWORD(v242) < (unsigned int)v242 )
        {
          v6 = 1;
          ++HIDWORD(v242);
          *(_QWORD *)v12 = v195;
          v8 = v244;
          ++v240;
        }
        else
        {
LABEL_240:
          v6 = 1;
          sub_C8CC70((__int64)&v240, v195, v12, v244, (__int64)a5, (__int64)a6);
          v8 = v244;
        }
LABEL_63:
        if ( v192 == v201 )
          goto LABEL_8;
      }
    }
    while ( v10 != *v11 )
    {
      if ( (__int64 *)v12 == ++v11 )
        goto LABEL_17;
    }
  }
  while ( v192 != v201 );
LABEL_8:
  v13 = v241;
  if ( v8 )
    v14 = &v241[HIDWORD(v242)];
  else
    v14 = &v241[(unsigned int)v242];
  if ( v241 != v14 )
  {
    while ( 1 )
    {
      v15 = *v13;
      v16 = v13;
      if ( (unsigned __int64)*v13 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v14 == ++v13 )
        goto LABEL_13;
    }
    if ( v13 != v14 )
    {
      do
      {
        sub_22C2BE0((__int64)a2, v15);
        sub_F34560(v15, 0, 0);
        v118 = v16 + 1;
        if ( v16 + 1 == v14 )
          break;
        v15 = *v118;
        for ( ++v16; (unsigned __int64)*v118 >= 0xFFFFFFFFFFFFFFFELL; v16 = v118 )
        {
          if ( v14 == ++v118 )
            goto LABEL_13;
          v15 = *v118;
        }
      }
      while ( v16 != v14 );
    }
  }
LABEL_13:
  if ( !v244 )
    _libc_free((unsigned __int64)v241);
  return v6;
}
