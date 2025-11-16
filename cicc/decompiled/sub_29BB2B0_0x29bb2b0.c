// Function: sub_29BB2B0
// Address: 0x29bb2b0
//
_QWORD *__fastcall sub_29BB2B0(
        _QWORD *a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        const __m128i *a7,
        unsigned __int64 a8)
{
  __int128 v8; // rax
  unsigned __int64 *v9; // r12
  unsigned __int64 v10; // rdx
  unsigned __int64 *v11; // r12
  unsigned __int64 *v12; // rsi
  unsigned __int64 v13; // rcx
  size_t v14; // r13
  char *v15; // rax
  const __m128i *v16; // rbx
  const __m128i *v17; // r14
  __m128i v18; // xmm5
  __int64 v19; // rax
  __int64 v20; // rdi
  _BYTE *v21; // rsi
  __int64 v22; // rdi
  void **v23; // rsi
  __int64 *v24; // r12
  __int64 *v25; // r13
  __int64 v26; // rdx
  _BYTE *v27; // rsi
  _BYTE *v28; // rsi
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // rcx
  const __m128i *v31; // rcx
  const __m128i *i; // rax
  void *v33; // r8
  unsigned __int64 v34; // r13
  signed __int64 v35; // rbx
  __int64 v36; // rax
  char *v37; // r15
  signed __int64 v38; // rdx
  __int64 *v39; // r12
  bool v40; // zf
  char *v41; // rdx
  __int64 v42; // r15
  void **v43; // r14
  _QWORD **v44; // rbx
  _QWORD **j; // r15
  _QWORD *v46; // rdx
  _QWORD *v47; // rsi
  __int64 v48; // r12
  _QWORD *v49; // rax
  _QWORD *v50; // rsi
  __int64 v51; // rax
  _BYTE *v52; // rsi
  __int64 *v53; // rbx
  __int64 *v54; // rsi
  unsigned __int64 *v55; // rdi
  unsigned __int64 *v56; // r8
  __int64 *v57; // rax
  unsigned __int64 *v58; // rdx
  __int64 v59; // rcx
  __int64 *v60; // rsi
  __int64 *v61; // rdx
  __int64 *v62; // rax
  _QWORD *v63; // rcx
  __int64 *k; // r13
  __int64 v65; // r12
  __int64 v66; // r15
  double v67; // xmm0_8
  __int64 v68; // rax
  double v69; // xmm1_8
  double v70; // xmm2_8
  double v71; // xmm0_8
  double v72; // rax
  __int64 v73; // rax
  double v74; // xmm0_8
  __int64 v75; // r12
  __int64 v76; // rbx
  __int64 v77; // rax
  __int64 v78; // rdx
  double v79; // xmm0_8
  _QWORD *v80; // rax
  _QWORD *v81; // rdx
  __int64 v82; // rax
  __int64 v83; // rax
  __int64 **v84; // r13
  __int64 v85; // rdx
  __int64 v86; // r11
  __int64 **v87; // r12
  __int64 v88; // rax
  __int64 v89; // r14
  _QWORD *v90; // rax
  __int64 v91; // r11
  unsigned __int64 v92; // r10
  __int64 v93; // rdx
  __int64 v94; // rcx
  __int64 v95; // rax
  __int64 v96; // r13
  __int64 v97; // rdx
  __int64 v98; // r11
  __int64 v99; // r12
  __int64 v100; // rax
  __int64 v101; // rbx
  _QWORD *v102; // rax
  __int64 v103; // r11
  unsigned __int64 v104; // r10
  __int64 v105; // rdx
  __int64 v106; // rcx
  __int64 v107; // rcx
  __int64 v108; // rdx
  unsigned __int64 v109; // r12
  __int64 v110; // rsi
  __int64 v111; // rax
  __int64 v112; // rsi
  unsigned __int64 v113; // r14
  __int64 v114; // rdx
  __int64 v115; // r8
  int *v116; // r14
  int v117; // r9d
  int *v118; // rbx
  __int64 v119; // r8
  int *v120; // r14
  int v121; // r9d
  int *v122; // rbx
  unsigned __int64 *v123; // rax
  __m128i *v124; // rsi
  unsigned __int64 *v125; // rax
  __m128i *v126; // rsi
  unsigned __int64 v127; // rax
  int *v128; // r13
  __int64 v129; // r14
  int *v130; // r15
  __int64 v131; // r13
  unsigned __int64 v132; // r9
  __int64 v133; // r14
  unsigned __int64 *v134; // rdx
  unsigned __int64 v135; // rax
  __int64 v136; // r15
  __int64 v137; // rbx
  __int64 v138; // r12
  __int64 v139; // rdx
  __int64 v140; // rsi
  __int64 v141; // r10
  __int64 **v142; // rax
  __int64 *v143; // rdi
  __int64 v144; // rcx
  __int64 v145; // r11
  char v146; // dl
  __int64 v147; // r13
  __int64 v148; // rbx
  char *v149; // rsi
  char *v150; // r13
  char *v151; // r12
  __int64 v152; // rbx
  unsigned __int64 v153; // rax
  char *v154; // r9
  __int64 v155; // rsi
  char *n; // rax
  __int64 v157; // rdx
  __int64 v158; // rcx
  __int64 v159; // rcx
  double v160; // xmm0_8
  __int64 v161; // rcx
  double v162; // xmm1_8
  double v163; // xmm2_8
  double v164; // xmm0_8
  unsigned __int64 v165; // rsi
  char *v166; // r13
  char *v167; // r14
  _QWORD **v168; // r15
  _QWORD **ii; // rbx
  _BYTE *v170; // rsi
  _QWORD *v171; // rdx
  __int64 v172; // rbx
  unsigned __int64 v173; // r12
  unsigned __int64 v174; // rdi
  __int64 v175; // rbx
  unsigned __int64 v176; // r12
  unsigned __int64 v177; // rdi
  unsigned __int64 v178; // rdi
  __int64 *v179; // rbx
  unsigned __int64 v180; // r12
  unsigned __int64 v181; // rdi
  unsigned __int64 v182; // rdi
  unsigned __int64 *v183; // rbx
  unsigned __int64 *v184; // r12
  unsigned __int64 *v185; // rbx
  unsigned __int64 *v186; // r12
  __int64 v188; // rcx
  __int64 **v189; // rcx
  char v190; // r8
  __int64 **v191; // rax
  unsigned __int64 *v192; // r15
  unsigned __int64 *v193; // r13
  unsigned __int64 *v194; // r15
  unsigned __int64 *v195; // r13
  __int64 v196; // rdi
  __int64 v197; // rdi
  void *v198; // r9
  signed __int64 v199; // rdx
  __int64 v200; // rax
  __int64 v201; // rsi
  bool v202; // cf
  unsigned __int64 v203; // rax
  unsigned __int64 v204; // rbx
  char *v205; // r10
  __int64 v206; // r15
  char *v207; // rax
  unsigned __int64 v208; // rbx
  __int64 v209; // rax
  unsigned __int64 v211; // [rsp+10h] [rbp-1B0h]
  unsigned __int64 v212; // [rsp+10h] [rbp-1B0h]
  int v213; // [rsp+1Ch] [rbp-1A4h]
  __int64 v214; // [rsp+20h] [rbp-1A0h]
  __int64 v215; // [rsp+28h] [rbp-198h]
  __int64 v216; // [rsp+28h] [rbp-198h]
  __int64 v217; // [rsp+28h] [rbp-198h]
  __int64 v218; // [rsp+30h] [rbp-190h]
  __int64 v219; // [rsp+30h] [rbp-190h]
  __int64 v220; // [rsp+30h] [rbp-190h]
  __int64 v221; // [rsp+30h] [rbp-190h]
  unsigned __int64 v222; // [rsp+30h] [rbp-190h]
  char *v223; // [rsp+38h] [rbp-188h]
  _QWORD *v224; // [rsp+40h] [rbp-180h]
  _QWORD *v225; // [rsp+48h] [rbp-178h]
  __int64 *v226; // [rsp+50h] [rbp-170h]
  __int64 v227; // [rsp+60h] [rbp-160h]
  __int64 v228; // [rsp+60h] [rbp-160h]
  __int64 v229; // [rsp+60h] [rbp-160h]
  __int64 v230; // [rsp+60h] [rbp-160h]
  __int64 v231; // [rsp+60h] [rbp-160h]
  __int64 v232; // [rsp+60h] [rbp-160h]
  int *v233; // [rsp+60h] [rbp-160h]
  double v234; // [rsp+68h] [rbp-158h]
  __int64 *v235; // [rsp+78h] [rbp-148h]
  signed __int64 v236; // [rsp+78h] [rbp-148h]
  double v237; // [rsp+80h] [rbp-140h]
  __int64 v238; // [rsp+80h] [rbp-140h]
  __int64 v239; // [rsp+80h] [rbp-140h]
  __int64 *v240; // [rsp+88h] [rbp-138h]
  __int64 *m; // [rsp+88h] [rbp-138h]
  void *v242; // [rsp+88h] [rbp-138h]
  char *v243; // [rsp+88h] [rbp-138h]
  void *v244; // [rsp+88h] [rbp-138h]
  unsigned __int64 v246; // [rsp+90h] [rbp-130h]
  void *v247; // [rsp+90h] [rbp-130h]
  unsigned __int64 v248; // [rsp+90h] [rbp-130h]
  __int64 *v249; // [rsp+98h] [rbp-128h]
  int v250; // [rsp+98h] [rbp-128h]
  void *v251; // [rsp+98h] [rbp-128h]
  __int64 v252; // [rsp+A0h] [rbp-120h] BYREF
  int v253; // [rsp+A8h] [rbp-118h]
  __int64 v254[2]; // [rsp+B0h] [rbp-110h] BYREF
  void *v255[2]; // [rsp+C0h] [rbp-100h] BYREF
  unsigned __int64 v256; // [rsp+D0h] [rbp-F0h] BYREF
  unsigned __int64 v257; // [rsp+E0h] [rbp-E0h] BYREF
  unsigned __int64 *v258; // [rsp+E8h] [rbp-D8h] BYREF
  unsigned __int64 *v259; // [rsp+F0h] [rbp-D0h]
  __int64 v260; // [rsp+F8h] [rbp-C8h]
  unsigned __int64 *v261; // [rsp+100h] [rbp-C0h] BYREF
  unsigned __int64 *v262; // [rsp+108h] [rbp-B8h]
  __int64 v263; // [rsp+110h] [rbp-B0h]
  __int64 *v264; // [rsp+118h] [rbp-A8h] BYREF
  __int64 *v265; // [rsp+120h] [rbp-A0h]
  __int64 v266; // [rsp+128h] [rbp-98h]
  const __m128i *v267; // [rsp+130h] [rbp-90h] BYREF
  const __m128i *v268; // [rsp+138h] [rbp-88h]
  __int64 v269; // [rsp+140h] [rbp-80h]
  unsigned __int64 v270; // [rsp+148h] [rbp-78h] BYREF
  __int64 v271; // [rsp+150h] [rbp-70h]
  __int64 v272; // [rsp+158h] [rbp-68h]
  unsigned __int64 v273; // [rsp+160h] [rbp-60h] BYREF
  __int64 v274; // [rsp+168h] [rbp-58h]
  __int64 v275; // [rsp+170h] [rbp-50h]
  void *src; // [rsp+178h] [rbp-48h]
  char *v277; // [rsp+180h] [rbp-40h]
  char *v278; // [rsp+188h] [rbp-38h]

  v257 = a3;
  v258 = 0;
  v259 = 0;
  v260 = 0;
  v261 = 0;
  v262 = 0;
  v263 = 0;
  v264 = 0;
  v265 = 0;
  v266 = 0;
  v267 = 0;
  v268 = 0;
  v269 = 0;
  v270 = 0;
  v271 = 0;
  v272 = 0;
  v273 = 0;
  v274 = 0;
  v275 = 0;
  src = 0;
  v277 = 0;
  v278 = 0;
  sub_29B95F0((__int64)&v264, a3);
  v252 = 0;
  if ( v257 )
  {
    *((_QWORD *)&v8 + 1) = 0;
    do
    {
      *(_QWORD *)&v8 = *(_QWORD *)(a2 + 8LL * *((_QWORD *)&v8 + 1));
      if ( (unsigned __int64)v8 <= 1 )
        *(_QWORD *)&v8 = 1;
      v254[0] = v8;
      *(_QWORD *)&v8 = *(_QWORD *)(a4 + 8LL * *((_QWORD *)&v8 + 1));
      if ( v8 == 0 )
        *(_QWORD *)&v8 = 1;
      v255[0] = (void *)v8;
      sub_29B8FA0((unsigned __int64 *)&v264, &v252, v254, v255);
      *((_QWORD *)&v8 + 1) = v252 + 1;
      *(_QWORD *)&v8 = v257;
      v252 = *((_QWORD *)&v8 + 1);
    }
    while ( v257 > *((_QWORD *)&v8 + 1) );
    v9 = v259;
    v10 = 0xAAAAAAAAAAAAAAABLL * (v259 - v258);
    if ( v257 > v10 )
    {
      sub_29BB070((__int64)&v258, v257 - v10);
      *(_QWORD *)&v8 = v257;
    }
    else if ( v257 < v10 )
    {
      v194 = &v258[3 * v257];
      if ( v194 != v259 )
      {
        v195 = &v258[3 * v257];
        do
        {
          if ( *v195 )
            j_j___libc_free_0(*v195);
          v195 += 3;
        }
        while ( v195 != v9 );
        v259 = v194;
        *(_QWORD *)&v8 = v257;
      }
    }
    v11 = v262;
    v12 = v261;
    *((_QWORD *)&v8 + 1) = 0xAAAAAAAAAAAAAAABLL * (v262 - v261);
    v13 = *((_QWORD *)&v8 + 1);
    if ( (unsigned __int64)v8 > *((_QWORD *)&v8 + 1) )
    {
      sub_29BB070((__int64)&v261, v8 - *((_QWORD *)&v8 + 1));
      *(_QWORD *)&v8 = v257;
      goto LABEL_12;
    }
  }
  else
  {
    v11 = v262;
    v12 = v261;
    *(_QWORD *)&v8 = 0;
    v13 = 0xAAAAAAAAAAAAAAABLL * (v262 - v261);
  }
  if ( v13 > (unsigned __int64)v8 )
  {
    v192 = &v12[3 * v8];
    if ( v192 != v11 )
    {
      v193 = &v12[3 * v8];
      do
      {
        if ( *v193 )
          j_j___libc_free_0(*v193);
        v193 += 3;
      }
      while ( v193 != v11 );
      v262 = v192;
      *(_QWORD *)&v8 = v257;
    }
  }
LABEL_12:
  if ( (unsigned __int64)v8 > 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v14 = 8 * v8;
  if ( (_QWORD)v8 )
  {
    v15 = (char *)sub_22077B0(8 * v8);
    v246 = (unsigned __int64)v15;
    if ( v15 != &v15[v14] )
      memset(v15, 0, v14);
  }
  else
  {
    v246 = 0;
  }
  sub_29B7B30(&v267, a8);
  v16 = a7;
  v17 = (const __m128i *)((char *)a7 + 24 * a8);
  if ( v17 != a7 )
  {
    do
    {
      while ( 1 )
      {
        v18 = _mm_loadu_si128(v16);
        v19 = v16->m128i_i64[0];
        v256 = v16[1].m128i_u64[0];
        *(__m128i *)v255 = v18;
        ++*(_QWORD *)(v246 + 8 * v19);
        if ( v19 != v18.m128i_i64[1] )
        {
          v20 = (__int64)&v258[3 * v19];
          v21 = *(_BYTE **)(v20 + 8);
          if ( v21 == *(_BYTE **)(v20 + 16) )
          {
            sub_9CA200(v20, v21, &v255[1]);
          }
          else
          {
            if ( v21 )
            {
              *(_QWORD *)v21 = v18.m128i_i64[1];
              v21 = *(_BYTE **)(v20 + 8);
            }
            *(_QWORD *)(v20 + 8) = v21 + 8;
          }
          v22 = (__int64)&v261[3 * (__int64)v255[1]];
          v23 = *(void ***)(v22 + 8);
          if ( v23 == *(void ***)(v22 + 16) )
          {
            sub_9CA200(v22, v23, v255);
          }
          else
          {
            if ( v23 )
            {
              *v23 = v255[0];
              v23 = *(void ***)(v22 + 8);
            }
            *(_QWORD *)(v22 + 8) = v23 + 1;
          }
          if ( v256 )
            break;
        }
        v16 = (const __m128i *)((char *)v16 + 24);
        if ( v17 == v16 )
          goto LABEL_42;
      }
      v24 = &v264[14 * (__int64)v255[0]];
      v252 = (__int64)v24;
      v25 = &v264[14 * (__int64)v255[1]];
      v254[0] = (__int64)v25;
      sub_29B7DB0((unsigned __int64 *)&v267, &v252, v254, (__int64 *)&v256);
      v26 = (__int64)&v268[-3].m128i_i64[1];
      v254[0] = (__int64)&v268[-3].m128i_i64[1];
      v27 = (_BYTE *)v25[12];
      if ( v27 == (_BYTE *)v25[13] )
      {
        sub_29B8510((__int64)(v25 + 11), v27, v254);
        v26 = (__int64)&v268[-3].m128i_i64[1];
      }
      else
      {
        if ( v27 )
        {
          *(_QWORD *)v27 = v26;
          v27 = (_BYTE *)v25[12];
          v26 = (__int64)&v268[-3].m128i_i64[1];
        }
        v25[12] = (__int64)(v27 + 8);
      }
      v254[0] = v26;
      v28 = (_BYTE *)v24[9];
      if ( v28 == (_BYTE *)v24[10] )
      {
        sub_29B8510((__int64)(v24 + 8), v28, v254);
      }
      else
      {
        if ( v28 )
        {
          *(_QWORD *)v28 = v26;
          v28 = (_BYTE *)v24[9];
        }
        v24[9] = (__int64)(v28 + 8);
      }
      v29 = v256;
      v30 = v256;
      if ( v24[3] >= v256 )
        v30 = v24[3];
      v24[3] = v30;
      if ( v25[3] >= v29 )
        v29 = v25[3];
      v16 = (const __m128i *)((char *)v16 + 24);
      v25[3] = v29;
    }
    while ( v17 != v16 );
  }
LABEL_42:
  v31 = v268;
  for ( i = v267; v31 != i; i = (const __m128i *)((char *)i + 40) )
    i[1].m128i_i8[8] = *(_QWORD *)(v246 + 8LL * *(_QWORD *)i->m128i_i64[0]) > 1u;
  sub_29B9440((__int64)&v270, v257);
  if ( v257 > 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::reserve");
  v33 = src;
  if ( v257 > (v278 - (_BYTE *)src) >> 3 )
  {
    v34 = 8 * v257;
    v35 = v277 - (_BYTE *)src;
    if ( v257 )
    {
      v36 = sub_22077B0(8 * v257);
      v33 = src;
      v37 = (char *)v36;
      v38 = v277 - (_BYTE *)src;
    }
    else
    {
      v38 = v277 - (_BYTE *)src;
      v37 = 0;
    }
    if ( v38 > 0 )
    {
      v251 = v33;
      memmove(v37, v33, v38);
      v33 = v251;
    }
    else if ( !v33 )
    {
LABEL_50:
      src = v37;
      v277 = &v37[v35];
      v278 = &v37[v34];
      goto LABEL_51;
    }
    j_j___libc_free_0((unsigned __int64)v33);
    goto LABEL_50;
  }
LABEL_51:
  v249 = v265;
  v39 = v264;
  if ( v264 != v265 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v255[0] = v39;
        sub_29B8A50(&v270, v39, v255);
        v40 = v39[3] == 0;
        v39[4] = v271 - 80;
        if ( !v40 )
          break;
LABEL_53:
        v39 += 14;
        if ( v249 == v39 )
          goto LABEL_59;
      }
      v41 = v277;
      v42 = v271 - 80;
      if ( v277 == v278 )
        break;
      if ( v277 )
      {
        *(_QWORD *)v277 = v42;
        v41 = v277;
      }
      v39 += 14;
      v277 = v41 + 8;
      if ( v249 == v39 )
        goto LABEL_59;
    }
    v198 = src;
    v199 = v277 - (_BYTE *)src;
    v200 = (v277 - (_BYTE *)src) >> 3;
    if ( v200 == 0xFFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"vector::_M_realloc_insert");
    v201 = 1;
    if ( v200 )
      v201 = v199 >> 3;
    v202 = __CFADD__(v201, v200);
    v203 = v201 + v200;
    if ( v202 )
    {
      v208 = 0x7FFFFFFFFFFFFFF8LL;
    }
    else
    {
      if ( !v203 )
      {
        v204 = 0;
        v205 = 0;
        goto LABEL_330;
      }
      if ( v203 > 0xFFFFFFFFFFFFFFFLL )
        v203 = 0xFFFFFFFFFFFFFFFLL;
      v208 = 8 * v203;
    }
    v236 = v277 - (_BYTE *)src;
    v244 = src;
    v209 = sub_22077B0(v208);
    v198 = v244;
    v199 = v236;
    v205 = (char *)v209;
    v204 = v209 + v208;
LABEL_330:
    if ( &v205[v199] )
      *(_QWORD *)&v205[v199] = v42;
    v206 = (__int64)&v205[v199 + 8];
    if ( v199 > 0 )
    {
      v242 = v198;
      v207 = (char *)memmove(v205, v198, v199);
      v198 = v242;
      v205 = v207;
    }
    else if ( !v198 )
    {
LABEL_334:
      src = v205;
      v277 = (char *)v206;
      v278 = (char *)v204;
      goto LABEL_53;
    }
    v243 = v205;
    j_j___libc_free_0((unsigned __int64)v198);
    v205 = v243;
    goto LABEL_334;
  }
LABEL_59:
  sub_29B8DF0(&v273, 0xCCCCCCCCCCCCCCCDLL * (((char *)v268 - (char *)v267) >> 3));
  v240 = v265;
  v43 = (void **)v264;
  if ( v264 != v265 )
  {
    do
    {
      v44 = (_QWORD **)v43[9];
      for ( j = (_QWORD **)v43[8]; v44 != j; ++j )
      {
        v46 = *j;
        v254[0] = (__int64)v46;
        v47 = v43[4];
        v48 = v46[1];
        v49 = (_QWORD *)v47[7];
        v50 = (_QWORD *)v47[8];
        if ( v49 == v50 )
          goto LABEL_188;
        while ( *(_QWORD *)(v48 + 32) != *v49 )
        {
          v49 += 2;
          if ( v50 == v49 )
            goto LABEL_188;
        }
        v51 = v49[1];
        if ( !v51 )
        {
LABEL_188:
          sub_29B86D0(&v273, v254);
          v123 = (unsigned __int64 *)v43[4];
          v255[0] = *(void **)(v48 + 32);
          v255[1] = (void *)(v274 - 120);
          v124 = (__m128i *)v123[8];
          if ( v124 == (__m128i *)v123[9] )
          {
            sub_29B79B0(v123 + 7, v124, (const __m128i *)v255);
          }
          else
          {
            if ( v124 )
            {
              *v124 = _mm_loadu_si128((const __m128i *)v255);
              v124 = (__m128i *)v123[8];
            }
            v123[8] = (unsigned __int64)&v124[1];
          }
          v125 = *(unsigned __int64 **)(v48 + 32);
          v255[0] = v43[4];
          v255[1] = (void *)(v274 - 120);
          v126 = (__m128i *)v125[8];
          if ( v126 == (__m128i *)v125[9] )
          {
            sub_29B79B0(v125 + 7, v126, (const __m128i *)v255);
          }
          else
          {
            if ( v126 )
            {
              *v126 = _mm_loadu_si128((const __m128i *)v255);
              v126 = (__m128i *)v125[8];
            }
            v125[8] = (unsigned __int64)&v126[1];
          }
        }
        else
        {
          v255[0] = v46;
          v52 = *(_BYTE **)(v51 + 24);
          if ( v52 == *(_BYTE **)(v51 + 32) )
          {
            sub_29B8510(v51 + 16, v52, v255);
          }
          else
          {
            if ( v52 )
            {
              *(_QWORD *)v52 = v46;
              v52 = *(_BYTE **)(v51 + 24);
            }
            *(_QWORD *)(v51 + 24) = v52 + 8;
          }
        }
      }
      v43 += 14;
    }
    while ( v240 != (__int64 *)v43 );
  }
  if ( v246 )
    j_j___libc_free_0(v246);
  v53 = v264;
  v54 = v265;
  if ( v264 != v265 )
  {
    v55 = v258;
    v56 = v261;
    v57 = v264;
    do
    {
      while ( 1 )
      {
        v58 = &v55[3 * *v57];
        if ( v58[1] - *v58 == 8 )
        {
          v59 = *(_QWORD *)*v58;
          if ( v56[3 * v59 + 1] - v56[3 * v59] == 8 )
          {
            if ( v59 )
              break;
          }
        }
        v57 += 14;
        if ( v54 == v57 )
          goto LABEL_81;
      }
      v57[6] = (__int64)&v53[14 * v59];
      v264[14 * v59 + 7] = (__int64)v57;
      v57 += 14;
      v53 = v264;
    }
    while ( v54 != v57 );
LABEL_81:
    v60 = v265;
    if ( v265 != v53 )
    {
      v61 = v53;
      do
      {
        v62 = (__int64 *)v61[6];
        if ( v62 )
        {
          v63 = (_QWORD *)v61[7];
          if ( v63 )
          {
            if ( v62 == v61 )
            {
LABEL_88:
              v53[14 * *v63 + 6] = 0;
              v61[7] = 0;
              v53 = v264;
            }
            else
            {
              while ( 1 )
              {
                v62 = (__int64 *)v62[6];
                if ( !v62 )
                  break;
                if ( v62 == v61 )
                  goto LABEL_88;
              }
            }
          }
        }
        v61 += 14;
      }
      while ( v60 != v61 );
      for ( k = v265; k != v53; v53 += 14 )
      {
        while ( 1 )
        {
          if ( !v53[7] )
          {
            v65 = v53[6];
            if ( v65 )
              break;
          }
          v53 += 14;
          if ( k == v53 )
            goto LABEL_97;
        }
        do
        {
          sub_29BA160((__int64)&v257, v53[4], *(_QWORD *)(v65 + 32), 0, 0);
          v65 = *(_QWORD *)(v65 + 48);
        }
        while ( v65 );
      }
    }
  }
LABEL_97:
  while ( 2 )
  {
    v223 = v277;
    if ( (unsigned __int64)(v277 - (_BYTE *)src) <= 8 || src == v277 )
      break;
    v235 = (__int64 *)src;
    v213 = 0;
    v214 = 0;
    v234 = -1.0;
    v224 = 0;
    v225 = 0;
    do
    {
      v66 = *v235;
      v226 = *(__int64 **)(*v235 + 64);
      if ( *(__int64 **)(*v235 + 56) != v226 )
      {
        for ( m = *(__int64 **)(*v235 + 56); v226 != m; m += 2 )
        {
          v75 = m[1];
          if ( *(_QWORD *)v75 == *(_QWORD *)(v75 + 8) )
            continue;
          v76 = *m;
          if ( ((__int64)(*(_QWORD *)(v66 + 40) - *(_QWORD *)(v66 + 32)) >> 3)
             + ((__int64)(*(_QWORD *)(*m + 40) - *(_QWORD *)(*m + 32)) >> 3) >= (unsigned __int64)(unsigned int)qword_50081E8 )
            continue;
          v77 = *(_QWORD *)(v66 + 24);
          if ( v77 >= 0 )
          {
            v67 = (double)(int)v77;
          }
          else
          {
            v78 = *(_QWORD *)(v66 + 24) & 1LL | (*(_QWORD *)(v66 + 24) >> 1);
            v67 = (double)(int)v78 + (double)(int)v78;
          }
          v68 = *(_QWORD *)(v76 + 24);
          v69 = *(double *)(v66 + 16) / v67;
          if ( v68 < 0 )
          {
            v79 = (double)(int)(*(_DWORD *)(v76 + 24) & 1 | (*(_QWORD *)(v76 + 24) >> 1));
            v70 = v79 + v79;
          }
          else
          {
            v70 = (double)(int)v68;
          }
          v71 = *(double *)(v76 + 16) / v70;
          if ( v69 > v71 )
          {
            v72 = v69;
            v69 = *(double *)(v76 + 16) / v70;
            v71 = v72;
          }
          if ( v71 / v69 > *(double *)&qword_5008028 )
            continue;
          if ( v66 == *(_QWORD *)v75 )
          {
            if ( *(_BYTE *)(v75 + 112) )
            {
              v73 = v75 + 64;
LABEL_110:
              v74 = *(double *)v73;
              v247 = *(void **)(v73 + 8);
              v250 = *(_DWORD *)(v73 + 16);
              goto LABEL_111;
            }
          }
          else
          {
            v73 = v75 + 88;
            if ( *(_BYTE *)(v75 + 113) )
              goto LABEL_110;
          }
          v80 = *(_QWORD **)(v66 + 56);
          v81 = *(_QWORD **)(v66 + 64);
          if ( v80 == v81 )
          {
LABEL_196:
            v82 = 0;
          }
          else
          {
            while ( v66 != *v80 )
            {
              v80 += 2;
              if ( v81 == v80 )
                goto LABEL_196;
            }
            v82 = v80[1];
            if ( v82 )
              v82 += 16;
          }
          v254[1] = v82;
          v254[0] = v75 + 16;
          sub_29B8380((__int64)v255, v66, v76, (__int64)v254, 0, 0);
          v74 = *(double *)v255;
          if ( *(double *)v255 <= 0.00000001 || *(double *)v255 <= -0.9999999899999999 )
          {
            v250 = 0;
            v247 = 0;
            v74 = -1.0;
          }
          else
          {
            v247 = v255[1];
            v250 = v256;
          }
          v83 = **(_QWORD **)(v76 + 32);
          v84 = *(__int64 ***)(v83 + 88);
          if ( v84 != *(__int64 ***)(v83 + 96) )
          {
            v85 = v76;
            v86 = v75;
            v87 = *(__int64 ***)(v83 + 96);
            do
            {
              while ( 1 )
              {
                v88 = **v84;
                if ( v66 == *(_QWORD *)(v88 + 32) )
                  break;
                if ( v87 == ++v84 )
                  goto LABEL_143;
              }
              v89 = *(_QWORD *)(v88 + 8);
              v218 = v85;
              v227 = v86;
              v90 = (_QWORD *)sub_22077B0(8u);
              v91 = v227;
              v92 = (unsigned __int64)v90;
              v93 = v218;
              *v90 = 0x400000002LL;
              if ( v89 != -1 )
              {
                v94 = *(_QWORD *)(v66 + 32);
                if ( v89 + 1 != (*(_QWORD *)(v66 + 40) - v94) >> 3 && !*(_QWORD *)(*(_QWORD *)(v94 + 8 * v89) + 48LL) )
                {
                  v115 = v89 + 1;
                  v116 = (int *)v90 + 1;
                  v117 = 2;
                  v118 = (int *)(v90 + 1);
                  while ( 1 )
                  {
                    v211 = v92;
                    v216 = v91;
                    v231 = v115;
                    v238 = v93;
                    sub_29B8380((__int64)v255, v66, v93, (__int64)v254, v115, v117);
                    v93 = v238;
                    v115 = v231;
                    v91 = v216;
                    v92 = v211;
                    if ( *(double *)v255 > 0.00000001 && *(double *)v255 > v74 + 0.00000001 )
                    {
                      v74 = *(double *)v255;
                      v247 = v255[1];
                      v250 = v256;
                    }
                    if ( v118 == v116 )
                      break;
                    v117 = *v116++;
                  }
                }
              }
              ++v84;
              v219 = v93;
              v228 = v91;
              j_j___libc_free_0(v92);
              v86 = v228;
              v85 = v219;
            }
            while ( v87 != v84 );
LABEL_143:
            v75 = v86;
            v76 = v85;
          }
          v95 = *(_QWORD *)(*(_QWORD *)(v76 + 40) - 8LL);
          v96 = *(_QWORD *)(v95 + 64);
          if ( v96 != *(_QWORD *)(v95 + 72) )
          {
            v97 = v76;
            v98 = v75;
            v99 = *(_QWORD *)(v95 + 72);
            do
            {
              while ( 1 )
              {
                v100 = *(_QWORD *)(*(_QWORD *)v96 + 8LL);
                if ( v66 == *(_QWORD *)(v100 + 32) )
                  break;
                v96 += 8;
                if ( v99 == v96 )
                  goto LABEL_152;
              }
              v101 = *(_QWORD *)(v100 + 8);
              v220 = v97;
              v229 = v98;
              v102 = (_QWORD *)sub_22077B0(8u);
              v103 = v229;
              v104 = (unsigned __int64)v102;
              v105 = v220;
              *v102 = 0x300000002LL;
              if ( v101 )
              {
                v106 = *(_QWORD *)(v66 + 32);
                if ( v101 != (*(_QWORD *)(v66 + 40) - v106) >> 3
                  && !*(_QWORD *)(*(_QWORD *)(v106 + 8 * v101 - 8) + 48LL) )
                {
                  v119 = v101;
                  v120 = (int *)v102 + 1;
                  v121 = 2;
                  v122 = (int *)(v102 + 1);
                  while ( 1 )
                  {
                    v212 = v104;
                    v217 = v103;
                    v232 = v119;
                    v239 = v105;
                    sub_29B8380((__int64)v255, v66, v105, (__int64)v254, v119, v121);
                    v105 = v239;
                    v119 = v232;
                    v103 = v217;
                    v104 = v212;
                    if ( *(double *)v255 > 0.00000001 && *(double *)v255 > v74 + 0.00000001 )
                    {
                      v74 = *(double *)v255;
                      v247 = v255[1];
                      v250 = v256;
                    }
                    if ( v122 == v120 )
                      break;
                    v121 = *v120++;
                  }
                }
              }
              v96 += 8;
              v221 = v105;
              v230 = v103;
              j_j___libc_free_0(v104);
              v98 = v230;
              v97 = v221;
            }
            while ( v99 != v96 );
LABEL_152:
            v75 = v98;
            v76 = v97;
          }
          v107 = *(_QWORD *)(v66 + 40);
          v108 = *(_QWORD *)(v66 + 32);
          if ( (v107 - v108) >> 3 <= (unsigned __int64)(unsigned int)qword_5008108
            && (unsigned __int64)(*(_QWORD *)(v66 + 40) - v108) > 8 )
          {
            v215 = v75;
            v237 = v74;
            v109 = 1;
            do
            {
              v110 = *(_QWORD *)(v108 + 8 * v109 - 8);
              v111 = *(_QWORD *)(v110 + 64);
              v112 = *(_QWORD *)(v110 + 72);
              if ( v111 == v112 )
              {
LABEL_164:
                v253 = 4;
                v252 = 0x300000002LL;
                v113 = sub_22077B0(0xCu);
                *(_QWORD *)v113 = v252;
                *(_DWORD *)(v113 + 8) = v253;
                if ( v109 )
                {
                  v114 = *(_QWORD *)(v66 + 32);
                  if ( (*(_QWORD *)(v66 + 40) - v114) >> 3 != v109
                    && !*(_QWORD *)(*(_QWORD *)(v114 + 8 * v109 - 8) + 48LL) )
                  {
                    v127 = v113 + 12;
                    v128 = (int *)v113;
                    v222 = v113;
                    v129 = v66;
                    v233 = (int *)v127;
                    v130 = v128;
                    do
                    {
                      sub_29B8380((__int64)v255, v129, v76, (__int64)v254, v109, *v130);
                      if ( *(double *)v255 > 0.00000001 && *(double *)v255 > v237 + 0.00000001 )
                      {
                        v237 = *(double *)v255;
                        v247 = v255[1];
                        v250 = v256;
                      }
                      ++v130;
                    }
                    while ( v233 != v130 );
                    v66 = v129;
                    v113 = v222;
                  }
                }
                j_j___libc_free_0(v113);
                v107 = *(_QWORD *)(v66 + 40);
                v108 = *(_QWORD *)(v66 + 32);
              }
              else
              {
                while ( *(_QWORD *)(v108 + 8 * v109) != *(_QWORD *)(*(_QWORD *)v111 + 8LL) )
                {
                  v111 += 8;
                  if ( v112 == v111 )
                    goto LABEL_164;
                }
              }
              ++v109;
            }
            while ( (v107 - v108) >> 3 > v109 );
            v75 = v215;
            v74 = v237;
          }
          if ( v66 == *(_QWORD *)v75 )
          {
            *(_BYTE *)(v75 + 112) = 1;
            *(double *)(v75 + 64) = v74;
            *(_QWORD *)(v75 + 72) = v247;
            *(_DWORD *)(v75 + 80) = v250;
          }
          else
          {
            *(_BYTE *)(v75 + 113) = 1;
            *(double *)(v75 + 88) = v74;
            *(_QWORD *)(v75 + 96) = v247;
            *(_DWORD *)(v75 + 104) = v250;
          }
LABEL_111:
          if ( v74 > 0.00000001 )
          {
            if ( v74 <= v234 + 0.00000001 )
            {
              if ( fabs(v74 - v234) >= 0.00000001
                || *v225 <= *(_QWORD *)v66 && (*v224 <= *(_QWORD *)*m || *v225 != *(_QWORD *)v66) )
              {
                continue;
              }
              v224 = (_QWORD *)*m;
            }
            else
            {
              v224 = (_QWORD *)*m;
            }
            v225 = (_QWORD *)v66;
            v234 = v74;
            v213 = v250;
            v214 = (__int64)v247;
          }
        }
      }
      ++v235;
    }
    while ( v223 != (char *)v235 );
    if ( v234 > 0.00000001 )
    {
      sub_29BA160((__int64)&v257, (__int64)v225, (__int64)v224, v214, v213);
      continue;
    }
    break;
  }
  if ( v257 )
  {
    v131 = 0;
    v132 = 0;
    while ( 1 )
    {
      v133 = 3 * v132;
      v134 = &v258[3 * v132];
      v135 = *v134;
      v136 = (__int64)(v134[1] - *v134) >> 3;
      if ( v136 )
        break;
LABEL_216:
      ++v132;
      v131 += 14;
      if ( v257 <= v132 )
        goto LABEL_217;
    }
    v137 = v134[1] - *v134 - 8;
    v138 = 0;
    while ( 2 )
    {
      v139 = *(_QWORD *)(v135 + v137);
      v140 = v264[v131 + 4];
      v141 = v264[14 * v139 + 4];
      if ( v140 == v141
        || (v142 = *(__int64 ***)(v141 + 32), v143 = *v142, (v144 = **v142) == 0)
        || (v145 = *(_QWORD *)(v140 + 40), (v146 = **(_QWORD **)(v145 - 8) == v132 && v139 == v144) == 0) )
      {
LABEL_215:
        ++v138;
        v137 -= 8;
        if ( v136 == v138 )
          goto LABEL_216;
        v135 = v258[v133];
        continue;
      }
      break;
    }
    v188 = *(_QWORD *)(v140 + 32);
    if ( v145 == v188 )
    {
LABEL_335:
      v189 = *(__int64 ***)(v141 + 40);
      v190 = v146;
      if ( v142 == v189 )
      {
LABEL_308:
        v248 = v132;
        sub_29BA160((__int64)&v257, v140, v141, 0, 0);
        v132 = v248;
        goto LABEL_215;
      }
    }
    else
    {
      while ( !*(_QWORD *)(*(_QWORD *)v188 + 24LL) )
      {
        v188 += 8;
        if ( v145 == v188 )
          goto LABEL_335;
      }
      v189 = *(__int64 ***)(v141 + 40);
      v190 = 0;
      if ( v142 == v189 )
        goto LABEL_215;
    }
    v191 = v142 + 1;
    while ( !v143[3] )
    {
      if ( v189 == v191 )
        goto LABEL_307;
      v143 = *v191++;
    }
    v146 = 0;
LABEL_307:
    if ( v146 != v190 )
      goto LABEL_215;
    goto LABEL_308;
  }
LABEL_217:
  v147 = v271;
  v148 = v270;
  v255[0] = 0;
  v255[1] = 0;
  v149 = 0;
  v256 = 0;
  if ( v270 != v271 )
  {
    do
    {
      if ( *(_QWORD *)(v148 + 40) != *(_QWORD *)(v148 + 32) )
      {
        v254[0] = v148;
        if ( (char *)v256 == v149 )
        {
          sub_29B7C20((__int64)v255, v149, v254);
          v149 = (char *)v255[1];
        }
        else
        {
          if ( v149 )
          {
            *(_QWORD *)v149 = v148;
            v149 = (char *)v255[1];
          }
          v149 += 8;
          v255[1] = v149;
        }
      }
      v148 += 80;
    }
    while ( v147 != v148 );
    v150 = (char *)v255[0];
    v151 = v149;
    if ( v255[0] != v149 )
    {
      v152 = v149 - (char *)v255[0];
      _BitScanReverse64(&v153, (v149 - (char *)v255[0]) >> 3);
      sub_29BA560((char *)v255[0], v149, 2LL * (int)(63 - (v153 ^ 0x3F)));
      if ( v152 > 128 )
      {
        sub_29B97C0(v150, v150 + 128);
        v154 = v150 + 128;
        if ( v150 + 128 == v149 )
          goto LABEL_239;
        while ( 1 )
        {
          v155 = *(_QWORD *)v154;
          for ( n = v154; ; n -= 8 )
          {
            v157 = *((_QWORD *)n - 1);
            v158 = ***(_QWORD ***)(v155 + 32);
            if ( (v158 == 0) == (***(_QWORD ***)(v157 + 32) == 0) )
              break;
            if ( v158 )
              goto LABEL_238;
LABEL_229:
            *(_QWORD *)n = v157;
          }
          v159 = *(_QWORD *)(v157 + 24);
          if ( v159 < 0 )
          {
            v196 = *(_QWORD *)(v157 + 24) & 1LL | (*(_QWORD *)(v157 + 24) >> 1);
            v160 = (double)(int)v196 + (double)(int)v196;
          }
          else
          {
            v160 = (double)(int)v159;
          }
          v161 = *(_QWORD *)(v155 + 24);
          v162 = *(double *)(v157 + 16) / v160;
          if ( v161 < 0 )
          {
            v197 = *(_QWORD *)(v155 + 24) & 1LL | (*(_QWORD *)(v155 + 24) >> 1);
            v163 = (double)(int)v197 + (double)(int)v197;
          }
          else
          {
            v163 = (double)(int)v161;
          }
          v164 = *(double *)(v155 + 16) / v163;
          if ( v164 > v162 || v162 <= v164 && *(_QWORD *)v157 > *(_QWORD *)v155 )
            goto LABEL_229;
LABEL_238:
          v154 += 8;
          *(_QWORD *)n = v155;
          if ( v151 == v154 )
            goto LABEL_239;
        }
      }
      sub_29B97C0(v150, v149);
    }
  }
LABEL_239:
  v165 = v257;
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  sub_9C9810((__int64)a1, v165);
  v166 = (char *)v255[0];
  v167 = (char *)v255[1];
  if ( v255[0] != v255[1] )
  {
    do
    {
      v168 = *(_QWORD ***)(*(_QWORD *)v166 + 40LL);
      for ( ii = *(_QWORD ***)(*(_QWORD *)v166 + 32LL); v168 != ii; a1[1] = v170 + 8 )
      {
        while ( 1 )
        {
          v171 = *ii;
          v170 = (_BYTE *)a1[1];
          if ( v170 != (_BYTE *)a1[2] )
            break;
          ++ii;
          sub_9CA200((__int64)a1, v170, v171);
          if ( v168 == ii )
            goto LABEL_247;
        }
        if ( v170 )
        {
          *(_QWORD *)v170 = *v171;
          v170 = (_BYTE *)a1[1];
        }
        ++ii;
      }
LABEL_247:
      v166 += 8;
    }
    while ( v167 != v166 );
    v167 = (char *)v255[0];
  }
  if ( v167 )
    j_j___libc_free_0((unsigned __int64)v167);
  if ( src )
    j_j___libc_free_0((unsigned __int64)src);
  v172 = v274;
  v173 = v273;
  if ( v274 != v273 )
  {
    do
    {
      v174 = *(_QWORD *)(v173 + 16);
      if ( v174 )
        j_j___libc_free_0(v174);
      v173 += 120LL;
    }
    while ( v172 != v173 );
    v173 = v273;
  }
  if ( v173 )
    j_j___libc_free_0(v173);
  v175 = v271;
  v176 = v270;
  if ( v271 != v270 )
  {
    do
    {
      v177 = *(_QWORD *)(v176 + 56);
      if ( v177 )
        j_j___libc_free_0(v177);
      v178 = *(_QWORD *)(v176 + 32);
      if ( v178 )
        j_j___libc_free_0(v178);
      v176 += 80LL;
    }
    while ( v175 != v176 );
    v176 = v270;
  }
  if ( v176 )
    j_j___libc_free_0(v176);
  if ( v267 )
    j_j___libc_free_0((unsigned __int64)v267);
  v179 = v265;
  v180 = (unsigned __int64)v264;
  if ( v265 != v264 )
  {
    do
    {
      v181 = *(_QWORD *)(v180 + 88);
      if ( v181 )
        j_j___libc_free_0(v181);
      v182 = *(_QWORD *)(v180 + 64);
      if ( v182 )
        j_j___libc_free_0(v182);
      v180 += 112LL;
    }
    while ( v179 != (__int64 *)v180 );
    v180 = (unsigned __int64)v264;
  }
  if ( v180 )
    j_j___libc_free_0(v180);
  v183 = v262;
  v184 = v261;
  if ( v262 != v261 )
  {
    do
    {
      if ( *v184 )
        j_j___libc_free_0(*v184);
      v184 += 3;
    }
    while ( v183 != v184 );
    v184 = v261;
  }
  if ( v184 )
    j_j___libc_free_0((unsigned __int64)v184);
  v185 = v259;
  v186 = v258;
  if ( v259 != v258 )
  {
    do
    {
      if ( *v186 )
        j_j___libc_free_0(*v186);
      v186 += 3;
    }
    while ( v185 != v186 );
    v186 = v258;
  }
  if ( v186 )
    j_j___libc_free_0((unsigned __int64)v186);
  return a1;
}
