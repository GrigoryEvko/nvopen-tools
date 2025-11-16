// Function: sub_9E7B10
// Address: 0x9e7b10
//
__int64 *__fastcall sub_9E7B10(__int64 *a1, const __m128i *a2, unsigned __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v5; // r14
  char v8; // al
  char v9; // al
  void (__fastcall *v10)(const __m128i *, const __m128i *, __int64); // rax
  __int64 m128i_i64; // r15
  unsigned __int64 v12; // rax
  __int64 *v13; // r12
  __int64 v14; // rax
  _BYTE *v15; // rsi
  __int64 v16; // rdx
  _QWORD *v17; // rcx
  __int64 *v18; // rbx
  int v19; // edx
  __int64 v20; // rcx
  __int64 v21; // rdx
  char v22; // r10
  __int64 v23; // rax
  __m128i v24; // xmm0
  __int64 v25; // rdx
  __m128i v26; // xmm1
  __int64 v27; // rax
  __int64 v29; // rcx
  __int64 v30; // rdx
  __m128i v31; // xmm2
  __m128i v32; // xmm0
  void (__fastcall *v33)(_QWORD, _QWORD, _QWORD); // rax
  __m128i v34; // xmm3
  size_t *v35; // rsi
  size_t v36; // rax
  __int64 *v37; // r11
  size_t v38; // rcx
  __int64 v39; // rsi
  __int64 v40; // rax
  __int64 *v41; // r11
  size_t v42; // rcx
  __int64 v43; // rdi
  __int64 v44; // rsi
  __int64 v45; // rax
  __int64 v46; // r9
  __int64 v47; // rdx
  __int64 v48; // rax
  __int64 *v49; // r11
  __int64 v50; // r8
  unsigned __int8 v51; // cl
  unsigned int v52; // eax
  char v53; // r10
  __int64 v54; // rsi
  unsigned int v55; // r9d
  __int64 v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rdx
  char v60; // al
  unsigned __int64 v61; // rdx
  __int64 v62; // rax
  __int64 *v63; // r11
  char v64; // dl
  _QWORD *v65; // rcx
  __int64 *v66; // rdi
  size_t v67; // r9
  __int64 v68; // rsi
  __int64 v69; // rdx
  bool v70; // al
  char v71; // dl
  _QWORD *v72; // rsi
  _QWORD *v73; // rcx
  _BYTE *v74; // rdi
  size_t v75; // r8
  __int64 v76; // rdx
  char v77; // al
  __int64 *v78; // r11
  unsigned __int64 v79; // rdi
  char v80; // al
  __int64 *v81; // rdi
  char v82; // al
  _QWORD *v83; // rcx
  __int64 *v84; // rdi
  __int64 v85; // rsi
  size_t v86; // r9
  __int64 v87; // rdx
  __int64 v88; // rax
  const char *v89; // rax
  size_t v90; // rax
  __int64 *v91; // rdi
  char v92; // cl
  _QWORD *v93; // r10
  __int64 v94; // rsi
  __int64 v95; // rax
  const char *v96; // rdx
  unsigned __int64 v97; // r9
  __int64 v98; // rdx
  unsigned int v99; // edi
  int v100; // edx
  bool v101; // cf
  int v102; // edi
  const char *v103; // rax
  unsigned __int64 v104; // rax
  char v105; // al
  __int64 *v106; // rdi
  __int64 v107; // rsi
  unsigned __int64 v108; // rax
  __int64 v109; // rax
  unsigned __int64 v110; // rdx
  bool v111; // r10
  __int64 v112; // rax
  __int64 v113; // rcx
  unsigned __int64 v114; // rax
  size_t v115; // rdx
  unsigned __int64 v116; // rdx
  _BYTE *v117; // rsi
  _BYTE *v118; // rax
  size_t v119; // rax
  size_t v120; // rax
  size_t v121; // rax
  __int64 v122; // rcx
  __int64 v123; // rdx
  __m128i *v124; // rsi
  bool v125; // zf
  const char *v126; // rax
  __int64 v127; // rdx
  char *v128; // rcx
  __int64 v129; // rax
  _BYTE *v130; // rsi
  __int64 *v131; // rax
  __int64 *v132; // rdx
  __int64 *j; // rax
  __int64 v134; // rcx
  __int64 v135; // rsi
  size_t v136; // rdx
  size_t v137; // rdx
  size_t v138; // rdx
  unsigned int v139; // r14d
  _QWORD *v140; // rbx
  int v141; // r12d
  _BYTE *v142; // r11
  unsigned __int64 v143; // rdx
  size_t v144; // rsi
  size_t v145; // rcx
  __int64 v146; // rax
  size_t v147; // r15
  __int64 v148; // rax
  int v149; // r9d
  __int64 v150; // rdx
  __int64 v151; // r8
  __int64 v152; // rax
  __int64 v153; // rax
  __int64 v154; // rsi
  char v155; // dl
  char v156; // dl
  __int64 v157; // rdx
  __int64 v158; // rax
  __int64 v159; // rax
  size_t v160; // r11
  char v161; // al
  int v162; // ecx
  char v163; // dl
  __int64 v164; // rdx
  char v165; // al
  __int64 v166; // r11
  __int64 v167; // rax
  __int64 v168; // rdx
  __int64 v169; // rsi
  const char *v170; // rax
  __int64 v171; // rdx
  char v172; // al
  char v173; // al
  __int64 v174; // rcx
  __int64 v175; // rdx
  __int64 v176; // rdi
  __int64 v177; // rax
  __int64 v178; // rax
  unsigned int v179; // esi
  __int64 v180; // rdi
  _QWORD *v181; // rdx
  unsigned int v182; // r10d
  int i; // ecx
  _QWORD *v184; // rax
  __int64 v185; // r8
  int v186; // edx
  _QWORD *v187; // rax
  __m128i *v188; // rsi
  __int64 v189; // rcx
  size_t v190; // rdx
  unsigned __int64 v191; // rsi
  __int64 v192; // rax
  unsigned int v193; // eax
  __int64 v194; // rsi
  unsigned int v195; // [rsp+Ch] [rbp-434h]
  bool v196; // [rsp+13h] [rbp-42Dh]
  int v197; // [rsp+14h] [rbp-42Ch]
  char v198; // [rsp+18h] [rbp-428h]
  char v199; // [rsp+1Ch] [rbp-424h]
  unsigned __int64 v200; // [rsp+20h] [rbp-420h]
  char v201; // [rsp+28h] [rbp-418h]
  unsigned int v202; // [rsp+30h] [rbp-410h]
  __int64 *v203; // [rsp+30h] [rbp-410h]
  unsigned int v204; // [rsp+30h] [rbp-410h]
  const char *v205; // [rsp+38h] [rbp-408h]
  __int64 *v206; // [rsp+38h] [rbp-408h]
  __int64 *v207; // [rsp+38h] [rbp-408h]
  __int64 v208; // [rsp+38h] [rbp-408h]
  __int64 v209; // [rsp+38h] [rbp-408h]
  unsigned int v210; // [rsp+40h] [rbp-400h]
  __int64 v211; // [rsp+40h] [rbp-400h]
  unsigned __int64 v212; // [rsp+40h] [rbp-400h]
  size_t v213; // [rsp+48h] [rbp-3F8h]
  __int64 v214; // [rsp+48h] [rbp-3F8h]
  size_t v215; // [rsp+48h] [rbp-3F8h]
  __int64 v216; // [rsp+48h] [rbp-3F8h]
  __int64 v217; // [rsp+48h] [rbp-3F8h]
  size_t v218; // [rsp+48h] [rbp-3F8h]
  size_t v219; // [rsp+48h] [rbp-3F8h]
  size_t v220; // [rsp+48h] [rbp-3F8h]
  __int64 v221; // [rsp+48h] [rbp-3F8h]
  __int64 v222; // [rsp+48h] [rbp-3F8h]
  __int64 v223; // [rsp+48h] [rbp-3F8h]
  __int64 v224; // [rsp+48h] [rbp-3F8h]
  int v225; // [rsp+50h] [rbp-3F0h]
  __int64 *v226; // [rsp+50h] [rbp-3F0h]
  int v227; // [rsp+50h] [rbp-3F0h]
  __int64 v228; // [rsp+50h] [rbp-3F0h]
  __int64 v229; // [rsp+50h] [rbp-3F0h]
  _BYTE *v230; // [rsp+50h] [rbp-3F0h]
  __int64 *v231; // [rsp+58h] [rbp-3E8h]
  __int64 v232; // [rsp+58h] [rbp-3E8h]
  unsigned __int64 v233; // [rsp+58h] [rbp-3E8h]
  _QWORD *v234; // [rsp+58h] [rbp-3E8h]
  char v235; // [rsp+58h] [rbp-3E8h]
  _QWORD *v236; // [rsp+60h] [rbp-3E0h]
  _QWORD *v237; // [rsp+60h] [rbp-3E0h]
  __int64 v238; // [rsp+60h] [rbp-3E0h]
  size_t v239; // [rsp+60h] [rbp-3E0h]
  const char *src; // [rsp+68h] [rbp-3D8h]
  _QWORD *srca; // [rsp+68h] [rbp-3D8h]
  _QWORD *srcb; // [rsp+68h] [rbp-3D8h]
  unsigned __int64 v243; // [rsp+70h] [rbp-3D0h]
  void *v244; // [rsp+70h] [rbp-3D0h]
  void *v245; // [rsp+70h] [rbp-3D0h]
  char v246; // [rsp+90h] [rbp-3B0h]
  __int64 *v247; // [rsp+90h] [rbp-3B0h]
  size_t v248; // [rsp+90h] [rbp-3B0h]
  __int64 *v249; // [rsp+90h] [rbp-3B0h]
  char v251; // [rsp+A9h] [rbp-397h] BYREF
  __int16 v252; // [rsp+AAh] [rbp-396h] BYREF
  unsigned int v253; // [rsp+ACh] [rbp-394h] BYREF
  __int64 v254; // [rsp+B0h] [rbp-390h] BYREF
  char v255; // [rsp+B8h] [rbp-388h]
  __int64 v256; // [rsp+C0h] [rbp-380h] BYREF
  char v257; // [rsp+C8h] [rbp-378h]
  __int64 v258[2]; // [rsp+D0h] [rbp-370h] BYREF
  _QWORD v259[2]; // [rsp+E0h] [rbp-360h] BYREF
  _QWORD v260[4]; // [rsp+F0h] [rbp-350h] BYREF
  __int64 v261[2]; // [rsp+110h] [rbp-330h] BYREF
  _QWORD v262[2]; // [rsp+120h] [rbp-320h] BYREF
  unsigned __int64 v263; // [rsp+130h] [rbp-310h] BYREF
  size_t v264; // [rsp+138h] [rbp-308h]
  _QWORD v265[2]; // [rsp+140h] [rbp-300h] BYREF
  __int16 v266; // [rsp+150h] [rbp-2F0h]
  size_t n[2]; // [rsp+160h] [rbp-2E0h] BYREF
  __int64 v268; // [rsp+170h] [rbp-2D0h] BYREF
  char v269; // [rsp+178h] [rbp-2C8h] BYREF
  __int64 v270; // [rsp+180h] [rbp-2C0h]
  __int64 v271; // [rsp+188h] [rbp-2B8h]
  __int64 v272; // [rsp+190h] [rbp-2B0h]
  __m128i v273; // [rsp+200h] [rbp-240h] BYREF
  void (__fastcall *v274)(_QWORD, _QWORD, _QWORD); // [rsp+210h] [rbp-230h] BYREF
  __int64 v275; // [rsp+218h] [rbp-228h]

  v5 = (unsigned __int64)a2;
  v246 = a4;
  if ( LODWORD(qword_4F80E68[8]) != 1 )
  {
    v8 = qword_4F80F48[8];
    if ( LOBYTE(qword_4F80F48[8]) )
      v8 = LODWORD(qword_4F80268[8]) != 2;
    *(_BYTE *)(a2[27].m128i_i64[1] + 872) = v8;
  }
  v9 = *(_BYTE *)(a5 + 72);
  if ( a2[125].m128i_i8[0] )
  {
    if ( v9 )
    {
      v29 = *(_QWORD *)(a5 + 56);
      v30 = *(_QWORD *)(a5 + 64);
      *(_QWORD *)(a5 + 56) = 0;
      v31 = _mm_loadu_si128(&v273);
      v32 = _mm_loadu_si128((const __m128i *)(a5 + 40));
      *(_QWORD *)(a5 + 64) = v275;
      *(__m128i *)(a5 + 40) = v31;
      v33 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))a2[124].m128i_i64[0];
      v34 = _mm_loadu_si128(a2 + 123);
      a2[124].m128i_i64[0] = v29;
      a4 = a2[124].m128i_i64[1];
      v274 = v33;
      v275 = a4;
      a2[124].m128i_i64[1] = v30;
      v273 = v34;
      a2[123] = v32;
      if ( v33 )
        v33(&v273, &v273, 3);
    }
    else
    {
      v10 = (void (__fastcall *)(const __m128i *, const __m128i *, __int64))a2[124].m128i_i64[0];
      a2[125].m128i_i8[0] = 0;
      if ( v10 )
        v10(a2 + 123, a2 + 123, 3);
    }
LABEL_9:
    m128i_i64 = (__int64)a2[2].m128i_i64;
    if ( a3 )
      goto LABEL_10;
    goto LABEL_20;
  }
  if ( !v9 )
    goto LABEL_9;
  a2[124].m128i_i64[0] = 0;
  v23 = *(_QWORD *)(a5 + 56);
  m128i_i64 = (__int64)a2[2].m128i_i64;
  v24 = _mm_loadu_si128((const __m128i *)(a5 + 40));
  v25 = a2[124].m128i_i64[1];
  *(_QWORD *)(a5 + 56) = 0;
  v26 = _mm_loadu_si128(a2 + 123);
  a2[124].m128i_i64[0] = v23;
  v27 = *(_QWORD *)(a5 + 64);
  a2[123] = v24;
  *(_QWORD *)(a5 + 64) = v25;
  *(__m128i *)(a5 + 40) = v26;
  a2[124].m128i_i64[1] = v27;
  a2[125].m128i_i8[0] = 1;
  if ( a3 )
  {
LABEL_10:
    sub_9CDFE0(v273.m128i_i64, m128i_i64, a3, a4);
    v12 = v273.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v273.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
      goto LABEL_11;
LABEL_21:
    *a1 = v12 | 1;
    return a1;
  }
LABEL_20:
  sub_A4DCE0(&v273, m128i_i64, 8, 0);
  v12 = v273.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v273.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    goto LABEL_21;
LABEL_11:
  v13 = v258;
  v251 = 0;
  v273.m128i_i64[0] = (__int64)&v274;
  v273.m128i_i64[1] = 0x4000000000LL;
  v14 = a2[27].m128i_i64[1];
  v15 = *(_BYTE **)(v14 + 760);
  v16 = *(_QWORD *)(v14 + 768);
  v258[0] = (__int64)v259;
  sub_9C36C0(v258, v15, (__int64)&v15[v16]);
  v260[1] = v258;
  v260[0] = &v251;
  v260[2] = v5;
  v260[3] = a5;
  v18 = &v254;
  while ( 2 )
  {
    sub_9CEA50((__int64)v18, m128i_i64, 0, (__int64)v17);
    v19 = v255 & 1;
    v20 = (unsigned int)(2 * v19);
    v255 = (2 * v19) | v255 & 0xFD;
    if ( (_BYTE)v19 )
    {
      v35 = (size_t *)v18;
      sub_9C9090(a1, v18);
      goto LABEL_26;
    }
    v21 = HIDWORD(v254);
    if ( (_DWORD)v254 == 1 )
    {
      v35 = v260;
      sub_9D2B60(n, (__int64)v260);
      if ( (n[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        *a1 = n[0] & 0xFFFFFFFFFFFFFFFELL | 1;
        goto LABEL_26;
      }
LABEL_34:
      v35 = (size_t *)v5;
      sub_9E01F0(a1, (_QWORD *)v5);
      goto LABEL_26;
    }
    if ( (_DWORD)v254 != 2 )
    {
      if ( !(_DWORD)v254 )
      {
        v35 = (size_t *)(v5 + 8);
        n[0] = (size_t)"Malformed block";
        LOWORD(v270) = 259;
        sub_9C81F0(a1, v5 + 8, (__int64)n);
        goto LABEL_26;
      }
      sub_A4B600(&v256, m128i_i64, HIDWORD(v254), &v273, 0);
      v22 = v257 & 1;
      v257 = (2 * (v257 & 1)) | v257 & 0xFD;
      if ( v22 )
      {
        v35 = (size_t *)&v256;
        sub_9C8CD0(a1, &v256);
        goto LABEL_77;
      }
      switch ( (int)v256 )
      {
        case 1:
          sub_9C8860((__int64)n, v5 + 8, (__int64 *)v273.m128i_i64[0], v273.m128i_u32[2]);
          v71 = n[1] & 1;
          LOBYTE(n[1]) = (2 * (n[1] & 1)) | n[1] & 0xFD;
          if ( !v71 )
          {
            *(_BYTE *)(v5 + 1832) = LODWORD(n[0]) != 0;
            sub_9CE2A0(n);
            v64 = v257;
            v70 = (v257 & 2) != 0;
            goto LABEL_95;
          }
          v35 = n;
          sub_9C8CD0(a1, (__int64 *)n);
          sub_9CE2A0(n);
          goto LABEL_77;
        case 2:
          if ( v251 )
          {
            BYTE1(v270) = 1;
            v89 = "target triple too late in module";
LABEL_131:
            v35 = (size_t *)(v5 + 8);
            n[0] = (size_t)v89;
            LOBYTE(v270) = 3;
            sub_9C81F0(a1, v5 + 8, (__int64)n);
            goto LABEL_77;
          }
          v261[0] = (__int64)v262;
          v261[1] = 0;
          LOBYTE(v262[0]) = 0;
          if ( !(unsigned __int8)sub_9C57B0((__int64 *)v273.m128i_i64[0], v273.m128i_u32[2], v261) )
          {
            v65 = *(_QWORD **)(v5 + 440);
            v266 = 260;
            srca = v65;
            v263 = (unsigned __int64)v261;
            sub_CC9F70(n, &v263);
            v17 = srca;
            v66 = (__int64 *)srca[29];
            if ( (__int64 *)n[0] == &v268 )
            {
              v138 = n[1];
              if ( n[1] )
              {
                if ( n[1] == 1 )
                {
                  *(_BYTE *)v66 = v268;
                }
                else
                {
                  memcpy(v66, &v268, n[1]);
                  v17 = srca;
                }
                v138 = n[1];
                v66 = (__int64 *)srca[29];
              }
              v17[30] = v138;
              *((_BYTE *)v66 + v138) = 0;
              v66 = (__int64 *)n[0];
              goto LABEL_90;
            }
            v67 = n[1];
            v68 = v268;
            if ( v66 == srca + 31 )
            {
              srca[29] = n[0];
              srca[30] = v67;
              srca[31] = v68;
            }
            else
            {
              v69 = srca[31];
              srca[29] = n[0];
              srca[30] = v67;
              srca[31] = v68;
              if ( v66 )
              {
                n[0] = (size_t)v66;
                v268 = v69;
LABEL_90:
                n[1] = 0;
                *(_BYTE *)v66 = 0;
                v17[33] = v270;
                v17[34] = v271;
                v17[35] = v272;
                if ( (__int64 *)n[0] != &v268 )
                  j_j___libc_free_0(n[0], v268 + 1);
                if ( (_QWORD *)v261[0] != v262 )
                  j_j___libc_free_0(v261[0], v262[0] + 1LL);
                goto LABEL_94;
              }
            }
            n[0] = (size_t)&v268;
            v66 = &v268;
            goto LABEL_90;
          }
          v78 = v261;
LABEL_299:
          v35 = (size_t *)(v5 + 8);
          v249 = v78;
          n[0] = (size_t)"Invalid record";
          LOWORD(v270) = 259;
          sub_9C81F0(a1, v5 + 8, (__int64)n);
          sub_2240A30(v249);
          goto LABEL_77;
        case 3:
          if ( v251 )
          {
            BYTE1(v270) = 1;
            v89 = "datalayout too late in module";
          }
          else
          {
            if ( !(unsigned __int8)sub_9C57B0((__int64 *)v273.m128i_i64[0], v273.m128i_u32[2], v13) )
            {
LABEL_94:
              v64 = v257;
              v70 = (v257 & 2) != 0;
LABEL_95:
              v273.m128i_i32[2] = 0;
              if ( v70 )
                sub_9CE230(&v256);
LABEL_81:
              if ( (v64 & 1) != 0 && v256 )
                (*(void (__fastcall **)(__int64))(*(_QWORD *)v256 + 8LL))(v256);
              goto LABEL_40;
            }
LABEL_130:
            BYTE1(v270) = 1;
            v89 = "Invalid record";
          }
          goto LABEL_131;
        case 4:
          v263 = (unsigned __int64)v265;
          v264 = 0;
          LOBYTE(v265[0]) = 0;
          v82 = sub_9C57B0((__int64 *)v273.m128i_i64[0], v273.m128i_u32[2], (__int64 *)&v263);
          v78 = (__int64 *)&v263;
          if ( v82 )
            goto LABEL_299;
          v83 = *(_QWORD **)(v5 + 440);
          n[0] = (size_t)&v268;
          v237 = v83;
          sub_9C2D70((__int64 *)n, (_BYTE *)v263, v263 + v264);
          v17 = v237;
          v84 = (__int64 *)v237[11];
          if ( (__int64 *)n[0] == &v268 )
          {
            v136 = n[1];
            if ( n[1] )
            {
              if ( n[1] == 1 )
              {
                *(_BYTE *)v84 = v268;
              }
              else
              {
                memcpy(v84, &v268, n[1]);
                v17 = v237;
              }
              v136 = n[1];
              v84 = (__int64 *)v237[11];
            }
            v17[12] = v136;
            *((_BYTE *)v84 + v136) = 0;
            v84 = (__int64 *)n[0];
          }
          else
          {
            v85 = v268;
            v86 = n[1];
            if ( v84 == v237 + 13 )
            {
              v237[11] = n[0];
              v237[12] = v86;
              v237[13] = v85;
            }
            else
            {
              v87 = v237[13];
              v237[11] = n[0];
              v237[12] = v86;
              v237[13] = v85;
              if ( v84 )
              {
                n[0] = (size_t)v84;
                v268 = v87;
                goto LABEL_123;
              }
            }
            n[0] = (size_t)&v268;
            v84 = &v268;
          }
LABEL_123:
          n[1] = 0;
          *(_BYTE *)v84 = 0;
          if ( (__int64 *)n[0] != &v268 )
          {
            srcb = v17;
            j_j___libc_free_0(n[0], v268 + 1);
            v17 = srcb;
          }
          v88 = v17[12];
          if ( v88 && *(_BYTE *)(v17[11] + v88 - 1) != 10 )
            sub_2240F50(v17 + 11, 10);
LABEL_116:
          v79 = v263;
          if ( (_QWORD *)v263 != v265 )
LABEL_110:
            j_j___libc_free_0(v79, v265[0] + 1LL);
          goto LABEL_94;
        case 5:
          v263 = (unsigned __int64)v265;
          v264 = 0;
          LOBYTE(v265[0]) = 0;
          v80 = sub_9C57B0((__int64 *)v273.m128i_i64[0], v273.m128i_u32[2], (__int64 *)&v263);
          v78 = (__int64 *)&v263;
          if ( v80 )
            goto LABEL_299;
          v81 = *(__int64 **)(v5 + 488);
          if ( v81 == *(__int64 **)(v5 + 496) )
          {
            sub_8FD760((__m128i **)(v5 + 480), *(const __m128i **)(v5 + 488), (__int64)&v263);
          }
          else
          {
            if ( v81 )
            {
              *v81 = (__int64)(v81 + 2);
              sub_9C36C0(v81, (_BYTE *)v263, v263 + v264);
              v81 = *(__int64 **)(v5 + 488);
            }
            *(_QWORD *)(v5 + 488) = v81 + 4;
          }
          goto LABEL_116;
        case 6:
          v263 = (unsigned __int64)v265;
          v264 = 0;
          LOBYTE(v265[0]) = 0;
          v77 = sub_9C57B0((__int64 *)v273.m128i_i64[0], v273.m128i_u32[2], (__int64 *)&v263);
          v78 = (__int64 *)&v263;
          v17 = v265;
          if ( v77 )
            goto LABEL_299;
          v79 = v263;
          if ( (_QWORD *)v263 != v265 )
            goto LABEL_110;
          goto LABEL_94;
        case 7:
          v228 = v5 + 8;
          v238 = v273.m128i_i64[0];
          v233 = v273.m128i_u32[2];
          if ( *(_BYTE *)(v5 + 392) )
          {
            v215 = *(_QWORD *)(v273.m128i_i64[0] + 8);
            if ( *(_QWORD *)v273.m128i_i64[0] + v215 > *(_QWORD *)(v5 + 384) )
              goto LABEL_229;
            v205 = (const char *)(*(_QWORD *)(v5 + 376) + *(_QWORD *)v273.m128i_i64[0]);
            v233 = v273.m128i_u32[2] - 2LL;
            v238 = v273.m128i_i64[0] + 16;
          }
          else
          {
            v215 = 0;
            v205 = byte_3F871B3;
          }
          if ( v233 <= 5
            || (v107 = *(_QWORD *)v238, v253 = *(_QWORD *)v238, (v211 = sub_9CAD80((_QWORD *)v5, v107)) == 0) )
          {
LABEL_229:
            BYTE1(v270) = 1;
            v126 = "Invalid record";
LABEL_230:
            n[0] = (size_t)v126;
            LOBYTE(v270) = 3;
            sub_9C81F0(v261, v228, (__int64)n);
            v104 = v261[0] & 0xFFFFFFFFFFFFFFFELL;
            goto LABEL_173;
          }
          v108 = *(_QWORD *)(v238 + 8);
          v201 = v108;
          if ( (v108 & 2) != 0 )
          {
            v197 = v108 >> 2;
          }
          else
          {
            if ( *(_BYTE *)(v211 + 8) != 14 )
            {
              BYTE1(v270) = 1;
              v126 = "Invalid type for value";
              goto LABEL_230;
            }
            v197 = *(_DWORD *)(v211 + 8) >> 8;
            v253 = sub_9C2A90(v5, v253, 0);
            v211 = sub_9CAD80((_QWORD *)v5, v253);
            if ( !v211 )
            {
              BYTE1(v270) = 1;
              v126 = "Missing element type for old-style global";
              goto LABEL_230;
            }
          }
          v202 = 0;
          v200 = *(_QWORD *)(v238 + 24);
          v109 = (unsigned int)(v200 - 1);
          if ( (unsigned int)v109 <= 0x12 )
            v202 = dword_3F22240[v109];
          v110 = *(_QWORD *)(v238 + 32);
          v252 = 0;
          sub_9C88F0((__int64 *)n, v5, v110, &v252, 0);
          v111 = 0;
          v104 = n[0] & 0xFFFFFFFFFFFFFFFELL;
          if ( (n[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          {
            v63 = v261;
LABEL_150:
            v261[0] = v104 | 1;
            goto LABEL_76;
          }
          LOBYTE(v265[0]) = 0;
          v263 = (unsigned __int64)v265;
          v264 = 0;
          v112 = *(_QWORD *)(v238 + 40);
          if ( !v112 )
            goto LABEL_309;
          v113 = *(_QWORD *)(v5 + 480);
          v114 = v112 - 1;
          if ( v114 >= (*(_QWORD *)(v5 + 488) - v113) >> 5 )
          {
            n[0] = (size_t)"Invalid ID";
            LOWORD(v270) = 259;
LABEL_169:
            sub_9C81F0(v261, v228, (__int64)n);
            goto LABEL_170;
          }
          sub_2240AE0(&v263, v113 + 32 * v114);
          v111 = 0;
LABEL_309:
          if ( v202 - 7 <= 1 || v233 == 6 )
          {
            v199 = 0;
          }
          else
          {
            v178 = *(_QWORD *)(v238 + 48);
            v199 = v178;
            if ( (_DWORD)v178 != 1 )
            {
              v125 = (_DWORD)v178 == 2;
              LOBYTE(v178) = 0;
              if ( v125 )
                LODWORD(v178) = *(_QWORD *)(v238 + 48);
              v199 = v178;
            }
          }
          LODWORD(v157) = 0;
          if ( v233 > 7 )
          {
            v157 = *(_QWORD *)(v238 + 56);
            if ( (unsigned int)v157 >= 5 )
              LODWORD(v157) = 1;
          }
          v198 = 0;
          if ( v233 > 8 )
          {
            v158 = *(_QWORD *)(v238 + 64);
            if ( (_DWORD)v158 == 1 )
              v198 = 2;
            else
              v198 = (_DWORD)v158 == 2;
          }
          if ( v233 > 9 )
            v111 = *(_QWORD *)(v238 + 72) != 0;
          v195 = v157;
          v196 = v111;
          n[0] = (size_t)v205;
          LOWORD(v270) = 261;
          n[1] = v215;
          BYTE4(v261[0]) = 1;
          LODWORD(v261[0]) = v197;
          v159 = sub_BD2C40(88, unk_3F0FAE8);
          v160 = v159;
          if ( v159 )
          {
            v217 = v159;
            sub_B30000(v159, *(_QWORD *)(v5 + 440), v211, v201 & 1, v202, 0, n, 0, v195, v261[0], v196);
            v160 = v217;
          }
          if ( HIBYTE(v252) )
          {
            v219 = v160;
            sub_B2F770(v160, (unsigned __int8)v252);
            v160 = v219;
          }
          if ( v264 )
          {
            v220 = v160;
            sub_B31A00(v160, v263, v264);
            v160 = v220;
          }
          v161 = (16 * (v199 & 3)) | *(_BYTE *)(v160 + 32) & 0xCF;
          *(_BYTE *)(v160 + 32) = v161;
          v162 = v161 & 0xF;
          v163 = v198 & 3;
          if ( (unsigned int)(v162 - 7) <= 1 )
          {
            *(_BYTE *)(v160 + 33) |= 0x40u;
            *(_BYTE *)(v160 + 32) = (v163 << 6) | v161 & 0x3F;
            if ( v233 <= 0xA )
              goto LABEL_334;
LABEL_328:
            if ( v162 == 7 )
              goto LABEL_334;
LABEL_329:
            if ( v162 != 8 )
            {
              v164 = *(_QWORD *)(v238 + 80);
              v165 = v164;
              if ( (_DWORD)v164 != 1 && (_DWORD)v164 != 2 )
                v165 = 0;
              *(_BYTE *)(v160 + 33) = v165 & 3 | *(_BYTE *)(v160 + 33) & 0xFC;
            }
            goto LABEL_334;
          }
          if ( (v161 & 0x30) == 0 || (v161 & 0xF) == 9 )
          {
            *(_BYTE *)(v160 + 32) = *(_BYTE *)(v160 + 32) & 0x3F | (v163 << 6);
            if ( v233 > 0xA )
              goto LABEL_329;
          }
          else
          {
            *(_BYTE *)(v160 + 33) |= 0x40u;
            *(_BYTE *)(v160 + 32) = (v163 << 6) | v161 & 0x3F;
            if ( v233 > 0xA )
              goto LABEL_328;
          }
          if ( (_DWORD)v200 == 5 )
          {
            *(_BYTE *)(v160 + 33) = *(_BYTE *)(v160 + 33) & 0xFC | 1;
          }
          else if ( (_DWORD)v200 == 6 )
          {
            *(_BYTE *)(v160 + 33) = *(_BYTE *)(v160 + 33) & 0xFC | 2;
          }
LABEL_334:
          v218 = v160;
          LODWORD(v261[0]) = sub_9E2F80(v5, *(_QWORD *)(v160 + 8), (int *)&v253, 1);
          n[0] = v218;
          sub_9C9EA0((__int64 *)(v5 + 744), (__int64 *)n, v261);
          v166 = v218;
          v167 = *(_QWORD *)(v238 + 16);
          if ( (_DWORD)v167 )
          {
            n[0] = v218;
            v188 = *(__m128i **)(v5 + 1416);
            LODWORD(n[1]) = v167 - 1;
            if ( v188 == *(__m128i **)(v5 + 1424) )
            {
              sub_9D2740((const __m128i **)(v5 + 1408), v188, (const __m128i *)n);
              v166 = v218;
            }
            else
            {
              if ( v188 )
                *v188 = _mm_loadu_si128((const __m128i *)n);
              *(_QWORD *)(v5 + 1416) += 16LL;
            }
          }
          if ( v233 <= 0xB )
          {
            if ( v200 > 0xB || ((1LL << v200) & 0xC12) == 0 )
              goto LABEL_355;
            v179 = *(_DWORD *)(v5 + 872);
            v261[0] = v166;
            v180 = v5 + 848;
            if ( v179 )
            {
              v181 = 0;
              v182 = (v179 - 1) & (((unsigned int)v166 >> 9) ^ ((unsigned int)v166 >> 4));
              for ( i = 1; ; ++i )
              {
                v184 = (_QWORD *)(*(_QWORD *)(v5 + 856) + 8LL * v182);
                v185 = *v184;
                if ( v166 == *v184 )
                  break;
                if ( v185 == -4096 )
                {
                  if ( v181 )
                    v184 = v181;
                  ++*(_QWORD *)(v5 + 848);
                  n[0] = (size_t)v184;
                  v186 = *(_DWORD *)(v5 + 864) + 1;
                  if ( 4 * v186 < 3 * v179 )
                  {
                    if ( v179 - *(_DWORD *)(v5 + 868) - v186 <= v179 >> 3 )
                    {
                      v209 = v166;
                      sub_9E6990(v180, v179);
                      sub_9D28C0(v5 + 848, v261, n);
                      v166 = v209;
                    }
                    goto LABEL_396;
                  }
                  goto LABEL_416;
                }
                if ( v181 || v185 != -8192 )
                  v184 = v181;
                v181 = v184;
                v182 = (v179 - 1) & (i + v182);
              }
            }
            else
            {
              ++*(_QWORD *)(v5 + 848);
              n[0] = 0;
LABEL_416:
              v208 = v166;
              sub_9E6990(v180, 2 * v179);
              sub_9D28C0(v5 + 848, v261, n);
              v166 = v208;
LABEL_396:
              v187 = (_QWORD *)n[0];
              ++*(_DWORD *)(v5 + 864);
              if ( *v187 != -4096 )
                --*(_DWORD *)(v5 + 868);
              *v187 = v261[0];
            }
          }
          else
          {
            v168 = *(_QWORD *)(v238 + 88);
            if ( (_DWORD)v168 )
            {
              v169 = *(_QWORD *)(v5 + 824);
              if ( (unsigned int)v168 > (unsigned __int64)((*(_QWORD *)(v5 + 832) - v169) >> 3) )
              {
                BYTE1(v270) = 1;
                v170 = "Invalid global variable comdat ID";
LABEL_339:
                n[0] = (size_t)v170;
                LOBYTE(v270) = 3;
                goto LABEL_169;
              }
              v222 = v166;
              sub_B2F990(v166, *(_QWORD *)(v169 + 8LL * (unsigned int)(v168 - 1)));
              v166 = v222;
            }
            if ( v233 != 12 )
            {
              v189 = *(_QWORD *)(v5 + 1480);
              v190 = 0;
              v191 = (unsigned int)*(_QWORD *)(v238 + 96) - 1;
              if ( v191 < (*(_QWORD *)(v5 + 1488) - v189) >> 3 )
                v190 = *(_QWORD *)(v189 + 8 * v191);
              v223 = v166;
              n[0] = v190;
              v192 = sub_A74680(n);
              v166 = v223;
              *(_QWORD *)(v223 + 72) = v192;
LABEL_355:
              if ( v233 > 0xD )
                *(_BYTE *)(v166 + 33) = ((*(_DWORD *)(v238 + 104) == 1) << 6) | *(_BYTE *)(v166 + 33) & 0xBF;
            }
          }
          v173 = *(_BYTE *)(v166 + 32);
          v174 = v173 & 0xF;
          v175 = (v173 & 0xFu) - 7;
          if ( (unsigned int)v175 <= 1 || (v173 & 0x30) != 0 && (_BYTE)v174 != 9 )
            *(_BYTE *)(v166 + 33) |= 0x40u;
          if ( v233 > 0xF )
          {
            v221 = v166;
            sub_B30D10(v166, *(_QWORD *)(v5 + 376) + *(_QWORD *)(v238 + 112));
            v166 = v221;
          }
          if ( v233 > 0x10 )
          {
            v176 = *(_QWORD *)(v238 + 128);
            if ( v176 )
            {
              v224 = v166;
              v193 = sub_9C7FF0(v176);
              sub_B311F0(v224, v193);
              v166 = v224;
            }
          }
          if ( v233 > 0x11 )
          {
            v177 = *(_QWORD *)(v238 + 136);
            if ( v177 )
            {
              switch ( (int)v177 )
              {
                case 1:
                  v194 = 0;
                  break;
                case 2:
                  v194 = 1;
                  break;
                case 3:
                  v194 = 2;
                  break;
                case 4:
                  v194 = 3;
                  break;
                case 5:
                  v194 = 4;
                  break;
                default:
                  BYTE1(v270) = 1;
                  v170 = "Invalid global variable code model";
                  goto LABEL_339;
              }
              sub_B30310(v166, v194, v175, v174);
            }
          }
          v261[0] = 1;
LABEL_170:
          if ( (_QWORD *)v263 != v265 )
            j_j___libc_free_0(v263, v265[0] + 1LL);
          v104 = v261[0] & 0xFFFFFFFFFFFFFFFELL;
LABEL_173:
          if ( v104 )
            goto LABEL_149;
LABEL_174:
          v261[0] = 0;
          v91 = v261;
LABEL_175:
          sub_9C66B0(v91);
          v64 = v257;
          v70 = (v257 & 2) != 0;
          goto LABEL_95;
        case 8:
          sub_9D2B60(n, (__int64)v260);
          v90 = n[0] & 0xFFFFFFFFFFFFFFFELL;
          if ( (n[0] & 0xFFFFFFFFFFFFFFFELL) != 0
            || (n[0] = 0,
                sub_9C66B0((__int64 *)n),
                sub_9E6B60((__int64 *)n, v5, (__int64 *)v273.m128i_i64[0], v273.m128i_u32[2]),
                v91 = (__int64 *)n,
                v90 = n[0] & 0xFFFFFFFFFFFFFFFELL,
                (n[0] & 0xFFFFFFFFFFFFFFFELL) != 0) )
          {
            n[0] = v90 | 1;
            v35 = n;
            *a1 = 0;
            sub_9C6670(a1, n);
            sub_9C66B0((__int64 *)n);
            goto LABEL_77;
          }
          n[0] = 0;
          goto LABEL_175;
        case 9:
        case 14:
        case 18:
          v37 = (__int64 *)v273.m128i_i64[0];
          v243 = v273.m128i_u32[2];
          if ( *(_BYTE *)(v5 + 392) )
          {
            v38 = *(_QWORD *)(v273.m128i_i64[0] + 8);
            if ( *(_QWORD *)v273.m128i_i64[0] + v38 > *(_QWORD *)(v5 + 384) )
              goto LABEL_228;
            v37 = (__int64 *)(v273.m128i_i64[0] + 16);
            src = (const char *)(*(_QWORD *)(v5 + 376) + *(_QWORD *)v273.m128i_i64[0]);
            v243 = v273.m128i_u32[2] - 2LL;
          }
          else
          {
            v38 = 0;
            src = byte_3F871B3;
          }
          if ( v243 < (unsigned __int64)((_DWORD)v256 != 9) + 3 )
            goto LABEL_228;
          v39 = *v37;
          v213 = v38;
          v225 = v256;
          v231 = v37;
          LODWORD(v261[0]) = *v37;
          v40 = sub_9CAD80((_QWORD *)v5, v39);
          v41 = v231;
          v42 = v213;
          v43 = v40;
          if ( !v40 )
          {
LABEL_228:
            n[0] = (size_t)"Invalid record";
            LOWORD(v270) = 259;
            goto LABEL_74;
          }
          if ( v225 == 9 )
          {
            if ( *(_BYTE *)(v40 + 8) != 14 )
            {
              n[0] = (size_t)"Invalid type for value";
              LOWORD(v270) = 259;
              goto LABEL_74;
            }
            v210 = *(_DWORD *)(v40 + 8) >> 8;
            LODWORD(v261[0]) = sub_9C2A90(v5, v261[0], 0);
            v148 = sub_9CAD80((_QWORD *)v5, LODWORD(v261[0]));
            v42 = v213;
            v41 = v231;
            v43 = v148;
            if ( !v148 )
            {
              n[0] = (size_t)"Missing element type for old-style indirect symbol";
              LOWORD(v270) = 259;
              goto LABEL_74;
            }
            v149 = 3;
            v227 = 1;
            v214 = v231[1];
            v232 = v231[2];
          }
          else
          {
            v44 = v231[1];
            v214 = v231[2];
            v210 = v44;
            v232 = v231[3];
            if ( v225 != 14 )
            {
              n[1] = v42;
              LOWORD(v270) = 261;
              v45 = (unsigned int)(v232 - 1);
              v46 = *(_QWORD *)(v5 + 440);
              n[0] = (size_t)src;
              v47 = 0;
              if ( (unsigned int)v45 <= 0x12 )
                v47 = (unsigned int)dword_3F22240[v45];
              v226 = v41;
              v48 = sub_B30730(v43, v44, v47, n, 0, v46);
              v49 = v226;
              v50 = v48;
              v51 = *(_BYTE *)(v48 + 32) & 0xF;
              v52 = v51 - 7;
              if ( v243 == 4 )
              {
                v55 = 4;
                goto LABEL_67;
              }
              v227 = 2;
              v53 = 0;
              v54 = 4;
              goto LABEL_55;
            }
            v227 = 2;
            v149 = 4;
          }
          n[1] = v42;
          v150 = 0;
          LOWORD(v270) = 261;
          v151 = *(_QWORD *)(v5 + 440);
          n[0] = (size_t)src;
          v152 = (unsigned int)(v232 - 1);
          if ( (unsigned int)v152 <= 0x12 )
            v150 = (unsigned int)dword_3F22240[v152];
          v204 = v149;
          v207 = v41;
          v153 = sub_B30580(v43, v210, v150, n, v151);
          v56 = v204;
          v49 = v207;
          v53 = 1;
          v50 = v153;
          v55 = v204;
          v54 = v204;
          v51 = *(_BYTE *)(v153 + 32) & 0xF;
          v52 = v51 - 7;
          if ( v243 == v204 )
            goto LABEL_294;
LABEL_55:
          v55 = v227 + 3;
          if ( v52 > 1 )
          {
            v154 = v49[v54];
            v155 = v154;
            if ( (_DWORD)v154 != 1 && (_DWORD)v154 != 2 )
              v155 = 0;
            v156 = *(_BYTE *)(v50 + 32) & 0xCF | (16 * (v155 & 3));
            *(_BYTE *)(v50 + 32) = v156;
            if ( (v156 & 0x30) != 0 && v51 != 9 )
              *(_BYTE *)(v50 + 33) |= 0x40u;
          }
          if ( !v53 )
            goto LABEL_67;
          v56 = v55;
          if ( v243 != v55 )
          {
            v55 = v227 + 4;
            if ( v52 > 1 )
            {
              v171 = v49[v56];
              v172 = v171;
              if ( (_DWORD)v171 != 1 && (_DWORD)v171 != 2 )
                v172 = 0;
              *(_BYTE *)(v50 + 33) = v172 & 3 | *(_BYTE *)(v50 + 33) & 0xFC;
            }
            v56 = v55;
            goto LABEL_60;
          }
LABEL_294:
          if ( v52 <= 1 )
            goto LABEL_295;
          if ( (_DWORD)v232 == 5 )
          {
            *(_BYTE *)(v50 + 33) = *(_BYTE *)(v50 + 33) & 0xFC | 1;
            goto LABEL_295;
          }
          if ( (_DWORD)v232 == 6 )
          {
            *(_BYTE *)(v50 + 33) = *(_BYTE *)(v50 + 33) & 0xFC | 2;
LABEL_295:
            v57 = v55;
            goto LABEL_69;
          }
LABEL_60:
          v57 = v55 + 1;
          if ( v243 == v56 )
            goto LABEL_295;
          v58 = v49[v56];
          if ( (unsigned int)v58 >= 5 )
            LOBYTE(v58) = 1;
          *(_BYTE *)(v50 + 33) = *(_BYTE *)(v50 + 33) & 0xE3 | (4 * (v58 & 7));
          if ( v243 != (unsigned int)v57 )
          {
            v59 = v49[(unsigned int)v57];
            v55 += 2;
            v60 = 2;
            if ( (_DWORD)v59 != 1 )
              v60 = (_DWORD)v59 == 2;
            *(_BYTE *)(v50 + 32) = (v60 << 6) | *(_BYTE *)(v50 + 32) & 0x3F;
LABEL_67:
            v57 = v55 + 1;
            if ( v243 != v55 )
            {
              *(_BYTE *)(v50 + 33) = ((LODWORD(v49[v55]) == 1) << 6) | *(_BYTE *)(v50 + 33) & 0xBF;
              goto LABEL_69;
            }
            goto LABEL_295;
          }
LABEL_69:
          if ( (*(_BYTE *)(v50 + 32) & 0xFu) - 7 <= 1
            || (*(_BYTE *)(v50 + 32) & 0x30) != 0 && (*(_BYTE *)(v50 + 32) & 0xF) != 9 )
          {
            *(_BYTE *)(v50 + 33) |= 0x40u;
          }
          v61 = (unsigned int)(v57 + 1);
          if ( v243 <= v61 )
            goto LABEL_223;
          v62 = v49[v57];
          if ( (unsigned __int64)(v62 + v49[v61]) <= *(_QWORD *)(v5 + 384) )
          {
            v244 = (void *)v50;
            sub_B30D10(v50, *(_QWORD *)(v5 + 376) + v62);
            v50 = (__int64)v244;
LABEL_223:
            v245 = (void *)v50;
            LODWORD(v263) = sub_9E2F80(v5, *(_QWORD *)(v50 + 8), (int *)v261, 1);
            n[0] = (size_t)v245;
            sub_9C9EA0((__int64 *)(v5 + 744), (__int64 *)n, &v263);
            v124 = *(__m128i **)(v5 + 1440);
            v125 = v124 == *(__m128i **)(v5 + 1448);
            n[0] = (size_t)v245;
            LODWORD(n[1]) = v214;
            if ( v125 )
            {
              sub_9D2970((const __m128i **)(v5 + 1432), v124, (const __m128i *)n);
            }
            else
            {
              if ( v124 )
                *v124 = _mm_loadu_si128((const __m128i *)n);
              *(_QWORD *)(v5 + 1440) += 16LL;
            }
            goto LABEL_227;
          }
          n[0] = (size_t)"Malformed partition, too large.";
          LOWORD(v270) = 259;
LABEL_74:
          sub_9C81F0((__int64 *)&v263, v5 + 8, (__int64)n);
          v63 = (__int64 *)&v263;
          if ( (v263 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          {
            v263 = v263 & 0xFFFFFFFFFFFFFFFELL | 1;
LABEL_76:
            v35 = (size_t *)v63;
            *a1 = 0;
            v247 = v63;
            sub_9C6670(a1, v63);
            sub_9C66B0(v247);
            goto LABEL_77;
          }
LABEL_227:
          v263 = 0;
          v91 = (__int64 *)&v263;
          goto LABEL_175;
        case 11:
          v263 = (unsigned __int64)v265;
          v264 = 0;
          LOBYTE(v265[0]) = 0;
          v105 = sub_9C57B0((__int64 *)v273.m128i_i64[0], v273.m128i_u32[2], (__int64 *)&v263);
          v78 = (__int64 *)&v263;
          if ( v105 )
            goto LABEL_299;
          v106 = *(__int64 **)(v5 + 512);
          if ( v106 == *(__int64 **)(v5 + 520) )
          {
            sub_8FD760((__m128i **)(v5 + 504), *(const __m128i **)(v5 + 512), (__int64)&v263);
          }
          else
          {
            if ( v106 )
            {
              *v106 = (__int64)(v106 + 2);
              sub_9C36C0(v106, (_BYTE *)v263, v263 + v264);
              v106 = *(__int64 **)(v5 + 512);
            }
            *(_QWORD *)(v5 + 512) = v106 + 4;
          }
          goto LABEL_116;
        case 12:
          v92 = *(_BYTE *)(v5 + 392);
          v93 = (_QWORD *)v273.m128i_i64[0];
          v94 = v5 + 8;
          v95 = v273.m128i_u32[2];
          if ( v92 )
          {
            if ( *(_QWORD *)v273.m128i_i64[0] + *(_QWORD *)(v273.m128i_i64[0] + 8) > *(_QWORD *)(v5 + 384) )
              goto LABEL_252;
            v96 = (const char *)(*(_QWORD *)(v5 + 376) + *(_QWORD *)v273.m128i_i64[0]);
            v95 = v273.m128i_u32[2] - 2LL;
            v93 = (_QWORD *)(v273.m128i_i64[0] + 16);
          }
          else
          {
            v96 = byte_3F871B3;
          }
          v97 = (unsigned __int64)v96;
          if ( !v95 )
          {
LABEL_252:
            n[0] = (size_t)"Invalid record";
            LOWORD(v270) = 259;
            sub_9C81F0(v261, v94, (__int64)n);
            goto LABEL_148;
          }
          v98 = *v93;
          v264 = 0;
          LOBYTE(v265[0]) = 0;
          v99 = v98 - 2;
          v100 = v98 - 1;
          v101 = v99 < 4;
          v102 = 0;
          if ( v101 )
            v102 = v100;
          v263 = (unsigned __int64)v265;
          if ( v92 )
            goto LABEL_240;
          if ( v95 == 1 )
          {
            BYTE1(v270) = 1;
            v103 = "Invalid record";
            goto LABEL_145;
          }
          if ( (unsigned int)v93[1] > (unsigned __int64)(v95 - 2) )
          {
            BYTE1(v270) = 1;
            v103 = "Comdat name size too large";
LABEL_145:
            n[0] = (size_t)v103;
            LOBYTE(v270) = 3;
            sub_9C81F0(v261, v94, (__int64)n);
            goto LABEL_146;
          }
          v229 = v93[1];
          v234 = v93;
          sub_2240E30(&v263, (unsigned int)v229);
          if ( (_DWORD)v229 )
          {
            v212 = v5;
            v139 = 2;
            v206 = v18;
            v140 = v234;
            v203 = v13;
            v141 = v229 + 2;
            v142 = v265;
            v216 = m128i_i64;
            do
            {
              v143 = v263;
              v144 = v264;
              v145 = 15;
              v146 = v140[v139];
              if ( (_BYTE *)v263 != v142 )
                v145 = v265[0];
              v147 = v264 + 1;
              if ( v264 + 1 > v145 )
              {
                v230 = v142;
                v235 = v140[v139];
                v239 = v264;
                sub_2240BB0(&v263, v264, 0, 0, 1);
                v143 = v263;
                v142 = v230;
                LOBYTE(v146) = v235;
                v144 = v239;
              }
              *(_BYTE *)(v143 + v144) = v146;
              ++v139;
              v264 = v147;
              *(_BYTE *)(v263 + v144 + 1) = 0;
            }
            while ( v141 != v139 );
            m128i_i64 = v216;
            v5 = v212;
            v18 = v206;
            v13 = v203;
          }
          v97 = v263;
LABEL_240:
          v129 = sub_BAA410(*(_QWORD *)(v5 + 440), v97);
          n[0] = v129;
          *(_DWORD *)(v129 + 8) = v102;
          v130 = *(_BYTE **)(v5 + 832);
          if ( v130 == *(_BYTE **)(v5 + 840) )
          {
            sub_9CC430(v5 + 824, v130, n);
          }
          else
          {
            if ( v130 )
              *(_QWORD *)v130 = v129;
            *(_QWORD *)(v5 + 832) += 8LL;
          }
          v261[0] = 1;
LABEL_146:
          if ( (_QWORD *)v263 != v265 )
            j_j___libc_free_0(v263, v265[0] + 1LL);
LABEL_148:
          v104 = v261[0] & 0xFFFFFFFFFFFFFFFELL;
          if ( (v261[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          {
LABEL_149:
            v63 = v261;
            goto LABEL_150;
          }
          goto LABEL_174;
        case 13:
          if ( !v273.m128i_i32[2] )
            goto LABEL_130;
          *(_QWORD *)(v5 + 472) = *(_QWORD *)v273.m128i_i64[0] - 1LL;
LABEL_80:
          v273.m128i_i32[2] = 0;
          v64 = v257;
          goto LABEL_81;
        case 16:
          n[0] = (size_t)&v269;
          n[1] = 0;
          v268 = 128;
          if ( (unsigned __int8)sub_9C3E90(v273.m128i_i64[0], v273.m128i_u32[2], 0, n) )
          {
            v35 = (size_t *)(v5 + 8);
            v263 = (unsigned __int64)"Invalid record";
            v266 = 259;
            sub_9C81F0(a1, v5 + 8, (__int64)&v263);
            if ( (char *)n[0] != &v269 )
              _libc_free(n[0], v35);
LABEL_77:
            sub_9CE2A0(&v256);
            goto LABEL_26;
          }
          v72 = (_QWORD *)n[0];
          v73 = *(_QWORD **)(v5 + 440);
          v263 = (unsigned __int64)v265;
          v236 = v73;
          sub_9C2D70((__int64 *)&v263, (_BYTE *)n[0], n[0] + n[1]);
          v17 = v236;
          v74 = (_BYTE *)v236[25];
          if ( (_QWORD *)v263 == v265 )
          {
            v137 = v264;
            if ( v264 )
            {
              if ( v264 == 1 )
              {
                *v74 = v265[0];
              }
              else
              {
                v72 = v265;
                memcpy(v74, v265, v264);
                v17 = v236;
              }
              v137 = v264;
              v74 = (_BYTE *)v236[25];
            }
            v17[26] = v137;
            v74[v137] = 0;
            v74 = (_BYTE *)v263;
            goto LABEL_104;
          }
          v72 = (_QWORD *)v265[0];
          v75 = v264;
          if ( v74 == (_BYTE *)(v236 + 27) )
          {
            v236[25] = v263;
            v236[26] = v75;
            v236[27] = v72;
          }
          else
          {
            v76 = v236[27];
            v236[25] = v263;
            v236[26] = v75;
            v236[27] = v72;
            if ( v74 )
            {
              v263 = (unsigned __int64)v74;
              v265[0] = v76;
LABEL_104:
              v264 = 0;
              *v74 = 0;
              if ( (_QWORD *)v263 != v265 )
              {
                v72 = (_QWORD *)(v265[0] + 1LL);
                j_j___libc_free_0(v263, v265[0] + 1LL);
              }
              if ( (char *)n[0] != &v269 )
                _libc_free(n[0], v72);
              goto LABEL_94;
            }
          }
          v263 = (unsigned __int64)v265;
          v74 = v265;
          goto LABEL_104;
        default:
          goto LABEL_80;
      }
    }
    switch ( HIDWORD(v254) )
    {
      case 0:
        sub_9D23D0((__int64 *)n, (_QWORD *)(v5 + 8));
        v36 = n[0] & 0xFFFFFFFFFFFFFFFELL;
        if ( (n[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_187;
        goto LABEL_39;
      case 9:
        sub_9D6F70((__int64 *)n, v5);
        v36 = n[0] & 0xFFFFFFFFFFFFFFFELL;
        if ( (n[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_187;
        goto LABEL_39;
      case 0xA:
        sub_9D5A00((__int64 *)n, (unsigned __int64 *)v5);
        v36 = n[0] & 0xFFFFFFFFFFFFFFFELL;
        if ( (n[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_187;
        goto LABEL_39;
      case 0xB:
        sub_9E3720((__int64 *)n, (_QWORD *)v5);
        v120 = n[0] & 0xFFFFFFFFFFFFFFFELL;
        if ( (n[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
        {
          v35 = n;
          *a1 = 0;
          n[0] = v120 | 1;
          sub_9C6670(a1, n);
          sub_9C66B0((__int64 *)n);
          goto LABEL_26;
        }
        n[0] = 0;
        sub_9C66B0((__int64 *)n);
        sub_9DD2C0((__int64 *)n, v5);
        v36 = n[0] & 0xFFFFFFFFFFFFFFFELL;
        if ( (n[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
        {
LABEL_187:
          v35 = (size_t *)a1;
          n[0] = 0;
          *a1 = v36 | 1;
          sub_9C66B0((__int64 *)n);
          goto LABEL_26;
        }
        goto LABEL_39;
      case 0xC:
        sub_9D2B60(n, (__int64)v260);
        v121 = n[0] & 0xFFFFFFFFFFFFFFFELL;
        if ( (n[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_206;
        n[0] = 0;
        sub_9C66B0((__int64 *)n);
        if ( *(_BYTE *)(v5 + 1632) )
          goto LABEL_213;
        v131 = *(__int64 **)(v5 + 1584);
        v132 = *(__int64 **)(v5 + 1576);
        if ( v132 != v131 )
        {
          for ( j = v131 - 1; j > v132; j[1] = v134 )
          {
            v134 = *v132;
            v135 = *j;
            ++v132;
            --j;
            *(v132 - 1) = v135;
          }
        }
        sub_9E01F0((__int64 *)n, (_QWORD *)v5);
        v121 = n[0] & 0xFFFFFFFFFFFFFFFELL;
        if ( (n[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_206;
        n[0] = 0;
        sub_9C66B0((__int64 *)n);
        *(_BYTE *)(v5 + 1632) = 1;
LABEL_213:
        v123 = *(_QWORD *)(v5 + 472);
        if ( !v123 )
          goto LABEL_217;
        if ( *(_BYTE *)(v5 + 464) )
        {
          sub_9CE5C0((__int64 *)n, m128i_i64, v123, v122);
          v121 = n[0] & 0xFFFFFFFFFFFFFFFELL;
          if ( (n[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
            goto LABEL_206;
LABEL_39:
          n[0] = 0;
          sub_9C66B0((__int64 *)n);
        }
        else
        {
          sub_9DE180((__int64 *)n, v5, v123, v122);
          v121 = n[0] & 0xFFFFFFFFFFFFFFFELL;
          if ( (n[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
            goto LABEL_206;
          n[0] = 0;
          sub_9C66B0((__int64 *)n);
          *(_BYTE *)(v5 + 464) = 1;
LABEL_217:
          sub_9DDE80((__int64 *)n, v5);
          v121 = n[0] & 0xFFFFFFFFFFFFFFFELL;
          if ( (n[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          {
LABEL_206:
            n[0] = v121 | 1;
            *a1 = 0;
            v35 = n;
            sub_9C6670(a1, n);
            sub_9C66B0((__int64 *)n);
            goto LABEL_26;
          }
          n[0] = 0;
          sub_9C66B0((__int64 *)n);
          if ( *(_BYTE *)(v5 + 464) )
          {
            *(_QWORD *)(v5 + 448) = 8LL * *(_QWORD *)(v5 + 48) - *(unsigned int *)(v5 + 64);
            goto LABEL_34;
          }
        }
        goto LABEL_40;
      case 0xE:
        if ( *(_BYTE *)(v5 + 464) )
          goto LABEL_184;
        v35 = (size_t *)v5;
        sub_9DE180((__int64 *)n, v5, 0, v20);
        v119 = n[0] & 0xFFFFFFFFFFFFFFFELL;
        if ( (n[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_185;
        n[0] = 0;
        sub_9C66B0((__int64 *)n);
        *(_BYTE *)(v5 + 464) = 1;
        goto LABEL_40;
      case 0xF:
        if ( v246 )
        {
          v21 = *(unsigned int *)(v5 + 64);
          v117 = *(_BYTE **)(v5 + 1680);
          v118 = (_BYTE *)(8LL * *(_QWORD *)(v5 + 48) - v21);
          v263 = (unsigned __int64)v118;
          if ( v117 == *(_BYTE **)(v5 + 1688) )
          {
            sub_9CA200(v5 + 1672, v117, &v263);
          }
          else
          {
            if ( v117 )
            {
              *(_QWORD *)v117 = v118;
              v117 = *(_BYTE **)(v5 + 1680);
            }
            *(_QWORD *)(v5 + 1680) = v117 + 8;
          }
LABEL_184:
          v35 = (size_t *)m128i_i64;
          sub_9CE5C0((__int64 *)n, m128i_i64, v21, v20);
          v119 = n[0] & 0xFFFFFFFFFFFFFFFELL;
          if ( (n[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          {
LABEL_185:
            n[0] = 0;
            *a1 = v119 | 1;
            sub_9C66B0((__int64 *)n);
            goto LABEL_26;
          }
        }
        else
        {
          v35 = (size_t *)(v5 + 808);
          sub_A14940(n, v5 + 808, 1);
          v119 = n[0] & 0xFFFFFFFFFFFFFFFELL;
          if ( (n[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
            goto LABEL_185;
        }
        goto LABEL_39;
      case 0x11:
        v35 = (size_t *)m128i_i64;
        sub_A4DCE0(n, m128i_i64, 17, 0);
        v115 = n[0] & 0xFFFFFFFFFFFFFFFELL;
        if ( (n[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
        {
          n[0] = 0;
          sub_9C66B0((__int64 *)n);
          v35 = (size_t *)v5;
          sub_9D8800((__int64 *)&v263, v5, v127, v128);
          v116 = v263 & 0xFFFFFFFFFFFFFFFELL;
          if ( (v263 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            goto LABEL_178;
          v263 = 0;
          sub_9C66B0((__int64 *)&v263);
LABEL_40:
          if ( (v255 & 2) != 0 )
            goto LABEL_208;
          if ( (v255 & 1) != 0 )
          {
            if ( v254 )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v254 + 8LL))(v254);
          }
          continue;
        }
        v248 = n[0] & 0xFFFFFFFFFFFFFFFELL;
        n[0] = 0;
        v263 = v115 | 1;
        sub_9C66B0((__int64 *)n);
        v116 = v248;
LABEL_178:
        v263 = 0;
        *a1 = v116 | 1;
        sub_9C66B0((__int64 *)&v263);
LABEL_26:
        if ( (v255 & 2) != 0 )
LABEL_208:
          sub_9CEF10(v18);
        if ( (v255 & 1) != 0 && v254 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v254 + 8LL))(v254);
        if ( (_QWORD *)v258[0] != v259 )
        {
          v35 = (size_t *)(v259[0] + 1LL);
          j_j___libc_free_0(v258[0], v259[0] + 1LL);
        }
        if ( (void (__fastcall **)(_QWORD, _QWORD, _QWORD))v273.m128i_i64[0] != &v274 )
          _libc_free(v273.m128i_i64[0], v35);
        return a1;
      case 0x12:
        sub_9DF3F0((__int64 *)n, (__int64 *)v5);
        v36 = n[0] & 0xFFFFFFFFFFFFFFFELL;
        if ( (n[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_187;
        goto LABEL_39;
      case 0x15:
        v35 = (size_t *)v5;
        sub_9CFE90((__int64 *)n, v5);
        v119 = n[0] & 0xFFFFFFFFFFFFFFFELL;
        if ( (n[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_185;
        goto LABEL_39;
      case 0x16:
        v35 = (size_t *)(v5 + 808);
        sub_A09F60(n, v5 + 808);
        v119 = n[0] & 0xFFFFFFFFFFFFFFFELL;
        if ( (n[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_185;
        goto LABEL_39;
      case 0x1A:
        sub_9CF9C0((__int64 *)n, (_QWORD *)v5);
        v36 = n[0] & 0xFFFFFFFFFFFFFFFELL;
        if ( (n[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_187;
        goto LABEL_39;
      default:
        sub_9CE5C0((__int64 *)n, m128i_i64, HIDWORD(v254), v20);
        v36 = n[0] & 0xFFFFFFFFFFFFFFFELL;
        if ( (n[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
          goto LABEL_39;
        goto LABEL_187;
    }
  }
}
