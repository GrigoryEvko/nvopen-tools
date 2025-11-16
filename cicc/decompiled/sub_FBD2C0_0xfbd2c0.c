// Function: sub_FBD2C0
// Address: 0xfbd2c0
//
__int64 __fastcall sub_FBD2C0(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5, char a6, __int64 a7)
{
  __int64 v7; // r9
  __int64 v8; // rbx
  __int64 v9; // r15
  __int64 v10; // rdx
  __m128i *v11; // r12
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdi
  __int64 *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rax
  __int64 v23; // r13
  _QWORD *v24; // rdi
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rax
  _BYTE *v28; // r15
  unsigned int v29; // r14d
  _BYTE *v30; // r13
  _BYTE *v31; // rdi
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rax
  __int64 v36; // r14
  __int64 v37; // r14
  __int64 v38; // rdx
  __int64 *v39; // r12
  __int64 *v40; // rbx
  __int64 *v41; // rdx
  __int64 *v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  __m128i v47; // xmm3
  __m128i v48; // xmm5
  __m128i v49; // xmm7
  __int64 v50; // r13
  __int64 v51; // r12
  __m128i v52; // xmm1
  const char **v53; // rdi
  __int64 v54; // rdx
  __int64 v55; // rsi
  __int64 v56; // r12
  unsigned __int8 v57; // r13
  __int64 v58; // rdi
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 *v61; // r14
  __int64 v62; // r12
  __int64 *v63; // r13
  _QWORD *v64; // rdi
  __int64 v65; // r10
  __int64 v66; // r15
  __int64 v67; // rbx
  __int64 v68; // rax
  __int64 v69; // rbx
  __int64 v70; // rax
  __int64 v71; // rsi
  __int64 v72; // r14
  __int64 v73; // r12
  __int64 *v74; // rbx
  __int64 *v75; // rdx
  __int64 v76; // rax
  __int64 v77; // r13
  __int64 *v78; // r14
  _QWORD *v79; // rax
  int v80; // r15d
  __int64 v81; // r12
  char v82; // r13
  unsigned __int64 v83; // rax
  __int64 v84; // r15
  __int64 *v85; // rbx
  unsigned __int64 v86; // rax
  __int64 v87; // r15
  unsigned __int64 v88; // rax
  __int64 v89; // r15
  unsigned __int64 v90; // rax
  __int64 *v91; // rbx
  __int64 v92; // r12
  unsigned __int64 v93; // rax
  unsigned __int8 *v94; // r15
  __int64 v95; // rdx
  unsigned __int8 v96; // dl
  char v97; // dh
  char v98; // al
  __int64 v99; // rcx
  char *v100; // rax
  __int64 v101; // r13
  __int64 v102; // r12
  __int64 v103; // r14
  __int64 v104; // rsi
  __int64 v105; // rax
  __int64 v106; // rax
  __int64 v107; // rbx
  int v108; // esi
  int v109; // r11d
  char *v110; // rdi
  unsigned int v111; // edx
  char *v112; // rax
  __int64 v113; // rcx
  __int64 v114; // rbx
  __int64 v115; // r8
  __int64 v116; // r9
  __int64 v117; // rdx
  __int64 v118; // rax
  __int64 v119; // rdi
  unsigned int v120; // esi
  __int64 v121; // r10
  __int64 v122; // rcx
  __int64 v123; // rcx
  __int64 v124; // r9
  int v125; // esi
  int v126; // r10d
  char *v127; // rcx
  unsigned int v128; // edx
  char *v129; // rax
  __int64 v130; // rdi
  __int64 v131; // rsi
  __int64 v132; // rdx
  __int64 v133; // r8
  __int64 *v134; // r9
  __int64 v135; // rax
  __int64 v136; // rcx
  __int64 v137; // r10
  const char *v138; // rax
  __int64 v139; // rdx
  __int64 v140; // r14
  __int64 v141; // rsi
  __int64 v142; // rax
  unsigned __int64 v143; // r13
  __int64 v144; // rcx
  __int64 v145; // r8
  __int64 v146; // r9
  const char *v147; // rsi
  __int64 v148; // r12
  __int64 v149; // rcx
  __int64 v150; // r8
  __int64 v151; // r9
  const char *v152; // rsi
  __int64 v153; // rsi
  unsigned __int8 *v154; // rsi
  _QWORD *v155; // rax
  __int64 v156; // rdx
  __int32 v157; // edx
  __int32 v158; // edx
  __int64 v159; // rcx
  __int64 v160; // r12
  unsigned __int64 v161; // rax
  __int64 v162; // r12
  unsigned __int64 v163; // rax
  _BYTE *v164; // [rsp+8h] [rbp-478h]
  __int64 **v165; // [rsp+10h] [rbp-470h]
  __int64 *v166; // [rsp+18h] [rbp-468h]
  __int64 v167; // [rsp+20h] [rbp-460h]
  __int64 v168; // [rsp+28h] [rbp-458h]
  __int64 v169; // [rsp+30h] [rbp-450h]
  __m128i *v170; // [rsp+40h] [rbp-440h]
  int v171; // [rsp+40h] [rbp-440h]
  __int64 v172; // [rsp+40h] [rbp-440h]
  _BYTE *v173; // [rsp+40h] [rbp-440h]
  unsigned int v174; // [rsp+40h] [rbp-440h]
  __int64 v175; // [rsp+40h] [rbp-440h]
  __int64 v176; // [rsp+48h] [rbp-438h]
  __int64 v178; // [rsp+50h] [rbp-430h]
  __int64 v179; // [rsp+58h] [rbp-428h]
  __int64 v180; // [rsp+58h] [rbp-428h]
  const char *v181; // [rsp+60h] [rbp-420h]
  __int64 v182; // [rsp+68h] [rbp-418h]
  __int64 v183; // [rsp+70h] [rbp-410h]
  __int64 *v184; // [rsp+70h] [rbp-410h]
  unsigned __int64 *v185; // [rsp+70h] [rbp-410h]
  __int64 *v187; // [rsp+78h] [rbp-408h]
  __int64 v189; // [rsp+98h] [rbp-3E8h] BYREF
  __m128i v190; // [rsp+A0h] [rbp-3E0h] BYREF
  __m128i v191; // [rsp+B0h] [rbp-3D0h] BYREF
  _BYTE v192[16]; // [rsp+C0h] [rbp-3C0h] BYREF
  void (__fastcall *v193)(_BYTE *, _BYTE *, __int64); // [rsp+D0h] [rbp-3B0h]
  unsigned __int8 (__fastcall *v194)(_BYTE *); // [rsp+D8h] [rbp-3A8h]
  __m128i v195; // [rsp+E0h] [rbp-3A0h] BYREF
  __m128i v196; // [rsp+F0h] [rbp-390h] BYREF
  _BYTE v197[16]; // [rsp+100h] [rbp-380h] BYREF
  void (__fastcall *v198)(_BYTE *, _BYTE *, __int64); // [rsp+110h] [rbp-370h]
  __m128i v199; // [rsp+120h] [rbp-360h]
  __m128i v200; // [rsp+130h] [rbp-350h]
  _BYTE v201[16]; // [rsp+140h] [rbp-340h] BYREF
  void (__fastcall *v202)(_BYTE *, _BYTE *, __int64); // [rsp+150h] [rbp-330h]
  __m128i v203; // [rsp+160h] [rbp-320h] BYREF
  __m128i v204; // [rsp+170h] [rbp-310h] BYREF
  _BYTE v205[16]; // [rsp+180h] [rbp-300h] BYREF
  void (__fastcall *v206)(_BYTE *, _BYTE *, __int64); // [rsp+190h] [rbp-2F0h]
  __m128i v207; // [rsp+1A0h] [rbp-2E0h] BYREF
  __m128i v208; // [rsp+1B0h] [rbp-2D0h] BYREF
  _BYTE v209[16]; // [rsp+1C0h] [rbp-2C0h] BYREF
  void (__fastcall *v210)(_BYTE *, _BYTE *, __int64); // [rsp+1D0h] [rbp-2B0h]
  __m128i v211; // [rsp+1E0h] [rbp-2A0h] BYREF
  __m128i v212; // [rsp+1F0h] [rbp-290h]
  _BYTE v213[16]; // [rsp+200h] [rbp-280h] BYREF
  void (__fastcall *v214)(_BYTE *, _BYTE *, __int64); // [rsp+210h] [rbp-270h]
  unsigned __int8 (__fastcall *v215)(_BYTE *, __int64); // [rsp+218h] [rbp-268h]
  _BYTE v216[16]; // [rsp+240h] [rbp-240h] BYREF
  void (__fastcall *v217)(_BYTE *, _BYTE *, __int64); // [rsp+250h] [rbp-230h]
  __int64 v218; // [rsp+260h] [rbp-220h] BYREF
  __int64 v219; // [rsp+268h] [rbp-218h]
  __int64 v220; // [rsp+270h] [rbp-210h]
  unsigned int v221; // [rsp+278h] [rbp-208h]
  __int64 v222; // [rsp+280h] [rbp-200h]
  _BYTE *v223; // [rsp+290h] [rbp-1F0h]
  __int64 v224; // [rsp+298h] [rbp-1E8h]
  _BYTE v225[144]; // [rsp+2A0h] [rbp-1E0h] BYREF
  char *v226; // [rsp+330h] [rbp-150h] BYREF
  __int64 v227; // [rsp+338h] [rbp-148h]
  char *v228; // [rsp+340h] [rbp-140h] BYREF
  __int64 v229; // [rsp+348h] [rbp-138h]
  _QWORD v230[4]; // [rsp+350h] [rbp-130h] BYREF
  __int16 v231; // [rsp+370h] [rbp-110h]

  v7 = *(_QWORD *)(a1 - 96);
  v223 = v225;
  v8 = *(_QWORD *)(a1 + 40);
  v218 = 0;
  v219 = 1;
  v220 = -4096;
  v222 = -4096;
  v224 = 0x200000000LL;
  v189 = v7;
  v182 = a2;
  v181 = (const char *)a3;
  v179 = a4;
  v183 = 0;
  if ( *(_BYTE *)v7 == 84 )
  {
    v58 = *(_QWORD *)(v7 + 40);
    if ( v58 == v8 )
    {
      v59 = *(_DWORD *)(v7 + 4) & 0x7FFFFFF;
      if ( (_DWORD)v59 == 1 )
      {
        a2 = 0;
        v29 = 257;
        sub_F34590(v58, 0);
        v28 = v223;
        v30 = &v223[72 * (unsigned int)v224];
        goto LABEL_26;
      }
      v60 = 4 * v59;
      if ( (*(_BYTE *)(v7 + 7) & 0x40) != 0 )
      {
        v61 = *(__int64 **)(v7 - 8);
        v184 = &v61[v60];
      }
      else
      {
        v184 = (__int64 *)v7;
        v61 = (__int64 *)(v7 - v60 * 8);
      }
      if ( v61 == v184 )
      {
        v30 = v225;
        goto LABEL_33;
      }
      v172 = v8;
      v62 = v7;
      v63 = v61;
      do
      {
        if ( *(_BYTE *)*v63 == 17 )
        {
          v203.m128i_i64[0] = *v63;
          v66 = sub_FBCEF0((__int64)&v218, v203.m128i_i64, a3, a4, a5, v7);
          v67 = *(_QWORD *)(*(_QWORD *)(v62 - 8)
                          + 32LL * *(unsigned int *)(v62 + 72)
                          + 8LL * (unsigned int)(((__int64)v63 - *(_QWORD *)(v62 - 8)) >> 5));
          v211.m128i_i64[0] = v67;
          if ( *(_DWORD *)(v66 + 16) )
          {
            a2 = v66;
            sub_D6CB10((__int64)&v226, v66, v211.m128i_i64);
            if ( LOBYTE(v230[0]) )
            {
              v68 = *(unsigned int *)(v66 + 40);
              a4 = *(unsigned int *)(v66 + 44);
              v69 = v211.m128i_i64[0];
              if ( v68 + 1 > a4 )
              {
                a2 = v66 + 48;
                sub_C8D5F0(v66 + 32, (const void *)(v66 + 48), v68 + 1, 8u, a5, v7);
                v68 = *(unsigned int *)(v66 + 40);
              }
              a3 = *(_QWORD *)(v66 + 32);
              *(_QWORD *)(a3 + 8 * v68) = v69;
              ++*(_DWORD *)(v66 + 40);
            }
          }
          else
          {
            v64 = *(_QWORD **)(v66 + 32);
            a2 = (__int64)&v64[*(unsigned int *)(v66 + 40)];
            if ( (_QWORD *)a2 == sub_F8ED40(v64, a2, v211.m128i_i64) )
            {
              if ( v65 + 1 > (unsigned __int64)*(unsigned int *)(v66 + 44) )
              {
                sub_C8D5F0(v66 + 32, (const void *)(v66 + 48), v65 + 1, 8u, a5, v7);
                a2 = *(_QWORD *)(v66 + 32) + 8LL * *(unsigned int *)(v66 + 40);
              }
              *(_QWORD *)a2 = v67;
              a3 = (unsigned int)(*(_DWORD *)(v66 + 40) + 1);
              *(_DWORD *)(v66 + 40) = a3;
              if ( (unsigned int)a3 > 2 )
              {
                v72 = v62;
                v73 = *(_QWORD *)(v66 + 32) + 8 * a3;
                v74 = *(__int64 **)(v66 + 32);
                do
                {
                  v75 = v74;
                  a2 = v66;
                  ++v74;
                  sub_D6CB10((__int64)&v226, v66, v75);
                }
                while ( (__int64 *)v73 != v74 );
                v62 = v72;
              }
            }
          }
        }
        v63 += 4;
      }
      while ( v184 != v63 );
      v183 = v62;
      v8 = v172;
      v27 = (unsigned int)v224;
      goto LABEL_23;
    }
    v183 = v7;
  }
  v9 = *(_QWORD *)(v8 + 16);
  if ( !v9 )
  {
LABEL_32:
    v30 = v223;
LABEL_33:
    v29 = 256;
    goto LABEL_34;
  }
  while ( 1 )
  {
    v10 = *(_QWORD *)(v9 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v10 - 30) <= 0xAu )
      break;
    v9 = *(_QWORD *)(v9 + 8);
    if ( !v9 )
      goto LABEL_32;
  }
  v11 = &v211;
LABEL_7:
  v12 = *(_QWORD *)(v10 + 40);
  v203.m128i_i64[0] = v12;
  if ( *(_BYTE *)v7 > 0x1Cu && v8 == *(_QWORD *)(v7 + 40) )
    goto LABEL_21;
  v13 = *(_QWORD *)(v12 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v13 == v12 + 48 || !v13 || (unsigned int)*(unsigned __int8 *)(v13 - 24) - 30 > 0xA )
    goto LABEL_330;
  if ( *(_BYTE *)(v13 - 24) == 31 && (*(_DWORD *)(v13 - 20) & 0x7FFFFFF) == 3 && *(_QWORD *)(v13 - 120) == v7 )
  {
    v14 = *(_QWORD *)(v13 - 56);
    if ( v14 != *(_QWORD *)(v13 - 88) )
    {
      v15 = v13 - 24;
      if ( v8 == v14 )
      {
        v42 = (__int64 *)sub_BD5C60(v15);
        v17 = sub_ACD6D0(v42);
      }
      else
      {
        v16 = (__int64 *)sub_BD5C60(v15);
        v17 = sub_ACD720(v16);
      }
      v211.m128i_i64[0] = v17;
      if ( v17 )
      {
        v22 = sub_FBCEF0((__int64)&v218, v11->m128i_i64, v18, v19, v20, v21);
        v23 = v22;
        if ( *(_DWORD *)(v22 + 16) )
        {
          a2 = v22;
          sub_D6CB10((__int64)&v226, v22, v203.m128i_i64);
          if ( LOBYTE(v230[0]) )
          {
            v35 = *(unsigned int *)(v23 + 40);
            v36 = v203.m128i_i64[0];
            if ( v35 + 1 > (unsigned __int64)*(unsigned int *)(v23 + 44) )
            {
              a2 = v23 + 48;
              sub_C8D5F0(v23 + 32, (const void *)(v23 + 48), v35 + 1, 8u, v33, v34);
              v35 = *(unsigned int *)(v23 + 40);
            }
            *(_QWORD *)(*(_QWORD *)(v23 + 32) + 8 * v35) = v36;
            ++*(_DWORD *)(v23 + 40);
          }
        }
        else
        {
          v24 = *(_QWORD **)(v22 + 32);
          a2 = (__int64)&v24[*(unsigned int *)(v22 + 40)];
          if ( (_QWORD *)a2 == sub_F8ED40(v24, a2, v203.m128i_i64) )
          {
            v37 = v203.m128i_i64[0];
            if ( v25 + 1 > (unsigned __int64)*(unsigned int *)(v23 + 44) )
            {
              sub_C8D5F0(v23 + 32, (const void *)(v23 + 48), v25 + 1, 8u, v25, v26);
              a2 = *(_QWORD *)(v23 + 32) + 8LL * *(unsigned int *)(v23 + 40);
            }
            *(_QWORD *)a2 = v37;
            v38 = (unsigned int)(*(_DWORD *)(v23 + 40) + 1);
            *(_DWORD *)(v23 + 40) = v38;
            if ( (unsigned int)v38 > 2 )
            {
              v176 = v8;
              v170 = v11;
              v39 = *(__int64 **)(v23 + 32);
              v40 = &v39[v38];
              do
              {
                v41 = v39;
                a2 = v23;
                ++v39;
                sub_D6CB10((__int64)&v226, v23, v41);
              }
              while ( v40 != v39 );
              v8 = v176;
              v11 = v170;
            }
          }
        }
      }
    }
  }
LABEL_21:
  while ( 1 )
  {
    v9 = *(_QWORD *)(v9 + 8);
    if ( !v9 )
      break;
    v10 = *(_QWORD *)(v9 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v10 - 30) <= 0xAu )
    {
      v7 = v189;
      goto LABEL_7;
    }
  }
  v27 = (unsigned int)v224;
LABEL_23:
  if ( !(_DWORD)v27 )
    goto LABEL_32;
  if ( a6 || (_BYTE)qword_4F8C468 )
  {
LABEL_25:
    v28 = v223;
    v29 = 256;
    v30 = &v223[72 * v27];
    goto LABEL_26;
  }
  v227 = (__int64)v230;
  v226 = 0;
  v228 = (char *)32;
  LODWORD(v229) = 0;
  BYTE4(v229) = 1;
  sub_AA72C0(&v211, v8, 0);
  sub_F94A80(&v203, &v211, v43, v44, v45, v46);
  if ( v217 )
    v217(v216, v216, 3);
  if ( v214 )
    v214(v213, v213, 3);
  v190 = _mm_loadu_si128(&v203);
  v191 = _mm_loadu_si128(&v204);
  sub_F99F40((__int64)v192, (__int64)v205);
  v47 = _mm_loadu_si128(&v208);
  v195 = _mm_loadu_si128(&v207);
  v196 = v47;
  sub_F99F40((__int64)v197, (__int64)v209);
  v171 = 0;
LABEL_56:
  v48 = _mm_loadu_si128(&v196);
  v211 = _mm_loadu_si128(&v195);
  v212 = v48;
  sub_F99F40((__int64)v213, (__int64)v197);
  v49 = _mm_loadu_si128(&v191);
  a2 = (__int64)v192;
  v199 = _mm_loadu_si128(&v190);
  v200 = v49;
  sub_F99F40((__int64)v201, (__int64)v192);
  v50 = v199.m128i_i64[0];
  v51 = v211.m128i_i64[0];
  if ( v202 )
  {
    a2 = (__int64)v201;
    v202(v201, v201, 3);
  }
  if ( v214 )
  {
    a2 = (__int64)v213;
    v214(v213, v213, 3);
  }
  if ( v50 != v51 )
  {
    v52 = _mm_loadu_si128(&v191);
    v53 = (const char **)v213;
    v211 = _mm_loadu_si128(&v190);
    v212 = v52;
    sub_F99F40((__int64)v213, (__int64)v192);
    do
    {
      v55 = *(_QWORD *)v211.m128i_i64[0];
      v211.m128i_i16[4] = 0;
      a2 = v55 & 0xFFFFFFFFFFFFFFF8LL;
      v211.m128i_i64[0] = a2;
      if ( a2 )
        a2 -= 24;
      if ( !v214 )
        goto LABEL_115;
      v53 = (const char **)v213;
    }
    while ( !v215(v213, a2) );
    v56 = v211.m128i_i64[0];
    if ( v211.m128i_i64[0] )
      v56 = v211.m128i_i64[0] - 24;
    if ( v214 )
    {
      a2 = (__int64)v213;
      v214(v213, v213, 3);
    }
    v57 = *(_BYTE *)v56;
    if ( *(_BYTE *)v56 != 85 )
      goto LABEL_102;
    a2 = 27;
    if ( (unsigned __int8)sub_A73ED0((_QWORD *)(v56 + 72), 27) )
      goto LABEL_72;
    a2 = 27;
    if ( (unsigned __int8)sub_B49560(v56, 27) )
      goto LABEL_72;
    a2 = 6;
    if ( (unsigned __int8)sub_A73ED0((_QWORD *)(v56 + 72), 6) )
      goto LABEL_72;
    a2 = 6;
    if ( (unsigned __int8)sub_B49560(v56, 6) )
      goto LABEL_72;
    v57 = *(_BYTE *)v56;
    if ( *(_BYTE *)v56 == 85 )
    {
      v76 = *(_QWORD *)(v56 - 32);
      if ( v76
        && !*(_BYTE *)v76
        && *(_QWORD *)(v76 + 24) == *(_QWORD *)(v56 + 80)
        && (*(_BYTE *)(v76 + 33) & 0x20) != 0
        && *(_DWORD *)(v76 + 36) == 11 )
      {
        goto LABEL_136;
      }
      v53 = (const char **)v56;
      if ( (unsigned __int8)sub_B46970((unsigned __int8 *)v56) )
      {
LABEL_124:
        if ( v171 <= (int)qword_4F8C8C8 )
        {
          ++v171;
          goto LABEL_105;
        }
LABEL_72:
        if ( v198 )
        {
          a2 = (__int64)v197;
          v198(v197, v197, 3);
        }
        if ( v193 )
        {
          a2 = (__int64)v192;
          v193(v192, v192, 3);
        }
        if ( v210 )
        {
          a2 = (__int64)v209;
          v210(v209, v209, 3);
        }
        if ( v206 )
        {
          a2 = (__int64)v205;
          v206(v205, v205, 3);
        }
        if ( !BYTE4(v229) )
          _libc_free(v227, a2);
        v27 = (unsigned int)v224;
        goto LABEL_25;
      }
    }
    else
    {
LABEL_102:
      v53 = (const char **)v56;
      if ( (unsigned __int8)sub_B46970((unsigned __int8 *)v56) || (unsigned int)v57 - 30 <= 0xA )
      {
LABEL_104:
        if ( v57 == 84 )
        {
LABEL_105:
          v70 = *(_QWORD *)(v56 + 16);
          if ( !v70 )
          {
            while ( 1 )
            {
LABEL_112:
              v71 = *(_QWORD *)v190.m128i_i64[0];
              v190.m128i_i16[4] = 0;
              a2 = v71 & 0xFFFFFFFFFFFFFFF8LL;
              v190.m128i_i64[0] = a2;
              if ( a2 )
                a2 -= 24;
              if ( !v193 )
                break;
              v53 = (const char **)v192;
              if ( v194(v192) )
                goto LABEL_56;
            }
LABEL_115:
            sub_4263D6(v53, a2, v54);
          }
          while ( 1 )
          {
            v54 = *(_QWORD *)(v70 + 24);
            if ( v8 != *(_QWORD *)(v54 + 40) || *(_BYTE *)v54 == 84 )
              goto LABEL_72;
            v70 = *(_QWORD *)(v70 + 8);
            if ( !v70 )
              goto LABEL_112;
          }
        }
        goto LABEL_124;
      }
    }
    v53 = *(const char ***)(v56 + 16);
    a2 = 0;
    if ( !sub_F90050((__int64)v53, 0, (__int64)&v226) )
    {
      v57 = *(_BYTE *)v56;
      goto LABEL_104;
    }
LABEL_136:
    v53 = (const char **)&v226;
    a2 = v56;
    sub_AE6EC0((__int64)&v226, v56);
    goto LABEL_105;
  }
  sub_A17130((__int64)v197);
  sub_A17130((__int64)v192);
  sub_A17130((__int64)v209);
  sub_A17130((__int64)v205);
  if ( !BYTE4(v229) )
    _libc_free(v227, a2);
  v28 = v223;
  v164 = &v223[72 * (unsigned int)v224];
  if ( v164 == v223 )
  {
    v30 = v223;
    goto LABEL_176;
  }
  v173 = v223;
  v77 = a5;
  v168 = v8;
  while ( 1 )
  {
    v78 = (__int64 *)*((_QWORD *)v173 + 5);
    v167 = *(_QWORD *)v173;
    v169 = *((unsigned int *)v173 + 12);
    v79 = *(_QWORD **)(*(_QWORD *)v173 + 24LL);
    v165 = (__int64 **)v78;
    if ( *(_DWORD *)(*(_QWORD *)v173 + 32LL) > 0x40u )
      v79 = (_QWORD *)*v79;
    v178 = *(_QWORD *)(a1 + -32 - 32LL * (v79 == 0));
    if ( v168 != v178 )
      break;
LABEL_174:
    v173 += 72;
    if ( v164 == v173 )
    {
      v28 = v223;
      v30 = &v223[72 * (unsigned int)v224];
LABEL_176:
      v29 = 256;
      goto LABEL_26;
    }
  }
  LOBYTE(v80) = sub_DF9710(v179);
  if ( (_BYTE)v80 )
  {
    LOBYTE(v80) = 0;
    if ( v77 )
      v80 = sub_F94F50(v168, v183) ^ 1;
  }
  a2 = (8 * v169) >> 3;
  v166 = &v78[v169];
  if ( (8 * v169) >> 5 )
  {
    v187 = &v78[4 * ((8 * v169) >> 5)];
    v81 = v77;
    v82 = v80;
    while ( 1 )
    {
      v91 = (__int64 *)*v78;
      if ( v82 )
      {
        a2 = *v78;
        if ( (unsigned __int8)sub_1056460(v81, *v78) )
        {
LABEL_172:
          v77 = v81;
          v85 = v78;
          goto LABEL_173;
        }
      }
      v83 = v91[6] & 0xFFFFFFFFFFFFFFF8LL;
      if ( (__int64 *)v83 == v91 + 6 || !v83 || (unsigned int)*(unsigned __int8 *)(v83 - 24) - 30 > 0xA )
        goto LABEL_330;
      if ( *(_BYTE *)(v83 - 24) == 33 )
        goto LABEL_172;
      v84 = v78[1];
      v85 = v78 + 1;
      if ( v82 )
      {
        a2 = v78[1];
        if ( (unsigned __int8)sub_1056460(v81, a2) )
        {
LABEL_178:
          v77 = v81;
          goto LABEL_173;
        }
      }
      v86 = *(_QWORD *)(v84 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v86 == v84 + 48 || !v86 || (unsigned int)*(unsigned __int8 *)(v86 - 24) - 30 > 0xA )
        goto LABEL_330;
      if ( *(_BYTE *)(v86 - 24) == 33 )
        goto LABEL_178;
      v87 = v78[2];
      v85 = v78 + 2;
      if ( v82 )
      {
        a2 = v78[2];
        if ( (unsigned __int8)sub_1056460(v81, a2) )
        {
          v77 = v81;
          goto LABEL_173;
        }
      }
      v88 = *(_QWORD *)(v87 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v88 == v87 + 48 || !v88 || (unsigned int)*(unsigned __int8 *)(v88 - 24) - 30 > 0xA )
        goto LABEL_330;
      if ( *(_BYTE *)(v88 - 24) == 33 )
        goto LABEL_178;
      v89 = v78[3];
      v85 = v78 + 3;
      if ( v82 )
      {
        a2 = v78[3];
        if ( (unsigned __int8)sub_1056460(v81, a2) )
          break;
      }
      v90 = *(_QWORD *)(v89 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v90 == v89 + 48 || !v90 || (unsigned int)*(unsigned __int8 *)(v90 - 24) - 30 > 0xA )
        goto LABEL_330;
      if ( *(_BYTE *)(v90 - 24) == 33 )
        goto LABEL_178;
      v78 += 4;
      if ( v187 == v78 )
      {
        LOBYTE(v80) = v82;
        v77 = v81;
        v85 = v187;
        a2 = v166 - v187;
        goto LABEL_184;
      }
    }
    v77 = v81;
LABEL_173:
    if ( v166 == v85 )
      goto LABEL_192;
    goto LABEL_174;
  }
  v85 = v78;
LABEL_184:
  if ( a2 != 2 )
  {
    if ( a2 != 3 )
    {
      if ( a2 != 1 )
        goto LABEL_192;
      goto LABEL_187;
    }
    v160 = *v85;
    if ( (_BYTE)v80 )
    {
      a2 = *v85;
      if ( (unsigned __int8)sub_1056460(v77, *v85) )
        goto LABEL_173;
    }
    v161 = *(_QWORD *)(v160 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v161 == v160 + 48 || !v161 || (unsigned int)*(unsigned __int8 *)(v161 - 24) - 30 > 0xA )
      goto LABEL_330;
    if ( *(_BYTE *)(v161 - 24) == 33 )
      goto LABEL_173;
    ++v85;
  }
  v162 = *v85;
  if ( (_BYTE)v80 )
  {
    a2 = *v85;
    if ( (unsigned __int8)sub_1056460(v77, *v85) )
      goto LABEL_173;
  }
  v163 = *(_QWORD *)(v162 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v163 == v162 + 48 || !v163 || (unsigned int)*(unsigned __int8 *)(v163 - 24) - 30 > 0xA )
    goto LABEL_330;
  if ( *(_BYTE *)(v163 - 24) == 33 )
    goto LABEL_173;
  ++v85;
LABEL_187:
  v92 = *v85;
  if ( (_BYTE)v80 )
  {
    a2 = *v85;
    if ( (unsigned __int8)sub_1056460(v77, *v85) )
      goto LABEL_173;
  }
  v93 = *(_QWORD *)(v92 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v93 == v92 + 48 || !v93 || (unsigned int)*(unsigned __int8 *)(v93 - 24) - 30 > 0xA )
    goto LABEL_330;
  if ( *(_BYTE *)(v93 - 24) == 33 )
    goto LABEL_173;
LABEL_192:
  v94 = (unsigned __int8 *)sub_F41DE0(v168, v165, v169, ".critedge", v182, 0, 0, 0);
  v226 = (char *)sub_BD5D20(v178);
  v228 = ".critedge";
  LOWORD(v230[0]) = 773;
  v227 = v95;
  sub_BD6B50(v94, (const char **)&v226);
  sub_AA4AC0((__int64)v94, v178 + 24);
  sub_F91F00(v178, (__int64)v94, v168, 0);
  v185 = (unsigned __int64 *)sub_AA5190((__int64)v94);
  if ( v185 )
  {
    v98 = v97;
  }
  else
  {
    v98 = 0;
    v96 = 0;
  }
  v211 = 0u;
  v99 = v96;
  v212.m128i_i64[0] = 0;
  BYTE1(v99) = v98;
  v180 = v99;
  v212.m128i_i32[2] = 0;
  if ( (unsigned __int8)sub_F9D990((__int64)&v211, &v189, &v226) )
  {
    v100 = v226 + 8;
  }
  else
  {
    v155 = sub_FAA5E0((__int64)&v211, &v189, v226);
    v156 = v189;
    v155[1] = 0;
    v100 = (char *)(v155 + 1);
    *((_QWORD *)v100 - 1) = v156;
  }
  *(_QWORD *)v100 = v167;
  v101 = *(_QWORD *)(v168 + 56);
  v102 = v101;
  while ( 2 )
  {
    if ( !v102 )
LABEL_330:
      BUG();
    v103 = v102 - 24;
    if ( a1 != v102 - 24 )
    {
      if ( *(_BYTE *)(v102 - 24) == 84 )
      {
        v104 = *(_QWORD *)(v102 - 32);
        v105 = 0x1FFFFFFFE0LL;
        if ( (*(_DWORD *)(v102 - 20) & 0x7FFFFFF) != 0 )
        {
          v106 = 0;
          do
          {
            if ( v94 == *(unsigned __int8 **)(v104 + 32LL * *(unsigned int *)(v102 + 48) + 8 * v106) )
            {
              v105 = 32 * v106;
              goto LABEL_205;
            }
            ++v106;
          }
          while ( (*(_DWORD *)(v102 - 20) & 0x7FFFFFF) != (_DWORD)v106 );
          v105 = 0x1FFFFFFFE0LL;
        }
LABEL_205:
        v107 = *(_QWORD *)(v104 + v105);
        v108 = v212.m128i_i32[2];
        v203.m128i_i64[0] = v102 - 24;
        if ( v212.m128i_i32[2] )
        {
          v109 = 1;
          v110 = 0;
          v111 = (v212.m128i_i32[2] - 1) & (((unsigned int)v103 >> 9) ^ ((unsigned int)v103 >> 4));
          v112 = (char *)(v211.m128i_i64[1] + 16LL * v111);
          v113 = *(_QWORD *)v112;
          if ( v103 == *(_QWORD *)v112 )
            goto LABEL_207;
          while ( v113 != -4096 )
          {
            if ( v113 == -8192 && !v110 )
              v110 = v112;
            v111 = (v212.m128i_i32[2] - 1) & (v109 + v111);
            v112 = (char *)(v211.m128i_i64[1] + 16LL * v111);
            v113 = *(_QWORD *)v112;
            if ( v103 == *(_QWORD *)v112 )
              goto LABEL_207;
            ++v109;
          }
          if ( v110 )
            v112 = v110;
          ++v211.m128i_i64[0];
          v157 = v212.m128i_i32[0] + 1;
          v226 = v112;
          if ( 4 * (v212.m128i_i32[0] + 1) < (unsigned int)(3 * v212.m128i_i32[2]) )
          {
            if ( v212.m128i_i32[2] - v212.m128i_i32[1] - v157 > (unsigned __int32)v212.m128i_i32[2] >> 3 )
            {
LABEL_285:
              v212.m128i_i32[0] = v157;
              if ( *(_QWORD *)v112 != -4096 )
                --v212.m128i_i32[1];
              *(_QWORD *)v112 = v103;
              *((_QWORD *)v112 + 1) = 0;
LABEL_207:
              *((_QWORD *)v112 + 1) = v107;
LABEL_208:
              v102 = *(_QWORD *)(v102 + 8);
              continue;
            }
LABEL_290:
            sub_FAA400((__int64)&v211, v108);
            sub_F9D990((__int64)&v211, v203.m128i_i64, &v226);
            v103 = v203.m128i_i64[0];
            v157 = v212.m128i_i32[0] + 1;
            v112 = v226;
            goto LABEL_285;
          }
        }
        else
        {
          ++v211.m128i_i64[0];
          v226 = 0;
        }
        v108 = 2 * v212.m128i_i32[2];
        goto LABEL_290;
      }
      v114 = sub_B47F80((_BYTE *)(v102 - 24));
      sub_B44240((_QWORD *)v114, (__int64)v94, v185, v180);
      v116 = v114;
      if ( (*(_BYTE *)(v102 - 17) & 0x10) != 0 )
      {
        v138 = sub_BD5D20(v102 - 24);
        LOWORD(v230[0]) = 773;
        v226 = (char *)v138;
        v227 = v139;
        v228 = ".c";
        sub_BD6B50((unsigned __int8 *)v114, (const char **)&v226);
        v116 = v114;
      }
      v117 = 32LL * (*(_DWORD *)(v114 + 4) & 0x7FFFFFF);
      v118 = v114 - v117;
      if ( (*(_BYTE *)(v114 + 7) & 0x40) != 0 )
      {
        v118 = *(_QWORD *)(v114 - 8);
        v116 = v118 + v117;
      }
      if ( v118 != v116 )
      {
        while ( 2 )
        {
          v119 = *(_QWORD *)v118;
          v115 = v211.m128i_i64[1];
          if ( v212.m128i_i32[2] )
          {
            v120 = (v212.m128i_i32[2] - 1) & (((unsigned int)v119 >> 9) ^ ((unsigned int)v119 >> 4));
            v117 = v211.m128i_i64[1] + 16LL * v120;
            v121 = *(_QWORD *)v117;
            if ( v119 == *(_QWORD *)v117 )
              goto LABEL_216;
            v117 = 1;
            if ( v121 != -4096 )
            {
              while ( 1 )
              {
                v120 = (v212.m128i_i32[2] - 1) & (v117 + v120);
                v174 = v117 + 1;
                v117 = v211.m128i_i64[1] + 16LL * v120;
                v137 = *(_QWORD *)v117;
                if ( v119 == *(_QWORD *)v117 )
                  break;
                v117 = v174;
                if ( v137 == -4096 )
                  goto LABEL_224;
              }
LABEL_216:
              if ( v117 != v211.m128i_i64[1] + 16LL * v212.m128i_u32[2] )
              {
                v117 = *(_QWORD *)(v117 + 8);
                if ( v119 )
                {
                  v122 = *(_QWORD *)(v118 + 8);
                  **(_QWORD **)(v118 + 16) = v122;
                  if ( v122 )
                    *(_QWORD *)(v122 + 16) = *(_QWORD *)(v118 + 16);
                }
                *(_QWORD *)v118 = v117;
                if ( v117 )
                {
                  v123 = *(_QWORD *)(v117 + 16);
                  *(_QWORD *)(v118 + 8) = v123;
                  if ( v123 )
                    *(_QWORD *)(v123 + 16) = v118 + 8;
                  *(_QWORD *)(v118 + 16) = v117 + 16;
                  *(_QWORD *)(v117 + 16) = v118;
                }
              }
            }
          }
LABEL_224:
          v118 += 32;
          if ( v116 == v118 )
            break;
          continue;
        }
      }
      v227 = 0;
      v226 = (char *)v181;
      v228 = 0;
      v229 = 0;
      v230[0] = a7;
      memset(&v230[1], 0, 24);
      v231 = 257;
      v124 = sub_1020E10(v114, &v226, v117, 257, v115, v116);
      if ( !v124 )
      {
        if ( *(_QWORD *)(v102 - 8) )
        {
          v226 = (char *)(v102 - 24);
          *sub_FAA780((__int64)&v211, (__int64 *)&v226) = v114;
        }
        goto LABEL_231;
      }
      if ( !*(_QWORD *)(v102 - 8) )
        goto LABEL_230;
      v125 = v212.m128i_i32[2];
      v203.m128i_i64[0] = v102 - 24;
      if ( v212.m128i_i32[2] )
      {
        v126 = 1;
        v127 = 0;
        v128 = (v212.m128i_i32[2] - 1) & (((unsigned int)v103 >> 9) ^ ((unsigned int)v103 >> 4));
        v129 = (char *)(v211.m128i_i64[1] + 16LL * v128);
        v130 = *(_QWORD *)v129;
        if ( v103 == *(_QWORD *)v129 )
          goto LABEL_229;
        while ( v130 != -4096 )
        {
          if ( !v127 && v130 == -8192 )
            v127 = v129;
          v128 = (v212.m128i_i32[2] - 1) & (v126 + v128);
          v129 = (char *)(v211.m128i_i64[1] + 16LL * v128);
          v130 = *(_QWORD *)v129;
          if ( v103 == *(_QWORD *)v129 )
            goto LABEL_229;
          ++v126;
        }
        if ( v127 )
          v129 = v127;
        ++v211.m128i_i64[0];
        v158 = v212.m128i_i32[0] + 1;
        v226 = v129;
        if ( 4 * (v212.m128i_i32[0] + 1) < (unsigned int)(3 * v212.m128i_i32[2]) )
        {
          v159 = v102 - 24;
          if ( v212.m128i_i32[2] - v212.m128i_i32[1] - v158 > (unsigned __int32)v212.m128i_i32[2] >> 3 )
          {
LABEL_304:
            v212.m128i_i32[0] = v158;
            if ( *(_QWORD *)v129 != -4096 )
              --v212.m128i_i32[1];
            *(_QWORD *)v129 = v159;
            *((_QWORD *)v129 + 1) = 0;
LABEL_229:
            *((_QWORD *)v129 + 1) = v124;
LABEL_230:
            if ( !(unsigned __int8)sub_B46970((unsigned __int8 *)v114) )
            {
              sub_B43D60((_QWORD *)v114);
              goto LABEL_208;
            }
LABEL_231:
            while ( v102 != v101 )
            {
              v131 = v101 - 24;
              LOBYTE(v227) = 0;
              if ( !v101 )
                v131 = 0;
              sub_B43F50(v114, v131, (__int64)v226, v227, 0);
              v101 = *(_QWORD *)(v101 + 8);
            }
            v101 = *(_QWORD *)(v102 + 8);
            LOBYTE(v227) = 0;
            sub_B43F50(v114, v102 - 24, (__int64)v226, 0, 0);
            if ( *(_BYTE *)v114 == 85 )
            {
              v135 = *(_QWORD *)(v114 - 32);
              if ( v135 )
              {
                if ( !*(_BYTE *)v135 )
                {
                  v136 = *(_QWORD *)(v114 + 80);
                  if ( *(_QWORD *)(v135 + 24) == v136
                    && (*(_BYTE *)(v135 + 33) & 0x20) != 0
                    && *(_DWORD *)(v135 + 36) == 11
                    && a7 )
                  {
                    sub_CFEAE0(a7, v114, v132, v136, v133, v134);
                  }
                }
              }
            }
            goto LABEL_208;
          }
          v175 = v124;
LABEL_309:
          sub_FAA400((__int64)&v211, v125);
          sub_F9D990((__int64)&v211, v203.m128i_i64, &v226);
          v159 = v203.m128i_i64[0];
          v124 = v175;
          v158 = v212.m128i_i32[0] + 1;
          v129 = v226;
          goto LABEL_304;
        }
      }
      else
      {
        ++v211.m128i_i64[0];
        v226 = 0;
      }
      v175 = v124;
      v125 = 2 * v212.m128i_i32[2];
      goto LABEL_309;
    }
    break;
  }
  v140 = 0;
  if ( v185 )
    v140 = (__int64)(v185 - 3);
  while ( 2 )
  {
    if ( !v101 )
    {
      v141 = 0;
      goto LABEL_254;
    }
    v141 = v101 - 24;
    if ( v102 - 24 != v101 - 24 )
    {
LABEL_254:
      LOBYTE(v227) = 0;
      sub_B43F50(v140, v141, (__int64)v226, 0, 0);
      v101 = *(_QWORD *)(v101 + 8);
      continue;
    }
    break;
  }
  v142 = 0;
  if ( v185 )
    v142 = (__int64)(v185 - 3);
  LOBYTE(v227) = 0;
  sub_B43F50(v142, v102 - 24, (__int64)v226, 0, 0);
  sub_AA5980(v168, (__int64)v94, 0);
  v143 = sub_986580((__int64)v94);
  sub_AC2B30(v143 - 32, v178);
  v147 = *(const char **)(v102 + 24);
  v226 = (char *)v147;
  if ( !v147 )
  {
    v148 = v143 + 48;
    if ( (char **)(v143 + 48) == &v226 )
      goto LABEL_263;
    v153 = *(_QWORD *)(v143 + 48);
    if ( !v153 )
      goto LABEL_263;
LABEL_270:
    sub_B91220(v148, v153);
    goto LABEL_271;
  }
  v148 = v143 + 48;
  sub_B96E90((__int64)&v226, (__int64)v147, 1);
  if ( (char **)(v143 + 48) == &v226 )
  {
    if ( v226 )
      sub_B91220((__int64)&v226, (__int64)v226);
    goto LABEL_263;
  }
  v153 = *(_QWORD *)(v143 + 48);
  if ( v153 )
    goto LABEL_270;
LABEL_271:
  v154 = (unsigned __int8 *)v226;
  *(_QWORD *)(v143 + 48) = v226;
  if ( v154 )
    sub_B976B0((__int64)&v226, v154, v148);
LABEL_263:
  if ( v182 )
  {
    v227 = 0x200000000LL;
    v226 = (char *)&v228;
    sub_F35FA0((__int64)&v226, (__int64)v94, v168 | 4, v144, v145, v146);
    sub_F35FA0((__int64)&v226, (__int64)v94, v178 & 0xFFFFFFFFFFFFFFFBLL, v149, v150, v151);
    v152 = v226;
    sub_FFB3D0(v182, v226, (unsigned int)v227);
    if ( v226 != (char *)&v228 )
      _libc_free(v226, v152);
  }
  sub_F39690((__int64)v94, v182, 0, 0, 0, 0, 0);
  a2 = 16LL * v212.m128i_u32[2];
  v29 = 0;
  sub_C7D6A0(v211.m128i_i64[1], a2, 8);
  v28 = v223;
  v30 = &v223[72 * (unsigned int)v224];
LABEL_26:
  if ( v28 != v30 )
  {
    do
    {
      v30 -= 72;
      v31 = (_BYTE *)*((_QWORD *)v30 + 5);
      if ( v31 != v30 + 56 )
        _libc_free(v31, a2);
      a2 = 8LL * *((unsigned int *)v30 + 8);
      sub_C7D6A0(*((_QWORD *)v30 + 2), a2, 8);
    }
    while ( v28 != v30 );
    v30 = v223;
  }
LABEL_34:
  if ( v30 != v225 )
    _libc_free(v30, a2);
  if ( (v219 & 1) == 0 )
    sub_C7D6A0(v220, 16LL * v221, 8);
  return v29;
}
