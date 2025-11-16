// Function: sub_E3D320
// Address: 0xe3d320
//
__int64 __fastcall sub_E3D320(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r13
  __int64 v6; // r9
  const void *v7; // r15
  size_t v8; // r14
  __int64 v9; // r12
  _BYTE *v10; // rbx
  int v11; // eax
  _BYTE *v12; // rdi
  unsigned __int64 v13; // rsi
  __int64 v14; // rax
  unsigned __int64 v15; // r12
  __int64 i; // r13
  __m128i v17; // xmm0
  unsigned __int64 v18; // r8
  __int64 v19; // r9
  unsigned __int64 v20; // r13
  _BYTE *v21; // r11
  __int64 v22; // rdx
  _BYTE *v23; // r10
  __int64 v24; // rax
  signed __int64 v25; // r12
  __int64 v26; // r14
  unsigned __int64 v27; // rdx
  unsigned __int64 *v28; // rbx
  unsigned __int64 v29; // r12
  unsigned __int64 *v30; // rax
  __int64 v31; // r13
  unsigned __int64 *v32; // rax
  unsigned __int64 v33; // rbx
  unsigned __int64 v34; // rbx
  __int8 v35; // dl
  __int64 v36; // r14
  unsigned __int64 v37; // r12
  __int64 v38; // rdx
  __int8 v39; // al
  unsigned __int64 *v40; // rbx
  __int64 v41; // rax
  unsigned __int64 v42; // rdx
  bool v43; // al
  unsigned __int64 *v44; // rbx
  bool v45; // r14
  __int64 v46; // rbx
  unsigned __int64 *v47; // rax
  bool v48; // zf
  __int64 v49; // rax
  unsigned __int64 v50; // rdx
  unsigned __int64 **v51; // r14
  _QWORD *v52; // rax
  unsigned __int64 *v53; // r14
  __int8 v54; // al
  int v55; // ebx
  __int64 v56; // rax
  unsigned __int64 v57; // rdx
  _QWORD *v58; // rax
  unsigned __int64 *v59; // r14
  __int8 v60; // al
  int v61; // ebx
  __int64 v62; // rax
  unsigned __int64 v63; // rdx
  __int8 v64; // al
  unsigned __int64 *v65; // r14
  int v66; // ebx
  __int64 v67; // rax
  unsigned __int64 v68; // rdx
  _QWORD *v69; // rax
  __int64 v70; // rdx
  unsigned __int64 *v71; // rbx
  unsigned __int64 *v72; // rax
  unsigned __int64 v73; // rbx
  unsigned int v74; // eax
  unsigned __int64 *v75; // r12
  unsigned __int64 v76; // r14
  unsigned __int64 *v77; // r15
  int v78; // eax
  __int64 v79; // r8
  __int64 v80; // r9
  unsigned __int64 *v81; // rcx
  unsigned __int64 *v82; // rax
  unsigned __int64 *v83; // r15
  __int8 v84; // al
  unsigned __int64 v85; // r15
  __int64 v86; // r14
  unsigned __int64 *v87; // r12
  signed __int64 v88; // r10
  __int64 v89; // rax
  __int64 v90; // r14
  unsigned __int64 v91; // rdx
  _BYTE *v92; // rsi
  __int64 v93; // rdx
  __int64 v94; // rax
  __int32 v95; // ebx
  _BYTE *v96; // rdi
  __int64 v97; // r12
  _BYTE *v98; // rsi
  __int64 v99; // rdx
  __int64 v100; // rdx
  __int64 *v101; // rdi
  __int64 v102; // r12
  __int64 v104; // rax
  unsigned __int64 v105; // rdx
  __int64 v106; // rax
  __int8 v107; // al
  __int64 v108; // r13
  __m128i v109; // xmm0
  unsigned __int64 *v110; // rbx
  unsigned __int64 **v111; // r12
  unsigned __int64 *v112; // rax
  unsigned __int64 v113; // rbx
  _BYTE *v114; // rdx
  __int64 v115; // rsi
  char *v116; // rcx
  __int64 v117; // rbx
  char *v118; // rax
  size_t v119; // rbx
  char *v120; // rax
  __m128i v121; // rax
  unsigned __int64 v122; // rsi
  _BYTE *v123; // rdx
  __int64 v124; // rdi
  __int64 v125; // rbx
  char *v126; // rcx
  char *v127; // rax
  size_t v128; // rbx
  char *v129; // rax
  int v130; // eax
  unsigned __int64 *v131; // r12
  __int8 v132; // al
  unsigned __int64 v133; // rax
  unsigned __int64 *v134; // r13
  unsigned __int64 *v135; // rbx
  signed __int64 v136; // r14
  __int64 v137; // rax
  unsigned __int64 v138; // rdx
  __int64 v139; // rax
  unsigned __int64 v140; // rdx
  __int64 v141; // rax
  unsigned __int64 v142; // r15
  __int8 v143; // al
  unsigned __int64 *v144; // r15
  unsigned int v145; // eax
  unsigned __int64 *v146; // r15
  __int8 v147; // al
  __int64 v148; // rax
  unsigned __int64 v149; // rdx
  __int8 v150; // al
  __int64 v151; // rax
  unsigned __int64 v152; // rdx
  __int64 v153; // rax
  __int64 v154; // r13
  __int64 v155; // rax
  __int8 v156; // al
  __int64 v157; // rdx
  __m128i v158; // rax
  __int64 v159; // rdi
  char *v160; // rcx
  __int64 v161; // rbx
  char *v162; // rax
  size_t v163; // rbx
  char *v164; // rax
  __m128i v165; // rax
  __int64 v166; // [rsp+0h] [rbp-2B0h]
  __int64 v167; // [rsp+8h] [rbp-2A8h]
  __int64 v168; // [rsp+10h] [rbp-2A0h]
  __int64 v170; // [rsp+20h] [rbp-290h]
  _BYTE *v171; // [rsp+30h] [rbp-280h]
  _BYTE *v172; // [rsp+38h] [rbp-278h]
  signed __int64 v173; // [rsp+38h] [rbp-278h]
  unsigned __int64 v174; // [rsp+38h] [rbp-278h]
  __int64 v175; // [rsp+38h] [rbp-278h]
  __int64 v176; // [rsp+40h] [rbp-270h]
  void *src; // [rsp+50h] [rbp-260h]
  __int64 v178; // [rsp+58h] [rbp-258h]
  unsigned __int64 v179; // [rsp+58h] [rbp-258h]
  unsigned __int64 v180; // [rsp+68h] [rbp-248h] BYREF
  unsigned __int64 *v181; // [rsp+70h] [rbp-240h] BYREF
  unsigned __int64 *v182; // [rsp+78h] [rbp-238h]
  __m128i v183[2]; // [rsp+80h] [rbp-230h] BYREF
  __m128i v184; // [rsp+A0h] [rbp-210h] BYREF
  __m128i v185[2]; // [rsp+B0h] [rbp-200h] BYREF
  unsigned __int64 *v186; // [rsp+D0h] [rbp-1E0h] BYREF
  unsigned __int64 *v187; // [rsp+D8h] [rbp-1D8h]
  __m128i v188; // [rsp+E0h] [rbp-1D0h] BYREF
  __m128i v189; // [rsp+F0h] [rbp-1C0h] BYREF
  __m128i v190; // [rsp+100h] [rbp-1B0h] BYREF
  __m128i v191; // [rsp+110h] [rbp-1A0h] BYREF
  __m128i v192; // [rsp+120h] [rbp-190h] BYREF
  __m128i v193; // [rsp+130h] [rbp-180h]
  __m128i v194; // [rsp+140h] [rbp-170h]
  __m128i v195; // [rsp+150h] [rbp-160h] BYREF
  void *v196; // [rsp+160h] [rbp-150h] BYREF
  __int64 v197; // [rsp+168h] [rbp-148h]
  _BYTE v198[48]; // [rsp+170h] [rbp-140h] BYREF
  __m128i v199; // [rsp+1A0h] [rbp-110h] BYREF
  _BYTE v200[48]; // [rsp+1B0h] [rbp-100h] BYREF
  _BYTE *v201; // [rsp+1E0h] [rbp-D0h] BYREF
  __int64 v202; // [rsp+1E8h] [rbp-C8h]
  _BYTE v203[64]; // [rsp+1F0h] [rbp-C0h] BYREF
  __m128i v204; // [rsp+230h] [rbp-80h] BYREF
  _BYTE v205[112]; // [rsp+240h] [rbp-70h] BYREF

  v6 = a1[3];
  v7 = (const void *)a1[2];
  v201 = v203;
  v8 = v6 - (_QWORD)v7;
  v202 = 0x800000000LL;
  v9 = (v6 - (__int64)v7) >> 3;
  if ( (unsigned __int64)(v6 - (_QWORD)v7) > 0x40 )
  {
    src = (void *)v6;
    sub_C8D5F0((__int64)&v201, v203, (v6 - (__int64)v7) >> 3, 8u, a5, v6);
    v10 = v201;
    v11 = v202;
    v6 = (__int64)src;
    v12 = &v201[8 * (unsigned int)v202];
  }
  else
  {
    v10 = v203;
    v11 = 0;
    v12 = v203;
  }
  if ( (const void *)v6 != v7 )
  {
    memmove(v12, v7, v8);
    v10 = v201;
    v11 = v202;
  }
  v180 = 0;
  v13 = (unsigned int)(v11 + v9);
  v196 = v198;
  v14 = (__int64)&v10[8 * v13];
  LODWORD(v202) = v13;
  v15 = 0;
  v195.m128i_i64[0] = (__int64)v10;
  v195.m128i_i64[1] = v14;
  v197 = 0x600000000LL;
  if ( !v13 )
  {
    v199.m128i_i64[1] = 0x600000000LL;
    v181 = (unsigned __int64 *)v198;
    v182 = (unsigned __int64 *)v198;
    v204.m128i_i64[0] = (__int64)v205;
    v204.m128i_i64[1] = 0x800000000LL;
    v199.m128i_i64[0] = (__int64)v200;
    goto LABEL_93;
  }
  v168 = v5;
  for ( i = (__int64)v10; i != v14; i = v195.m128i_i64[0] )
  {
    v204.m128i_i8[8] = 1;
    v204.m128i_i64[0] = v195.m128i_i64[0];
    v17 = _mm_loadu_si128(&v204);
    v185[1] = v17;
    v18 = *(_QWORD *)v17.m128i_i64[0] - 48LL;
    if ( v18 <= 0x1F )
    {
      v104 = (unsigned int)v197;
      v105 = (unsigned int)v197 + 1LL;
      if ( v105 > HIDWORD(v197) )
      {
        v175 = *(_QWORD *)v17.m128i_i64[0] - 48LL;
        sub_C8D5F0((__int64)&v196, v198, v105, 8u, v18, v6);
        v104 = (unsigned int)v197;
        v18 = v175;
      }
      *((_QWORD *)v196 + v104) = 16;
      LODWORD(v197) = v197 + 1;
      v106 = (unsigned int)v197;
      if ( (unsigned __int64)(unsigned int)v197 + 1 > HIDWORD(v197) )
      {
        v174 = v18;
        sub_C8D5F0((__int64)&v196, v198, (unsigned int)v197 + 1LL, 8u, v18, v6);
        v106 = (unsigned int)v197;
        v18 = v174;
      }
      *((_QWORD *)v196 + v106) = v18;
      v107 = 0;
      v108 = v195.m128i_i64[0];
      LODWORD(v197) = v197 + 1;
      if ( v195.m128i_i64[0] != v195.m128i_i64[1] )
      {
        v204.m128i_i64[0] = v195.m128i_i64[0];
        v107 = 1;
        v204.m128i_i8[8] = 1;
        v185[0] = _mm_loadu_si128(&v204);
      }
      v185[0].m128i_i8[8] = v107;
      v109 = _mm_loadu_si128(v185);
      goto LABEL_119;
    }
    if ( *(_QWORD *)v17.m128i_i64[0] == 35 )
    {
      v151 = (unsigned int)v197;
      v152 = (unsigned int)v197 + 1LL;
      if ( v152 > HIDWORD(v197) )
      {
        sub_C8D5F0((__int64)&v196, v198, v152, 8u, v18, v6);
        v151 = (unsigned int)v197;
      }
      *((_QWORD *)v196 + v151) = 16;
      LODWORD(v197) = v197 + 1;
      v153 = (unsigned int)v197;
      v154 = *(_QWORD *)(v17.m128i_i64[0] + 8);
      if ( (unsigned __int64)(unsigned int)v197 + 1 > HIDWORD(v197) )
      {
        sub_C8D5F0((__int64)&v196, v198, (unsigned int)v197 + 1LL, 8u, v18, v6);
        v153 = (unsigned int)v197;
      }
      *((_QWORD *)v196 + v153) = v154;
      LODWORD(v197) = v197 + 1;
      v155 = (unsigned int)v197;
      if ( (unsigned __int64)(unsigned int)v197 + 1 > HIDWORD(v197) )
      {
        sub_C8D5F0((__int64)&v196, v198, (unsigned int)v197 + 1LL, 8u, v18, v6);
        v155 = (unsigned int)v197;
      }
      *((_QWORD *)v196 + v155) = 34;
      v156 = 0;
      v108 = v195.m128i_i64[0];
      LODWORD(v197) = v197 + 1;
      if ( v195.m128i_i64[0] != v195.m128i_i64[1] )
      {
        v204.m128i_i64[0] = v195.m128i_i64[0];
        v156 = 1;
        v204.m128i_i8[8] = 1;
        v184 = _mm_loadu_si128(&v204);
      }
      v184.m128i_i8[8] = v156;
      v109 = _mm_loadu_si128(&v184);
LABEL_119:
      v199 = v109;
      v204 = v109;
      v195.m128i_i64[0] = v108 + 8LL * (unsigned int)sub_AF4160((unsigned __int64 **)&v195);
      v15 += (unsigned int)sub_AF4160((unsigned __int64 **)&v204);
      goto LABEL_14;
    }
    v183[1] = v17;
    v199 = v17;
    v195.m128i_i64[0] = i + 8LL * (unsigned int)sub_AF4160((unsigned __int64 **)&v195);
    a5 = (unsigned int)sub_AF4160((unsigned __int64 **)&v204);
    v20 = a5 + v15;
    v21 = &v10[8 * v15];
    v22 = 8 * (a5 + v15);
    v23 = &v10[v22];
    v24 = (unsigned int)v197;
    v25 = v22 - 8 * v15;
    v26 = v25 >> 3;
    v27 = (v25 >> 3) + (unsigned int)v197;
    if ( v27 > HIDWORD(v197) )
    {
      v171 = v21;
      v172 = v23;
      sub_C8D5F0((__int64)&v196, v198, v27, 8u, a5, v19);
      v24 = (unsigned int)v197;
      v21 = v171;
      v23 = v172;
    }
    if ( v23 != v21 )
    {
      memcpy((char *)v196 + 8 * v24, v21, v25);
      v24 = (unsigned int)v197;
    }
    v6 = v26 + v24;
    v15 = v20;
    LODWORD(v197) = v26 + v24;
LABEL_14:
    if ( v13 <= v15 )
      break;
    v14 = v195.m128i_i64[1];
  }
  v28 = (unsigned __int64 *)v196;
  v29 = (unsigned int)v197;
  v204.m128i_i64[0] = (__int64)v205;
  v30 = (unsigned __int64 *)((char *)v196 + 8 * (unsigned int)v197);
  v31 = v168;
  v181 = (unsigned __int64 *)v196;
  v182 = v30;
  v204.m128i_i64[1] = 0x800000000LL;
  if ( v180 < (unsigned int)v197 )
  {
    while ( 1 )
    {
      v204.m128i_i32[2] = 0;
      v35 = 0;
      if ( v30 != v28 )
      {
        v199.m128i_i8[8] = 1;
        v35 = 1;
        v199.m128i_i64[0] = (__int64)v181;
        v183[0] = _mm_loadu_si128(&v199);
      }
      v183[0].m128i_i8[8] = v35;
      v195 = _mm_loadu_si128(v183);
      v199 = v195;
      if ( !v195.m128i_i8[8] )
        break;
      v36 = v195.m128i_i64[0];
      if ( *(_QWORD *)v195.m128i_i64[0] == 16 )
      {
        v37 = *(_QWORD *)(v195.m128i_i64[0] + 8);
        v38 = 0;
        if ( !v204.m128i_i32[3] )
        {
          sub_C8D5F0((__int64)&v204, v205, 1u, 8u, a5, v6);
          v38 = 8LL * v204.m128i_u32[2];
        }
        v39 = 0;
        *(_QWORD *)(v204.m128i_i64[0] + v38) = v36;
        v40 = v181;
        ++v204.m128i_i32[2];
        if ( v181 != v182 )
        {
          v199.m128i_i64[0] = (__int64)v181;
          v31 = (__int64)&v40[(unsigned int)sub_AF4160((unsigned __int64 **)&v199)];
          v39 = 1;
          if ( v182 == (unsigned __int64 *)v31 )
          {
            v31 = v176;
            v39 = 0;
          }
          v176 = v31;
        }
        v195.m128i_i64[0] = v31;
        v195.m128i_i8[8] = v39;
        if ( !v39 )
          break;
        v41 = v204.m128i_u32[2];
        v42 = v204.m128i_u32[2] + 1LL;
        if ( v42 > v204.m128i_u32[3] )
        {
          sub_C8D5F0((__int64)&v204, v205, v42, 8u, a5, v6);
          v41 = v204.m128i_u32[2];
        }
        *(_QWORD *)(v204.m128i_i64[0] + 8 * v41) = v31;
        ++v204.m128i_i32[2];
        switch ( **(_QWORD **)(v204.m128i_i64[0] + 8) )
        {
          case 0x1BLL:
          case 0x1ELL:
            v43 = v37 == 1;
            goto LABEL_37;
          case 0x1CLL:
          case 0x22LL:
          case 0x24LL:
          case 0x25LL:
            v43 = v37 == 0;
LABEL_37:
            if ( !v43 )
              goto LABEL_38;
            v114 = v196;
            v115 = 8 * v180 + 24;
            v116 = (char *)v196 + 8 * v180;
            v117 = 8LL * (unsigned int)v197;
            v118 = (char *)v196 + v117;
            v119 = v117 - v115;
            if ( (char *)v196 + v115 != v118 )
            {
              v120 = (char *)memmove((char *)v196 + 8 * v180, (char *)v196 + v115, v119);
              v114 = v196;
              v116 = v120;
            }
            LODWORD(v197) = (&v116[v119] - v114) >> 3;
            sub_E3D300(&v180, &v181, (__int64)v114, (unsigned int)v197);
            v34 = v180;
            goto LABEL_20;
          default:
LABEL_38:
            v44 = v181;
            v45 = 0;
            if ( v181 != v182 )
            {
              v199.m128i_i64[0] = (__int64)v181;
              v46 = (__int64)&v44[(unsigned int)sub_AF4160((unsigned __int64 **)&v199)];
              if ( v182 != (unsigned __int64 *)v46 )
              {
                v199.m128i_i64[0] = v46;
                v47 = (unsigned __int64 *)(v46 + 8LL * (unsigned int)sub_AF4160((unsigned __int64 **)&v199));
                v48 = v182 == v47;
                if ( v182 == v47 )
                  v47 = (unsigned __int64 *)v178;
                v45 = !v48;
                v178 = (__int64)v47;
              }
            }
            v195.m128i_i8[8] = v45;
            v195.m128i_i64[0] = v178;
            if ( !v45 )
              goto LABEL_120;
            v49 = v204.m128i_u32[2];
            v50 = v204.m128i_u32[2] + 1LL;
            if ( v50 > v204.m128i_u32[3] )
            {
              sub_C8D5F0((__int64)&v204, v205, v50, 8u, a5, v6);
              v49 = v204.m128i_u32[2];
            }
            *(_QWORD *)(v204.m128i_i64[0] + 8 * v49) = v178;
            v51 = (unsigned __int64 **)v204.m128i_i64[0];
            ++v204.m128i_i32[2];
            v52 = *(_QWORD **)(v204.m128i_i64[0] + 8);
            if ( *v52 == 16 )
            {
              v121.m128i_i64[0] = sub_E3D160(v37, v52[1], **(_QWORD **)(v204.m128i_i64[0] + 16));
              v199 = v121;
              if ( v121.m128i_i8[8] )
              {
                v122 = v180;
                v123 = v196;
                v124 = 8 * v180 + 40;
                v125 = 8LL * (unsigned int)v197;
                v126 = (char *)v196 + v124 - 24;
                v127 = (char *)v196 + v125;
                v128 = v125 - v124;
                if ( (char *)v196 + v124 != v127 )
                {
                  v129 = (char *)memmove((char *)v196 + v124 - 24, (char *)v196 + v124, v128);
                  v122 = v180;
                  v123 = v196;
                  v126 = v129;
                }
                LODWORD(v197) = (&v126[v128] - v123) >> 3;
                goto LABEL_128;
              }
            }
            else
            {
              v53 = v181;
              v54 = 0;
              if ( v181 != v182 )
              {
                v55 = 3;
                while ( 1 )
                {
                  v199.m128i_i64[0] = (__int64)v53;
                  v53 += (unsigned int)sub_AF4160((unsigned __int64 **)&v199);
                  if ( v182 == v53 )
                    break;
                  if ( !--v55 )
                  {
                    v170 = (__int64)v53;
                    v54 = 1;
                    goto LABEL_52;
                  }
                }
                v54 = 0;
              }
LABEL_52:
              v195.m128i_i8[8] = v54;
              v195.m128i_i64[0] = v170;
              if ( !v54 )
              {
LABEL_120:
                v110 = v181;
                v111 = (unsigned __int64 **)v204.m128i_i64[0];
                v112 = &v110[(unsigned int)sub_AF4160(&v181)];
                v113 = v180;
                v181 = v112;
                v34 = (unsigned int)sub_AF4160(v111) + v113;
                v180 = v34;
                goto LABEL_20;
              }
              v56 = v204.m128i_u32[2];
              v57 = v204.m128i_u32[2] + 1LL;
              if ( v57 > v204.m128i_u32[3] )
              {
                sub_C8D5F0((__int64)&v204, v205, v57, 8u, a5, v6);
                v56 = v204.m128i_u32[2];
              }
              *(_QWORD *)(v204.m128i_i64[0] + 8 * v56) = v170;
              v51 = (unsigned __int64 **)v204.m128i_i64[0];
              ++v204.m128i_i32[2];
              v58 = *(_QWORD **)(v204.m128i_i64[0] + 16);
              if ( *v58 == 16
                && (v157 = **(_QWORD **)(v204.m128i_i64[0] + 8), **(_DWORD **)(v204.m128i_i64[0] + 24) == (_DWORD)v157)
                && (((_DWORD)v157 - 30) & 0xFFFFFFFB) == 0 )
              {
                v158.m128i_i64[0] = sub_E3D160(v37, v58[1], v157);
                v199 = v158;
                if ( v158.m128i_i8[8] )
                {
                  v122 = v180;
                  v123 = v196;
                  v159 = 8 * v180 + 48;
                  goto LABEL_169;
                }
              }
              else
              {
                v59 = v181;
                v60 = 0;
                if ( v181 != v182 )
                {
                  v61 = 4;
                  while ( 1 )
                  {
                    v199.m128i_i64[0] = (__int64)v59;
                    v59 += (unsigned int)sub_AF4160((unsigned __int64 **)&v199);
                    if ( v182 == v59 )
                      break;
                    if ( !--v61 )
                    {
                      v167 = (__int64)v59;
                      v60 = 1;
                      goto LABEL_61;
                    }
                  }
                  v60 = 0;
                }
LABEL_61:
                v195.m128i_i8[8] = v60;
                v195.m128i_i64[0] = v167;
                if ( !v60 )
                  goto LABEL_120;
                v62 = v204.m128i_u32[2];
                v63 = v204.m128i_u32[2] + 1LL;
                if ( v63 > v204.m128i_u32[3] )
                {
                  sub_C8D5F0((__int64)&v204, v205, v63, 8u, a5, v6);
                  v62 = v204.m128i_u32[2];
                }
                *(_QWORD *)(v204.m128i_i64[0] + 8 * v62) = v167;
                v64 = 0;
                v65 = v181;
                ++v204.m128i_i32[2];
                if ( v181 != v182 )
                {
                  v66 = 5;
                  while ( 1 )
                  {
                    v199.m128i_i64[0] = (__int64)v65;
                    v65 += (unsigned int)sub_AF4160((unsigned __int64 **)&v199);
                    if ( v182 == v65 )
                      break;
                    if ( !--v66 )
                    {
                      v166 = (__int64)v65;
                      v64 = 1;
                      goto LABEL_69;
                    }
                  }
                  v64 = 0;
                }
LABEL_69:
                v195.m128i_i8[8] = v64;
                v195.m128i_i64[0] = v166;
                if ( !v64 )
                  goto LABEL_120;
                v67 = v204.m128i_u32[2];
                v68 = v204.m128i_u32[2] + 1LL;
                if ( v68 > v204.m128i_u32[3] )
                {
                  sub_C8D5F0((__int64)&v204, v205, v68, 8u, a5, v6);
                  v67 = v204.m128i_u32[2];
                }
                *(_QWORD *)(v204.m128i_i64[0] + 8 * v67) = v166;
                v51 = (unsigned __int64 **)v204.m128i_i64[0];
                ++v204.m128i_i32[2];
                v69 = *(_QWORD **)(v204.m128i_i64[0] + 32);
                if ( *v69 == 16 && **(_QWORD **)(v204.m128i_i64[0] + 16) == 4101 )
                {
                  v70 = **(_QWORD **)(v204.m128i_i64[0] + 24);
                  if ( **(_DWORD **)(v204.m128i_i64[0] + 8) == (_DWORD)v70
                    && (((_DWORD)v70 - 30) & 0xFFFFFFFB) == 0
                    && **(_DWORD **)(v204.m128i_i64[0] + 40) == (_DWORD)v70 )
                  {
                    v165.m128i_i64[0] = sub_E3D160(v37, v69[1], v70);
                    v199 = v165;
                    if ( v165.m128i_i8[8] )
                    {
                      v122 = v180;
                      v123 = v196;
                      v159 = 8 * v180 + 72;
LABEL_169:
                      v160 = &v123[v159 - 24];
                      v161 = 8LL * (unsigned int)v197;
                      v162 = &v123[v161];
                      v163 = v161 - v159;
                      if ( &v123[v159] != v162 )
                      {
                        v164 = (char *)memmove(v160, &v123[v159], v163);
                        v122 = v180;
                        v123 = v196;
                        v160 = v164;
                      }
                      LODWORD(v197) = (&v160[v163] - v123) >> 3;
LABEL_128:
                      *(_QWORD *)&v123[8 * v122] = 16;
                      *((_QWORD *)v196 + v180 + 1) = v199.m128i_i64[0];
                      sub_E3D300(&v180, &v181, (__int64)v196, (unsigned int)v197);
                      v34 = v180;
                      goto LABEL_20;
                    }
                  }
                }
              }
            }
            v71 = v181;
            v72 = &v71[(unsigned int)sub_AF4160(&v181)];
            v73 = v180;
            v181 = v72;
            v74 = sub_AF4160(v51);
            v29 = (unsigned int)v197;
            v180 = v74 + v73;
            if ( (unsigned int)v197 <= v180 )
            {
LABEL_76:
              v28 = (unsigned __int64 *)v196;
              v30 = (unsigned __int64 *)((char *)v196 + 8 * v29);
              goto LABEL_77;
            }
            break;
        }
      }
      else
      {
        v32 = &v28[(unsigned int)sub_AF4160(&v181)];
        v33 = v180;
        v181 = v32;
        v34 = (unsigned int)sub_AF4160((unsigned __int64 **)&v195) + v33;
        v180 = v34;
LABEL_20:
        v29 = (unsigned int)v197;
        if ( (unsigned int)v197 <= v34 )
          goto LABEL_76;
      }
      v30 = v182;
      v28 = v181;
    }
    v28 = (unsigned __int64 *)v196;
    v29 = (unsigned int)v197;
    v30 = (unsigned __int64 *)((char *)v196 + 8 * (unsigned int)v197);
  }
LABEL_77:
  v186 = v28;
  v199.m128i_i64[0] = (__int64)v200;
  v187 = v30;
  v199.m128i_i64[1] = 0x600000000LL;
  if ( !v29 )
    goto LABEL_93;
  v179 = v29;
  v75 = v28;
  v76 = 0;
  if ( v28 == v30 )
  {
LABEL_92:
    v92 = (_BYTE *)v199.m128i_i64[0];
    if ( (_BYTE *)v199.m128i_i64[0] != v200 )
      goto LABEL_137;
    goto LABEL_93;
  }
  while ( 1 )
  {
    v195.m128i_i8[8] = 1;
    v195.m128i_i64[0] = (__int64)v186;
    v77 = v186;
    v48 = *v186 == 16;
    v194 = _mm_loadu_si128(&v195);
    if ( !v48 )
      break;
    if ( v186[1] )
    {
      v78 = sub_AF4160((unsigned __int64 **)&v195);
      v81 = v187;
      v82 = &v75[v78];
      if ( v187 == v82 )
        goto LABEL_130;
      if ( *v82 != 34 )
        goto LABEL_83;
      v139 = v199.m128i_u32[2];
      v140 = v199.m128i_u32[2] + 1LL;
      if ( v140 > v199.m128i_u32[3] )
      {
        sub_C8D5F0((__int64)&v199, v200, v140, 8u, v79, v80);
        v139 = v199.m128i_u32[2];
      }
      *(_QWORD *)(v199.m128i_i64[0] + 8 * v139) = 35;
      ++v199.m128i_i32[2];
      v141 = v199.m128i_u32[2];
      v142 = v77[1];
      if ( (unsigned __int64)v199.m128i_u32[2] + 1 > v199.m128i_u32[3] )
      {
        sub_C8D5F0((__int64)&v199, v200, v199.m128i_u32[2] + 1LL, 8u, v79, v80);
        v141 = v199.m128i_u32[2];
      }
      *(_QWORD *)(v199.m128i_i64[0] + 8 * v141) = v142;
      v143 = 0;
      v144 = v186;
      ++v199.m128i_i32[2];
      if ( v186 != v187 )
      {
        v195.m128i_i64[0] = (__int64)v186;
        v143 = 1;
        v195.m128i_i8[8] = 1;
        v189 = _mm_loadu_si128(&v195);
      }
      v189.m128i_i8[8] = v143;
      v193 = _mm_loadu_si128(&v189);
      v195 = v193;
      v186 = &v144[(unsigned int)sub_AF4160(&v186)];
      v145 = sub_AF4160((unsigned __int64 **)&v195);
      v146 = v186;
      v76 += v145;
      v147 = 0;
      if ( v186 != v187 )
      {
        v195.m128i_i64[0] = (__int64)v186;
        v147 = 1;
        v195.m128i_i8[8] = 1;
        v190 = _mm_loadu_si128(&v195);
      }
      v190.m128i_i8[8] = v147;
      v193 = _mm_loadu_si128(&v190);
      v195 = v193;
    }
    else
    {
      v148 = v199.m128i_u32[2];
      v149 = v199.m128i_u32[2] + 1LL;
      if ( v149 > v199.m128i_u32[3] )
      {
        sub_C8D5F0((__int64)&v199, v200, v149, 8u, a5, v6);
        v148 = v199.m128i_u32[2];
      }
      *(_QWORD *)(v199.m128i_i64[0] + 8 * v148) = 48;
      v150 = 0;
      v146 = v186;
      ++v199.m128i_i32[2];
      if ( v186 != v187 )
      {
        v195.m128i_i64[0] = (__int64)v186;
        v150 = 1;
        v195.m128i_i8[8] = 1;
        v192 = _mm_loadu_si128(&v195);
      }
      v192.m128i_i8[8] = v150;
      v193 = _mm_loadu_si128(&v192);
      v195 = v193;
    }
    v186 = &v146[(unsigned int)sub_AF4160(&v186)];
    v76 += (unsigned int)sub_AF4160((unsigned __int64 **)&v195);
LABEL_90:
    if ( v76 < v179 )
    {
      v75 = v186;
      if ( v186 != v187 )
        continue;
    }
    goto LABEL_92;
  }
  v130 = sub_AF4160((unsigned __int64 **)&v195);
  v81 = v187;
  if ( v187 != &v75[v130] )
  {
LABEL_83:
    v83 = v186;
    v84 = 0;
    if ( v186 != v81 )
    {
      v195.m128i_i64[0] = (__int64)v186;
      v84 = 1;
      v195.m128i_i8[8] = 1;
      v188 = _mm_loadu_si128(&v195);
    }
    v188.m128i_i8[8] = v84;
    v193 = _mm_loadu_si128(&v188);
    v195 = v193;
    v186 = &v83[(unsigned int)sub_AF4160(&v186)];
    v85 = v76 + (unsigned int)sub_AF4160((unsigned __int64 **)&v195);
    v86 = v76;
    v87 = &v28[v86];
    a5 = (__int64)&v28[v85];
    v88 = 8 * v85 - v86 * 8;
    v89 = v199.m128i_u32[2];
    v90 = v88 >> 3;
    v91 = (v88 >> 3) + v199.m128i_u32[2];
    if ( v91 > v199.m128i_u32[3] )
    {
      v173 = v88;
      sub_C8D5F0((__int64)&v199, v200, v91, 8u, a5, v6);
      v89 = v199.m128i_u32[2];
      v88 = v173;
      a5 = (__int64)&v28[v85];
    }
    if ( (unsigned __int64 *)a5 != v87 )
    {
      memcpy((void *)(v199.m128i_i64[0] + 8 * v89), v87, v88);
      LODWORD(v89) = v199.m128i_i32[2];
    }
    v199.m128i_i32[2] = v89 + v90;
    v76 = v85;
    goto LABEL_90;
  }
LABEL_130:
  v131 = v186;
  v132 = 0;
  if ( v186 != v81 )
  {
    v195.m128i_i64[0] = (__int64)v186;
    v132 = 1;
    v195.m128i_i8[8] = 1;
    v191 = _mm_loadu_si128(&v195);
  }
  v191.m128i_i8[8] = v132;
  v193 = _mm_loadu_si128(&v191);
  v195 = v193;
  v186 = &v131[(unsigned int)sub_AF4160(&v186)];
  v133 = v76 + (unsigned int)sub_AF4160((unsigned __int64 **)&v195);
  v134 = &v28[v133];
  v135 = &v28[v76];
  v136 = v133 * 8 - 8 * v76;
  v137 = v199.m128i_u32[2];
  v138 = (v136 >> 3) + v199.m128i_u32[2];
  if ( v138 > v199.m128i_u32[3] )
  {
    sub_C8D5F0((__int64)&v199, v200, v138, 8u, a5, v6);
    v137 = v199.m128i_u32[2];
  }
  v92 = (_BYTE *)v199.m128i_i64[0];
  if ( v134 != v135 )
  {
    memcpy((void *)(v199.m128i_i64[0] + 8 * v137), v135, v136);
    LODWORD(v137) = v199.m128i_i32[2];
    v92 = (_BYTE *)v199.m128i_i64[0];
  }
  v199.m128i_i32[2] = v137 + (v136 >> 3);
  if ( v92 == v200 )
  {
LABEL_93:
    v93 = v199.m128i_u32[2];
    v94 = (unsigned int)v197;
    v95 = v199.m128i_i32[2];
    if ( v199.m128i_u32[2] <= (unsigned __int64)(unsigned int)v197 )
    {
      v98 = v200;
      v96 = v200;
      if ( v199.m128i_i32[2] )
      {
        memmove(v196, v200, 8LL * v199.m128i_u32[2]);
        v96 = (_BYTE *)v199.m128i_i64[0];
      }
    }
    else
    {
      if ( v199.m128i_u32[2] > (unsigned __int64)HIDWORD(v197) )
      {
        LODWORD(v197) = 0;
        sub_C8D5F0((__int64)&v196, v198, v199.m128i_u32[2], 8u, a5, v6);
        v96 = (_BYTE *)v199.m128i_i64[0];
        v93 = v199.m128i_u32[2];
        v94 = 0;
        v98 = (_BYTE *)v199.m128i_i64[0];
      }
      else
      {
        v96 = v200;
        v97 = 8LL * (unsigned int)v197;
        v98 = v200;
        if ( (_DWORD)v197 )
        {
          memmove(v196, v200, 8LL * (unsigned int)v197);
          v96 = (_BYTE *)v199.m128i_i64[0];
          v93 = v199.m128i_u32[2];
          v94 = v97;
          v98 = (_BYTE *)(v199.m128i_i64[0] + v97);
        }
      }
      v99 = 8 * v93;
      if ( v98 != &v96[v99] )
      {
        memcpy((char *)v196 + v94, v98, v99 - v94);
        v96 = (_BYTE *)v199.m128i_i64[0];
      }
    }
    LODWORD(v197) = v95;
    if ( v96 != v200 )
      _libc_free(v96, v98);
    v92 = v196;
    v100 = (unsigned int)v197;
    goto LABEL_102;
  }
LABEL_137:
  if ( v196 != v198 )
  {
    _libc_free(v196, v92);
    v92 = (_BYTE *)v199.m128i_i64[0];
  }
  v196 = v92;
  v100 = v199.m128i_u32[2];
  v197 = v199.m128i_i64[1];
LABEL_102:
  v101 = (__int64 *)(a1[1] & 0xFFFFFFFFFFFFFFF8LL);
  if ( (a1[1] & 4) != 0 )
    v101 = (__int64 *)*v101;
  v102 = sub_B0D000(v101, (__int64 *)v92, v100, 0, 1);
  if ( (_BYTE *)v204.m128i_i64[0] != v205 )
    _libc_free(v204.m128i_i64[0], v92);
  if ( v196 != v198 )
    _libc_free(v196, v92);
  if ( v201 != v203 )
    _libc_free(v201, v92);
  return v102;
}
