// Function: sub_1A0D7B0
// Address: 0x1a0d7b0
//
__int64 __fastcall sub_1A0D7B0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        __m128 a5,
        __m128 a6,
        __m128i a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 *v11; // r14
  __int64 v14; // r13
  unsigned __int64 v15; // rax
  _BYTE *v16; // rsi
  __int64 *v17; // rdi
  const __m128i *v18; // rcx
  const __m128i *v19; // r8
  unsigned __int64 v20; // rdx
  __int64 v21; // rax
  __m128 *v22; // rdi
  __m128 *v23; // rdx
  const __m128i *v24; // rax
  const __m128i *v25; // r9
  const __m128i *v26; // r8
  unsigned __int64 v27; // r15
  __int64 v28; // rax
  __m128 *v29; // rdi
  __m128 *v30; // rdx
  const __m128i *v31; // rax
  __m128 *v32; // rax
  __m128 *v33; // rax
  char *v34; // rax
  __int64 v35; // rax
  __int64 v36; // rdi
  __int64 v37; // r13
  __int64 v38; // r14
  __int64 v39; // r15
  __int64 v40; // r13
  __int64 v41; // rdx
  __int64 v42; // rcx
  int v43; // r8d
  int v44; // r9d
  double v45; // xmm4_8
  double v46; // xmm5_8
  __int64 v47; // rax
  void *v48; // rax
  const void *v49; // rsi
  __int64 v50; // rax
  __int64 v51; // rsi
  __int64 v52; // rsi
  __int64 v53; // rax
  __int64 v54; // rsi
  _QWORD *v55; // rax
  __int64 *v56; // rsi
  _QWORD *v57; // r8
  _QWORD *v58; // rdx
  _QWORD *v59; // rcx
  _QWORD **v60; // rdi
  __int64 v61; // r9
  int v62; // edx
  __int64 v63; // r13
  __int64 v64; // rcx
  __int64 v65; // rsi
  int v66; // ecx
  unsigned int v67; // edx
  __int64 *v68; // r9
  __int64 v69; // r10
  __int64 i; // rax
  unsigned __int64 v71; // rdi
  __int64 v72; // rcx
  __int64 v73; // rdx
  __int64 *v74; // rax
  int v75; // esi
  __int64 v76; // r15
  __int64 v77; // rdi
  char *v78; // rdx
  int v79; // esi
  __int64 v80; // r10
  unsigned int v81; // ecx
  __int64 *v82; // r8
  __int64 v83; // r9
  __int64 v84; // r8
  char *v85; // rsi
  __int64 v86; // rcx
  __int64 v87; // rcx
  __int64 v88; // rsi
  __int64 v89; // rax
  __int64 v90; // rcx
  __int64 v91; // rax
  __int64 v92; // rax
  __int64 v93; // rdx
  __int64 v94; // rcx
  int v95; // r8d
  int v96; // r9d
  double v97; // xmm4_8
  double v98; // xmm5_8
  int v99; // eax
  unsigned int v100; // ecx
  __int64 v101; // rdx
  _QWORD *v102; // rax
  _QWORD *j; // rdx
  int v104; // eax
  __int64 v105; // rdx
  _QWORD *v106; // rax
  _QWORD *n; // rdx
  __int64 v108; // r14
  int v109; // eax
  __int64 v110; // rdx
  _QWORD *v111; // rax
  _QWORD *ii; // rdx
  __int64 v113; // rsi
  __int64 v114; // r12
  int v116; // r9d
  int v117; // r11d
  int v118; // r8d
  __int64 v119; // rax
  int v120; // r11d
  unsigned int v121; // ecx
  _QWORD *v122; // rdi
  unsigned int v123; // eax
  __int64 v124; // rax
  unsigned __int64 v125; // rax
  __int64 v126; // rax
  int v127; // r12d
  __int64 v128; // r13
  _QWORD *v129; // rax
  __int64 v130; // rdx
  _QWORD *jj; // rdx
  unsigned int v132; // ecx
  _QWORD *v133; // rdi
  unsigned int v134; // eax
  __int64 v135; // rax
  unsigned __int64 v136; // rax
  unsigned __int64 v137; // rax
  int v138; // r13d
  __int64 v139; // r12
  _QWORD *v140; // rax
  __int64 v141; // rdx
  _QWORD *m; // rdx
  _QWORD *v143; // rax
  _QWORD *v144; // rdi
  unsigned int v145; // eax
  __int64 v146; // rax
  unsigned __int64 v147; // rax
  unsigned __int64 v148; // rax
  int v149; // r13d
  __int64 v150; // r12
  _QWORD *v151; // rax
  __int64 v152; // rdx
  _QWORD *k; // rdx
  _QWORD *v154; // rax
  _QWORD *v155; // rax
  __int64 v157; // [rsp+10h] [rbp-380h]
  __int64 *v158; // [rsp+20h] [rbp-370h]
  __int64 v159; // [rsp+30h] [rbp-360h]
  unsigned __int64 v160; // [rsp+38h] [rbp-358h]
  __int64 v161; // [rsp+40h] [rbp-350h] BYREF
  __int64 v162; // [rsp+48h] [rbp-348h]
  __int64 v163; // [rsp+50h] [rbp-340h]
  __int64 v164; // [rsp+60h] [rbp-330h] BYREF
  _QWORD *v165; // [rsp+68h] [rbp-328h]
  _QWORD *v166; // [rsp+70h] [rbp-320h]
  __int64 v167; // [rsp+78h] [rbp-318h]
  int v168; // [rsp+80h] [rbp-310h]
  _QWORD v169[8]; // [rsp+88h] [rbp-308h] BYREF
  const __m128i *v170; // [rsp+C8h] [rbp-2C8h] BYREF
  const __m128i *v171; // [rsp+D0h] [rbp-2C0h]
  __int64 v172; // [rsp+D8h] [rbp-2B8h]
  _QWORD v173[16]; // [rsp+E0h] [rbp-2B0h] BYREF
  _QWORD v174[2]; // [rsp+160h] [rbp-230h] BYREF
  unsigned __int64 v175; // [rsp+170h] [rbp-220h]
  _BYTE v176[64]; // [rsp+188h] [rbp-208h] BYREF
  __m128 *v177; // [rsp+1C8h] [rbp-1C8h]
  __m128 *v178; // [rsp+1D0h] [rbp-1C0h]
  char *v179; // [rsp+1D8h] [rbp-1B8h]
  __int64 v180[2]; // [rsp+1E0h] [rbp-1B0h] BYREF
  unsigned __int64 v181; // [rsp+1F0h] [rbp-1A0h]
  char v182[64]; // [rsp+208h] [rbp-188h] BYREF
  __m128 *v183; // [rsp+248h] [rbp-148h]
  __m128 *v184; // [rsp+250h] [rbp-140h]
  char *v185; // [rsp+258h] [rbp-138h]
  __int64 v186; // [rsp+260h] [rbp-130h] BYREF
  __int64 v187; // [rsp+268h] [rbp-128h]
  unsigned __int64 v188; // [rsp+270h] [rbp-120h]
  char *v189; // [rsp+278h] [rbp-118h]
  _BYTE v190[64]; // [rsp+288h] [rbp-108h] BYREF
  unsigned __int64 *v191; // [rsp+2C8h] [rbp-C8h]
  __m128 *v192; // [rsp+2D0h] [rbp-C0h]
  char *v193; // [rsp+2D8h] [rbp-B8h]
  __m128i v194; // [rsp+2E0h] [rbp-B0h] BYREF
  void **v195; // [rsp+2F0h] [rbp-A0h]
  __int64 v196; // [rsp+2F8h] [rbp-98h]
  __int64 v197; // [rsp+300h] [rbp-90h] BYREF
  void *v198; // [rsp+308h] [rbp-88h] BYREF
  _QWORD *v199; // [rsp+310h] [rbp-80h]
  __int64 v200; // [rsp+318h] [rbp-78h] BYREF
  __int64 *v201; // [rsp+320h] [rbp-70h]
  __int64 *v202; // [rsp+328h] [rbp-68h]
  __int64 v203; // [rsp+330h] [rbp-60h]
  unsigned __int64 v204; // [rsp+338h] [rbp-58h]
  __int64 v205; // [rsp+340h] [rbp-50h] BYREF
  unsigned __int64 *v206; // [rsp+348h] [rbp-48h]
  __m128 *v207; // [rsp+350h] [rbp-40h]
  char *v208; // [rsp+358h] [rbp-38h]

  v11 = &v186;
  v14 = *(_QWORD *)(a3 + 80);
  v161 = 0;
  v162 = 0;
  if ( v14 )
    v14 -= 24;
  memset(v173, 0, sizeof(v173));
  v163 = 0;
  v173[1] = &v173[5];
  v173[2] = &v173[5];
  v165 = v169;
  v166 = v169;
  v169[0] = v14;
  LODWORD(v173[3]) = 8;
  v170 = 0;
  v171 = 0;
  v172 = 0;
  v167 = 0x100000008LL;
  v168 = 0;
  v164 = 1;
  v15 = sub_157EBA0(v14);
  v194.m128i_i64[0] = v14;
  v194.m128i_i64[1] = v15;
  LODWORD(v195) = 0;
  sub_13FDF40(&v170, 0, &v194);
  sub_13FE0F0((__int64)&v164);
  v16 = v190;
  v17 = &v186;
  sub_16CCCB0(&v186, (__int64)v190, (__int64)v173);
  v18 = (const __m128i *)v173[14];
  v19 = (const __m128i *)v173[13];
  v191 = 0;
  v192 = 0;
  v193 = 0;
  v20 = v173[14] - v173[13];
  if ( v173[14] == v173[13] )
  {
    v22 = 0;
  }
  else
  {
    if ( v20 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_199;
    v160 = v173[14] - v173[13];
    v21 = sub_22077B0(v173[14] - v173[13]);
    v18 = (const __m128i *)v173[14];
    v19 = (const __m128i *)v173[13];
    v20 = v160;
    v22 = (__m128 *)v21;
  }
  v191 = (unsigned __int64 *)v22;
  v192 = v22;
  v193 = (char *)v22 + v20;
  if ( v18 != v19 )
  {
    v23 = v22;
    v24 = v19;
    do
    {
      if ( v23 )
      {
        a4 = (__m128)_mm_loadu_si128(v24);
        *v23 = a4;
        v23[1].m128_u64[0] = v24[1].m128i_u64[0];
      }
      v24 = (const __m128i *)((char *)v24 + 24);
      v23 = (__m128 *)((char *)v23 + 24);
    }
    while ( v24 != v18 );
    v22 = (__m128 *)((char *)v22 + 8 * ((unsigned __int64)((char *)&v24[-2].m128i_u64[1] - (char *)v19) >> 3) + 24);
  }
  v192 = v22;
  sub_16CCEE0(&v194, (__int64)&v198, 8, (__int64)&v186);
  v17 = v174;
  v16 = v176;
  v206 = v191;
  v191 = 0;
  v207 = v192;
  v192 = 0;
  v208 = v193;
  v193 = 0;
  sub_16CCCB0(v174, (__int64)v176, (__int64)&v164);
  v25 = v171;
  v26 = v170;
  v177 = 0;
  v178 = 0;
  v179 = 0;
  v27 = (char *)v171 - (char *)v170;
  if ( v171 != v170 )
  {
    if ( v27 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v28 = sub_22077B0((char *)v171 - (char *)v170);
      v25 = v171;
      v26 = v170;
      v29 = (__m128 *)v28;
      goto LABEL_15;
    }
LABEL_199:
    sub_4261EA(v17, v16, v20);
  }
  v29 = 0;
LABEL_15:
  v177 = v29;
  v178 = v29;
  v179 = (char *)v29 + v27;
  if ( v26 != v25 )
  {
    v30 = v29;
    v31 = v26;
    do
    {
      if ( v30 )
      {
        a5 = (__m128)_mm_loadu_si128(v31);
        *v30 = a5;
        v30[1].m128_u64[0] = v31[1].m128i_u64[0];
      }
      v31 = (const __m128i *)((char *)v31 + 24);
      v30 = (__m128 *)((char *)v30 + 24);
    }
    while ( v31 != v25 );
    v29 = (__m128 *)((char *)v29 + 8 * ((unsigned __int64)((char *)&v31[-2].m128i_u64[1] - (char *)v26) >> 3) + 24);
  }
  v178 = v29;
  sub_16CCEE0(v180, (__int64)v182, 8, (__int64)v174);
  v32 = v177;
  v177 = 0;
  v183 = v32;
  v33 = v178;
  v178 = 0;
  v184 = v33;
  v34 = v179;
  v179 = 0;
  v185 = v34;
  sub_19380F0((__int64)v180, (__int64)&v194, (__int64)&v161);
  if ( v183 )
    j_j___libc_free_0(v183, v185 - (char *)v183);
  if ( v181 != v180[1] )
    _libc_free(v181);
  if ( v177 )
    j_j___libc_free_0(v177, v179 - (char *)v177);
  if ( v175 != v174[1] )
    _libc_free(v175);
  if ( v206 )
    j_j___libc_free_0(v206, v208 - (char *)v206);
  if ( v195 != (void **)v194.m128i_i64[1] )
    _libc_free((unsigned __int64)v195);
  if ( v191 )
    j_j___libc_free_0(v191, v193 - (char *)v191);
  if ( v188 != v187 )
    _libc_free(v188);
  if ( v170 )
    j_j___libc_free_0(v170, v172 - (_QWORD)v170);
  if ( v166 != v165 )
    _libc_free((unsigned __int64)v166);
  if ( v173[13] )
    j_j___libc_free_0(v173[13], v173[15] - v173[13]);
  if ( v173[2] != v173[1] )
    _libc_free(v173[2]);
  sub_1A040C0(a2, a3, &v161);
  sub_1A029A0(a2, &v161);
  v35 = v162;
  v36 = v161;
  *(_BYTE *)(a2 + 752) = 0;
  v159 = v35;
  v157 = v36;
  if ( v36 != v35 )
  {
    while ( 1 )
    {
      v37 = *(_QWORD *)(v159 - 8);
      if ( *(_QWORD *)(v37 + 48) != v37 + 40 )
      {
        v158 = v11;
        v38 = *(_QWORD *)(v37 + 48);
        v39 = v37 + 40;
        do
        {
          while ( 1 )
          {
            v40 = v38 - 24;
            if ( !v38 )
              v40 = 0;
            if ( !(unsigned __int8)sub_1AE9990(v40, 0) )
              break;
            v38 = *(_QWORD *)(v38 + 8);
            sub_1A0CF30(a2, v40, v41, v42, v43, v44);
            if ( v39 == v38 )
              goto LABEL_53;
          }
          sub_1A0C020(a2, v40, a4, a5, a6, a7, v45, v46, a10, a11);
          v38 = *(_QWORD *)(v38 + 8);
        }
        while ( v39 != v38 );
LABEL_53:
        v11 = v158;
      }
      v194 = 0u;
      v195 = 0;
      v196 = 0;
      j___libc_free_0(0);
      v47 = *(unsigned int *)(a2 + 88);
      LODWORD(v196) = v47;
      if ( (_DWORD)v47 )
      {
        v48 = (void *)sub_22077B0(8 * v47);
        v49 = *(const void **)(a2 + 72);
        v194.m128i_i64[1] = (__int64)v48;
        v195 = *(void ***)(a2 + 80);
        memcpy(v48, v49, 8LL * (unsigned int)v196);
      }
      else
      {
        v194.m128i_i64[1] = 0;
        v195 = 0;
      }
      v50 = *(_QWORD *)(a2 + 168) - *(_QWORD *)(a2 + 136);
      v197 = 0;
      v51 = *(_QWORD *)(a2 + 144);
      v199 = 0;
      v52 = v51 - *(_QWORD *)(a2 + 152);
      v200 = 0;
      v201 = 0;
      v53 = (v52 >> 3) + (((v50 >> 3) - 1) << 6);
      v54 = *(_QWORD *)(a2 + 128) - *(_QWORD *)(a2 + 112);
      v202 = 0;
      v198 = 0;
      v203 = 0;
      v204 = 0;
      v205 = 0;
      v206 = 0;
      sub_1A02210(&v197, v53 + (v54 >> 3));
      v55 = v199;
      v56 = v201;
      v57 = *(_QWORD **)(a2 + 144);
      v58 = *(_QWORD **)(a2 + 112);
      v59 = *(_QWORD **)(a2 + 128);
      v60 = (_QWORD **)(v202 + 1);
      v61 = *(_QWORD *)(a2 + 136);
      while ( v57 != v58 )
      {
        if ( v55 )
          *v55 = *v58;
        if ( ++v58 == v59 )
        {
          v58 = *(_QWORD **)(v61 + 8);
          v61 += 8;
          v59 = v58 + 64;
        }
        if ( v56 == ++v55 )
        {
          v55 = *v60++;
          v56 = v55 + 64;
        }
      }
LABEL_70:
      for ( i = v203; (_QWORD *)v203 != v199; i = v203 )
      {
        v71 = v204;
        if ( i == v204 )
        {
          v72 = *(v206 - 1);
          v62 = v196;
          v63 = *(_QWORD *)(v72 + 504);
          v64 = v72 + 512;
          if ( !(_DWORD)v196 )
            goto LABEL_73;
        }
        else
        {
          v62 = v196;
          v63 = *(_QWORD *)(i - 8);
          if ( !(_DWORD)v196 )
            goto LABEL_69;
          v64 = i;
        }
        v65 = *(_QWORD *)(v64 - 8);
        v66 = v62 - 1;
        v67 = (v62 - 1) & (((unsigned int)v65 >> 9) ^ ((unsigned int)v65 >> 4));
        v68 = (__int64 *)(v194.m128i_i64[1] + 8LL * v67);
        v69 = *v68;
        if ( v65 == *v68 )
        {
LABEL_67:
          *v68 = -16;
          i = v203;
          LODWORD(v195) = (_DWORD)v195 - 1;
          v71 = v204;
          ++HIDWORD(v195);
        }
        else
        {
          v116 = 1;
          while ( v69 != -8 )
          {
            v117 = v116 + 1;
            v67 = v66 & (v116 + v67);
            v68 = (__int64 *)(v194.m128i_i64[1] + 8LL * v67);
            v69 = *v68;
            if ( v65 == *v68 )
              goto LABEL_67;
            v116 = v117;
          }
        }
        if ( i == v71 )
        {
LABEL_73:
          j_j___libc_free_0(v71, 512);
          v73 = *--v206 + 512;
          v204 = *v206;
          v205 = v73;
          v203 = v204 + 504;
          if ( !(unsigned __int8)sub_1AE9990(v63, 0) )
            goto LABEL_70;
          goto LABEL_74;
        }
LABEL_69:
        v203 = i - 8;
        if ( !(unsigned __int8)sub_1AE9990(v63, 0) )
          goto LABEL_70;
LABEL_74:
        sub_1A0D410(a2, v63, (__int64)&v194);
        *(_BYTE *)(a2 + 752) = 1;
      }
      v74 = *(__int64 **)(a2 + 112);
      if ( *(__int64 **)(a2 + 144) != v74 )
        break;
LABEL_86:
      sub_1A019D0(&v197);
      j___libc_free_0(v194.m128i_i64[1]);
      v159 -= 8;
      if ( v157 == v159 )
        goto LABEL_87;
    }
    while ( 1 )
    {
      v75 = *(_DWORD *)(a2 + 88);
      v76 = *v74;
      v77 = *(_QWORD *)(a2 + 120);
      v78 = *(char **)(a2 + 136);
      if ( v75 )
      {
        v79 = v75 - 1;
        v80 = *(_QWORD *)(a2 + 72);
        v81 = v79 & (((unsigned int)v76 >> 9) ^ ((unsigned int)v76 >> 4));
        v82 = (__int64 *)(v80 + 8LL * v81);
        v83 = *v82;
        if ( v76 == *v82 )
        {
LABEL_80:
          *v82 = -16;
          v84 = *(_QWORD *)(a2 + 120);
          v85 = *(char **)(a2 + 136);
          --*(_DWORD *)(a2 + 80);
          ++*(_DWORD *)(a2 + 84);
          v86 = v78 - v85;
          v78 = v85;
          v87 = ((v86 >> 3) - 1) << 6;
          goto LABEL_81;
        }
        v118 = 1;
        while ( v83 != -8 )
        {
          v120 = v118 + 1;
          v81 = v79 & (v118 + v81);
          v82 = (__int64 *)(v80 + 8LL * v81);
          v83 = *v82;
          if ( v76 == *v82 )
            goto LABEL_80;
          v118 = v120;
        }
      }
      v84 = *(_QWORD *)(a2 + 120);
      v87 = -64;
LABEL_81:
      v88 = *(_QWORD *)(a2 + 112);
      v89 = (((__int64)v74 - v77) >> 3) + v87 + ((*(_QWORD *)(a2 + 128) - v88) >> 3);
      v90 = v89 + ((v88 - v84) >> 3);
      if ( v90 < 0 )
      {
        v119 = ~((unsigned __int64)~v90 >> 6);
        goto LABEL_129;
      }
      if ( v90 > 63 )
      {
        v119 = v90 >> 6;
LABEL_129:
        v78 += 8 * v119;
        v91 = *(_QWORD *)v78 + 8 * (v90 - (v119 << 6));
        goto LABEL_84;
      }
      v91 = v88 + 8 * v89;
LABEL_84:
      v186 = v91;
      v92 = *(_QWORD *)v78;
      v189 = v78;
      v187 = v92;
      v188 = v92 + 512;
      sub_1A0CA40(v180, (_QWORD *)(a2 + 96), v11);
      if ( (unsigned __int8)sub_1AE9990(v76, 0) )
      {
        sub_1A0CF30(a2, v76, v93, v94, v95, v96);
        v74 = *(__int64 **)(a2 + 112);
        if ( *(__int64 **)(a2 + 144) == v74 )
          goto LABEL_86;
      }
      else
      {
        sub_1A0C020(a2, v76, a4, a5, a6, a7, v97, v98, a10, a11);
        v74 = *(__int64 **)(a2 + 112);
        if ( *(__int64 **)(a2 + 144) == v74 )
          goto LABEL_86;
      }
    }
  }
LABEL_87:
  v99 = *(_DWORD *)(a2 + 16);
  ++*(_QWORD *)a2;
  if ( !v99 )
  {
    if ( !*(_DWORD *)(a2 + 20) )
      goto LABEL_94;
    v101 = *(unsigned int *)(a2 + 24);
    if ( (unsigned int)v101 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a2 + 8));
      *(_QWORD *)(a2 + 8) = 0;
      *(_QWORD *)(a2 + 16) = 0;
      *(_DWORD *)(a2 + 24) = 0;
      goto LABEL_94;
    }
    goto LABEL_91;
  }
  v100 = 4 * v99;
  v101 = *(unsigned int *)(a2 + 24);
  if ( (unsigned int)(4 * v99) < 0x40 )
    v100 = 64;
  if ( v100 >= (unsigned int)v101 )
  {
LABEL_91:
    v102 = *(_QWORD **)(a2 + 8);
    for ( j = &v102[2 * v101]; j != v102; v102 += 2 )
      *v102 = -8;
    *(_QWORD *)(a2 + 16) = 0;
    goto LABEL_94;
  }
  v144 = *(_QWORD **)(a2 + 8);
  v145 = v99 - 1;
  if ( !v145 )
  {
    v150 = 2048;
    v149 = 128;
LABEL_182:
    j___libc_free_0(v144);
    *(_DWORD *)(a2 + 24) = v149;
    v151 = (_QWORD *)sub_22077B0(v150);
    v152 = *(unsigned int *)(a2 + 24);
    *(_QWORD *)(a2 + 16) = 0;
    *(_QWORD *)(a2 + 8) = v151;
    for ( k = &v151[2 * v152]; k != v151; v151 += 2 )
    {
      if ( v151 )
        *v151 = -8;
    }
    goto LABEL_94;
  }
  _BitScanReverse(&v145, v145);
  v146 = (unsigned int)(1 << (33 - (v145 ^ 0x1F)));
  if ( (int)v146 < 64 )
    v146 = 64;
  if ( (_DWORD)v146 != (_DWORD)v101 )
  {
    v147 = (4 * (int)v146 / 3u + 1) | ((unsigned __int64)(4 * (int)v146 / 3u + 1) >> 1);
    v148 = ((v147 | (v147 >> 2)) >> 4) | v147 | (v147 >> 2) | ((((v147 | (v147 >> 2)) >> 4) | v147 | (v147 >> 2)) >> 8);
    v149 = (v148 | (v148 >> 16)) + 1;
    v150 = 16 * ((v148 | (v148 >> 16)) + 1);
    goto LABEL_182;
  }
  *(_QWORD *)(a2 + 16) = 0;
  v154 = &v144[2 * v146];
  do
  {
    if ( v144 )
      *v144 = -8;
    v144 += 2;
  }
  while ( v154 != v144 );
LABEL_94:
  v104 = *(_DWORD *)(a2 + 48);
  ++*(_QWORD *)(a2 + 32);
  if ( v104 )
  {
    v132 = 4 * v104;
    v105 = *(unsigned int *)(a2 + 56);
    if ( (unsigned int)(4 * v104) < 0x40 )
      v132 = 64;
    if ( v132 >= (unsigned int)v105 )
      goto LABEL_97;
    v133 = *(_QWORD **)(a2 + 40);
    v134 = v104 - 1;
    if ( v134 )
    {
      _BitScanReverse(&v134, v134);
      v135 = (unsigned int)(1 << (33 - (v134 ^ 0x1F)));
      if ( (int)v135 < 64 )
        v135 = 64;
      if ( (_DWORD)v135 == (_DWORD)v105 )
      {
        *(_QWORD *)(a2 + 48) = 0;
        v155 = &v133[2 * v135];
        do
        {
          if ( v133 )
            *v133 = -8;
          v133 += 2;
        }
        while ( v155 != v133 );
        goto LABEL_100;
      }
      v136 = (4 * (int)v135 / 3u + 1) | ((unsigned __int64)(4 * (int)v135 / 3u + 1) >> 1);
      v137 = ((v136 | (v136 >> 2)) >> 4)
           | v136
           | (v136 >> 2)
           | ((((v136 | (v136 >> 2)) >> 4) | v136 | (v136 >> 2)) >> 8);
      v138 = (v137 | (v137 >> 16)) + 1;
      v139 = 16 * ((v137 | (v137 >> 16)) + 1);
    }
    else
    {
      v139 = 2048;
      v138 = 128;
    }
    j___libc_free_0(v133);
    *(_DWORD *)(a2 + 56) = v138;
    v140 = (_QWORD *)sub_22077B0(v139);
    v141 = *(unsigned int *)(a2 + 56);
    *(_QWORD *)(a2 + 48) = 0;
    *(_QWORD *)(a2 + 40) = v140;
    for ( m = &v140[2 * v141]; m != v140; v140 += 2 )
    {
      if ( v140 )
        *v140 = -8;
    }
  }
  else if ( *(_DWORD *)(a2 + 52) )
  {
    v105 = *(unsigned int *)(a2 + 56);
    if ( (unsigned int)v105 <= 0x40 )
    {
LABEL_97:
      v106 = *(_QWORD **)(a2 + 40);
      for ( n = &v106[2 * v105]; n != v106; v106 += 2 )
        *v106 = -8;
      *(_QWORD *)(a2 + 48) = 0;
      goto LABEL_100;
    }
    j___libc_free_0(*(_QWORD *)(a2 + 40));
    *(_QWORD *)(a2 + 40) = 0;
    *(_QWORD *)(a2 + 48) = 0;
    *(_DWORD *)(a2 + 56) = 0;
  }
LABEL_100:
  v108 = a2 + 176;
  do
  {
    v109 = *(_DWORD *)(v108 + 16);
    ++*(_QWORD *)v108;
    if ( !v109 )
    {
      if ( !*(_DWORD *)(v108 + 20) )
        goto LABEL_107;
      v110 = *(unsigned int *)(v108 + 24);
      if ( (unsigned int)v110 > 0x40 )
      {
        j___libc_free_0(*(_QWORD *)(v108 + 8));
        *(_DWORD *)(v108 + 24) = 0;
        *(_QWORD *)(v108 + 8) = 0;
      }
      else
      {
LABEL_104:
        v111 = *(_QWORD **)(v108 + 8);
        for ( ii = &v111[3 * v110]; ii != v111; *(v111 - 2) = -8 )
        {
          *v111 = -8;
          v111 += 3;
        }
      }
      *(_DWORD *)(v108 + 16) = 0;
      *(_DWORD *)(v108 + 20) = 0;
      goto LABEL_107;
    }
    v121 = 4 * v109;
    v110 = *(unsigned int *)(v108 + 24);
    if ( (unsigned int)(4 * v109) < 0x40 )
      v121 = 64;
    if ( v121 >= (unsigned int)v110 )
      goto LABEL_104;
    v122 = *(_QWORD **)(v108 + 8);
    v123 = v109 - 1;
    if ( !v123 )
    {
      v128 = 3072;
      v127 = 128;
LABEL_145:
      j___libc_free_0(v122);
      *(_DWORD *)(v108 + 24) = v127;
      v129 = (_QWORD *)sub_22077B0(v128);
      v130 = *(unsigned int *)(v108 + 24);
      *(_DWORD *)(v108 + 16) = 0;
      *(_QWORD *)(v108 + 8) = v129;
      *(_DWORD *)(v108 + 20) = 0;
      for ( jj = &v129[3 * v130]; jj != v129; v129 += 3 )
      {
        if ( v129 )
        {
          *v129 = -8;
          v129[1] = -8;
        }
      }
      goto LABEL_107;
    }
    _BitScanReverse(&v123, v123);
    v124 = (unsigned int)(1 << (33 - (v123 ^ 0x1F)));
    if ( (int)v124 < 64 )
      v124 = 64;
    if ( (_DWORD)v124 != (_DWORD)v110 )
    {
      v125 = (4 * (int)v124 / 3u + 1) | ((unsigned __int64)(4 * (int)v124 / 3u + 1) >> 1);
      v126 = ((((v125 >> 2) | v125 | (((v125 >> 2) | v125) >> 4)) >> 8)
            | (v125 >> 2)
            | v125
            | (((v125 >> 2) | v125) >> 4)
            | (((((v125 >> 2) | v125 | (((v125 >> 2) | v125) >> 4)) >> 8)
              | (v125 >> 2)
              | v125
              | (((v125 >> 2) | v125) >> 4)) >> 16))
           + 1;
      v127 = v126;
      v128 = 24 * v126;
      goto LABEL_145;
    }
    *(_DWORD *)(v108 + 16) = 0;
    *(_DWORD *)(v108 + 20) = 0;
    v143 = &v122[3 * v124];
    do
    {
      if ( v122 )
      {
        *v122 = -8;
        v122[1] = -8;
      }
      v122 += 3;
    }
    while ( v143 != v122 );
LABEL_107:
    v108 += 32;
  }
  while ( v108 != a2 + 752 );
  v113 = a1 + 40;
  v114 = a1 + 96;
  if ( *(_BYTE *)(a2 + 752) )
  {
    v200 = 0;
    v196 = 0x100000002LL;
    v194.m128i_i64[1] = (__int64)&v198;
    v195 = &v198;
    v201 = &v205;
    v202 = &v205;
    v203 = 2;
    LODWORD(v197) = 0;
    LODWORD(v204) = 0;
    v198 = &unk_4F9EE50;
    v194.m128i_i64[0] = 1;
    if ( (&unk_4F9EE50 != &unk_4F9EE48 || (unsigned __int64)&unk_4F9EE52 <= 1) && &unk_4F9EE50 != &unk_4F98E60 )
    {
      if ( &unk_4F9EE50 == (_UNKNOWN *)-2LL )
      {
        v198 = &unk_4F98E60;
        LODWORD(v197) = -1;
      }
      else
      {
        HIDWORD(v196) = 2;
        v199 = &unk_4F98E60;
      }
      v194.m128i_i64[0] = 2;
    }
    sub_16CCEE0((_QWORD *)a1, v113, 2, (__int64)&v194);
    sub_16CCEE0((_QWORD *)(a1 + 56), v114, 2, (__int64)&v200);
    if ( v202 != v201 )
      _libc_free((unsigned __int64)v202);
    if ( v195 != (void **)v194.m128i_i64[1] )
      _libc_free((unsigned __int64)v195);
  }
  else
  {
    *(_QWORD *)(a1 + 24) = 0x100000002LL;
    *(_QWORD *)(a1 + 8) = v113;
    *(_QWORD *)(a1 + 16) = v113;
    *(_QWORD *)(a1 + 56) = 0;
    *(_QWORD *)(a1 + 64) = v114;
    *(_QWORD *)(a1 + 72) = v114;
    *(_QWORD *)(a1 + 80) = 2;
    *(_DWORD *)(a1 + 88) = 0;
    *(_DWORD *)(a1 + 32) = 0;
    *(_QWORD *)a1 = 1;
    *(_QWORD *)(a1 + 40) = &unk_4F9EE48;
  }
  if ( v161 )
    j_j___libc_free_0(v161, v163 - v161);
  return a1;
}
