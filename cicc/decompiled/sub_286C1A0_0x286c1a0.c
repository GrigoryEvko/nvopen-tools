// Function: sub_286C1A0
// Address: 0x286c1a0
//
__int64 __fastcall sub_286C1A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // r13
  __int64 v8; // r12
  __int64 v9; // rbx
  unsigned __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  const __m128i *v13; // r14
  __int64 v14; // r12
  unsigned __int64 v15; // rdx
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rbx
  __int64 v20; // r8
  __int64 v21; // r13
  __int64 v22; // rdi
  __int64 v23; // rbx
  __int64 *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 *v27; // r14
  __int64 v28; // rbx
  __int64 *v29; // r12
  __int64 v30; // rax
  __int64 v31; // rbx
  __int64 v32; // rbx
  __int64 v33; // r14
  __int64 v34; // rbx
  __int64 v35; // r12
  __int64 *v36; // rax
  __int64 v37; // rcx
  unsigned __int64 v38; // rdx
  unsigned __int64 i; // r14
  unsigned __int64 v40; // rsi
  __int64 v41; // rdi
  __int64 *v42; // rax
  __int64 *v43; // r12
  __int64 *v44; // rbx
  __int64 v45; // r13
  __int64 v46; // rdx
  __int64 v47; // rcx
  __m128i v48; // xmm5
  __m128i v49; // xmm5
  unsigned __int64 v50; // rax
  __int64 v51; // r15
  unsigned __int64 v52; // rdi
  __int64 *v53; // rbx
  __m128i *v54; // r12
  __int64 v56; // r15
  __int64 v57; // rbx
  __int64 v58; // rcx
  __m128i v59; // xmm7
  __m128i v60; // xmm0
  __m128i v61; // xmm6
  __int64 v62; // rbx
  __int64 v63; // r9
  __int64 **v64; // r12
  __int64 v65; // r12
  __int64 v66; // r15
  __int64 v67; // rcx
  __int64 v68; // r14
  _BYTE *v69; // r13
  __int64 v70; // rdi
  __int64 v71; // rbx
  __int64 v72; // rax
  __int64 v73; // rbx
  __int64 *v74; // r12
  __int64 v75; // rbx
  __int64 v76; // r12
  __int64 v77; // r13
  __int64 v78; // r14
  __m128i v79; // xmm7
  __m128i v80; // xmm3
  _QWORD *v81; // rax
  __int64 v82; // rdx
  __int64 v83; // rcx
  __int64 v84; // r8
  __int64 v85; // r9
  __int64 v86; // r15
  __m128i v87; // xmm3
  __m128i v88; // xmm4
  unsigned __int64 v89; // rdi
  __int64 v90; // rbx
  __int64 v91; // rax
  __int64 v92; // rsi
  __int64 v93; // rcx
  __m128i v94; // xmm6
  __int64 *v95; // rbx
  __int64 *v96; // r15
  __int64 v97; // r14
  __int64 v98; // rax
  unsigned __int64 v99; // rdx
  __int64 v100; // r12
  __int64 v101; // rax
  unsigned __int64 v102; // rdx
  __int64 v103; // rax
  const __m128i *v104; // r13
  __int64 v105; // r14
  __int64 v106; // rbx
  __int64 v107; // rdi
  __int64 v108; // r15
  __int64 *v109; // rdi
  __int64 v110; // r13
  __int64 v111; // rax
  __int64 v112; // r15
  __int64 v113; // rbx
  __int64 v114; // r14
  _QWORD *v115; // rax
  bool v116; // al
  __m128i v117; // xmm0
  __int64 v118; // rsi
  int v119; // ecx
  __int64 *v120; // rax
  __m128i v121; // xmm1
  unsigned __int64 *v122; // r12
  unsigned __int64 *v123; // rax
  __int64 v124; // rsi
  char v125; // al
  __int64 v126; // rax
  __int64 v127; // rbx
  unsigned __int64 v128; // rdx
  __int64 v129; // rax
  __int64 v130; // rcx
  unsigned __int64 v131; // rdx
  __int64 v132; // rdx
  __int64 v133; // rsi
  __int64 v134; // r8
  __int64 v135; // r9
  unsigned __int64 v136; // r12
  _QWORD *v137; // rbx
  __int64 v138; // r9
  __int64 v139; // rax
  unsigned __int64 v140; // rdx
  __int64 *v141; // rdi
  __int64 v142; // rdx
  __int64 v143; // rcx
  _QWORD *v144; // rbx
  __int64 v145; // r8
  __m128i v146; // xmm4
  __int64 v147; // rax
  __int64 v148; // rcx
  unsigned __int64 v149; // rdx
  __int64 v150; // rdx
  __int64 v151; // rsi
  __int64 v152; // r8
  __int64 v153; // r9
  unsigned __int64 *v154; // rax
  unsigned __int64 *v155; // r14
  unsigned __int64 *v156; // rbx
  unsigned __int64 v157; // rax
  unsigned __int64 v158; // r15
  char v159; // al
  __int64 v160; // rdx
  __int64 v161; // rcx
  __int64 *v162; // r14
  __int64 *v163; // rax
  __int64 *v164; // r12
  __int64 v165; // rbx
  __int64 *v166; // r13
  __int64 v167; // r8
  __int64 v168; // r9
  __int64 v169; // rdi
  __int64 v170; // rax
  __int64 *v171; // r14
  __int64 v172; // r13
  char v173; // al
  __int64 v174; // rsi
  __int64 v175; // r10
  char v176; // al
  __m128i v177; // xmm4
  __int64 *v178; // rdi
  __m128i v179; // xmm7
  char v180; // dl
  __int64 v181; // rax
  __int64 v182; // rdi
  _QWORD *v183; // rax
  __int64 v184; // r10
  __int64 v185; // r13
  __int64 v186; // rbx
  __int64 v187; // r14
  __int64 v188; // r12
  __int64 *v189; // rax
  _QWORD *v190; // rax
  __int64 *v191; // rax
  __int64 *v192; // rdx
  _QWORD *v193; // rax
  __int64 v194; // rsi
  unsigned __int64 v195; // [rsp-8h] [rbp-288h]
  __int64 v196; // [rsp+8h] [rbp-278h]
  __int64 v197; // [rsp+8h] [rbp-278h]
  __int64 v198; // [rsp+10h] [rbp-270h]
  __int64 v199; // [rsp+10h] [rbp-270h]
  _BYTE *v200; // [rsp+18h] [rbp-268h]
  __int64 *v201; // [rsp+20h] [rbp-260h]
  __int64 v202; // [rsp+28h] [rbp-258h]
  __int64 v203; // [rsp+28h] [rbp-258h]
  __int64 v204; // [rsp+30h] [rbp-250h]
  __int64 v205; // [rsp+30h] [rbp-250h]
  __int64 *v206; // [rsp+30h] [rbp-250h]
  unsigned __int64 v207; // [rsp+30h] [rbp-250h]
  __int64 v208; // [rsp+30h] [rbp-250h]
  unsigned __int64 v209; // [rsp+30h] [rbp-250h]
  __int64 v210; // [rsp+38h] [rbp-248h]
  __int64 v211; // [rsp+38h] [rbp-248h]
  __int64 v212; // [rsp+40h] [rbp-240h]
  __int64 v213; // [rsp+48h] [rbp-238h]
  __int64 v214; // [rsp+48h] [rbp-238h]
  __int64 v215; // [rsp+48h] [rbp-238h]
  __int64 v216; // [rsp+50h] [rbp-230h]
  __int64 v217; // [rsp+50h] [rbp-230h]
  __int64 v218; // [rsp+58h] [rbp-228h]
  __int64 v219; // [rsp+60h] [rbp-220h]
  __int64 v220; // [rsp+60h] [rbp-220h]
  __int8 v221; // [rsp+60h] [rbp-220h]
  __int64 v222; // [rsp+60h] [rbp-220h]
  __int64 *v223; // [rsp+60h] [rbp-220h]
  __m128i *v224; // [rsp+68h] [rbp-218h]
  __int64 v225; // [rsp+68h] [rbp-218h]
  __int64 *v226; // [rsp+68h] [rbp-218h]
  bool v227; // [rsp+68h] [rbp-218h]
  __int64 v228; // [rsp+70h] [rbp-210h]
  __int64 *v229; // [rsp+78h] [rbp-208h]
  __int64 v230; // [rsp+78h] [rbp-208h]
  __int64 v231; // [rsp+78h] [rbp-208h]
  __int64 v232; // [rsp+78h] [rbp-208h]
  char v233; // [rsp+78h] [rbp-208h]
  __int64 v234; // [rsp+78h] [rbp-208h]
  __int64 v235; // [rsp+78h] [rbp-208h]
  __int64 v236; // [rsp+80h] [rbp-200h]
  __int64 v237; // [rsp+80h] [rbp-200h]
  __int64 v238; // [rsp+80h] [rbp-200h]
  __int64 v239; // [rsp+80h] [rbp-200h]
  _QWORD *v240; // [rsp+80h] [rbp-200h]
  __int64 v241; // [rsp+80h] [rbp-200h]
  __int64 *v242; // [rsp+80h] [rbp-200h]
  __int64 v243; // [rsp+88h] [rbp-1F8h]
  __int64 v244; // [rsp+88h] [rbp-1F8h]
  __int64 v245; // [rsp+88h] [rbp-1F8h]
  __int64 v246; // [rsp+88h] [rbp-1F8h]
  __int64 v247; // [rsp+88h] [rbp-1F8h]
  __int64 v248; // [rsp+88h] [rbp-1F8h]
  __int64 v249; // [rsp+90h] [rbp-1F0h]
  __int64 v250; // [rsp+90h] [rbp-1F0h]
  __int64 v251; // [rsp+90h] [rbp-1F0h]
  __int64 v252; // [rsp+90h] [rbp-1F0h]
  __int64 v253; // [rsp+90h] [rbp-1F0h]
  __int64 v254; // [rsp+90h] [rbp-1F0h]
  __int64 v255; // [rsp+98h] [rbp-1E8h]
  __int64 v256; // [rsp+98h] [rbp-1E8h]
  __int64 **v257; // [rsp+98h] [rbp-1E8h]
  __int64 v258; // [rsp+98h] [rbp-1E8h]
  __int64 v259; // [rsp+98h] [rbp-1E8h]
  __int64 *v260; // [rsp+98h] [rbp-1E8h]
  _BYTE *v261; // [rsp+A0h] [rbp-1E0h] BYREF
  __int64 v262; // [rsp+A8h] [rbp-1D8h]
  _BYTE v263[32]; // [rsp+B0h] [rbp-1D0h] BYREF
  unsigned __int64 v264[2]; // [rsp+D0h] [rbp-1B0h] BYREF
  _BYTE v265[32]; // [rsp+E0h] [rbp-1A0h] BYREF
  __int64 *v266; // [rsp+100h] [rbp-180h] BYREF
  __m128i v267; // [rsp+108h] [rbp-178h] BYREF
  unsigned __int8 v268; // [rsp+118h] [rbp-168h]
  __int64 v269; // [rsp+120h] [rbp-160h]
  __int64 *v270; // [rsp+128h] [rbp-158h] BYREF
  __int64 v271; // [rsp+130h] [rbp-150h]
  _BYTE v272[32]; // [rsp+138h] [rbp-148h] BYREF
  __int64 v273; // [rsp+158h] [rbp-128h]
  __m128i v274; // [rsp+160h] [rbp-120h] BYREF
  __int64 *v275; // [rsp+170h] [rbp-110h] BYREF
  _BYTE v276[24]; // [rsp+178h] [rbp-108h] BYREF
  __m128i v277; // [rsp+190h] [rbp-F0h] BYREF
  __int64 v278; // [rsp+1A0h] [rbp-E0h]
  _BYTE v279[32]; // [rsp+1A8h] [rbp-D8h] BYREF
  _QWORD *v280; // [rsp+1C8h] [rbp-B8h]
  __m128i v281; // [rsp+1D0h] [rbp-B0h] BYREF
  __int64 *v282; // [rsp+1E0h] [rbp-A0h] BYREF
  __m128i v283; // [rsp+1E8h] [rbp-98h]
  char v284; // [rsp+1F8h] [rbp-88h]
  __int64 v285; // [rsp+200h] [rbp-80h]
  __int64 *v286; // [rsp+208h] [rbp-78h] BYREF
  __int64 v287; // [rsp+210h] [rbp-70h]
  _BYTE v288[32]; // [rsp+218h] [rbp-68h] BYREF
  _QWORD *v289; // [rsp+238h] [rbp-48h]
  __m128i v290; // [rsp+240h] [rbp-40h]

  v6 = a1;
  v202 = *(unsigned int *)(a1 + 1328);
  if ( !*(_DWORD *)(a1 + 1328) )
    return sub_2869060(v6);
  v204 = 0;
  v210 = 0;
  do
  {
    v7 = *(_QWORD *)(v6 + 1320) + v204;
    v255 = *(unsigned int *)(v7 + 768);
    if ( *(_DWORD *)(v7 + 768) )
    {
      v8 = 0;
      v9 = 0;
      do
      {
        v11 = v8 + *(_QWORD *)(v7 + 760);
        v282 = *(__int64 **)v11;
        v283 = _mm_loadu_si128((const __m128i *)(v11 + 8));
        v284 = *(_BYTE *)(v11 + 24);
        v12 = *(_QWORD *)(v11 + 32);
        v286 = (__int64 *)v288;
        v285 = v12;
        v287 = 0x400000000LL;
        if ( *(_DWORD *)(v11 + 48) )
        {
          v249 = v11;
          sub_2850210((__int64)&v286, v11 + 40, v12, 0x400000000LL, a5, a6);
          v11 = v249;
        }
        v289 = *(_QWORD **)(v11 + 88);
        v290 = _mm_loadu_si128((const __m128i *)(v11 + 96));
        sub_2864410(v6, v7, v210, (__int64)&v282, 0);
        if ( v286 != (__int64 *)v288 )
          _libc_free((unsigned __int64)v286);
        ++v9;
        v8 += 112;
      }
      while ( v9 != v255 );
      v218 = *(unsigned int *)(v7 + 768);
      if ( *(_DWORD *)(v7 + 768) )
      {
        v76 = v6;
        v219 = v7;
        v259 = 0;
        v77 = 0;
        while ( 1 )
        {
          v90 = v77 + *(_QWORD *)(v219 + 760);
          v266 = *(__int64 **)v90;
          v267 = _mm_loadu_si128((const __m128i *)(v90 + 8));
          v268 = *(_BYTE *)(v90 + 24);
          v91 = *(_QWORD *)(v90 + 32);
          v270 = (__int64 *)v272;
          v92 = 0;
          v269 = v91;
          v271 = 0x400000000LL;
          v93 = *(unsigned int *)(v90 + 48);
          if ( (_DWORD)v93 )
          {
            sub_2850210((__int64)&v270, v90 + 40, v10, v93, a5, a6);
            v92 = (unsigned int)v271;
            v91 = v269;
            v93 = (unsigned int)v271;
          }
          v78 = *(_QWORD *)(v90 + 88);
          v273 = v78;
          v274 = _mm_loadu_si128((const __m128i *)(v90 + 96));
          v10 = (v91 == 1) + v92 - ((v274.m128i_i64[0] == 0) - 1LL);
          if ( v10 <= 1 )
            goto LABEL_147;
          if ( v91 == 1 )
          {
            v269 = 0;
            if ( v92 + 1 > (unsigned __int64)HIDWORD(v271) )
            {
              sub_C8D5F0((__int64)&v270, v272, v92 + 1, 8u, a5, a6);
              v92 = (unsigned int)v271;
            }
            v270[v92] = v78;
            v273 = 0;
            v93 = (unsigned int)(v271 + 1);
            v91 = v269;
            LODWORD(v271) = v271 + 1;
          }
          v79 = _mm_loadu_si128(&v267);
          v277.m128i_i64[0] = v91;
          v275 = v266;
          v10 = v268;
          v261 = v263;
          v262 = 0x400000000LL;
          v276[16] = v268;
          v277.m128i_i64[1] = (__int64)v279;
          v278 = 0x400000000LL;
          *(__m128i *)v276 = v79;
          if ( (_DWORD)v93 )
          {
            sub_2850210((__int64)&v277.m128i_i64[1], (__int64)&v270, v268, v93, a5, a6);
            a5 = (__int64)v270;
            LODWORD(v278) = 0;
            v94 = _mm_load_si128(&v274);
            v280 = (_QWORD *)v273;
            v281 = v94;
            v95 = &v270[(unsigned int)v271];
            if ( v270 == v95 )
            {
              v246 = 0;
            }
            else
            {
              v96 = v270;
              v97 = v76;
              v246 = 0;
              do
              {
                while ( 1 )
                {
                  v100 = *v96;
                  if ( !sub_DAEB50(*(_QWORD *)(v97 + 8), *v96, **(_QWORD **)(*(_QWORD *)(v97 + 56) + 32LL))
                    || sub_DAE0A0(*(_QWORD *)(v97 + 8), v100, *(_QWORD *)(v97 + 56)) )
                  {
                    break;
                  }
                  if ( !v246 )
                  {
                    v247 = *(_QWORD *)(v97 + 8);
                    v103 = sub_D95540(v100);
                    v246 = sub_D97090(v247, v103);
                  }
                  v98 = (unsigned int)v262;
                  v93 = HIDWORD(v262);
                  v99 = (unsigned int)v262 + 1LL;
                  if ( v99 > HIDWORD(v262) )
                  {
                    sub_C8D5F0((__int64)&v261, v263, v99, 8u, a5, a6);
                    v98 = (unsigned int)v262;
                  }
                  v10 = (unsigned __int64)v261;
                  ++v96;
                  *(_QWORD *)&v261[8 * v98] = v100;
                  LODWORD(v262) = v262 + 1;
                  if ( v95 == v96 )
                    goto LABEL_164;
                }
                v101 = (unsigned int)v278;
                v93 = HIDWORD(v278);
                v102 = (unsigned int)v278 + 1LL;
                if ( v102 > HIDWORD(v278) )
                {
                  sub_C8D5F0((__int64)&v277.m128i_i64[1], v279, v102, 8u, a5, a6);
                  v101 = (unsigned int)v278;
                }
                v10 = v277.m128i_u64[1];
                ++v96;
                *(_QWORD *)(v277.m128i_i64[1] + 8 * v101) = v100;
                LODWORD(v278) = v278 + 1;
              }
              while ( v95 != v96 );
LABEL_164:
              v76 = v97;
            }
          }
          else
          {
            v80 = _mm_load_si128(&v274);
            LODWORD(v278) = 0;
            v246 = 0;
            v280 = (_QWORD *)v273;
            v281 = v80;
          }
          if ( (_DWORD)v262 )
            break;
LABEL_143:
          if ( (_BYTE *)v277.m128i_i64[1] != v279 )
            _libc_free(v277.m128i_u64[1]);
          if ( v261 != v263 )
            _libc_free((unsigned __int64)v261);
LABEL_147:
          if ( v270 != (__int64 *)v272 )
            _libc_free((unsigned __int64)v270);
          ++v259;
          v77 += 112;
          if ( v259 == v218 )
          {
            v6 = v76;
            goto LABEL_11;
          }
        }
        if ( (unsigned int)v262 == 1 )
        {
LABEL_141:
          v10 = v281.m128i_i64[0];
          if ( v281.m128i_i64[0] && !v281.m128i_i8[8] )
          {
            v137 = sub_DA2C50(*(_QWORD *)(v76 + 8), v246, v281.m128i_i64[0], 1u);
            v139 = (unsigned int)v262;
            v140 = (unsigned int)v262 + 1LL;
            if ( v140 > HIDWORD(v262) )
            {
              sub_C8D5F0((__int64)&v261, v263, v140, 8u, (__int64)&v261, v138);
              v139 = (unsigned int)v262;
            }
            *(_QWORD *)&v261[8 * v139] = v137;
            v141 = *(__int64 **)(v76 + 8);
            LODWORD(v262) = v262 + 1;
            v281.m128i_i64[0] = 0;
            v281.m128i_i8[8] = 0;
            v144 = sub_DC7EB0(v141, (__int64)&v261, 0, 0);
            v283 = _mm_loadu_si128((const __m128i *)v276);
            v282 = v275;
            v284 = v276[16];
            v285 = v277.m128i_i64[0];
            v286 = (__int64 *)v288;
            v287 = 0x400000000LL;
            if ( (_DWORD)v278 )
              sub_2850210((__int64)&v286, (__int64)&v277.m128i_i64[1], v142, v143, v145, (unsigned int)v278);
            v146 = _mm_load_si128(&v281);
            v289 = v280;
            v290 = v146;
            if ( !sub_D968A0((__int64)v144) )
            {
              v147 = (unsigned int)v287;
              v148 = HIDWORD(v287);
              v149 = (unsigned int)v287 + 1LL;
              if ( v149 > HIDWORD(v287) )
              {
                sub_C8D5F0((__int64)&v286, v288, v149, 8u, a5, a6);
                v147 = (unsigned int)v287;
              }
              v150 = (__int64)v286;
              v286[v147] = (__int64)v144;
              v151 = *(_QWORD *)(v76 + 56);
              LODWORD(v287) = v287 + 1;
              sub_2857080((__int64)&v282, v151, v150, v148, a5, a6);
              sub_2862B30(v76, v219, v210, (unsigned __int64)&v282, v152, v153);
            }
            if ( v286 != (__int64 *)v288 )
              _libc_free((unsigned __int64)v286);
          }
          goto LABEL_143;
        }
        v264[1] = 0x400000000LL;
        v264[0] = (unsigned __int64)v265;
        sub_2850210((__int64)v264, (__int64)&v261, v10, v93, (__int64)&v261, a6);
        v81 = sub_DC7EB0(*(__int64 **)(v76 + 8), (__int64)v264, 0, 0);
        v287 = 0x400000000LL;
        v86 = (__int64)v81;
        v87 = _mm_loadu_si128((const __m128i *)v276);
        v282 = v275;
        v283 = v87;
        v284 = v276[16];
        v285 = v277.m128i_i64[0];
        v286 = (__int64 *)v288;
        if ( (_DWORD)v278 )
          sub_2850210((__int64)&v286, (__int64)&v277.m128i_i64[1], v82, v83, v84, v85);
        v88 = _mm_load_si128(&v281);
        v289 = v280;
        v290 = v88;
        if ( sub_D968A0(v86) )
        {
          v89 = (unsigned __int64)v286;
          if ( v286 == (__int64 *)v288 )
            goto LABEL_139;
        }
        else
        {
          v129 = (unsigned int)v287;
          v130 = HIDWORD(v287);
          v131 = (unsigned int)v287 + 1LL;
          if ( v131 > HIDWORD(v287) )
          {
            sub_C8D5F0((__int64)&v286, v288, v131, 8u, a5, a6);
            v129 = (unsigned int)v287;
          }
          v132 = (__int64)v286;
          v286[v129] = v86;
          v133 = *(_QWORD *)(v76 + 56);
          LODWORD(v287) = v287 + 1;
          sub_2857080((__int64)&v282, v133, v132, v130, a5, a6);
          sub_2862B30(v76, v219, v210, (unsigned __int64)&v282, v134, v135);
          v89 = (unsigned __int64)v286;
          if ( v286 == (__int64 *)v288 )
            goto LABEL_139;
        }
        _libc_free(v89);
LABEL_139:
        if ( (_BYTE *)v264[0] != v265 )
          _libc_free(v264[0]);
        goto LABEL_141;
      }
    }
LABEL_11:
    ++v210;
    v204 += 2184;
  }
  while ( v210 != v202 );
  v203 = *(unsigned int *)(v6 + 1328);
  if ( !*(_DWORD *)(v6 + 1328) )
    return sub_2869060(v6);
  v205 = 0;
  v13 = (const __m128i *)v6;
  v211 = 0;
  while ( 2 )
  {
    v14 = v13[82].m128i_i64[1] + v205;
    v236 = *(unsigned int *)(v14 + 768);
    if ( !*(_DWORD *)(v14 + 768) )
      goto LABEL_27;
    v243 = 0;
    v250 = 0;
    do
    {
      v16 = *(_QWORD *)(v14 + 760) + v243;
      v17 = *(_QWORD *)v16;
      v282 = *(__int64 **)v16;
      v283 = _mm_loadu_si128((const __m128i *)(v16 + 8));
      v284 = *(_BYTE *)(v16 + 24);
      v18 = *(_QWORD *)(v16 + 32);
      v286 = (__int64 *)v288;
      v285 = v18;
      v287 = 0x400000000LL;
      a5 = *(unsigned int *)(v16 + 48);
      if ( (_DWORD)a5 )
      {
        sub_2850210((__int64)&v286, v16 + 40, v18, (__int64)v288, a5, a6);
        v17 = (__int64)v282;
      }
      v15 = *(_QWORD *)(v16 + 88);
      v289 = (_QWORD *)v15;
      v290 = _mm_loadu_si128((const __m128i *)(v16 + 96));
      if ( !v17 )
      {
        v19 = 0;
        v256 = (unsigned int)v287;
        if ( (_DWORD)v287 )
        {
          do
          {
            v20 = v19++;
            sub_28644C0((__int64)v13, v14, v211, (__int64)&v282, v20, 0);
          }
          while ( v256 != v19 );
        }
        if ( v285 == 1 )
          sub_28644C0((__int64)v13, v14, v211, (__int64)&v282, -1, 1);
      }
      if ( v286 != (__int64 *)v288 )
        _libc_free((unsigned __int64)v286);
      ++v250;
      v243 += 112;
    }
    while ( v250 != v236 );
    v225 = *(unsigned int *)(v14 + 768);
    if ( !*(_DWORD *)(v14 + 768) )
      goto LABEL_27;
    v56 = v14;
    v238 = 0;
    v245 = 0;
    do
    {
      v57 = *(_QWORD *)(v56 + 760) + v238;
      v252 = 0;
      v282 = *(__int64 **)v57;
      v283 = _mm_loadu_si128((const __m128i *)(v57 + 8));
      v284 = *(_BYTE *)(v57 + 24);
      v58 = *(_QWORD *)(v57 + 32);
      v286 = (__int64 *)v288;
      v285 = v58;
      v287 = 0x400000000LL;
      if ( *(_DWORD *)(v57 + 48) )
      {
        sub_2850210((__int64)&v286, v57 + 40, v15, v58, a5, a6);
        v252 = (unsigned int)v287;
      }
      v289 = *(_QWORD **)(v57 + 88);
      v59 = _mm_loadu_si128((const __m128i *)(v57 + 96));
      v275 = (__int64 *)&v276[8];
      v290 = v59;
      v60 = _mm_loadu_si128((const __m128i *)(v56 + 712));
      *(_QWORD *)v276 = 0x200000001LL;
      *(__m128i *)&v276[8] = v60;
      if ( *(_QWORD *)(v56 + 728) == *(_QWORD *)(v56 + 712) && *(_BYTE *)(v56 + 736) == *(_BYTE *)(v56 + 720) )
      {
        if ( !v252 )
        {
LABEL_110:
          v64 = &v275;
          if ( v285 != 1 )
            goto LABEL_97;
LABEL_111:
          sub_2864A10(v13, (const __m128i *)v56, v211, (__int64 *)&v282, (__int64)v64, -1, 1);
          v15 = v195;
          goto LABEL_95;
        }
      }
      else
      {
        v61 = _mm_loadu_si128((const __m128i *)(v56 + 728));
        *(_DWORD *)v276 = 2;
        v277 = v61;
        if ( !v252 )
          goto LABEL_110;
      }
      a5 = (__int64)&v275;
      v62 = 0;
      do
      {
        v63 = v62++;
        v257 = (__int64 **)a5;
        sub_2864A10(v13, (const __m128i *)v56, v211, (__int64 *)&v282, a5, v63, 0);
        a5 = (__int64)v257;
      }
      while ( v62 != v252 );
      v64 = v257;
      if ( v285 == 1 )
        goto LABEL_111;
LABEL_95:
      if ( v275 != (__int64 *)&v276[8] )
        _libc_free((unsigned __int64)v275);
LABEL_97:
      if ( v286 != (__int64 *)v288 )
        _libc_free((unsigned __int64)v286);
      ++v245;
      v238 += 112;
    }
    while ( v245 != v225 );
    v65 = v56;
    v258 = *(unsigned int *)(v56 + 768);
    if ( *(_DWORD *)(v56 + 768) )
    {
      v66 = 0;
      v253 = (__int64)v13;
      v67 = (__int64)&v277.m128i_i64[1];
      v68 = 0;
      v69 = v279;
      while ( 2 )
      {
        v71 = v66 + *(_QWORD *)(v65 + 760);
        v275 = *(__int64 **)v71;
        *(__m128i *)v276 = _mm_loadu_si128((const __m128i *)(v71 + 8));
        v276[16] = *(_BYTE *)(v71 + 24);
        v72 = *(_QWORD *)(v71 + 32);
        v277.m128i_i64[1] = (__int64)v69;
        v277.m128i_i64[0] = v72;
        v278 = 0x400000000LL;
        if ( *(_DWORD *)(v71 + 48) )
          sub_2850210((__int64)&v277.m128i_i64[1], v71 + 40, v15, v67, a5, a6);
        v70 = *(_QWORD *)(v71 + 88);
        v280 = (_QWORD *)v70;
        v281 = _mm_loadu_si128((const __m128i *)(v71 + 96));
        if ( *(_DWORD *)(v65 + 32) == 3 )
        {
          if ( (_DWORD)v278 )
          {
            v73 = sub_D95540(*(_QWORD *)v277.m128i_i64[1]);
          }
          else if ( v70 )
          {
            v73 = sub_D95540(v70);
          }
          else
          {
            if ( !v275 )
              goto LABEL_103;
            v73 = v275[1];
          }
          if ( v73
            && (unsigned __int64)sub_D97050(*(_QWORD *)(v253 + 8), v73) <= 0x40
            && *(_QWORD *)(v65 + 712) == *(_QWORD *)(v65 + 728)
            && *(_BYTE *)(v65 + 720) == *(_BYTE *)(v65 + 736)
            && (!v280 || *(_BYTE *)(sub_D95540((__int64)v280) + 8) != 14) )
          {
            if ( v277.m128i_i64[1] != v277.m128i_i64[1] + 8LL * (unsigned int)v278 )
            {
              v239 = v65;
              v74 = (__int64 *)v277.m128i_i64[1];
              v230 = v73;
              v75 = v277.m128i_i64[1] + 8LL * (unsigned int)v278;
              do
              {
                if ( *(_BYTE *)(sub_D95540(*v74) + 8) == 14 )
                {
                  v65 = v239;
                  goto LABEL_103;
                }
                ++v74;
              }
              while ( (__int64 *)v75 != v74 );
              v65 = v239;
              v73 = v230;
            }
            v170 = *(_QWORD *)(v253 + 1096);
            v15 = *(unsigned int *)(v253 + 1104);
            v67 = v170 + 8 * v15;
            v242 = (__int64 *)v67;
            if ( v170 != v67 )
            {
              v200 = v69;
              v215 = v68;
              v171 = *(__int64 **)(v253 + 1096);
              while ( 1 )
              {
                v172 = *v171;
                if ( sub_AC3670(v73, *v171) )
                {
                  v67 = *(_QWORD *)v276;
                  if ( *(_QWORD *)v276 == 0x8000000000000000LL )
                  {
                    if ( v172 == -1 )
                      goto LABEL_302;
                  }
                  else
                  {
                    v221 = v276[8];
                    if ( !*(_QWORD *)v276 )
                      goto LABEL_290;
                  }
                  if ( !v276[8] )
                  {
                    v15 = v172 * *(_QWORD *)v276 % v172;
                    if ( *(_QWORD *)v276 == v172 * *(_QWORD *)v276 / v172 )
                    {
                      v221 = 0;
                      v67 = v172 * *(_QWORD *)v276;
LABEL_290:
                      if ( *(_BYTE *)(v73 + 8) == 14 || (v232 = v67, v173 = sub_AC3670(v73, v67), v67 = v232, v173) )
                      {
                        v174 = *(_QWORD *)(v65 + 712);
                        v15 = 0x8000000000000000LL;
                        v227 = v172 == -1;
                        if ( v174 != 0x8000000000000000LL || v172 != -1 )
                        {
                          v175 = v172 * v174;
                          v15 = v172 * v174 % v172;
                          if ( v174 == v172 * v174 / v172 )
                          {
                            v233 = *(_BYTE *)(v65 + 720);
                            if ( *(_BYTE *)(v73 + 8) == 14
                              || (v196 = v67, v176 = sub_AC3670(v73, v172 * v174), v175 = v172 * v174, v67 = v196, v176) )
                            {
                              v177 = _mm_loadu_si128((const __m128i *)v276);
                              v282 = v275;
                              v283 = v177;
                              v284 = v276[16];
                              v285 = v277.m128i_i64[0];
                              v286 = (__int64 *)v288;
                              v287 = 0x400000000LL;
                              if ( (_DWORD)v278 )
                              {
                                v197 = v175;
                                v199 = v67;
                                sub_2850210((__int64)&v286, (__int64)&v277.m128i_i64[1], v15, v67, a5, a6);
                                v175 = v197;
                                v67 = v199;
                              }
                              v178 = *(__int64 **)(v253 + 48);
                              v179 = _mm_load_si128(&v281);
                              v283.m128i_i64[0] = v67;
                              v289 = v280;
                              v290 = v179;
                              v283.m128i_i8[8] = v221;
                              v222 = v175;
                              if ( sub_2850770(
                                     v178,
                                     v175,
                                     v233,
                                     v175,
                                     v233,
                                     *(_DWORD *)(v65 + 32),
                                     *(_QWORD *)(v65 + 40),
                                     *(_DWORD *)(v65 + 48),
                                     (__int64)&v282) )
                              {
                                v180 = 1;
                                v181 = v222 + v283.m128i_i64[0] - *(_QWORD *)(v65 + 712);
                                if ( !v283.m128i_i8[8] && !v233 )
                                  v180 = *(_BYTE *)(v65 + 720);
                                v182 = *(_QWORD *)(v253 + 8);
                                v283.m128i_i8[8] = v180;
                                v283.m128i_i64[0] = v181;
                                v183 = sub_DA2C50(v182, v73, v172, 0);
                                a6 = 0;
                                v184 = (__int64)v183;
                                if ( (_DWORD)v287 )
                                {
                                  v198 = v172;
                                  v185 = (unsigned int)v287;
                                  v234 = v73;
                                  v186 = 0;
                                  v223 = v171;
                                  v187 = v65;
                                  v188 = (__int64)v183;
                                  do
                                  {
                                    v189 = sub_DCA690(*(__int64 **)(v253 + 8), v286[v186], v188, 0, 0);
                                    v286[v186] = (__int64)v189;
                                    v190 = sub_285DD00(v286[v186], v188, *(__int64 **)(v253 + 8), 0);
                                    v15 = v277.m128i_u64[1];
                                    if ( *(_QWORD **)(v277.m128i_i64[1] + 8 * v186) != v190 )
                                    {
                                      v65 = v187;
                                      v73 = v234;
                                      v171 = v223;
                                      goto LABEL_300;
                                    }
                                    ++v186;
                                  }
                                  while ( v185 != v186 );
                                  v184 = v188;
                                  v172 = v198;
                                  v65 = v187;
                                  v73 = v234;
                                  v171 = v223;
                                }
                                if ( !v289
                                  || (v235 = v184,
                                      v191 = sub_DCA690(*(__int64 **)(v253 + 8), (__int64)v289, v184, 0, 0),
                                      v192 = *(__int64 **)(v253 + 8),
                                      v289 = v191,
                                      v193 = sub_285DD00((__int64)v191, v235, v192, 0),
                                      v280 == v193) )
                                {
                                  if ( !v290.m128i_i64[0]
                                    || (v290.m128i_i64[0] != 0x8000000000000000LL || !v227)
                                    && (v194 = v172 * v290.m128i_i64[0],
                                        v290.m128i_i64[0] = v194,
                                        v15 = v194 % v172,
                                        v194 / v172 == v281.m128i_i64[0])
                                    && (*(_BYTE *)(v73 + 8) == 14 || sub_AC3670(v73, v194)) )
                                  {
                                    sub_2862B30(v253, v65, v211, (unsigned __int64)&v282, a5, a6);
                                  }
                                }
                              }
LABEL_300:
                              if ( v286 != (__int64 *)v288 )
                                _libc_free((unsigned __int64)v286);
                            }
                          }
                        }
                      }
                    }
                  }
                }
LABEL_302:
                if ( v242 == ++v171 )
                {
                  v68 = v215;
                  v69 = v200;
                  break;
                }
              }
            }
          }
        }
LABEL_103:
        if ( (_BYTE *)v277.m128i_i64[1] != v69 )
          _libc_free(v277.m128i_u64[1]);
        ++v68;
        v66 += 112;
        if ( v68 != v258 )
          continue;
        break;
      }
      v13 = (const __m128i *)v253;
      v214 = *(unsigned int *)(v65 + 768);
      if ( *(_DWORD *)(v65 + 768) )
      {
        v217 = 0;
        v104 = (const __m128i *)v253;
        v105 = v65;
        v220 = 0;
        while ( 1 )
        {
          v106 = *(_QWORD *)(v105 + 760) + v217;
          v275 = *(__int64 **)v106;
          *(__m128i *)v276 = _mm_loadu_si128((const __m128i *)(v106 + 8));
          v276[16] = *(_BYTE *)(v106 + 24);
          v277.m128i_i64[0] = *(_QWORD *)(v106 + 32);
          v277.m128i_i64[1] = (__int64)v279;
          v278 = 0x400000000LL;
          a6 = *(unsigned int *)(v106 + 48);
          if ( (_DWORD)a6 )
          {
            sub_2850210((__int64)&v277.m128i_i64[1], v106 + 40, v15, v67, a5, a6);
            v107 = *(_QWORD *)(v106 + 88);
            a5 = (unsigned int)v278;
            v280 = (_QWORD *)v107;
            v281 = _mm_loadu_si128((const __m128i *)(v106 + 96));
            if ( (_DWORD)v278 )
            {
              v231 = sub_D95540(*(_QWORD *)v277.m128i_i64[1]);
              goto LABEL_178;
            }
          }
          else
          {
            v107 = *(_QWORD *)(v106 + 88);
            v280 = (_QWORD *)v107;
            v281 = _mm_loadu_si128((const __m128i *)(v106 + 96));
          }
          if ( v107 )
          {
            v231 = sub_D95540(v107);
          }
          else
          {
            if ( !v275 )
              goto LABEL_171;
            v231 = v275[1];
          }
LABEL_178:
          if ( v231 )
          {
            if ( !v277.m128i_i64[0] )
              goto LABEL_180;
            if ( v277.m128i_i64[0] == 1 )
            {
              v126 = (unsigned int)v278;
              v67 = HIDWORD(v278);
              v277.m128i_i64[0] = 0;
              v127 = (__int64)v280;
              v128 = (unsigned int)v278 + 1LL;
              if ( v128 > HIDWORD(v278) )
              {
                sub_C8D5F0((__int64)&v277.m128i_i64[1], v279, v128, 8u, a5, a6);
                v126 = (unsigned int)v278;
              }
              *(_QWORD *)(v277.m128i_i64[1] + 8 * v126) = v127;
              v280 = 0;
              LODWORD(v278) = v278 + 1;
LABEL_180:
              v15 = v104[68].m128i_u64[1];
              v226 = (__int64 *)(v15 + 8LL * v104[69].m128i_u32[0]);
              if ( (__int64 *)v15 != v226 )
              {
                v260 = (__int64 *)v104[68].m128i_i64[1];
                v108 = (__int64)v104;
                do
                {
                  v254 = *v260;
                  if ( !(_BYTE)qword_5000A48 || *v260 != -1 )
                  {
                    v276[16] = (unsigned int)v278 > 1;
                    v109 = *(__int64 **)(v108 + 48);
                    v277.m128i_i64[0] = v254;
                    if ( sub_2850770(
                           v109,
                           *(_QWORD *)(v105 + 712),
                           *(_BYTE *)(v105 + 720),
                           *(_QWORD *)(v105 + 728),
                           *(_BYTE *)(v105 + 736),
                           *(_DWORD *)(v105 + 32),
                           *(_QWORD *)(v105 + 40),
                           *(_DWORD *)(v105 + 48),
                           (__int64)&v275) )
                    {
                      if ( *(_DWORD *)(v105 + 32) != 3 || v276[16] || *(_QWORD *)v276 || v275 )
                        goto LABEL_189;
                    }
                    else if ( !*(_DWORD *)(v105 + 32)
                           && sub_2850770(
                                *(__int64 **)(v108 + 48),
                                *(_QWORD *)(v105 + 712),
                                *(_BYTE *)(v105 + 720),
                                *(_QWORD *)(v105 + 728),
                                *(_BYTE *)(v105 + 736),
                                1u,
                                *(_QWORD *)(v105 + 40),
                                *(_DWORD *)(v105 + 48),
                                (__int64)&v275)
                           && *(_BYTE *)(v105 + 744) )
                    {
                      *(_DWORD *)(v105 + 32) = 1;
LABEL_189:
                      v110 = (unsigned int)v278;
                      if ( !(_DWORD)v278 )
                        goto LABEL_199;
                      v248 = v105;
                      v111 = v108;
                      v112 = 0;
                      v113 = v111;
                      while ( 2 )
                      {
                        v114 = *(_QWORD *)(v277.m128i_i64[1] + 8 * v112);
                        if ( *(_WORD *)(v114 + 24) != 8
                          || *(_QWORD *)(v114 + 48) != *(_QWORD *)(v113 + 56) && !*(_BYTE *)(v248 + 744)
                          || (v240 = sub_DA2C50(*(_QWORD *)(v113 + 8), v231, v254, 0), sub_D968A0((__int64)v240))
                          || (v115 = sub_285DD00(v114, (__int64)v240, *(__int64 **)(v113 + 8), 1)) == 0
                          || (v241 = (__int64)v115, v116 = sub_D968A0((__int64)v115), v15 = v241, v116) )
                        {
LABEL_197:
                          if ( v110 == ++v112 )
                          {
                            v105 = v248;
                            v108 = v113;
                            goto LABEL_199;
                          }
                          continue;
                        }
                        break;
                      }
                      v117 = _mm_loadu_si128((const __m128i *)v276);
                      v118 = -8;
                      v287 = 0x400000000LL;
                      v119 = v278;
                      v282 = v275;
                      v283 = v117;
                      v284 = v276[16];
                      v285 = v277.m128i_i64[0];
                      v120 = (__int64 *)v288;
                      v286 = (__int64 *)v288;
                      if ( (_DWORD)v278 )
                      {
                        sub_2850210((__int64)&v286, (__int64)&v277.m128i_i64[1], v241, (unsigned int)v278, a5, a6);
                        v120 = v286;
                        v15 = v241;
                        v119 = v287;
                        v118 = 8LL * (unsigned int)v287 - 8;
                      }
                      v121 = _mm_load_si128(&v281);
                      v122 = (unsigned __int64 *)&v120[v112];
                      v123 = (unsigned __int64 *)((char *)v120 + v118);
                      v289 = (_QWORD *)v15;
                      v290 = v121;
                      if ( v122 != v123 )
                      {
                        v15 = *v122;
                        *v122 = *v123;
                        *v123 = v15;
                        v119 = v287;
                      }
                      v67 = (unsigned int)(v119 - 1);
                      LODWORD(v287) = v67;
                      if ( v285 == 1 )
                      {
                        if ( (_DWORD)v67 )
                        {
                          v124 = *(_QWORD *)(v113 + 56);
                          v125 = *(_BYTE *)(v248 + 744);
                          if ( *(_QWORD *)(v114 + 48) == v124 )
                          {
                            if ( v125 )
                              sub_2857080((__int64)&v282, v124, v15, v67, a5, a6);
                            goto LABEL_211;
                          }
                          if ( !v125 )
                            goto LABEL_211;
                        }
                      }
                      else
                      {
LABEL_211:
                        sub_2862B30(v113, v248, v211, (unsigned __int64)&v282, a5, a6);
                      }
                      if ( v286 != (__int64 *)v288 )
                        _libc_free((unsigned __int64)v286);
                      goto LABEL_197;
                    }
                  }
LABEL_199:
                  ++v260;
                }
                while ( v226 != v260 );
                v104 = (const __m128i *)v108;
              }
            }
          }
LABEL_171:
          if ( (_BYTE *)v277.m128i_i64[1] != v279 )
            _libc_free(v277.m128i_u64[1]);
          ++v220;
          v217 += 112;
          if ( v220 == v214 )
          {
            v13 = v104;
            break;
          }
        }
      }
    }
LABEL_27:
    ++v211;
    v205 += 2184;
    if ( v211 != v203 )
      continue;
    break;
  }
  v6 = (__int64)v13;
  v212 = v13[83].m128i_u32[0];
  if ( v13[83].m128i_i32[0] )
  {
    v224 = (__m128i *)v13;
    v216 = 0;
    v228 = 0;
    while ( 1 )
    {
      v21 = 0;
      v251 = v224[82].m128i_i64[1] + v216;
      v237 = *(unsigned int *)(v251 + 768);
      if ( *(_DWORD *)(v251 + 768) )
        break;
LABEL_83:
      ++v228;
      v216 += 2184;
      if ( v228 == v212 )
      {
        v6 = (__int64)v224;
        return sub_2869060(v6);
      }
    }
    while ( 2 )
    {
      v23 = *(_QWORD *)(v251 + 760) + 112 * v21;
      v24 = *(__int64 **)v23;
      v275 = *(__int64 **)v23;
      *(__m128i *)v276 = _mm_loadu_si128((const __m128i *)(v23 + 8));
      v276[16] = *(_BYTE *)(v23 + 24);
      v25 = *(_QWORD *)(v23 + 32);
      v277.m128i_i64[1] = (__int64)v279;
      v277.m128i_i64[0] = v25;
      v278 = 0x400000000LL;
      v26 = *(unsigned int *)(v23 + 48);
      if ( (_DWORD)v26 )
      {
        sub_2850210((__int64)&v277.m128i_i64[1], v23 + 40, v25, v26, a5, a6);
        v24 = v275;
      }
      v22 = *(_QWORD *)(v23 + 88);
      v280 = (_QWORD *)v22;
      v281 = _mm_loadu_si128((const __m128i *)(v23 + 96));
      if ( v24 )
        goto LABEL_33;
      if ( (_DWORD)v278 )
      {
        v244 = sub_D95540(*(_QWORD *)v277.m128i_i64[1]);
      }
      else
      {
        if ( !v22 )
          goto LABEL_33;
        v244 = sub_D95540(v22);
      }
      if ( !v244 || *(_BYTE *)(v244 + 8) == 14 || v280 && *(_BYTE *)(sub_D95540((__int64)v280) + 8) == 14 )
        goto LABEL_33;
      v27 = (__int64 *)v277.m128i_i64[1];
      v28 = 8LL * (unsigned int)v278;
      v29 = (__int64 *)(v277.m128i_i64[1] + v28);
      v30 = v28 >> 3;
      v31 = v28 >> 5;
      if ( !v31 )
        goto LABEL_260;
      v32 = v277.m128i_i64[1] + 32 * v31;
      do
      {
        if ( *(_BYTE *)(sub_D95540(*v27) + 8) == 14 )
          goto LABEL_52;
        if ( *(_BYTE *)(sub_D95540(v27[1]) + 8) == 14 )
        {
          ++v27;
          goto LABEL_52;
        }
        if ( *(_BYTE *)(sub_D95540(v27[2]) + 8) == 14 )
        {
          v27 += 2;
          goto LABEL_52;
        }
        if ( *(_BYTE *)(sub_D95540(v27[3]) + 8) == 14 )
        {
          v27 += 3;
          goto LABEL_52;
        }
        v27 += 4;
      }
      while ( (__int64 *)v32 != v27 );
      v30 = v29 - v27;
LABEL_260:
      if ( v30 != 2 )
      {
        if ( v30 != 3 )
        {
          if ( v30 == 1 && *(_BYTE *)(sub_D95540(*v27) + 8) == 14 )
            goto LABEL_52;
LABEL_53:
          v266 = &v267.m128i_i64[1];
          v267.m128i_i64[0] = 0x100000000LL;
          v33 = *(_QWORD *)(v251 + 56);
          v34 = v33 + 80LL * *(unsigned int *)(v251 + 64);
          if ( v33 == v34 )
          {
            v42 = (__int64 *)v224[79].m128i_i64[1];
            v229 = &v42[v224[80].m128i_u32[0]];
            if ( v42 != v229 )
              goto LABEL_61;
          }
          else
          {
            v35 = v33 + 16;
            v36 = &v267.m128i_i64[1];
            v37 = 0;
            v38 = 0;
            for ( i = v33 + 96; ; i += 80LL )
            {
              v41 = (__int64)&v36[6 * v38];
              if ( v41 )
              {
                sub_C8CD80(v41, v41 + 32, v35, v37, a5, a6);
                LODWORD(v37) = v267.m128i_i32[0];
              }
              v37 = (unsigned int)(v37 + 1);
              v267.m128i_i32[0] = v37;
              if ( v34 == i - 16 )
                break;
              v38 = (unsigned int)v37;
              v36 = v266;
              v35 = i;
              v40 = (unsigned int)v37 + 1LL;
              if ( v40 > v267.m128i_u32[1] )
              {
                if ( (unsigned __int64)v266 > i || (v38 = (unsigned __int64)&v266[6 * (unsigned int)v37], i >= v38) )
                {
                  sub_28665C0((__int64)&v266, v40, v38, v37, a5, a6);
                  v38 = v267.m128i_u32[0];
                  v36 = v266;
                  v37 = v267.m128i_u32[0];
                }
                else
                {
                  v136 = i - (_QWORD)v266;
                  sub_28665C0((__int64)&v266, v40, v38, v37, a5, a6);
                  v36 = v266;
                  v38 = v267.m128i_u32[0];
                  v35 = (__int64)v266 + v136;
                  v37 = v267.m128i_u32[0];
                }
              }
            }
            v42 = (__int64 *)v224[79].m128i_i64[1];
            v229 = &v42[v224[80].m128i_u32[0]];
            if ( v42 != v229 )
            {
LABEL_61:
              v213 = v21;
              v43 = (__int64 *)v224;
              v44 = v42;
              while ( 1 )
              {
                while ( 1 )
                {
                  v45 = *v44;
                  if ( *v44 == v244 || !(unsigned __int8)sub_DFA860(v43[6]) )
                    goto LABEL_62;
                  v48 = _mm_loadu_si128((const __m128i *)v276);
                  v282 = v275;
                  v283 = v48;
                  v284 = v276[16];
                  v285 = v277.m128i_i64[0];
                  v286 = (__int64 *)v288;
                  v287 = 0x400000000LL;
                  if ( (_DWORD)v278 )
                    sub_2850210((__int64)&v286, (__int64)&v277.m128i_i64[1], v46, v47, a5, a6);
                  v49 = _mm_load_si128(&v281);
                  v289 = v280;
                  v290 = v49;
                  if ( v280 )
                    break;
                  a6 = (__int64)v286;
                  v154 = (unsigned __int64 *)&v286[(unsigned int)v287];
                  if ( v286 == (__int64 *)v154 )
                    goto LABEL_71;
LABEL_250:
                  v206 = v44;
                  v155 = v154;
                  v156 = (unsigned __int64 *)a6;
                  do
                  {
                    v157 = sub_284F5A0((__int64)v266, v267.m128i_u32[0], *v156, v45, v43[1]);
                    v158 = v157;
                    if ( !v157 || sub_D968A0(v157) )
                    {
                      v44 = v206;
                      a6 = (__int64)v286;
                      goto LABEL_71;
                    }
                    *v156++ = v158;
                  }
                  while ( v155 != v156 );
                  v44 = v206;
                  v208 = (__int64)(v43 + 4535);
                  if ( v289 && (unsigned __int8)sub_2853AB0(v208, (__int64)v289, (unsigned int)v228) )
                    goto LABEL_274;
                  a6 = (__int64)v286;
                  v162 = &v286[(unsigned int)v287];
                  if ( v286 != v162 )
                  {
                    v163 = v43;
                    v201 = v44;
                    v164 = v286;
                    v165 = v208;
                    v166 = v163;
                    v209 = (unsigned __int64)v286;
                    do
                    {
                      if ( (unsigned __int8)sub_2853AB0(v165, *v164, (unsigned int)v228) )
                      {
                        v44 = v201;
                        v43 = v166;
                        goto LABEL_274;
                      }
                      ++v164;
                    }
                    while ( v162 != v164 );
                    v44 = v201;
                    a6 = v209;
                    v43 = v166;
                  }
LABEL_71:
                  if ( (_BYTE *)a6 == v288 )
                    goto LABEL_62;
                  v52 = a6;
LABEL_73:
                  _libc_free(v52);
                  if ( v229 == ++v44 )
                  {
LABEL_74:
                    v21 = v213;
                    v37 = v267.m128i_u32[0];
                    goto LABEL_75;
                  }
                }
                v50 = sub_284F5A0((__int64)v266, v267.m128i_u32[0], (unsigned __int64)v280, v45, v43[1]);
                v51 = v50;
                if ( !v50 || sub_D968A0(v50) )
                {
                  a6 = (__int64)v286;
                  goto LABEL_71;
                }
                a6 = (__int64)v286;
                v289 = (_QWORD *)v51;
                v154 = (unsigned __int64 *)&v286[(unsigned int)v287];
                if ( v286 != (__int64 *)v154 )
                  goto LABEL_250;
                v207 = (unsigned __int64)v286;
                v159 = sub_2853AB0((__int64)(v43 + 4535), v51, (unsigned int)v228);
                a6 = v207;
                if ( !v159 )
                  goto LABEL_71;
LABEL_274:
                sub_2857080((__int64)&v282, v43[7], v160, v161, a5, a6);
                sub_2862B30((__int64)v43, v251, v228, (unsigned __int64)&v282, v167, v168);
                v52 = (unsigned __int64)v286;
                if ( v286 != (__int64 *)v288 )
                  goto LABEL_73;
LABEL_62:
                if ( v229 == ++v44 )
                  goto LABEL_74;
              }
            }
LABEL_75:
            v53 = v266;
            v54 = (__m128i *)&v266[6 * v37];
            if ( v266 != (__int64 *)v54 )
            {
              do
              {
                while ( 1 )
                {
                  v54 -= 3;
                  if ( !v54[1].m128i_i8[12] )
                    break;
                  if ( v53 == (__int64 *)v54 )
                    goto LABEL_80;
                }
                _libc_free(v54->m128i_u64[1]);
              }
              while ( v53 != (__int64 *)v54 );
LABEL_80:
              v54 = (__m128i *)v266;
            }
            if ( v54 != (__m128i *)&v267.m128i_u64[1] )
              _libc_free((unsigned __int64)v54);
          }
LABEL_33:
          if ( (_BYTE *)v277.m128i_i64[1] != v279 )
            _libc_free(v277.m128i_u64[1]);
          if ( ++v21 == v237 )
            goto LABEL_83;
          continue;
        }
        if ( *(_BYTE *)(sub_D95540(*v27) + 8) == 14 )
          goto LABEL_52;
        ++v27;
      }
      break;
    }
    if ( *(_BYTE *)(sub_D95540(*v27) + 8) != 14 )
    {
      v169 = v27[1];
      ++v27;
      if ( *(_BYTE *)(sub_D95540(v169) + 8) != 14 )
        goto LABEL_53;
    }
LABEL_52:
    if ( v29 == v27 )
      goto LABEL_53;
    goto LABEL_33;
  }
  return sub_2869060(v6);
}
