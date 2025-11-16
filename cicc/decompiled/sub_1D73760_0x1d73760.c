// Function: sub_1D73760
// Address: 0x1d73760
//
__int64 __fastcall sub_1D73760(
        __int64 a1,
        __int64 a2,
        __int64 ***a3,
        __int64 a4,
        int a5,
        double a6,
        __m128 a7,
        __m128 a8,
        __m128 a9,
        double a10,
        double a11,
        __m128i a12,
        __m128 a13)
{
  __int64 *v15; // r13
  __int64 v16; // r12
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rdx
  _QWORD *v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rbx
  int v23; // r9d
  unsigned __int8 v24; // dl
  unsigned __int8 v25; // r14
  unsigned __int8 v26; // al
  __int64 v27; // rdx
  __int64 *v28; // rax
  __int64 v29; // rax
  unsigned __int64 **v30; // r9
  __int64 *v31; // r8
  _QWORD *v32; // rax
  _QWORD *v33; // rdx
  bool v34; // al
  __int64 v35; // r14
  unsigned __int64 v36; // rdx
  _QWORD *v37; // rax
  unsigned int v38; // ebx
  _QWORD *v39; // rcx
  _QWORD *v40; // rsi
  _QWORD *v41; // rdi
  __int64 v42; // rax
  int v43; // eax
  __m128 *v44; // rdx
  int v45; // eax
  unsigned int v46; // esi
  __int64 *v47; // rax
  __int64 v48; // rdi
  unsigned int v49; // edx
  __int64 v50; // r14
  __int64 v51; // rcx
  __int64 v52; // rax
  __m128 *v53; // rax
  __int64 v54; // r14
  __int64 v55; // r14
  __int64 v56; // rdi
  char v57; // r11
  __int64 v58; // rax
  int v59; // eax
  unsigned int v60; // r11d
  __int64 v61; // r14
  __int64 v62; // rax
  __int64 v63; // rsi
  _BYTE *v64; // r12
  _BYTE *v65; // rbx
  __int64 v66; // rdi
  __m128 *v68; // rax
  __int64 v69; // rdx
  int v70; // r8d
  __int64 v71; // r9
  __int64 *v72; // rbx
  __int64 v73; // r14
  __int64 *v74; // r13
  __int64 v75; // r12
  char v76; // dl
  __int64 v77; // rax
  unsigned __int32 v78; // r8d
  __int64 v79; // r9
  double v80; // xmm4_8
  double v81; // xmm5_8
  unsigned __int64 v82; // r13
  _BYTE *v83; // r14
  unsigned __int64 v84; // r12
  __int64 v85; // rbx
  __int64 ***v86; // r12
  int v87; // esi
  __int64 v88; // rdx
  int v89; // r10d
  unsigned __int64 v90; // rdi
  unsigned __int64 v91; // rdi
  unsigned int i; // eax
  __int64 *v93; // rdi
  __int64 v94; // r11
  unsigned int v95; // eax
  _QWORD *v96; // rax
  _QWORD *v97; // rsi
  __int64 v98; // r8
  __int64 v99; // rax
  __int64 v100; // rbx
  __int64 v101; // rax
  unsigned __int64 *v102; // rax
  const __m128i *v103; // r14
  __int64 v104; // rsi
  __int64 **v105; // r13
  __int64 v106; // rdx
  int v107; // r9d
  unsigned __int32 v108; // r8d
  unsigned __int64 v109; // rcx
  unsigned __int64 v110; // rcx
  int v111; // eax
  __int64 v112; // rcx
  unsigned __int32 j; // eax
  __int64 *v114; // rbx
  __int64 v115; // r8
  __int32 v116; // eax
  __int64 v117; // rax
  __int64 v118; // rdx
  unsigned int v119; // ecx
  _QWORD *v120; // rax
  __int64 v121; // r12
  _BYTE *v122; // rbx
  __int64 v123; // rdi
  __int64 *v124; // r14
  __int64 v125; // r13
  _QWORD *v126; // rax
  __int64 v127; // rdx
  _QWORD *v128; // rdi
  __int64 v129; // rcx
  _QWORD *v130; // rsi
  __int64 v131; // rdx
  __int64 v132; // rcx
  __int64 v133; // rcx
  __int64 v134; // rcx
  __int64 v135; // rdx
  _QWORD *v136; // rax
  __int64 v137; // rsi
  __int64 v138; // rax
  char v139; // al
  unsigned __int64 *v140; // rbx
  __int64 ***v141; // rbx
  __int64 v142; // rdx
  char v143; // al
  unsigned __int64 *v144; // r13
  __int64 v145; // rax
  __int64 v146; // rax
  __int64 v147; // rax
  __int64 v148; // rcx
  _QWORD *v149; // r8
  unsigned __int64 v150; // rdx
  _QWORD *v151; // rax
  __int64 v152; // rsi
  __int64 v153; // rdx
  __int64 **v154; // rcx
  unsigned int v155; // r8d
  __int64 v156; // rdi
  __int64 **v157; // rcx
  __int64 **v158; // rax
  char v159; // al
  char v160; // al
  __int64 v161; // r14
  __int64 v162; // r11
  __int64 v163; // rdx
  __int64 v164; // rax
  __int64 v165; // rsi
  __int64 v166; // rax
  __int64 v167; // rax
  __int64 **v168; // rdx
  __int64 v169; // rax
  __int64 v170; // rbx
  __int64 v171; // r13
  int v172; // ecx
  int v173; // edx
  unsigned __int64 v174; // rax
  __int64 v175; // r10
  __int64 v176; // rdi
  __int64 v177; // r9
  char v178; // al
  char v179; // al
  __int64 v180; // rsi
  char v181; // al
  char v182; // al
  __int64 v183; // rax
  __int64 v184; // rbx
  _QWORD *v185; // rax
  __int64 v186; // r10
  __int64 v187; // r11
  __int64 v188; // rax
  __int64 v189; // rax
  __int64 v190; // rax
  __int64 v191; // rax
  __int64 v192; // r13
  __int64 v193; // rax
  __int64 v194; // rcx
  int v195; // esi
  int v196; // edx
  unsigned int v197; // eax
  _QWORD *v198; // r13
  _QWORD *v199; // rbx
  __int64 v200; // rsi
  __int64 v201; // rax
  unsigned __int32 v202; // eax
  __int32 v203; // eax
  __int64 v204; // rax
  bool v205; // zf
  _QWORD *v206; // rcx
  _QWORD *v207; // rdx
  __int64 v208; // rsi
  int v209; // ecx
  __int64 v210; // rdx
  __int64 v211; // rax
  char v212; // al
  __int64 *v213; // rax
  char v214; // al
  __int64 v215; // rax
  __int64 v216; // rsi
  __int64 v217; // rax
  __int64 v218; // rsi
  __int64 v219; // rdx
  unsigned __int8 *v220; // rsi
  __int64 v221; // r10
  __int64 v222; // rax
  __int64 v223; // rsi
  __int64 v224; // rsi
  __int64 v225; // [rsp+0h] [rbp-7E0h]
  _QWORD *v226; // [rsp+8h] [rbp-7D8h]
  __int64 v227; // [rsp+8h] [rbp-7D8h]
  __int64 *v229; // [rsp+30h] [rbp-7B0h]
  __int64 v230; // [rsp+30h] [rbp-7B0h]
  __int64 *v231; // [rsp+30h] [rbp-7B0h]
  __int64 *v232; // [rsp+30h] [rbp-7B0h]
  __int64 v233; // [rsp+30h] [rbp-7B0h]
  __int64 v234; // [rsp+30h] [rbp-7B0h]
  __int64 v235; // [rsp+30h] [rbp-7B0h]
  __int64 v236; // [rsp+30h] [rbp-7B0h]
  __int64 *v237; // [rsp+30h] [rbp-7B0h]
  __int64 v238; // [rsp+30h] [rbp-7B0h]
  bool v239; // [rsp+38h] [rbp-7A8h]
  __int64 v240; // [rsp+40h] [rbp-7A0h]
  int v241; // [rsp+40h] [rbp-7A0h]
  unsigned __int8 v243; // [rsp+60h] [rbp-780h]
  __int64 v244; // [rsp+60h] [rbp-780h]
  __int64 v245; // [rsp+68h] [rbp-778h]
  __int64 v247; // [rsp+70h] [rbp-770h]
  __int64 **v249; // [rsp+78h] [rbp-768h]
  __int64 v250; // [rsp+78h] [rbp-768h]
  __int64 **v251; // [rsp+78h] [rbp-768h]
  __int64 **v252; // [rsp+78h] [rbp-768h]
  __int64 v253; // [rsp+78h] [rbp-768h]
  __int64 *v254; // [rsp+78h] [rbp-768h]
  __int64 v255; // [rsp+78h] [rbp-768h]
  __int64 v256; // [rsp+78h] [rbp-768h]
  __int64 *v257; // [rsp+80h] [rbp-760h]
  __int64 v258; // [rsp+80h] [rbp-760h]
  __int64 v259; // [rsp+80h] [rbp-760h]
  __int64 v260; // [rsp+80h] [rbp-760h]
  __int64 v261; // [rsp+80h] [rbp-760h]
  unsigned __int64 v262; // [rsp+88h] [rbp-758h]
  int v263; // [rsp+88h] [rbp-758h]
  unsigned __int8 v264; // [rsp+88h] [rbp-758h]
  __int64 v265; // [rsp+88h] [rbp-758h]
  const __m128i *v266; // [rsp+88h] [rbp-758h]
  __int64 v267; // [rsp+88h] [rbp-758h]
  __int64 v268; // [rsp+88h] [rbp-758h]
  __int64 v269; // [rsp+88h] [rbp-758h]
  __int64 v270; // [rsp+88h] [rbp-758h]
  __int64 v271; // [rsp+88h] [rbp-758h]
  __int64 v272; // [rsp+88h] [rbp-758h]
  __int64 v273; // [rsp+88h] [rbp-758h]
  unsigned __int64 *v274; // [rsp+98h] [rbp-748h] BYREF
  __int64 *v275[2]; // [rsp+A0h] [rbp-740h] BYREF
  __int64 v276; // [rsp+B0h] [rbp-730h]
  __m128i v277; // [rsp+C0h] [rbp-720h] BYREF
  __int64 ***v278; // [rsp+D0h] [rbp-710h]
  _QWORD v279[6]; // [rsp+E0h] [rbp-700h] BYREF
  __m128i v280; // [rsp+110h] [rbp-6D0h] BYREF
  __m128i v281; // [rsp+120h] [rbp-6C0h] BYREF
  __m128i v282; // [rsp+130h] [rbp-6B0h] BYREF
  __int64 v283; // [rsp+140h] [rbp-6A0h]
  _QWORD *v284; // [rsp+150h] [rbp-690h] BYREF
  __int64 v285; // [rsp+158h] [rbp-688h]
  _QWORD v286[8]; // [rsp+160h] [rbp-680h] BYREF
  unsigned __int64 *v287; // [rsp+1A0h] [rbp-640h] BYREF
  __int64 v288; // [rsp+1A8h] [rbp-638h]
  __int64 *v289; // [rsp+1B0h] [rbp-630h] BYREF
  _QWORD *v290; // [rsp+1B8h] [rbp-628h]
  __int64 v291; // [rsp+1C0h] [rbp-620h]
  int v292; // [rsp+1C8h] [rbp-618h]
  __int64 v293; // [rsp+1D0h] [rbp-610h]
  __m128i *v294; // [rsp+1D8h] [rbp-608h]
  __int64 v295; // [rsp+1E0h] [rbp-600h]
  __int64 v296; // [rsp+1E8h] [rbp-5F8h]
  _BYTE **v297; // [rsp+1F0h] [rbp-5F0h]
  __m128i *v298; // [rsp+1F8h] [rbp-5E8h]
  char v299; // [rsp+200h] [rbp-5E0h]
  _BYTE *v300; // [rsp+210h] [rbp-5D0h] BYREF
  __int64 v301; // [rsp+218h] [rbp-5C8h]
  _BYTE v302[128]; // [rsp+220h] [rbp-5C0h] BYREF
  _BYTE *v303; // [rsp+2A0h] [rbp-540h] BYREF
  __int64 v304; // [rsp+2A8h] [rbp-538h]
  _BYTE v305[128]; // [rsp+2B0h] [rbp-530h] BYREF
  __int64 v306; // [rsp+330h] [rbp-4B0h]
  __int64 v307; // [rsp+340h] [rbp-4A0h] BYREF
  _BYTE *v308; // [rsp+348h] [rbp-498h]
  _BYTE *v309; // [rsp+350h] [rbp-490h]
  __int64 v310; // [rsp+358h] [rbp-488h]
  int v311; // [rsp+360h] [rbp-480h]
  _BYTE v312[136]; // [rsp+368h] [rbp-478h] BYREF
  _BYTE *v313; // [rsp+3F0h] [rbp-3F0h] BYREF
  __int64 v314; // [rsp+3F8h] [rbp-3E8h]
  _BYTE v315[896]; // [rsp+400h] [rbp-3E0h] BYREF
  unsigned int v316; // [rsp+780h] [rbp-60h]
  char v317; // [rsp+784h] [rbp-5Ch]
  __int64 **v318; // [rsp+788h] [rbp-58h]
  __int64 *v319; // [rsp+790h] [rbp-50h]
  __int64 ***v320; // [rsp+798h] [rbp-48h]
  __int64 v321; // [rsp+7A0h] [rbp-40h]

  v15 = &v307;
  v16 = a2;
  v308 = v312;
  v309 = v312;
  v286[0] = a3;
  v17 = *(_QWORD *)(a1 + 200);
  v18 = *(_QWORD *)(a1 + 904);
  v285 = 0x800000001LL;
  v300 = v302;
  v279[1] = v17;
  v19 = *(_QWORD *)(a2 + 40);
  v301 = 0x1000000000LL;
  v279[0] = v18;
  v284 = v286;
  v307 = 0;
  v310 = 16;
  v311 = 0;
  memset(&v279[2], 0, 24);
  v313 = v315;
  v303 = v305;
  v316 = 0;
  v317 = 1;
  v318 = 0;
  v319 = v279;
  v320 = a3;
  v243 = 0;
  v314 = 0x1000000000LL;
  v304 = 0x1000000000LL;
  v321 = v19;
  v20 = v286;
  v306 = a1 + 520;
  LODWORD(v21) = 1;
  while ( 1 )
  {
    v22 = v20[(unsigned int)v21 - 1];
    LODWORD(v285) = v21 - 1;
    sub_1412190((__int64)v15, v22);
    v25 = v24;
    if ( !v24 )
      goto LABEL_51;
    v26 = *(_BYTE *)(v22 + 16);
    if ( v26 <= 0x17u )
      goto LABEL_6;
    if ( v26 == 77 )
      break;
    if ( v26 != 79 )
    {
LABEL_6:
      v27 = *(_QWORD *)(a1 + 176);
      a6 = 0.0;
      v28 = *(__int64 **)(a1 + 184);
      v280 = 0;
      v287 = (unsigned __int64 *)&v300;
      v288 = v27;
      v281 = 0;
      v282 = 0;
      LODWORD(v301) = 0;
      v277 = 0u;
      v283 = 0;
      v289 = v28;
      v29 = sub_15F2050(v16);
      v290 = (_QWORD *)sub_1632FA0(v29);
      v293 = v16;
      v291 = a4;
      v299 = 0;
      v292 = a5;
      v294 = &v280;
      v295 = a1 + 320;
      v296 = a1 + 488;
      v297 = &v303;
      v298 = &v277;
      sub_1D61F00((__int64)&v287, v22, 0);
      v31 = (__int64 *)v277.m128i_i64[0];
      if ( !v277.m128i_i64[0] || *(_QWORD *)(v277.m128i_i64[0] + 40) == *(_QWORD *)(v16 + 40) )
        goto LABEL_14;
      v30 = &v287;
      if ( *(_QWORD *)(a1 + 824) )
      {
        v96 = *(_QWORD **)(a1 + 800);
        if ( v96 )
        {
          v97 = (_QWORD *)(a1 + 792);
          do
          {
            if ( v277.m128i_i64[0] > v96[4] )
            {
              v96 = (_QWORD *)v96[3];
            }
            else
            {
              v97 = v96;
              v96 = (_QWORD *)v96[2];
            }
          }
          while ( v96 );
          if ( (_QWORD *)(a1 + 792) != v97 && v277.m128i_i64[0] >= v97[4] )
            goto LABEL_14;
        }
      }
      else
      {
        v32 = *(_QWORD **)(a1 + 752);
        v33 = &v32[*(unsigned int *)(a1 + 760)];
        if ( v32 != v33 )
        {
          while ( v277.m128i_i64[0] != *v32 )
          {
            if ( v33 == ++v32 )
              goto LABEL_56;
          }
          if ( v33 != v32 )
          {
LABEL_14:
            v34 = v317;
            v283 = v22;
            if ( v317 )
            {
              if ( v281.m128i_i64[1] | v280.m128i_i64[1] )
              {
                v35 = (unsigned int)v314;
                v317 = 0;
                if ( (_DWORD)v314 )
                  goto LABEL_19;
                goto LABEL_87;
              }
              if ( v280.m128i_i64[0] )
                v34 = v282.m128i_i64[0] == 0;
            }
            v35 = (unsigned int)v314;
            v317 = v34;
            if ( (_DWORD)v314 )
            {
LABEL_19:
              v36 = (unsigned __int64)v313;
              v37 = (_QWORD *)*((_QWORD *)v313 + 4);
              if ( v37 )
              {
                if ( v282.m128i_i64[0] )
                {
                  v38 = 255;
                  if ( *v37 != *(_QWORD *)v282.m128i_i64[0] )
                    goto LABEL_40;
                }
              }
              v39 = *(_QWORD **)v313;
              if ( *(_QWORD *)v313 )
              {
                if ( !v280.m128i_i64[0] )
                {
                  v40 = (_QWORD *)*((_QWORD *)v313 + 5);
                  if ( !v40 )
                  {
                    v41 = (_QWORD *)v282.m128i_i64[1];
                    v38 = v282.m128i_i64[0] != (_QWORD)v37;
                    goto LABEL_29;
                  }
                  v41 = (_QWORD *)v282.m128i_i64[1];
                  if ( !v282.m128i_i64[1] )
                  {
                    v38 = v282.m128i_i64[0] != (_QWORD)v37;
                    goto LABEL_29;
                  }
LABEL_27:
                  v38 = 255;
                  if ( *v40 != *v41 )
                    goto LABEL_40;
LABEL_28:
                  v38 = v282.m128i_i64[0] != (_QWORD)v37;
                  if ( v39 == (_QWORD *)v280.m128i_i64[0] )
                  {
LABEL_30:
                    if ( *((_QWORD *)v313 + 1) != v280.m128i_i64[1] )
                      v38 |= 4u;
                    if ( v40 != v41 )
                      v38 |= 8u;
                    v42 = *((_QWORD *)v313 + 3);
                    if ( v42 && v281.m128i_i64[1] && v42 != v281.m128i_i64[1] )
                      v38 |= 0x10u;
                    v262 = (unsigned __int64)v313;
                    v43 = sub_39FAC40(v38);
                    v36 = v262;
                    if ( v43 > 1 )
                      v38 = 255;
LABEL_40:
                    if ( v316 )
                    {
                      if ( v316 != v38 )
                      {
                        v316 = 255;
LABEL_68:
                        LODWORD(v314) = 0;
                        goto LABEL_69;
                      }
                    }
                    else
                    {
                      v316 = v38;
                    }
                    if ( v38 == 255 || v38 == 16 )
                      goto LABEL_68;
                    if ( v38 == 4 )
                    {
                      if ( v282.m128i_i64[1] )
                        goto LABEL_68;
                      if ( (unsigned int)v35 < HIDWORD(v314) )
                      {
LABEL_48:
                        v44 = (__m128 *)(v36 + 56 * v35);
LABEL_49:
                        a8 = (__m128)_mm_load_si128(&v280);
                        *v44 = a8;
                        a9 = (__m128)_mm_load_si128(&v281);
                        v44[1] = a9;
                        v44[2] = (__m128)_mm_load_si128(&v282);
                        v44[3].m128_u64[0] = v283;
                        v45 = v314;
LABEL_50:
                        LODWORD(v314) = v45 + 1;
LABEL_51:
                        LODWORD(v21) = v285;
                        goto LABEL_52;
                      }
                    }
                    else
                    {
                      if ( v38 == 2 && v281.m128i_i8[0] )
                        goto LABEL_68;
                      if ( (unsigned int)v35 < HIDWORD(v314) )
                        goto LABEL_48;
                    }
                    sub_16CD150((__int64)&v313, v315, 0, 56, (int)v31, (int)v30);
                    v45 = v314;
                    v44 = (__m128 *)&v313[56 * (unsigned int)v314];
                    if ( !v44 )
                      goto LABEL_50;
                    goto LABEL_49;
                  }
LABEL_29:
                  v38 |= 2u;
                  goto LABEL_30;
                }
                v38 = 255;
                if ( *v39 != *(_QWORD *)v280.m128i_i64[0] )
                  goto LABEL_40;
              }
              v40 = (_QWORD *)*((_QWORD *)v313 + 5);
              v41 = (_QWORD *)v282.m128i_i64[1];
              if ( !v40 || !v282.m128i_i64[1] )
                goto LABEL_28;
              goto LABEL_27;
            }
LABEL_87:
            if ( !HIDWORD(v314) )
            {
              sub_16CD150((__int64)&v313, v315, 0, 56, (int)v31, (int)v30);
              LODWORD(v35) = v314;
            }
            v68 = (__m128 *)&v313[56 * (unsigned int)v35];
            if ( v68 )
            {
              *(__m128i *)v68 = _mm_load_si128(&v280);
              a12 = _mm_load_si128(&v281);
              LODWORD(v35) = v314;
              v68[1] = (__m128)a12;
              a13 = (__m128)_mm_load_si128(&v282);
              v68[2] = a13;
              v68[3].m128_u64[0] = v283;
            }
            LODWORD(v314) = v35 + 1;
            goto LABEL_51;
          }
        }
      }
LABEL_56:
      v46 = *(_DWORD *)(a1 + 744);
      v47 = *(__int64 **)(v277.m128i_i64[0] - 24LL * (*(_DWORD *)(v277.m128i_i64[0] + 20) & 0xFFFFFFF));
      v275[0] = v47;
      if ( v46 )
      {
        v48 = *(_QWORD *)(a1 + 728);
        v49 = (v46 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
        v50 = v48 + 536LL * v49;
        v51 = *(_QWORD *)v50;
        if ( v47 == *(__int64 **)v50 )
        {
LABEL_58:
          v52 = *(unsigned int *)(v50 + 16);
          if ( (unsigned int)v52 >= *(_DWORD *)(v50 + 20) )
          {
            v270 = v277.m128i_i64[0];
            sub_16CD150(v50 + 8, (const void *)(v50 + 24), 0, 16, v277.m128i_i32[0], (int)&v287);
            v31 = (__int64 *)v270;
            v53 = (__m128 *)(*(_QWORD *)(v50 + 8) + 16LL * *(unsigned int *)(v50 + 16));
          }
          else
          {
            v53 = (__m128 *)(*(_QWORD *)(v50 + 8) + 16 * v52);
          }
          goto LABEL_60;
        }
        v241 = 1;
        v272 = 0;
        while ( v51 != -8 )
        {
          if ( v51 == -16 )
          {
            if ( v272 )
              v50 = v272;
            v272 = v50;
          }
          v49 = (v46 - 1) & (v241 + v49);
          v50 = v48 + 536LL * v49;
          v51 = *(_QWORD *)v50;
          if ( v47 == *(__int64 **)v50 )
            goto LABEL_58;
          ++v241;
        }
        if ( v272 )
          v50 = v272;
        v172 = *(_DWORD *)(a1 + 736);
        ++*(_QWORD *)(a1 + 720);
        v173 = v172 + 1;
        if ( 4 * (v172 + 1) < 3 * v46 )
        {
          if ( v46 - *(_DWORD *)(a1 + 740) - v173 <= v46 >> 3 )
          {
            v231 = v31;
            sub_1D6B340(a1 + 720, v46);
            sub_1D68000(a1 + 720, (__int64 *)v275, &v287);
            v50 = (__int64)v287;
            v47 = v275[0];
            v31 = v231;
            v173 = *(_DWORD *)(a1 + 736) + 1;
          }
          goto LABEL_293;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 720);
      }
      v232 = v31;
      sub_1D6B340(a1 + 720, 2 * v46);
      sub_1D68000(a1 + 720, (__int64 *)v275, &v287);
      v50 = (__int64)v287;
      v47 = v275[0];
      v31 = v232;
      v173 = *(_DWORD *)(a1 + 736) + 1;
LABEL_293:
      *(_DWORD *)(a1 + 736) = v173;
      if ( *(_QWORD *)v50 != -8 )
        --*(_DWORD *)(a1 + 740);
      *(_QWORD *)v50 = v47;
      v53 = (__m128 *)(v50 + 24);
      *(_QWORD *)(v50 + 8) = v50 + 24;
      *(_QWORD *)(v50 + 16) = 0x2000000000LL;
LABEL_60:
      a7 = (__m128)_mm_load_si128(&v277);
      v229 = v31;
      *v53 = a7;
      ++*(_DWORD *)(v50 + 16);
      v54 = *(unsigned int *)(a1 + 856);
      v275[0] = v31;
      v55 = *(_QWORD *)(a1 + 840) + 16 * v54;
      v56 = a1 + 832;
      v57 = sub_1D66AA0(a1 + 832, (__int64 *)v275, &v287);
      v30 = &v287;
      v58 = (__int64)v287;
      v31 = v229;
      if ( !v57 )
        v58 = *(_QWORD *)(a1 + 840) + 16LL * *(unsigned int *)(a1 + 856);
      if ( v55 != v58 )
        goto LABEL_14;
      v59 = *(_DWORD *)(a1 + 848);
      v60 = *(_DWORD *)(a1 + 856);
      v275[0] = v229;
      v263 = v59;
      if ( v60 )
      {
        v240 = *(_QWORD *)(a1 + 840);
        v61 = (v60 - 1) & (((unsigned int)v229 >> 9) ^ ((unsigned int)v229 >> 4));
        v62 = v240 + 16 * v61;
        v63 = *(_QWORD *)v62;
        if ( v229 == *(__int64 **)v62 )
        {
LABEL_65:
          *(_DWORD *)(v62 + 8) = v263;
          goto LABEL_14;
        }
        v209 = 1;
        v210 = 0;
        while ( v63 != -8 )
        {
          if ( !v210 && v63 == -16 )
            v210 = v62;
          LODWORD(v61) = (v60 - 1) & (v209 + v61);
          v62 = v240 + 16LL * (unsigned int)v61;
          v63 = *(_QWORD *)v62;
          if ( v229 == *(__int64 **)v62 )
            goto LABEL_65;
          ++v209;
        }
        if ( v210 )
          v62 = v210;
        ++*(_QWORD *)(a1 + 832);
        v196 = v263 + 1;
        if ( 4 * (v263 + 1) < 3 * v60 )
        {
          if ( v60 - *(_DWORD *)(a1 + 852) - v196 > v60 >> 3 )
            goto LABEL_355;
          v195 = v60;
LABEL_354:
          sub_1D6B640(v56, v195);
          sub_1D66AA0(v56, (__int64 *)v275, &v287);
          v62 = (__int64)v287;
          v31 = v275[0];
          v196 = *(_DWORD *)(a1 + 848) + 1;
LABEL_355:
          *(_DWORD *)(a1 + 848) = v196;
          if ( *(_QWORD *)v62 != -8 )
            --*(_DWORD *)(a1 + 852);
          *(_QWORD *)v62 = v31;
          *(_DWORD *)(v62 + 8) = 0;
          goto LABEL_65;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 832);
      }
      v195 = 2 * v60;
      goto LABEL_354;
    }
    v98 = *(_QWORD *)(v22 - 24);
    v99 = (unsigned int)v285;
    if ( (unsigned int)v285 >= HIDWORD(v285) )
    {
      v271 = *(_QWORD *)(v22 - 24);
      sub_16CD150((__int64)&v284, v286, 0, 8, v98, v23);
      v99 = (unsigned int)v285;
      v98 = v271;
    }
    v284[v99] = v98;
    v21 = (unsigned int)(v285 + 1);
    LODWORD(v285) = v21;
    v100 = *(_QWORD *)(v22 - 48);
    if ( HIDWORD(v285) <= (unsigned int)v21 )
    {
      sub_16CD150((__int64)&v284, v286, 0, 8, v98, v23);
      v21 = (unsigned int)v285;
    }
    v243 = v25;
    v284[v21] = v100;
    LODWORD(v21) = v285 + 1;
    LODWORD(v285) = v285 + 1;
LABEL_52:
    if ( !(_DWORD)v21 )
      goto LABEL_98;
LABEL_53:
    v20 = v284;
  }
  v69 = sub_13CF970(v22);
  v71 = v69 + 24LL * (*(_DWORD *)(v22 + 20) & 0xFFFFFFF);
  v21 = (unsigned int)v285;
  if ( v69 == v71 )
  {
    v243 = v25;
    goto LABEL_52;
  }
  v264 = v25;
  v72 = (__int64 *)v69;
  v73 = v16;
  v257 = v15;
  v74 = (__int64 *)v71;
  do
  {
    v75 = *v72;
    if ( HIDWORD(v285) <= (unsigned int)v21 )
    {
      sub_16CD150((__int64)&v284, v286, 0, 8, v70, v71);
      v21 = (unsigned int)v285;
    }
    v72 += 3;
    v284[v21] = v75;
    v21 = (unsigned int)(v285 + 1);
    LODWORD(v285) = v285 + 1;
  }
  while ( v74 != v72 );
  v16 = v73;
  v15 = v257;
  v243 = v264;
  if ( (_DWORD)v21 )
    goto LABEL_53;
LABEL_98:
  if ( !(_DWORD)v314 )
    goto LABEL_69;
  if ( (unsigned int)v314 == 1 || !v316 )
    goto LABEL_175;
  if ( v317 || byte_4FC26C0 )
    goto LABEL_69;
  if ( v316 == 4 )
  {
    v76 = byte_4FC2260;
LABEL_108:
    if ( !v76 )
      goto LABEL_69;
    v280 = 0u;
    v287 = (unsigned __int64 *)&v289;
    v288 = 0x200000000LL;
    v281.m128i_i64[0] = 0;
    v281.m128i_i32[2] = 0;
    v77 = sub_15A9650(*v319, **((_QWORD **)v313 + 6));
    v82 = (unsigned __int64)v313;
    v265 = v77;
    v83 = &v313[56 * (unsigned int)v314];
    if ( v313 != v83 )
    {
      do
      {
        v84 = *(_QWORD *)(v82 + 48);
        v85 = 0;
        if ( *(_BYTE *)(v84 + 16) > 0x17u )
          v85 = *(_QWORD *)(v84 + 40);
        if ( v316 == 4 )
        {
          v86 = (__int64 ***)sub_15A0680(v265, *(_QWORD *)(v82 + 8), 0);
        }
        else if ( v316 > 4 )
        {
          if ( v316 != 8 )
            goto LABEL_148;
          v86 = *(__int64 ****)(v82 + 40);
        }
        else if ( v316 == 1 )
        {
          v86 = *(__int64 ****)(v82 + 32);
        }
        else
        {
          if ( v316 != 2 )
            goto LABEL_148;
          v86 = *(__int64 ****)v82;
        }
        if ( !v86 )
        {
          v84 = *(_QWORD *)(v82 + 48);
LABEL_148:
          v101 = (unsigned int)v288;
          if ( (unsigned int)v288 >= HIDWORD(v288) )
          {
            sub_16CD150((__int64)&v287, &v289, 0, 16, v78, v79);
            v101 = (unsigned int)v288;
          }
          v102 = &v287[2 * v101];
          *v102 = v84;
          v102[1] = v85;
          LODWORD(v288) = v288 + 1;
          goto LABEL_151;
        }
        if ( v318 && *v86 != v318 )
        {
          if ( v287 != (unsigned __int64 *)&v289 )
            _libc_free((unsigned __int64)v287);
          goto LABEL_233;
        }
        v318 = *v86;
        v87 = v281.m128i_i32[2];
        v88 = *(_QWORD *)(v82 + 48);
        v277.m128i_i64[1] = v85;
        v277.m128i_i64[0] = v88;
        if ( v281.m128i_i32[2] )
        {
          v89 = 1;
          v79 = 0;
          v78 = v281.m128i_i32[2] - 1;
          v90 = (((((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4)
                 | ((unsigned __int64)(((unsigned int)v88 >> 9) ^ ((unsigned int)v88 >> 4)) << 32))
                - 1
                - ((unsigned __int64)(((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4)) << 32)) >> 22)
              ^ ((((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4)
                | ((unsigned __int64)(((unsigned int)v88 >> 9) ^ ((unsigned int)v88 >> 4)) << 32))
               - 1
               - ((unsigned __int64)(((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4)) << 32));
          v91 = ((9 * (((v90 - 1 - (v90 << 13)) >> 8) ^ (v90 - 1 - (v90 << 13)))) >> 15)
              ^ (9 * (((v90 - 1 - (v90 << 13)) >> 8) ^ (v90 - 1 - (v90 << 13))));
          for ( i = (v281.m128i_i32[2] - 1) & (((v91 - 1 - (v91 << 27)) >> 31) ^ (v91 - 1 - ((_DWORD)v91 << 27)));
                ;
                i = v78 & v95 )
          {
            v93 = (__int64 *)(v280.m128i_i64[1] + 24LL * i);
            v94 = *v93;
            if ( v88 == *v93 && v85 == v93[1] )
              break;
            if ( v94 == -8 )
            {
              if ( v93[1] == -8 )
              {
                if ( v79 )
                  v93 = (__int64 *)v79;
                ++v280.m128i_i64[0];
                v203 = v281.m128i_i32[0] + 1;
                if ( 4 * (v281.m128i_i32[0] + 1) >= (unsigned int)(3 * v281.m128i_i32[2]) )
                  goto LABEL_388;
                v78 = (unsigned __int32)v281.m128i_i32[2] >> 3;
                if ( v281.m128i_i32[2] - v281.m128i_i32[1] - v203 <= (unsigned __int32)v281.m128i_i32[2] >> 3 )
                  goto LABEL_389;
                goto LABEL_382;
              }
            }
            else if ( v94 == -16 && v93[1] == -16 && !v79 )
            {
              v79 = v280.m128i_i64[1] + 24LL * i;
            }
            v95 = v89 + i;
            ++v89;
          }
        }
        else
        {
          ++v280.m128i_i64[0];
LABEL_388:
          v87 = 2 * v281.m128i_i32[2];
LABEL_389:
          sub_1D6AF90((__int64)&v280, v87);
          sub_1D66970((__int64)&v280, v277.m128i_i64, v275);
          v93 = v275[0];
          v88 = v277.m128i_i64[0];
          v203 = v281.m128i_i32[0] + 1;
LABEL_382:
          v281.m128i_i32[0] = v203;
          if ( *v93 != -8 || v93[1] != -8 )
            --v281.m128i_i32[1];
          *v93 = v88;
          v204 = v277.m128i_i64[1];
          v93[2] = 0;
          v93[1] = v204;
        }
        v93[2] = (__int64)v86;
LABEL_151:
        v82 += 56LL;
      }
      while ( v83 != (_BYTE *)v82 );
    }
    v103 = (const __m128i *)v287;
    v266 = (const __m128i *)&v287[2 * (unsigned int)v288];
    if ( v287 != (unsigned __int64 *)v266 )
    {
      do
      {
        v104 = v281.m128i_u32[2];
        v105 = v318;
        v277 = _mm_loadu_si128(v103);
        if ( v281.m128i_i32[2] )
        {
          v106 = v277.m128i_i64[0];
          v107 = 1;
          v108 = (unsigned __int32)v277.m128i_i32[2] >> 9;
          v109 = (((v108 ^ ((unsigned __int32)v277.m128i_i32[2] >> 4)
                  | ((unsigned __int64)(((unsigned __int32)v277.m128i_i32[0] >> 9)
                                      ^ ((unsigned __int32)v277.m128i_i32[0] >> 4)) << 32))
                 - 1
                 - ((unsigned __int64)(v108 ^ ((unsigned __int32)v277.m128i_i32[2] >> 4)) << 32)) >> 22)
               ^ ((v108 ^ ((unsigned __int32)v277.m128i_i32[2] >> 4)
                 | ((unsigned __int64)(((unsigned __int32)v277.m128i_i32[0] >> 9)
                                     ^ ((unsigned __int32)v277.m128i_i32[0] >> 4)) << 32))
                - 1
                - ((unsigned __int64)(v108 ^ ((unsigned __int32)v277.m128i_i32[2] >> 4)) << 32));
          v110 = ((9 * (((v109 - 1 - (v109 << 13)) >> 8) ^ (v109 - 1 - (v109 << 13)))) >> 15)
               ^ (9 * (((v109 - 1 - (v109 << 13)) >> 8) ^ (v109 - 1 - (v109 << 13))));
          v111 = ((v110 - 1 - (v110 << 27)) >> 31) ^ (v110 - 1 - ((_DWORD)v110 << 27));
          v112 = 0;
          for ( j = (v281.m128i_i32[2] - 1) & v111; ; j = (v281.m128i_i32[2] - 1) & v202 )
          {
            v114 = (__int64 *)(v280.m128i_i64[1] + 24LL * j);
            v115 = *v114;
            if ( *(_OWORD *)v114 == *(_OWORD *)&v277 )
              break;
            if ( v115 == -8 )
            {
              if ( v114[1] == -8 )
              {
                if ( v112 )
                  v114 = (__int64 *)v112;
                ++v280.m128i_i64[0];
                v116 = v281.m128i_i32[0] + 1;
                if ( 4 * (v281.m128i_i32[0] + 1) >= (unsigned int)(3 * v281.m128i_i32[2]) )
                  goto LABEL_393;
                v112 = (unsigned int)(v281.m128i_i32[2] - v281.m128i_i32[1] - v116);
                if ( (unsigned int)v112 <= (unsigned __int32)v281.m128i_i32[2] >> 3 )
                  goto LABEL_394;
                goto LABEL_162;
              }
            }
            else if ( v115 == -16 && v114[1] == -16 && !v112 )
            {
              v112 = v280.m128i_i64[1] + 24LL * j;
            }
            v202 = v107 + j;
            ++v107;
          }
        }
        else
        {
          ++v280.m128i_i64[0];
LABEL_393:
          LODWORD(v104) = 2 * v281.m128i_i32[2];
LABEL_394:
          sub_1D6AF90((__int64)&v280, v104);
          v104 = (__int64)&v277;
          sub_1D66970((__int64)&v280, v277.m128i_i64, v275);
          v114 = v275[0];
          v106 = v277.m128i_i64[0];
          v116 = v281.m128i_i32[0] + 1;
LABEL_162:
          v281.m128i_i32[0] = v116;
          if ( *v114 != -8 || v114[1] != -8 )
            --v281.m128i_i32[1];
          *v114 = v106;
          v117 = v277.m128i_i64[1];
          v114[2] = 0;
          v114[1] = v117;
        }
        ++v103;
        v114[2] = sub_15A06D0(v105, v104, v106, v112);
      }
      while ( v266 != v103 );
      v266 = (const __m128i *)v287;
    }
    if ( v266 != (const __m128i *)&v289 )
      _libc_free((unsigned __int64)v266);
    v118 = sub_1D706F0(
             (__int64)&v313,
             (__int64)&v280,
             a6,
             *(double *)a7.m128_u64,
             a8,
             a9,
             v80,
             v81,
             *(double *)a12.m128i_i64,
             a13);
    if ( !v118 )
    {
LABEL_233:
      j___libc_free_0(v280.m128i_i64[1]);
      goto LABEL_69;
    }
    v119 = v316;
    v120 = v313;
    if ( v316 == 4 )
    {
      *((_QWORD *)v313 + 5) = v118;
      v120[3] = 1;
      v120[1] = 0;
    }
    else if ( v316 > 4 )
    {
      v205 = *((_QWORD *)v313 + 3) == 0;
      *((_QWORD *)v313 + 5) = v118;
      if ( v205 )
      {
        v206 = &v120[7 * (unsigned int)v314];
        v207 = v120 + 7;
        if ( v120 != v206 )
        {
          while ( v206 != v207 )
          {
            v208 = v207[3];
            v207 += 7;
            if ( v208 )
            {
              v120[3] = v208;
              break;
            }
          }
        }
      }
    }
    else
    {
      *((_QWORD *)v313 + 4) = v118;
      if ( v119 != 1 )
        *v120 = 0;
    }
    j___libc_free_0(v280.m128i_i64[1]);
LABEL_175:
    v121 = (__int64)v303;
    v122 = &v303[8 * (unsigned int)v304];
    while ( (_BYTE *)v121 != v122 )
    {
      while ( 1 )
      {
        v123 = *((_QWORD *)v122 - 1);
        v122 -= 8;
        if ( !v123 )
          break;
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v123 + 8LL))(v123);
        if ( (_BYTE *)v121 == v122 )
          goto LABEL_179;
      }
    }
LABEL_179:
    LODWORD(v304) = 0;
    v124 = *(__int64 **)v313;
    v125 = *((_QWORD *)v313 + 4);
    v245 = *((_QWORD *)v313 + 1);
    v267 = *((_QWORD *)v313 + 3);
    v258 = *((_QWORD *)v313 + 5);
    if ( v243 )
      goto LABEL_199;
    v126 = v300;
    v127 = 8LL * (unsigned int)v301;
    v128 = &v300[v127];
    v129 = v127 >> 3;
    if ( !(v127 >> 5) )
      goto LABEL_192;
    v130 = &v300[32 * (v127 >> 5)];
    v131 = *(_QWORD *)(a2 + 40);
    do
    {
      if ( *(_BYTE *)(*v126 + 16LL) > 0x17u && v131 != *(_QWORD *)(*v126 + 40LL) )
        goto LABEL_198;
      v132 = v126[1];
      if ( *(_BYTE *)(v132 + 16) > 0x17u && v131 != *(_QWORD *)(v132 + 40) )
      {
        ++v126;
        goto LABEL_198;
      }
      v133 = v126[2];
      if ( *(_BYTE *)(v133 + 16) > 0x17u && v131 != *(_QWORD *)(v133 + 40) )
      {
        v126 += 2;
        goto LABEL_198;
      }
      v134 = v126[3];
      if ( *(_BYTE *)(v134 + 16) > 0x17u && v131 != *(_QWORD *)(v134 + 40) )
      {
        v126 += 3;
        goto LABEL_198;
      }
      v126 += 4;
    }
    while ( v130 != v126 );
    v129 = v128 - v126;
LABEL_192:
    switch ( v129 )
    {
      case 2LL:
        v135 = *(_QWORD *)(a2 + 40);
        break;
      case 3LL:
        v135 = *(_QWORD *)(a2 + 40);
        if ( *(_BYTE *)(*v126 + 16LL) > 0x17u && v135 != *(_QWORD *)(*v126 + 40LL) )
        {
LABEL_198:
          if ( v128 == v126 )
            goto LABEL_74;
LABEL_199:
          v136 = (_QWORD *)sub_16498A0(a2);
          v137 = *(_QWORD *)(a2 + 48);
          v287 = 0;
          v290 = v136;
          v138 = *(_QWORD *)(a2 + 40);
          v291 = 0;
          v288 = v138;
          v292 = 0;
          v293 = 0;
          v294 = 0;
          v289 = (__int64 *)(a2 + 24);
          v280.m128i_i64[0] = v137;
          if ( v137 )
          {
            sub_1623A60((__int64)&v280, v137, 2);
            if ( v287 )
              sub_161E7C0((__int64)&v287, (__int64)v287);
            v287 = (unsigned __int64 *)v280.m128i_i64[0];
            if ( v280.m128i_i64[0] )
              sub_1623210((__int64)&v280, (unsigned __int8 *)v280.m128i_i64[0], (__int64)&v287);
          }
          v280.m128i_i64[1] = 2;
          v247 = a1 + 240;
          v281.m128i_i64[1] = (__int64)a3;
          v281.m128i_i64[0] = 0;
          v239 = a3 + 2 != 0 && a3 != 0 && a3 + 1 != 0;
          if ( v239 )
            sub_164C220((__int64)&v280.m128i_i64[1]);
          v280.m128i_i64[0] = (__int64)&unk_49F9E38;
          v282.m128i_i64[0] = a1 + 240;
          v139 = sub_1D682F0(v247, (__int64)&v280, &v277);
          v140 = (unsigned __int64 *)v277.m128i_i64[0];
          if ( !v139 )
          {
            v140 = (unsigned __int64 *)sub_1D736D0(v247, (__int64)&v280, v277.m128i_i64[0]);
            sub_1D5A8A0(v140 + 1, &v280.m128i_i64[1]);
            v146 = v282.m128i_i64[0];
            v140[5] = 6;
            v140[6] = 0;
            v140[4] = v146;
            v140[7] = 0;
          }
          v280.m128i_i64[0] = (__int64)&unk_49EE2B0;
          sub_1455FA0((__int64)&v280.m128i_i64[1]);
          v275[0] = (__int64 *)6;
          v275[1] = 0;
          v276 = v140[7];
          if ( v276 == -8
            || v276 == 0
            || v276 == -16
            || (sub_1649AC0((unsigned __int64 *)v275, v140[5] & 0xFFFFFFFFFFFFFFF8LL), v276 == 0 || v276 == -8)
            || v276 == -16 )
          {
            v280 = (__m128i)6uLL;
            v281.m128i_i64[0] = 0;
            sub_1455FA0((__int64)&v280);
          }
          else
          {
            v280 = (__m128i)6uLL;
            v281.m128i_i64[0] = v276;
            sub_1649AC0((unsigned __int64 *)&v280, (unsigned __int64)v275[0] & 0xFFFFFFFFFFFFFFF8LL);
            v141 = (__int64 ***)v281.m128i_i64[0];
            sub_1455FA0((__int64)&v280);
            if ( v141 )
            {
              v142 = (__int64)*a3;
              if ( *v141 != *a3 )
              {
                v281.m128i_i16[0] = 257;
                v141 = (__int64 ***)sub_12A95D0((__int64 *)&v287, (__int64)v141, v142, (__int64)&v280);
                goto LABEL_215;
              }
LABEL_266:
              sub_1648780(a2, (__int64)a3, (__int64)v141);
              v278 = v141;
              v277 = (__m128i)6uLL;
LABEL_267:
              if ( v141 != (__int64 ***)-16LL && v141 != (__int64 ***)-8LL )
                sub_164C220((__int64)&v277);
LABEL_216:
              v280.m128i_i64[1] = 2;
              v281.m128i_i64[0] = 0;
              v281.m128i_i64[1] = (__int64)a3;
              if ( v239 )
                sub_164C220((__int64)&v280.m128i_i64[1]);
              v282.m128i_i64[0] = a1 + 240;
              v280.m128i_i64[0] = (__int64)&unk_49F9E38;
              v143 = sub_1D682F0(v247, (__int64)&v280, &v274);
              v144 = v274;
              if ( !v143 )
              {
                v144 = (unsigned __int64 *)sub_1D736D0(v247, (__int64)&v280, (__int64)v274);
                sub_1D5A8A0(v144 + 1, &v280.m128i_i64[1]);
                v145 = v282.m128i_i64[0];
                v144[5] = 6;
                v144[6] = 0;
                v144[4] = v145;
                v144[7] = 0;
              }
              v280.m128i_i64[0] = (__int64)&unk_49EE2B0;
              sub_1455FA0((__int64)&v280.m128i_i64[1]);
              sub_1D5A8A0(v144 + 5, &v277);
              sub_1455FA0((__int64)&v277);
              v243 = 1;
              if ( a3[1] )
                goto LABEL_221;
              v169 = *(_QWORD *)(a1 + 232);
              if ( v169 )
              {
                v280 = (__m128i)6uLL;
                v170 = v169 - 24;
                v281.m128i_i64[0] = v169 - 24;
                if ( v169 == 16 || v169 == 8 || (sub_164C220((__int64)&v280), (v169 = *(_QWORD *)(a1 + 232)) != 0) )
                {
                  v171 = *(_QWORD *)(v169 + 16);
                  sub_1AEB370((__int64)a3, *(__int64 **)(a1 + 200));
                  if ( v281.m128i_i64[0] != v170 )
                  {
                    *(_QWORD *)(a1 + 232) = *(_QWORD *)(v171 + 48);
                    sub_1D672E0(v247);
                    if ( *(_BYTE *)(a1 + 304) )
                    {
                      v197 = *(_DWORD *)(a1 + 296);
                      if ( v197 )
                      {
                        v198 = *(_QWORD **)(a1 + 280);
                        v199 = &v198[2 * v197];
                        do
                        {
                          if ( *v198 != -8 && *v198 != -4 )
                          {
                            v200 = v198[1];
                            if ( v200 )
                              sub_161E7C0((__int64)(v198 + 1), v200);
                          }
                          v198 += 2;
                        }
                        while ( v199 != v198 );
                      }
                      j___libc_free_0(*(_QWORD *)(a1 + 280));
                      *(_BYTE *)(a1 + 304) = 0;
                    }
                  }
                  sub_1455FA0((__int64)&v280);
                  v243 = 1;
                  goto LABEL_221;
                }
              }
              else
              {
                v280 = (__m128i)6uLL;
                v281.m128i_i64[0] = 0;
              }
              BUG();
            }
          }
          v243 = byte_4FC3060;
          if ( !byte_4FC3060 )
          {
            v147 = sub_16D5D50();
            v149 = &qword_4FA0200;
            v150 = v147;
            v151 = *(_QWORD **)&dword_4FA0208[2];
            if ( *(_QWORD *)&dword_4FA0208[2] )
            {
              v148 = (__int64)dword_4FA0208;
              do
              {
                if ( v150 > v151[4] )
                {
                  v151 = (_QWORD *)v151[3];
                }
                else
                {
                  v148 = (__int64)v151;
                  v151 = (_QWORD *)v151[2];
                }
              }
              while ( v151 );
              if ( (_DWORD *)v148 != dword_4FA0208 && v150 >= *(_QWORD *)(v148 + 32) )
              {
                v174 = *(_QWORD *)(v148 + 56);
                v149 = (_QWORD *)(v148 + 48);
                if ( v174 )
                {
                  v148 = (unsigned int)dword_4FC2FC8;
                  v150 = (unsigned __int64)v149;
                  do
                  {
                    if ( *(_DWORD *)(v174 + 32) < dword_4FC2FC8 )
                    {
                      v174 = *(_QWORD *)(v174 + 24);
                    }
                    else
                    {
                      v150 = v174;
                      v174 = *(_QWORD *)(v174 + 16);
                    }
                  }
                  while ( v174 );
                  if ( (_QWORD *)v150 != v149 && dword_4FC2FC8 >= *(_DWORD *)(v150 + 32) && *(_DWORD *)(v150 + 36) )
                    goto LABEL_313;
                }
              }
            }
            if ( !*(_QWORD *)(a1 + 160) || !(unsigned __int8)sub_14A2D50(*(_QWORD *)(a1 + 192)) )
            {
LABEL_313:
              if ( v125 )
              {
                v149 = *(_QWORD **)v125;
                if ( v267 )
                {
                  v175 = *(_QWORD *)v258;
                  if ( v149 && *((_BYTE *)v149 + 8) != 15 )
                    v149 = 0;
LABEL_318:
                  if ( v175 && *(_BYTE *)(v175 + 8) != 15 )
                    v175 = 0;
                  v176 = *(_QWORD *)(a1 + 904);
                  v177 = (__int64)*a3;
                  if ( *((_BYTE *)*a3 + 8) != 15 )
                    goto LABEL_323;
                  goto LABEL_322;
                }
                if ( v149 && *((_BYTE *)v149 + 8) == 15 )
                {
                  v175 = 0;
                  v176 = *(_QWORD *)(a1 + 904);
                  v177 = (__int64)*a3;
                  if ( *((_BYTE *)*a3 + 8) != 15 )
                  {
LABEL_324:
                    v227 = v175;
                    v234 = v177;
                    v179 = sub_1D5D710(v176, (__int64)v149, v150, v148, (unsigned int)v149);
                    v177 = v234;
                    v175 = v227;
                    if ( v179 )
                      goto LABEL_371;
LABEL_325:
                    if ( v175 )
                    {
                      v235 = v177;
                      v180 = v175;
                      goto LABEL_327;
                    }
                    goto LABEL_328;
                  }
LABEL_322:
                  v225 = v175;
                  v226 = v149;
                  v233 = v177;
                  v178 = sub_1D5D710(v176, v177, v150, v148, (unsigned int)v149);
                  v177 = v233;
                  v149 = v226;
                  v175 = v225;
                  if ( v178 )
                    goto LABEL_371;
LABEL_323:
                  if ( !v149 )
                    goto LABEL_325;
                  goto LABEL_324;
                }
              }
              else if ( v267 )
              {
                v149 = 0;
                v175 = *(_QWORD *)v258;
                goto LABEL_318;
              }
              v176 = *(_QWORD *)(a1 + 904);
              v177 = (__int64)*a3;
              if ( *((_BYTE *)*a3 + 8) == 15 )
              {
                v235 = (__int64)*a3;
                v180 = (__int64)*a3;
LABEL_327:
                v181 = sub_1D5D710(v176, v180, v150, v148, (unsigned int)v149);
                v177 = v235;
                if ( v181 )
                  goto LABEL_371;
              }
LABEL_328:
              if ( v124 )
              {
                v236 = v177;
                v182 = sub_1D5D710(v176, *v124, v150, v148, (unsigned int)v149);
                v177 = v236;
                if ( v182 )
                  goto LABEL_371;
              }
              v152 = v177;
              v183 = sub_15A9650(v176, v177);
              v184 = v183;
              if ( v125 )
              {
                v185 = *(_QWORD **)v125;
                v186 = v125;
                if ( *(_BYTE *)(*(_QWORD *)v125 + 8LL) == 15 )
                {
                  v152 = 45;
                  v280.m128i_i64[0] = (__int64)"sunkaddr";
                  v281.m128i_i16[0] = 259;
                  v186 = sub_12AA3B0((__int64 *)&v287, 0x2Du, v125, v184, (__int64)&v280);
                  v185 = *(_QWORD **)v186;
                }
                if ( (_QWORD *)v184 != v185 )
                {
                  v277.m128i_i64[0] = (__int64)"sunkaddr";
                  LOWORD(v278) = 259;
                  if ( v184 != *(_QWORD *)v186 )
                  {
                    if ( *(_BYTE *)(v186 + 16) > 0x10u )
                    {
                      v281.m128i_i16[0] = 257;
                      v221 = sub_15FE0A0((_QWORD *)v186, v184, 1, (__int64)&v280, 0);
                      if ( v288 )
                      {
                        v255 = v221;
                        v237 = v289;
                        sub_157E9D0(v288 + 40, v221);
                        v221 = v255;
                        v222 = *(_QWORD *)(v255 + 24);
                        v223 = *v237;
                        *(_QWORD *)(v255 + 32) = v237;
                        v223 &= 0xFFFFFFFFFFFFFFF8LL;
                        *(_QWORD *)(v255 + 24) = v223 | v222 & 7;
                        *(_QWORD *)(v223 + 8) = v255 + 24;
                        *v237 = *v237 & 7 | (v255 + 24);
                      }
                      v256 = v221;
                      sub_164B780(v221, v277.m128i_i64);
                      v152 = (__int64)v287;
                      v186 = v256;
                      if ( v287 )
                      {
                        v274 = v287;
                        sub_1623A60((__int64)&v274, (__int64)v287, 2);
                        v186 = v256;
                        v224 = *(_QWORD *)(v256 + 48);
                        v153 = v256 + 48;
                        if ( v224 )
                        {
                          sub_161E7C0(v256 + 48, v224);
                          v186 = v256;
                          v153 = v256 + 48;
                        }
                        v152 = (__int64)v274;
                        *(_QWORD *)(v186 + 48) = v274;
                        if ( v152 )
                        {
                          v238 = v186;
                          sub_1623210((__int64)&v274, (unsigned __int8 *)v152, v153);
                          v186 = v238;
                        }
                      }
                    }
                    else
                    {
                      v152 = v184;
                      v186 = sub_15A4750((__int64 ***)v186, (__int64 **)v184, 1);
                    }
                  }
                }
                v187 = v186;
                if ( !v267 )
                  goto LABEL_346;
LABEL_338:
                v188 = *(_QWORD *)v258;
                if ( v184 == *(_QWORD *)v258 )
                {
                  v186 = v258;
                  goto LABEL_342;
                }
                if ( *(_BYTE *)(v188 + 8) == 15 )
                {
                  v253 = v187;
                  v280.m128i_i64[0] = (__int64)"sunkaddr";
                  v281.m128i_i16[0] = 259;
                  v215 = sub_12AA3B0((__int64 *)&v287, 0x2Du, v258, v184, (__int64)&v280);
                  v187 = v253;
                  v186 = v215;
                  goto LABEL_342;
                }
                if ( *(_DWORD *)(v188 + 8) >> 8 > *(_DWORD *)(v184 + 8) >> 8 )
                {
                  v250 = v187;
                  v280.m128i_i64[0] = (__int64)"sunkaddr";
                  v281.m128i_i16[0] = 259;
                  v189 = sub_12AA3B0((__int64 *)&v287, 0x24u, v258, v184, (__int64)&v280);
                  v187 = v250;
                  v186 = v189;
LABEL_342:
                  v152 = v267;
                  if ( v267 != 1 )
                  {
                    v260 = v187;
                    v273 = v186;
                    v280.m128i_i64[0] = (__int64)"sunkaddr";
                    v281.m128i_i16[0] = 259;
                    v190 = sub_15A0680(v184, v152, 0);
                    v152 = v273;
                    v191 = sub_156D130((__int64 *)&v287, v273, v190, (__int64)&v280, 0, 0);
                    v187 = v260;
                    v186 = v191;
                  }
                  if ( v187 )
                  {
                    v152 = v187;
                    v280.m128i_i64[0] = (__int64)"sunkaddr";
                    v281.m128i_i16[0] = 259;
                    v186 = sub_12899C0((__int64 *)&v287, v187, v186, (__int64)&v280, 0, 0);
                  }
LABEL_346:
                  v261 = v186;
                  v192 = v186;
                  if ( v124 )
                  {
                    v152 = 45;
                    v280.m128i_i64[0] = (__int64)"sunkaddr";
                    v281.m128i_i16[0] = 259;
                    v193 = sub_12AA3B0((__int64 *)&v287, 0x2Du, (__int64)v124, v184, (__int64)&v280);
                    v192 = v193;
                    if ( v261 )
                    {
                      v280.m128i_i64[0] = (__int64)"sunkaddr";
                      v152 = v261;
                      v281.m128i_i16[0] = 259;
                      v192 = sub_12899C0((__int64 *)&v287, v261, v193, (__int64)&v280, 0, 0);
                    }
                  }
                  goto LABEL_349;
                }
                if ( v187 && *(_BYTE *)(v187 + 16) > 0x17u && v125 != v187 )
                {
                  sub_15F20C0((_QWORD *)v187);
LABEL_221:
                  sub_1455FA0((__int64)v275);
                  if ( v287 )
                    sub_161E7C0((__int64)&v287, (__int64)v287);
                  goto LABEL_70;
                }
LABEL_371:
                v243 = 0;
                goto LABEL_221;
              }
              if ( v267 )
              {
                v187 = 0;
                goto LABEL_338;
              }
              if ( v124 )
              {
                v152 = 45;
                v280.m128i_i64[0] = (__int64)"sunkaddr";
                v281.m128i_i16[0] = 259;
                v192 = sub_12AA3B0((__int64 *)&v287, 0x2Du, (__int64)v124, v183, (__int64)&v280);
LABEL_349:
                if ( v245 )
                {
                  v152 = v245;
                  v211 = sub_15A0680(v184, v245, 0);
                  v153 = v211;
                  if ( v192 )
                  {
                    v152 = v192;
                    v280.m128i_i64[0] = (__int64)"sunkaddr";
                    v281.m128i_i16[0] = 259;
                    v192 = sub_12899C0((__int64 *)&v287, v192, v211, (__int64)&v280, 0, 0);
                  }
                  else
                  {
                    v192 = v211;
                  }
                }
LABEL_350:
                if ( v192 )
                {
                  v194 = (__int64)*a3;
                  v281.m128i_i16[0] = 259;
                  v280.m128i_i64[0] = (__int64)"sunkaddr";
                  v141 = (__int64 ***)sub_12AA3B0((__int64 *)&v287, 0x2Eu, v192, v194, (__int64)&v280);
                  goto LABEL_215;
                }
                goto LABEL_440;
              }
              if ( v245 )
              {
                v152 = v245;
                v192 = sub_15A0680(v183, v245, 0);
                goto LABEL_350;
              }
LABEL_440:
              v141 = (__int64 ***)sub_15A06D0(*a3, v152, v153, (__int64)v154);
              goto LABEL_215;
            }
          }
          v152 = (__int64)*a3;
          v244 = sub_15A9650(*(_QWORD *)(a1 + 904), (__int64)*a3);
          if ( v125 )
          {
            if ( *(_BYTE *)(*(_QWORD *)v125 + 8LL) == 15 )
            {
              if ( !v267 )
              {
                if ( v124 )
                  goto LABEL_371;
                v154 = *a3;
                if ( *((_BYTE *)*a3 + 8) != 15 )
                  goto LABEL_447;
                v152 = (__int64)*a3;
                v251 = *a3;
                v124 = (__int64 *)v125;
                v125 = 0;
                v212 = sub_1D5D710(*(_QWORD *)(a1 + 904), (__int64)*a3, v153, (__int64)v154, v155);
                v154 = v251;
                if ( v212 )
                  goto LABEL_242;
LABEL_446:
                v213 = v124;
                v124 = (__int64 *)v125;
                v125 = (__int64)v213;
LABEL_447:
                v141 = (__int64 ***)v125;
                v159 = *((_BYTE *)v154 + 8);
                v125 = (__int64)v124;
LABEL_244:
                if ( v159 != 16 )
                  goto LABEL_245;
                goto LABEL_469;
              }
              v153 = 0;
              v201 = *(_QWORD *)v258;
              if ( *(_BYTE *)(*(_QWORD *)v258 + 8LL) == 15 )
                goto LABEL_371;
              goto LABEL_368;
            }
            if ( !v267 )
            {
              v156 = *(_QWORD *)(a1 + 904);
              v157 = *a3;
              v153 = *((unsigned __int8 *)*a3 + 8);
              v158 = *a3;
              if ( !v124 )
              {
                if ( (_BYTE)v153 != 15 )
                {
                  v152 = 46;
                  v280.m128i_i64[0] = (__int64)"sunkaddr";
                  v281.m128i_i16[0] = 259;
                  v141 = (__int64 ***)sub_12AA3B0((__int64 *)&v287, 0x2Eu, v125, (__int64)v157, (__int64)&v280);
                  if ( !v141 )
                    goto LABEL_439;
LABEL_464:
                  v154 = *a3;
                  if ( *((_BYTE *)*a3 + 8) != 16 )
                  {
                    v161 = sub_16471D0(v290, *((_DWORD *)v154 + 2) >> 8);
                    v230 = sub_1643330(v290);
LABEL_466:
                    if ( v267 )
                    {
                      v162 = 0;
LABEL_251:
                      v125 = v258;
                      if ( v244 != *(_QWORD *)v258 )
                      {
                        v163 = v258;
                        v259 = v162;
                        v280.m128i_i64[0] = (__int64)"sunkaddr";
                        v281.m128i_i16[0] = 259;
                        v164 = sub_12AA3B0((__int64 *)&v287, 0x24u, v163, v244, (__int64)&v280);
                        v162 = v259;
                        v125 = v164;
                      }
                      v165 = v267;
                      if ( v267 != 1 )
                      {
                        v268 = v162;
                        v280.m128i_i64[0] = (__int64)"sunkaddr";
                        v281.m128i_i16[0] = 259;
                        v166 = sub_15A0680(v244, v165, 0);
                        v167 = sub_156D130((__int64 *)&v287, v125, v166, (__int64)&v280, 0, 0);
                        v162 = v268;
                        v125 = v167;
                      }
                      if ( v162 )
                      {
                        v280.m128i_i64[0] = (__int64)"sunkaddr";
                        v281.m128i_i16[0] = 259;
                        v125 = sub_12899C0((__int64 *)&v287, v162, v125, (__int64)&v280, 0, 0);
                      }
LABEL_257:
                      v269 = v125;
                      if ( v245 )
                      {
                        v269 = sub_15A0680(v244, v245, 0);
                        if ( v125 )
                        {
                          if ( *v141 != (__int64 **)v161 )
                          {
                            v281.m128i_i16[0] = 257;
                            v141 = (__int64 ***)sub_12A95D0((__int64 *)&v287, (__int64)v141, v161, (__int64)&v280);
                          }
                          v280.m128i_i64[0] = (__int64)"sunkaddr";
                          v281.m128i_i16[0] = 259;
                          v141 = (__int64 ***)sub_12815B0((__int64 *)&v287, v230, v141, v125, (__int64)&v280);
                        }
                      }
                      goto LABEL_258;
                    }
LABEL_475:
                    if ( !v245 )
                    {
LABEL_262:
                      v168 = *a3;
                      if ( *a3 == *v141 )
                        goto LABEL_266;
                      v281.m128i_i16[0] = 257;
                      v141 = (__int64 ***)sub_12A95D0((__int64 *)&v287, (__int64)v141, (__int64)v168, (__int64)&v280);
LABEL_215:
                      sub_1648780(a2, (__int64)a3, (__int64)v141);
                      v278 = v141;
                      v277 = (__m128i)6uLL;
                      if ( !v141 )
                        goto LABEL_216;
                      goto LABEL_267;
                    }
                    v269 = sub_15A0680(v244, v245, 0);
LABEL_258:
                    if ( v269 )
                    {
                      if ( (__int64 **)v161 != *v141 )
                      {
                        v281.m128i_i16[0] = 257;
                        v141 = (__int64 ***)sub_12A95D0((__int64 *)&v287, (__int64)v141, v161, (__int64)&v280);
                      }
                      v280.m128i_i64[0] = (__int64)"sunkaddr";
                      v281.m128i_i16[0] = 259;
                      v141 = (__int64 ***)sub_12815B0((__int64 *)&v287, v230, v141, v269, (__int64)&v280);
                    }
                    goto LABEL_262;
                  }
                  v125 = 0;
                  goto LABEL_469;
                }
                v252 = *a3;
                v214 = sub_1D5D710(v156, (__int64)*a3, v153, (__int64)v157, v155);
                v154 = v252;
                if ( v214 )
                  goto LABEL_371;
LABEL_463:
                v152 = 46;
                v280.m128i_i64[0] = (__int64)"sunkaddr";
                v281.m128i_i16[0] = 259;
                v141 = (__int64 ***)sub_12AA3B0((__int64 *)&v287, 0x2Eu, v125, (__int64)v154, (__int64)&v280);
                if ( !v141 )
                {
LABEL_438:
                  if ( v267 )
                    goto LABEL_371;
                  goto LABEL_439;
                }
                goto LABEL_464;
              }
              goto LABEL_240;
            }
            v201 = *(_QWORD *)v258;
            if ( *(_BYTE *)(*(_QWORD *)v258 + 8LL) != 15 )
            {
              v153 = v125;
              v125 = 0;
LABEL_368:
              if ( *(_DWORD *)(v201 + 8) >> 8 < *(_DWORD *)(v244 + 8) >> 8 )
                goto LABEL_371;
              goto LABEL_369;
            }
          }
          else
          {
            if ( !v267 )
            {
              if ( !v124 )
                goto LABEL_439;
LABEL_418:
              v156 = *(_QWORD *)(a1 + 904);
              v158 = *a3;
LABEL_240:
              v154 = v158;
              v159 = *((_BYTE *)v158 + 8);
              v141 = (__int64 ***)v124;
              if ( v159 != 15 )
                goto LABEL_244;
LABEL_241:
              v152 = (__int64)v154;
              v249 = v154;
              v160 = sub_1D5D710(v156, (__int64)v154, v153, (__int64)v154, v155);
              v154 = v249;
              if ( v160 )
              {
LABEL_242:
                if ( v124 )
                {
                  v159 = *((_BYTE *)v154 + 8);
                  v141 = (__int64 ***)v124;
                  goto LABEL_244;
                }
                if ( v125 )
                  goto LABEL_371;
                goto LABEL_438;
              }
LABEL_435:
              if ( !v124 )
              {
                if ( !v125 )
                {
                  if ( v267 != 1 )
                    goto LABEL_438;
                  v152 = 46;
                  v280.m128i_i64[0] = (__int64)"sunkaddr";
                  v281.m128i_i16[0] = 259;
                  v141 = (__int64 ***)sub_12AA3B0((__int64 *)&v287, 0x2Eu, v258, (__int64)v154, (__int64)&v280);
                  if ( v141 )
                  {
                    v154 = *a3;
                    if ( *((_BYTE *)*a3 + 8) != 16 )
                    {
                      v161 = sub_16471D0(v290, *((_DWORD *)v154 + 2) >> 8);
                      v230 = sub_1643330(v290);
                      goto LABEL_475;
                    }
                    v267 = 0;
LABEL_469:
                    v154 = (__int64 **)*v154[2];
LABEL_245:
                    v161 = sub_16471D0(v290, *((_DWORD *)v154 + 2) >> 8);
                    v230 = sub_1643330(v290);
                    if ( v125 )
                    {
                      if ( v244 != *(_QWORD *)v125 )
                      {
                        v277.m128i_i64[0] = (__int64)"sunkaddr";
                        LOWORD(v278) = 259;
                        if ( v244 != *(_QWORD *)v125 )
                        {
                          if ( *(_BYTE *)(v125 + 16) > 0x10u )
                          {
                            v281.m128i_i16[0] = 257;
                            v125 = sub_15FE0A0((_QWORD *)v125, v244, 1, (__int64)&v280, 0);
                            if ( v288 )
                            {
                              v254 = v289;
                              sub_157E9D0(v288 + 40, v125);
                              v216 = *v254;
                              v217 = *(_QWORD *)(v125 + 24) & 7LL;
                              *(_QWORD *)(v125 + 32) = v254;
                              v216 &= 0xFFFFFFFFFFFFFFF8LL;
                              *(_QWORD *)(v125 + 24) = v216 | v217;
                              *(_QWORD *)(v216 + 8) = v125 + 24;
                              *v254 = *v254 & 7 | (v125 + 24);
                            }
                            sub_164B780(v125, v277.m128i_i64);
                            if ( v287 )
                            {
                              v274 = v287;
                              sub_1623A60((__int64)&v274, (__int64)v287, 2);
                              v218 = *(_QWORD *)(v125 + 48);
                              v219 = v125 + 48;
                              if ( v218 )
                              {
                                sub_161E7C0(v125 + 48, v218);
                                v219 = v125 + 48;
                              }
                              v220 = (unsigned __int8 *)v274;
                              *(_QWORD *)(v125 + 48) = v274;
                              if ( v220 )
                                sub_1623210((__int64)&v274, v220, v219);
                            }
                          }
                          else
                          {
                            v125 = sub_15A4750((__int64 ***)v125, (__int64 **)v244, 1);
                          }
                        }
                      }
                      v162 = v125;
                      if ( !v267 )
                        goto LABEL_257;
                      goto LABEL_251;
                    }
                    goto LABEL_466;
                  }
LABEL_439:
                  if ( v245 )
                    goto LABEL_371;
                  goto LABEL_440;
                }
                goto LABEL_463;
              }
              goto LABEL_446;
            }
            v153 = 0;
            v201 = *(_QWORD *)v258;
            if ( *(_BYTE *)(*(_QWORD *)v258 + 8LL) != 15 )
              goto LABEL_368;
          }
          if ( v267 != 1 )
            goto LABEL_371;
          v267 = 0;
          v153 = v125;
          v125 = v258;
LABEL_369:
          if ( !v124 )
          {
            v124 = (__int64 *)v125;
            v156 = *(_QWORD *)(a1 + 904);
            v125 = v153;
            v154 = *a3;
            if ( *((_BYTE *)*a3 + 8) != 15 )
              goto LABEL_435;
            goto LABEL_241;
          }
          if ( v125 )
            goto LABEL_371;
          v125 = v153;
          goto LABEL_418;
        }
        ++v126;
        break;
      case 1LL:
        v135 = *(_QWORD *)(a2 + 40);
LABEL_196:
        v64 = v303;
        if ( *(_BYTE *)(*v126 + 16LL) <= 0x17u || *(_QWORD *)(*v126 + 40LL) == v135 )
          goto LABEL_75;
        goto LABEL_198;
      default:
        goto LABEL_74;
    }
    if ( *(_BYTE *)(*v126 + 16LL) <= 0x17u || v135 == *(_QWORD *)(*v126 + 40LL) )
    {
      ++v126;
      goto LABEL_196;
    }
    goto LABEL_198;
  }
  if ( v316 > 4 )
  {
    v76 = byte_4FC2180;
    if ( v316 != 8 )
      goto LABEL_69;
    goto LABEL_108;
  }
  if ( v316 == 1 )
  {
    v76 = byte_4FC2420;
    goto LABEL_108;
  }
  if ( v316 == 2 )
  {
    v76 = byte_4FC2340;
    goto LABEL_108;
  }
LABEL_69:
  sub_1D5ABA0((__int64 *)&v303, 0);
  v243 = 0;
LABEL_70:
  v64 = v303;
  v65 = &v303[8 * (unsigned int)v304];
  if ( v303 != v65 )
  {
    do
    {
      v66 = *((_QWORD *)v65 - 1);
      v65 -= 8;
      if ( v66 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v66 + 8LL))(v66);
    }
    while ( v64 != v65 );
LABEL_74:
    v64 = v303;
  }
LABEL_75:
  if ( v64 != v305 )
    _libc_free((unsigned __int64)v64);
  if ( v313 != v315 )
    _libc_free((unsigned __int64)v313);
  if ( v300 != v302 )
    _libc_free((unsigned __int64)v300);
  if ( v309 != v308 )
    _libc_free((unsigned __int64)v309);
  if ( v284 != v286 )
    _libc_free((unsigned __int64)v284);
  return v243;
}
