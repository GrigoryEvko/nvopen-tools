// Function: sub_19C4800
// Address: 0x19c4800
//
void __fastcall sub_19C4800(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        __m128 a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        __m128 a13)
{
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  _BYTE *v19; // rsi
  _BYTE *v20; // r15
  _BYTE *v21; // r10
  _BYTE *v22; // r8
  signed __int64 v23; // r8
  size_t v24; // r12
  __int64 v25; // r15
  __int64 v26; // r13
  int v27; // r8d
  int v28; // r9d
  __int64 v29; // rbx
  __int64 v30; // r12
  signed __int64 v31; // r12
  __int64 *v32; // rbx
  _QWORD *v33; // rax
  __int64 *v34; // rsi
  int v35; // eax
  __int64 v36; // rdx
  void *v37; // r9
  _BYTE *v38; // r13
  _BYTE *v39; // r15
  signed __int64 v40; // r12
  signed __int64 v41; // r14
  signed __int64 v42; // r14
  _BYTE *v43; // r14
  signed __int64 v44; // rbx
  __int64 v45; // rax
  char *v46; // r13
  signed __int64 v47; // rdx
  _QWORD *v48; // rax
  _QWORD *v49; // rdx
  char v50; // cl
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // r14
  char *v54; // rax
  _BYTE *v55; // rsi
  char *v56; // r12
  __int64 v57; // rdx
  __int64 v58; // rdx
  _QWORD *v59; // rbx
  int v60; // ecx
  __int64 v61; // rcx
  unsigned __int64 *v62; // rdi
  __int64 **v63; // rcx
  unsigned int v64; // ecx
  __int64 v65; // rdi
  char *v66; // rdx
  _QWORD *v67; // rdi
  __int64 v68; // rdi
  __int64 v69; // rax
  __int64 v70; // rcx
  __int64 v71; // r9
  unsigned __int64 v72; // rsi
  __int64 v73; // rdi
  _QWORD *v74; // r13
  _QWORD *v75; // rax
  _QWORD *v76; // r11
  _QWORD *v77; // rbx
  _QWORD *v78; // rax
  __int64 v79; // rcx
  __int64 v80; // rdx
  __int64 v81; // r10
  int v82; // edx
  int v83; // eax
  char *v84; // r15
  __int64 v85; // rax
  __int64 v86; // rax
  _QWORD *v87; // rbx
  int v88; // edx
  __int64 v89; // rdx
  __int64 **v90; // rdx
  __int64 v91; // rax
  _QWORD *v92; // r11
  __int64 v93; // r12
  unsigned __int64 *v94; // rsi
  __int64 v95; // rax
  bool v96; // al
  _BOOL8 v97; // rdi
  unsigned int v98; // edx
  __int64 v99; // rcx
  __int64 v100; // r13
  __int64 v101; // r8
  int v102; // ecx
  int v103; // ecx
  __int64 v104; // rdi
  __int64 v105; // rsi
  unsigned int v106; // edx
  __int64 *v107; // rax
  __int64 v108; // r10
  __int64 v109; // rdi
  unsigned __int64 v110; // rax
  __int64 v111; // rax
  __int64 v112; // rdx
  __int64 v113; // r14
  __int64 v114; // rbx
  __int64 v115; // rcx
  __int64 v116; // r8
  __int64 v117; // r9
  unsigned int v118; // eax
  int v119; // edx
  __int64 v120; // rsi
  __int64 v121; // rcx
  __int64 v122; // rdi
  __int64 v123; // rsi
  __int64 v124; // r12
  __int64 v125; // rcx
  __int64 v126; // r10
  __int64 v127; // rdx
  unsigned int v128; // ecx
  int v129; // eax
  __int64 v130; // rdx
  _QWORD *v131; // rax
  __int64 v132; // rcx
  unsigned __int64 v133; // rdx
  __int64 v134; // rdx
  __int64 v135; // rdx
  __int64 v136; // rcx
  __int64 v137; // rax
  __int64 *v138; // rbx
  __int64 *v139; // r13
  __int64 v140; // rax
  __int64 v141; // rbx
  __int64 v142; // r14
  __int64 v143; // rbx
  __int64 v144; // rax
  __int64 v145; // r12
  __int64 v146; // r13
  __int64 v147; // rax
  __int64 v148; // rbx
  __int64 v149; // r12
  _QWORD *v150; // rax
  __int64 v151; // rdx
  __int64 *v152; // rax
  __int64 v153; // rcx
  unsigned __int64 v154; // rdx
  __int64 v155; // rdx
  __int64 v156; // rax
  __int64 v157; // rsi
  __int64 v158; // r14
  __int64 v159; // r13
  double v160; // xmm4_8
  double v161; // xmm5_8
  __int64 v162; // rdx
  __int64 v163; // rcx
  __int64 v164; // r8
  __int64 v165; // r9
  int v166; // eax
  __int64 v167; // rax
  int v168; // edx
  _QWORD *v169; // r12
  double v170; // xmm4_8
  double v171; // xmm5_8
  _BYTE *v172; // rsi
  double v173; // xmm4_8
  double v174; // xmm5_8
  __int64 v175; // rax
  __int64 v176; // rdx
  int v177; // eax
  int v178; // r9d
  unsigned __int64 v179; // rcx
  _BYTE *v180; // r13
  size_t v181; // rdx
  unsigned __int64 v182; // rax
  bool v183; // cf
  unsigned __int64 v184; // rax
  __int64 v185; // r14
  __int64 v186; // rax
  char *v187; // rbx
  __int64 v188; // r14
  unsigned __int64 v189; // rdx
  unsigned __int64 v190; // rax
  unsigned __int64 v191; // rax
  __int64 v192; // rdx
  __int64 v193; // rax
  char *v194; // rbx
  size_t v195; // r12
  size_t v196; // r14
  char *v197; // r12
  _QWORD *v198; // r12
  __int64 v199; // rax
  _QWORD *v200; // rbx
  __int64 v201; // rdx
  __int64 v202; // rax
  int v203; // r10d
  _QWORD *v204; // rax
  int v205; // edi
  _QWORD *v206; // rsi
  __int64 v207; // rcx
  __int64 v208; // r8
  int v209; // r11d
  _QWORD *v210; // r9
  int v211; // esi
  _QWORD *v212; // rcx
  unsigned int v213; // edx
  __int64 v214; // r10
  void **v215; // rax
  void **v216; // r9
  void *v217; // r14
  void **v218; // rbx
  __int64 v219; // rax
  void *v220; // rdx
  void **v221; // r13
  __int64 v222; // r14
  __int64 v223; // r15
  unsigned int v224; // esi
  __int64 v225; // r8
  unsigned int v226; // edi
  _QWORD *v227; // r12
  void *v228; // rax
  unsigned int v229; // esi
  __int64 v230; // r12
  __int64 v231; // r8
  __int64 v232; // rcx
  __int64 v233; // rdx
  __int64 v234; // r9
  __int64 v235; // rax
  __int64 v236; // r10
  unsigned int v237; // esi
  _QWORD *v238; // rcx
  void *v239; // r9
  int v240; // eax
  int v241; // r9d
  __int64 v242; // r10
  unsigned int v243; // ecx
  int v244; // eax
  __int64 v245; // r8
  int v246; // edi
  _QWORD *v247; // rsi
  int v248; // edi
  int v249; // edx
  _QWORD *v250; // r10
  int v251; // eax
  _QWORD *v252; // rax
  int v253; // eax
  __int64 v254; // r10
  int v255; // edi
  __int64 v256; // rsi
  int v257; // eax
  int v258; // r9d
  __int64 v259; // r10
  int v260; // edi
  unsigned int v261; // ecx
  __int64 v262; // r8
  int v263; // eax
  __int64 v264; // r10
  int v265; // edi
  int v266; // ecx
  int v267; // r10d
  _QWORD *v268; // rbx
  _QWORD *v269; // r12
  __int64 v270; // rsi
  __int64 v271; // rax
  int v272; // esi
  unsigned int v273; // edx
  __int64 v274; // r10
  _QWORD *v275; // rax
  _QWORD *v276; // rdx
  int v277; // edi
  __int64 v278; // rcx
  __int64 v279; // r8
  size_t v280; // rax
  void *v281; // rdx
  size_t v282; // rdx
  __int64 v284; // [rsp+20h] [rbp-190h]
  unsigned int v287; // [rsp+38h] [rbp-178h]
  void *v288; // [rsp+38h] [rbp-178h]
  void *v289; // [rsp+38h] [rbp-178h]
  void *v290; // [rsp+38h] [rbp-178h]
  char *v291; // [rsp+38h] [rbp-178h]
  _BYTE *v292; // [rsp+40h] [rbp-170h]
  char *v293; // [rsp+40h] [rbp-170h]
  void *v294; // [rsp+40h] [rbp-170h]
  void *v295; // [rsp+40h] [rbp-170h]
  __int64 v296; // [rsp+48h] [rbp-168h]
  __int64 v297; // [rsp+48h] [rbp-168h]
  __int64 v298; // [rsp+48h] [rbp-168h]
  __int64 v299; // [rsp+48h] [rbp-168h]
  char *v300; // [rsp+48h] [rbp-168h]
  __int64 v301; // [rsp+50h] [rbp-160h]
  unsigned __int64 *v302; // [rsp+50h] [rbp-160h]
  __int64 v303; // [rsp+50h] [rbp-160h]
  __int64 v304; // [rsp+50h] [rbp-160h]
  void *v305; // [rsp+50h] [rbp-160h]
  void *v306; // [rsp+50h] [rbp-160h]
  void *v307; // [rsp+50h] [rbp-160h]
  int v308; // [rsp+50h] [rbp-160h]
  _QWORD *v309; // [rsp+50h] [rbp-160h]
  _QWORD *v310; // [rsp+50h] [rbp-160h]
  char *v312; // [rsp+60h] [rbp-150h]
  _BYTE *v313; // [rsp+60h] [rbp-150h]
  _QWORD *v314; // [rsp+60h] [rbp-150h]
  char *v315; // [rsp+60h] [rbp-150h]
  _BYTE *v316; // [rsp+60h] [rbp-150h]
  char *v317; // [rsp+60h] [rbp-150h]
  char *v318; // [rsp+60h] [rbp-150h]
  unsigned int v319; // [rsp+60h] [rbp-150h]
  _QWORD *v320; // [rsp+60h] [rbp-150h]
  _QWORD *v321; // [rsp+60h] [rbp-150h]
  _QWORD *v322; // [rsp+60h] [rbp-150h]
  _QWORD *v323; // [rsp+60h] [rbp-150h]
  int v324; // [rsp+60h] [rbp-150h]
  _DWORD *v325; // [rsp+60h] [rbp-150h]
  unsigned int v326; // [rsp+60h] [rbp-150h]
  _QWORD *v327; // [rsp+60h] [rbp-150h]
  char *v328; // [rsp+60h] [rbp-150h]
  __int64 v330; // [rsp+70h] [rbp-140h] BYREF
  _QWORD *v331; // [rsp+78h] [rbp-138h] BYREF
  char *v332; // [rsp+80h] [rbp-130h] BYREF
  _QWORD v333[2]; // [rsp+88h] [rbp-128h] BYREF
  __int64 v334; // [rsp+98h] [rbp-118h]
  __int64 v335; // [rsp+A0h] [rbp-110h]
  __int64 v336; // [rsp+B0h] [rbp-100h] BYREF
  __int64 v337; // [rsp+B8h] [rbp-F8h] BYREF
  __int64 v338; // [rsp+C0h] [rbp-F0h]
  __int64 v339; // [rsp+C8h] [rbp-E8h]
  __int64 **i; // [rsp+D0h] [rbp-E0h]
  void *src; // [rsp+E0h] [rbp-D0h] BYREF
  __int64 v342; // [rsp+E8h] [rbp-C8h]
  _BYTE v343[64]; // [rsp+F0h] [rbp-C0h] BYREF
  __int64 *v344; // [rsp+130h] [rbp-80h] BYREF
  __int64 v345; // [rsp+138h] [rbp-78h]
  __int64 v346; // [rsp+140h] [rbp-70h] BYREF
  unsigned int v347; // [rsp+148h] [rbp-68h]
  _QWORD *v348; // [rsp+158h] [rbp-58h]
  unsigned int v349; // [rsp+168h] [rbp-48h]
  char v350; // [rsp+170h] [rbp-40h]
  char v351; // [rsp+179h] [rbp-37h]

  v301 = *(_QWORD *)(*(_QWORD *)(a1 + 312) + 56LL);
  v14 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4F9A488, 1u);
  if ( v14 )
  {
    v15 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v14 + 104LL))(v14, &unk_4F9A488);
    if ( v15 )
      sub_1465DB0(*(_QWORD *)(v15 + 160), a4);
  }
  v16 = *(_QWORD *)(a1 + 376);
  if ( *(_QWORD *)(a1 + 384) != v16 )
    *(_QWORD *)(a1 + 384) = v16;
  v284 = a1 + 400;
  v17 = *(_QWORD *)(a1 + 400);
  if ( *(_QWORD *)(a1 + 408) != v17 )
    *(_QWORD *)(a1 + 408) = v17;
  v18 = sub_1AA91E0(*(_QWORD *)(a1 + 320), *(_QWORD *)(a1 + 312), *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 160));
  v19 = *(_BYTE **)(a1 + 384);
  v330 = v18;
  if ( v19 == *(_BYTE **)(a1 + 392) )
  {
    sub_1292090(a1 + 376, v19, &v330);
    v20 = *(_BYTE **)(a1 + 384);
  }
  else
  {
    if ( v19 )
    {
      *(_QWORD *)v19 = v18;
      v19 = *(_BYTE **)(a1 + 384);
    }
    v20 = v19 + 8;
    *(_QWORD *)(a1 + 384) = v19 + 8;
  }
  v21 = (_BYTE *)a4[4];
  v22 = (_BYTE *)a4[5];
  if ( v22 != v21 )
  {
    v23 = v22 - v21;
    v24 = v23;
    v296 = *(_QWORD *)(a1 + 392);
    if ( v296 - (__int64)v20 >= (unsigned __int64)v23 )
    {
      memmove(v20, v21, v23);
      *(_QWORD *)(a1 + 384) += v24;
      goto LABEL_15;
    }
    v179 = v23 >> 3;
    v180 = *(_BYTE **)(a1 + 376);
    v181 = v20 - v180;
    v182 = (v20 - v180) >> 3;
    if ( v23 >> 3 > 0xFFFFFFFFFFFFFFFLL - v182 )
      goto LABEL_502;
    if ( v179 < v182 )
      v179 = (v20 - v180) >> 3;
    v183 = __CFADD__(v179, v182);
    v184 = v179 + v182;
    if ( v183 )
    {
      v185 = 0x7FFFFFFFFFFFFFF8LL;
    }
    else
    {
      if ( !v184 )
      {
        v188 = 0;
        v187 = 0;
LABEL_442:
        v328 = &v187[v181 + v24];
        if ( v20 == v180 )
        {
          memcpy(&v187[v181], v21, v24);
          v281 = 0;
          v280 = *(_QWORD *)(a1 + 384) - (_QWORD)v20;
          if ( *(_BYTE **)(a1 + 384) == v20 )
          {
LABEL_445:
            v328 = &v328[(_QWORD)v281];
            if ( !v180 )
            {
LABEL_446:
              *(_QWORD *)(a1 + 376) = v187;
              *(_QWORD *)(a1 + 384) = v328;
              *(_QWORD *)(a1 + 392) = v188;
              goto LABEL_15;
            }
LABEL_456:
            j_j___libc_free_0(v180, v296 - (_QWORD)v180);
            goto LABEL_446;
          }
        }
        else
        {
          v291 = &v187[v181];
          v294 = v21;
          memmove(v187, v180, v181);
          memcpy(v291, v294, v24);
          v280 = *(_QWORD *)(a1 + 384) - (_QWORD)v20;
          if ( *(_BYTE **)(a1 + 384) == v20 )
            goto LABEL_456;
        }
        v295 = (void *)v280;
        memcpy(v328, v20, v280);
        v281 = v295;
        goto LABEL_445;
      }
      if ( v184 > 0xFFFFFFFFFFFFFFFLL )
        v184 = 0xFFFFFFFFFFFFFFFLL;
      v185 = 8 * v184;
    }
    v316 = (_BYTE *)a4[4];
    v186 = sub_22077B0(v185);
    v21 = v316;
    v187 = (char *)v186;
    v188 = v186 + v185;
    v180 = *(_BYTE **)(a1 + 376);
    v296 = *(_QWORD *)(a1 + 392);
    v181 = v20 - v180;
    goto LABEL_442;
  }
LABEL_15:
  src = v343;
  v342 = 0x800000000LL;
  sub_13FA0E0((__int64)a4, (__int64)&src);
  if ( (_DWORD)v342 )
  {
    v312 = 0;
    v297 = 8LL * (unsigned int)v342;
    do
    {
      v25 = *(_QWORD *)&v312[(_QWORD)src];
      v26 = *(_QWORD *)(v25 + 8);
      if ( v26 )
      {
        while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v26) + 16) - 25) > 9u )
        {
          v26 = *(_QWORD *)(v26 + 8);
          if ( !v26 )
            goto LABEL_75;
        }
        v344 = &v346;
        v29 = v26;
        v30 = 0;
        v345 = 0x400000000LL;
        while ( 1 )
        {
          v29 = *(_QWORD *)(v29 + 8);
          if ( !v29 )
            break;
          while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v29) + 16) - 25) <= 9u )
          {
            v29 = *(_QWORD *)(v29 + 8);
            ++v30;
            if ( !v29 )
              goto LABEL_23;
          }
        }
LABEL_23:
        v31 = v30 + 1;
        v32 = &v346;
        if ( v31 > 4 )
        {
          sub_16CD150((__int64)&v344, &v346, v31, 8, v27, v28);
          v32 = &v344[(unsigned int)v345];
        }
        v33 = sub_1648700(v26);
LABEL_27:
        if ( v32 )
          *v32 = v33[5];
        v26 = *(_QWORD *)(v26 + 8);
        if ( v26 )
        {
          do
          {
            v33 = sub_1648700(v26);
            if ( (unsigned __int8)(*((_BYTE *)v33 + 16) - 25) <= 9u )
            {
              ++v32;
              goto LABEL_27;
            }
            v26 = *(_QWORD *)(v26 + 8);
          }
          while ( v26 );
          v34 = v344;
          v35 = v345 + v31;
          v36 = (unsigned int)(v345 + v31);
        }
        else
        {
          v34 = v344;
          v35 = v345 + v31;
          v36 = (unsigned int)(v345 + v31);
        }
      }
      else
      {
LABEL_75:
        v344 = &v346;
        v34 = &v346;
        v36 = 0;
        v35 = 0;
        HIDWORD(v345) = 4;
      }
      LODWORD(v345) = v35;
      sub_1AAB350(v25, v34, v36, ".us-lcssa", *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 160), 1);
      if ( v344 != &v346 )
        _libc_free((unsigned __int64)v344);
      v312 += 8;
    }
    while ( (char *)v297 != v312 );
  }
  LODWORD(v342) = 0;
  sub_13FA0E0((__int64)a4, (__int64)&src);
  v37 = src;
  v38 = *(_BYTE **)(a1 + 384);
  v39 = *(_BYTE **)(a1 + 376);
  v40 = v38 - v39;
  v41 = 8LL * (unsigned int)v342;
  if ( v41 )
  {
    v292 = *(_BYTE **)(a1 + 392);
    if ( v41 <= (unsigned __int64)(v292 - v38) )
    {
      memmove(v38, src, 8LL * (unsigned int)v342);
      v42 = *(_QWORD *)(a1 + 384) + v41;
      *(_QWORD *)(a1 + 384) = v42;
      v40 = v42 - *(_QWORD *)(a1 + 376);
      goto LABEL_39;
    }
    v189 = v41 >> 3;
    v190 = v40 >> 3;
    if ( v41 >> 3 <= (unsigned __int64)(0xFFFFFFFFFFFFFFFLL - (v40 >> 3)) )
    {
      if ( v189 < v190 )
        v189 = v40 >> 3;
      v183 = __CFADD__(v189, v190);
      v191 = v189 + v190;
      if ( v183 )
      {
        v192 = 0x7FFFFFFFFFFFFFF8LL;
      }
      else
      {
        if ( !v191 )
        {
          v300 = 0;
          v194 = 0;
LABEL_265:
          v318 = &v194[v40 + v41];
          if ( v38 == v39 )
          {
            v282 = v41;
            v196 = 0;
            memcpy(&v194[v40], v37, v282);
            v195 = *(_QWORD *)(a1 + 384) - (_QWORD)v38;
            if ( *(_BYTE **)(a1 + 384) == v38 )
            {
LABEL_268:
              v197 = &v318[v196];
              if ( !v39 )
              {
LABEL_269:
                *(_QWORD *)(a1 + 384) = v197;
                v40 = v197 - v194;
                *(_QWORD *)(a1 + 376) = v194;
                *(_QWORD *)(a1 + 392) = v300;
                goto LABEL_39;
              }
LABEL_453:
              j_j___libc_free_0(v39, v292 - v39);
              goto LABEL_269;
            }
          }
          else
          {
            v289 = v37;
            memmove(v194, v39, v40);
            memcpy(&v194[v40], v289, v41);
            v195 = *(_QWORD *)(a1 + 384) - (_QWORD)v38;
            if ( v38 == *(_BYTE **)(a1 + 384) )
            {
              v197 = &v318[v195];
              goto LABEL_453;
            }
          }
          v196 = v195;
          memcpy(v318, v38, v195);
          goto LABEL_268;
        }
        if ( v191 > 0xFFFFFFFFFFFFFFFLL )
          v191 = 0xFFFFFFFFFFFFFFFLL;
        v192 = 8 * v191;
      }
      v288 = src;
      v317 = (char *)v192;
      v193 = sub_22077B0(v192);
      v37 = v288;
      v194 = (char *)v193;
      v39 = *(_BYTE **)(a1 + 376);
      v300 = &v317[v193];
      v292 = *(_BYTE **)(a1 + 392);
      v40 = v38 - v39;
      goto LABEL_265;
    }
LABEL_502:
    sub_4262D8((__int64)"vector::_M_range_insert");
  }
LABEL_39:
  if ( (unsigned __int64)v40 > 0x7FFFFFFFFFFFFFF8LL )
    sub_4262D8((__int64)"vector::reserve");
  v43 = *(_BYTE **)(a1 + 400);
  if ( (unsigned __int64)v40 > *(_QWORD *)(a1 + 416) - (_QWORD)v43 )
  {
    v313 = *(_BYTE **)(a1 + 408);
    v44 = v313 - v43;
    if ( v40 )
    {
      v45 = sub_22077B0(v40);
      v43 = *(_BYTE **)(a1 + 400);
      v46 = (char *)v45;
      v47 = *(_QWORD *)(a1 + 408) - (_QWORD)v43;
    }
    else
    {
      v47 = v313 - v43;
      v46 = 0;
    }
    if ( v47 > 0 )
    {
      memmove(v46, v43, v47);
    }
    else if ( !v43 )
    {
LABEL_45:
      *(_QWORD *)(a1 + 400) = v46;
      *(_QWORD *)(a1 + 408) = &v46[v44];
      *(_QWORD *)(a1 + 416) = &v46[v40];
      goto LABEL_46;
    }
    j_j___libc_free_0(v43, *(_QWORD *)(a1 + 416) - (_QWORD)v43);
    goto LABEL_45;
  }
LABEL_46:
  v344 = 0;
  v347 = 128;
  v48 = (_QWORD *)sub_22077B0(0x2000);
  v346 = 0;
  v345 = (__int64)v48;
  v337 = 2;
  v49 = &v48[8 * (unsigned __int64)v347];
  v336 = (__int64)&unk_49E6B50;
  v338 = 0;
  v339 = -8;
  for ( i = 0; v49 != v48; v48 += 8 )
  {
    if ( v48 )
    {
      v50 = v337;
      v48[2] = 0;
      v48[3] = -8;
      *v48 = &unk_49E6B50;
      v48[1] = v50 & 6;
      v48[4] = i;
    }
  }
  v350 = 0;
  v351 = 1;
  v51 = *(_QWORD *)(a1 + 376);
  v52 = (*(_QWORD *)(a1 + 384) - v51) >> 3;
  if ( (_DWORD)v52 )
  {
    v53 = 0;
    v298 = 8LL * (unsigned int)(v52 - 1);
    while ( 1 )
    {
      LOWORD(v338) = 259;
      v336 = (__int64)".us";
      v54 = (char *)sub_1AB5760(*(_QWORD *)(v51 + v53), &v344, &v336, v301, 0, 0);
      v55 = *(_BYTE **)(a1 + 408);
      v332 = v54;
      v56 = v54;
      if ( v55 == *(_BYTE **)(a1 + 416) )
      {
        sub_1292090(v284, v55, &v332);
        v56 = v332;
      }
      else
      {
        if ( v55 )
        {
          *(_QWORD *)v55 = v54;
          v55 = *(_BYTE **)(a1 + 408);
          v56 = v332;
        }
        *(_QWORD *)(a1 + 408) = v55 + 8;
      }
      v57 = *(_QWORD *)(*(_QWORD *)(a1 + 376) + v53);
      v337 = 2;
      v338 = 0;
      v339 = v57;
      if ( v57 != 0 && v57 != -8 && v57 != -16 )
        sub_164C220((__int64)&v337);
      i = &v344;
      v336 = (__int64)&unk_49E6B50;
      if ( !v347 )
        break;
      v58 = v339;
      v64 = (v347 - 1) & (((unsigned int)v339 >> 9) ^ ((unsigned int)v339 >> 4));
      v59 = (_QWORD *)(v345 + ((unsigned __int64)v64 << 6));
      v65 = v59[3];
      if ( v339 == v65 )
        goto LABEL_78;
      v203 = 1;
      v204 = 0;
      while ( v65 != -8 )
      {
        if ( v65 == -16 && !v204 )
          v204 = v59;
        v64 = (v347 - 1) & (v203 + v64);
        v59 = (_QWORD *)(v345 + ((unsigned __int64)v64 << 6));
        v65 = v59[3];
        if ( v339 == v65 )
          goto LABEL_78;
        ++v203;
      }
      if ( v204 )
        v59 = v204;
      v344 = (__int64 *)((char *)v344 + 1);
      v60 = v346 + 1;
      if ( 4 * ((int)v346 + 1) >= 3 * v347 )
        goto LABEL_61;
      if ( v347 - HIDWORD(v346) - v60 <= v347 >> 3 )
      {
        sub_12E48B0((__int64)&v344, v347);
        if ( v347 )
        {
          v58 = v339;
          v205 = 1;
          v206 = 0;
          LODWORD(v207) = (v347 - 1) & (((unsigned int)v339 >> 9) ^ ((unsigned int)v339 >> 4));
          v59 = (_QWORD *)(v345 + ((unsigned __int64)(unsigned int)v207 << 6));
          v208 = v59[3];
          if ( v339 != v208 )
          {
            while ( v208 != -8 )
            {
              if ( v208 == -16 && !v206 )
                v206 = v59;
              v207 = (v347 - 1) & ((_DWORD)v207 + v205);
              v59 = (_QWORD *)(v345 + (v207 << 6));
              v208 = v59[3];
              if ( v339 == v208 )
                goto LABEL_63;
              ++v205;
            }
LABEL_305:
            if ( v206 )
              v59 = v206;
          }
LABEL_63:
          v60 = v346 + 1;
          goto LABEL_64;
        }
LABEL_62:
        v58 = v339;
        v59 = 0;
        goto LABEL_63;
      }
LABEL_64:
      LODWORD(v346) = v60;
      v61 = v59[3];
      v62 = v59 + 1;
      if ( v61 == -8 )
      {
        if ( v58 != -8 )
          goto LABEL_69;
      }
      else
      {
        --HIDWORD(v346);
        if ( v61 != v58 )
        {
          if ( v61 != -16 && v61 )
          {
            sub_1649B30(v62);
            v58 = v339;
            v62 = v59 + 1;
          }
LABEL_69:
          v59[3] = v58;
          if ( v58 != -8 && v58 != 0 && v58 != -16 )
            sub_1649AC0(v62, v337 & 0xFFFFFFFFFFFFFFF8LL);
          v58 = v339;
        }
      }
      v63 = i;
      v59[5] = 6;
      v59[6] = 0;
      v59[4] = v63;
      v59[7] = 0;
LABEL_78:
      v336 = (__int64)&unk_49EE2B0;
      if ( v58 != -8 && v58 != 0 && v58 != -16 )
        sub_1649B30(&v337);
      v66 = (char *)v59[7];
      v67 = v59 + 5;
      if ( v66 != v56 )
      {
        if ( v66 != 0 && v66 + 8 != 0 && v66 != (char *)-16LL )
        {
          sub_1649B30(v67);
          v67 = v59 + 5;
        }
        v59[7] = v56;
        if ( v56 + 8 != 0 && v56 != 0 && v56 != (char *)-16LL )
          sub_164C220((__int64)v67);
      }
      sub_1404520(*(_QWORD *)(a1 + 168), *(_QWORD *)(*(_QWORD *)(a1 + 376) + v53), (__int64)v332, (__int64)a4);
      if ( v298 == v53 )
        goto LABEL_91;
      v51 = *(_QWORD *)(a1 + 376);
      v53 += 8;
    }
    v344 = (__int64 *)((char *)v344 + 1);
LABEL_61:
    sub_12E48B0((__int64)&v344, 2 * v347);
    if ( v347 )
    {
      v58 = v339;
      v277 = 1;
      v206 = 0;
      LODWORD(v278) = (v347 - 1) & (((unsigned int)v339 >> 9) ^ ((unsigned int)v339 >> 4));
      v59 = (_QWORD *)(v345 + ((unsigned __int64)(unsigned int)v278 << 6));
      v279 = v59[3];
      if ( v339 != v279 )
      {
        while ( v279 != -8 )
        {
          if ( v279 == -16 && !v206 )
            v206 = v59;
          v278 = (v347 - 1) & ((_DWORD)v278 + v277);
          v59 = (_QWORD *)(v345 + (v278 << 6));
          v279 = v59[3];
          if ( v339 == v279 )
            goto LABEL_63;
          ++v277;
        }
        goto LABEL_305;
      }
      goto LABEL_63;
    }
    goto LABEL_62;
  }
LABEL_91:
  v68 = v301 + 72;
  v69 = **(_QWORD **)(a1 + 400);
  if ( v301 + 72 != v69 + 24 )
  {
    v70 = v330;
    v71 = v330 + 24;
    if ( v68 != v330 + 24 )
    {
      v72 = *(_QWORD *)(v301 + 72) & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)((*(_QWORD *)(v69 + 24) & 0xFFFFFFFFFFFFFFF8LL) + 8) = v68;
      *(_QWORD *)(v301 + 72) = *(_QWORD *)(v301 + 72) & 7LL | *(_QWORD *)(v69 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      v73 = *(_QWORD *)(v70 + 24);
      *(_QWORD *)(v72 + 8) = v71;
      v73 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v69 + 24) = v73 | *(_QWORD *)(v69 + 24) & 7LL;
      *(_QWORD *)(v73 + 8) = v69 + 24;
      *(_QWORD *)(v70 + 24) = v72 | *(_QWORD *)(v70 + 24) & 7LL;
    }
  }
  v74 = (_QWORD *)(a1 + 224);
  v75 = sub_19C3400((__int64)a4, *a4, (__int64)&v344, *(_QWORD *)(a1 + 160), *(__int64 **)(a1 + 168));
  v76 = (_QWORD *)(a1 + 224);
  v331 = v75;
  v77 = v75;
  v78 = *(_QWORD **)(a1 + 232);
  if ( !v78 )
    goto LABEL_124;
  do
  {
    while ( 1 )
    {
      v79 = v78[2];
      v80 = v78[3];
      if ( v78[4] >= (unsigned __int64)v77 )
        break;
      v78 = (_QWORD *)v78[3];
      if ( !v80 )
        goto LABEL_99;
    }
    v76 = v78;
    v78 = (_QWORD *)v78[2];
  }
  while ( v79 );
LABEL_99:
  if ( v74 == v76 || v76[4] > (unsigned __int64)v77 )
  {
LABEL_124:
    v314 = v76;
    v91 = sub_22077B0(88);
    v92 = v314;
    a6 = 0;
    *(_QWORD *)(v91 + 32) = v77;
    v93 = v91;
    v94 = (unsigned __int64 *)(v91 + 32);
    *(_OWORD *)(v91 + 40) = 0;
    *(_OWORD *)(v91 + 56) = 0;
    *(_OWORD *)(v91 + 72) = 0;
    if ( v314 == v74 )
    {
      if ( *(_QWORD *)(a1 + 256) )
      {
        v92 = *(_QWORD **)(a1 + 248);
        if ( v92[4] < (unsigned __int64)v77 )
          goto LABEL_449;
      }
    }
    else
    {
      v302 = (unsigned __int64 *)(v91 + 32);
      if ( (unsigned __int64)v77 < v314[4] )
      {
        if ( v314 != *(_QWORD **)(a1 + 240) )
        {
          v95 = sub_220EF80(v314);
          v94 = v302;
          if ( *(_QWORD *)(v95 + 32) < (unsigned __int64)v77 )
          {
            v92 = v314;
            if ( !*(_QWORD *)(v95 + 24) )
              v92 = (_QWORD *)v95;
            v96 = *(_QWORD *)(v95 + 24) != 0;
            goto LABEL_131;
          }
          goto LABEL_430;
        }
LABEL_420:
        v96 = 1;
        goto LABEL_131;
      }
      if ( (unsigned __int64)v77 <= v314[4] )
        goto LABEL_440;
      if ( v314 == *(_QWORD **)(a1 + 248) )
      {
LABEL_449:
        v96 = 0;
LABEL_131:
        v97 = v92 == v74 || v96 || (unsigned __int64)v77 < v92[4];
        sub_220F040(v97, v93, v92, v74);
        v76 = (_QWORD *)v93;
        ++*(_QWORD *)(a1 + 256);
        goto LABEL_101;
      }
      v271 = sub_220EEE0(v314);
      v94 = v302;
      if ( *(_QWORD *)(v271 + 32) > (unsigned __int64)v77 )
      {
        v92 = v314;
        if ( v314[3] )
        {
          v92 = (_QWORD *)v271;
          goto LABEL_420;
        }
        goto LABEL_449;
      }
    }
LABEL_430:
    v275 = sub_19C0280(a1 + 216, v94);
    v92 = v276;
    if ( v276 )
    {
      v96 = v275 != 0;
      goto LABEL_131;
    }
    v92 = v275;
LABEL_440:
    v327 = v92;
    j___libc_free_0(0);
    j_j___libc_free_0(v93, 88);
    v76 = v327;
  }
LABEL_101:
  v81 = *(_QWORD *)(a1 + 272);
  --*(_DWORD *)v81;
  ++*(_DWORD *)(v81 + 4);
  *((_DWORD *)v76 + 11) = 0;
  v82 = *(_DWORD *)v81 >> 1;
  v83 = *(_DWORD *)v81 - v82;
  *((_DWORD *)v76 + 10) = v82;
  *(_DWORD *)v81 = v83;
  *((_DWORD *)v76 + 12) = *(_DWORD *)(v81 + 8);
  v299 = v81 + 16;
  if ( *(_DWORD *)(v81 + 32) )
  {
    v215 = *(void ***)(v81 + 24);
    v216 = &v215[14 * *(unsigned int *)(v81 + 40)];
    if ( v215 != v216 )
    {
      while ( 1 )
      {
        v217 = *v215;
        v218 = v215;
        if ( *v215 != (void *)-16LL && v217 != (void *)-8LL )
          break;
        v215 += 14;
        if ( v216 == v215 )
          goto LABEL_102;
      }
      if ( v216 != v215 )
      {
        v219 = v347;
        v220 = v217;
        v221 = v216;
        v222 = v81;
        if ( v347 )
          goto LABEL_340;
LABEL_327:
        v223 = 0;
LABEL_328:
        v224 = *(_DWORD *)(v222 + 40);
        if ( v224 )
        {
LABEL_329:
          v225 = *(_QWORD *)(v222 + 24);
          v226 = (v224 - 1) & (((unsigned int)v220 >> 9) ^ ((unsigned int)v220 >> 4));
          v227 = (_QWORD *)(v225 + 112LL * v226);
          v228 = (void *)*v227;
          if ( (void *)*v227 == v220 )
          {
LABEL_330:
            v229 = *((_DWORD *)v76 + 20);
            v230 = (__int64)(v227 + 1);
            v231 = (__int64)(v76 + 7);
            if ( v229 )
              goto LABEL_331;
LABEL_378:
            ++v76[7];
LABEL_379:
            v325 = v76;
            sub_19C24C0(v231, 2 * v229);
            v76 = v325;
            v253 = v325[20];
            if ( !v253 )
              goto LABEL_508;
            LODWORD(v234) = v253 - 1;
            v254 = *((_QWORD *)v325 + 8);
            LODWORD(v232) = (v253 - 1) & (((unsigned int)v223 >> 9) ^ ((unsigned int)v223 >> 4));
            v249 = v325[18] + 1;
            v235 = v254 + 112LL * (unsigned int)v232;
            v231 = *(_QWORD *)v235;
            if ( v223 != *(_QWORD *)v235 )
            {
              v255 = 1;
              v256 = 0;
              while ( v231 != -8 )
              {
                if ( !v256 && v231 == -16 )
                  v256 = v235;
                LODWORD(v232) = v234 & (v255 + v232);
                v235 = v254 + 112LL * (unsigned int)v232;
                v231 = *(_QWORD *)v235;
                if ( v223 == *(_QWORD *)v235 )
                  goto LABEL_366;
                ++v255;
              }
LABEL_383:
              if ( v256 )
                v235 = v256;
              goto LABEL_366;
            }
            goto LABEL_366;
          }
          v324 = 1;
          v250 = 0;
          while ( v228 != (void *)-8LL )
          {
            if ( v228 == (void *)-16LL && !v250 )
              v250 = v227;
            v226 = (v224 - 1) & (v324 + v226);
            v227 = (_QWORD *)(v225 + 112LL * v226);
            v228 = (void *)*v227;
            if ( (void *)*v227 == v220 )
              goto LABEL_330;
            ++v324;
          }
          v251 = *(_DWORD *)(v222 + 32);
          if ( v250 )
            v227 = v250;
          ++*(_QWORD *)(v222 + 16);
          v244 = v251 + 1;
          if ( 4 * v244 >= 3 * v224 )
            goto LABEL_350;
          if ( v224 - *(_DWORD *)(v222 + 36) - v244 > v224 >> 3 )
            goto LABEL_375;
          v290 = v220;
          v309 = v76;
          v326 = ((unsigned int)v220 >> 9) ^ ((unsigned int)v220 >> 4);
          sub_19C24C0(v299, v224);
          v257 = *(_DWORD *)(v222 + 40);
          if ( !v257 )
          {
LABEL_507:
            ++*(_DWORD *)(v222 + 32);
            BUG();
          }
          v258 = v257 - 1;
          v259 = *(_QWORD *)(v222 + 24);
          v247 = 0;
          v220 = v290;
          v76 = v309;
          v260 = 1;
          v261 = (v257 - 1) & v326;
          v227 = (_QWORD *)(v259 + 112LL * v261);
          v244 = *(_DWORD *)(v222 + 32) + 1;
          v262 = *v227;
          if ( (void *)*v227 == v290 )
            goto LABEL_375;
          while ( v262 != -8 )
          {
            if ( !v247 && v262 == -16 )
              v247 = v227;
            v261 = v258 & (v260 + v261);
            v227 = (_QWORD *)(v259 + 112LL * v261);
            v262 = *v227;
            if ( (void *)*v227 == v290 )
              goto LABEL_375;
            ++v260;
          }
        }
        else
        {
          while ( 1 )
          {
            ++*(_QWORD *)(v222 + 16);
LABEL_350:
            v307 = v220;
            v323 = v76;
            sub_19C24C0(v299, 2 * v224);
            v240 = *(_DWORD *)(v222 + 40);
            if ( !v240 )
              goto LABEL_507;
            v220 = v307;
            v241 = v240 - 1;
            v242 = *(_QWORD *)(v222 + 24);
            v76 = v323;
            v243 = (v240 - 1) & (((unsigned int)v307 >> 9) ^ ((unsigned int)v307 >> 4));
            v227 = (_QWORD *)(v242 + 112LL * v243);
            v244 = *(_DWORD *)(v222 + 32) + 1;
            v245 = *v227;
            if ( (void *)*v227 != v307 )
              break;
LABEL_375:
            *(_DWORD *)(v222 + 32) = v244;
            if ( *v227 != -8 )
              --*(_DWORD *)(v222 + 36);
            v252 = v227 + 6;
            *v227 = v220;
            v231 = (__int64)(v76 + 7);
            v230 = (__int64)(v227 + 1);
            *(_QWORD *)v230 = 0;
            *(_QWORD *)(v230 + 8) = v252;
            *(_QWORD *)(v230 + 16) = v252;
            *(_QWORD *)(v230 + 24) = 8;
            *(_DWORD *)(v230 + 32) = 0;
            v229 = *((_DWORD *)v76 + 20);
            if ( !v229 )
              goto LABEL_378;
LABEL_331:
            v232 = v76[8];
            v319 = ((unsigned int)v223 >> 9) ^ ((unsigned int)v223 >> 4);
            v233 = (v229 - 1) & v319;
            LODWORD(v234) = v233;
            v235 = v232 + 112 * v233;
            v236 = *(_QWORD *)v235;
            if ( v223 != *(_QWORD *)v235 )
            {
              v308 = 1;
              v234 = 0;
              while ( v236 != -8 )
              {
                if ( !v234 && v236 == -16 )
                  v234 = v235;
                v233 = (v229 - 1) & (v308 + (_DWORD)v233);
                LODWORD(v232) = v308 + 1;
                v235 = v76[8] + 112LL * (unsigned int)v233;
                v236 = *(_QWORD *)v235;
                if ( v223 == *(_QWORD *)v235 )
                  goto LABEL_332;
                ++v308;
              }
              v248 = *((_DWORD *)v76 + 18);
              if ( v234 )
                v235 = v234;
              ++v76[7];
              v249 = v248 + 1;
              if ( 4 * (v248 + 1) >= 3 * v229 )
                goto LABEL_379;
              LODWORD(v232) = v229 - *((_DWORD *)v76 + 19) - v249;
              if ( (unsigned int)v232 <= v229 >> 3 )
              {
                v310 = v76;
                sub_19C24C0(v231, v229);
                v76 = v310;
                v263 = *((_DWORD *)v310 + 20);
                if ( !v263 )
                {
LABEL_508:
                  ++*((_DWORD *)v76 + 18);
                  BUG();
                }
                LODWORD(v234) = v263 - 1;
                v264 = v310[8];
                v256 = 0;
                LODWORD(v232) = (v263 - 1) & v319;
                v249 = *((_DWORD *)v310 + 18) + 1;
                v265 = 1;
                v235 = v264 + 112LL * (unsigned int)v232;
                v231 = *(_QWORD *)v235;
                if ( v223 != *(_QWORD *)v235 )
                {
                  while ( v231 != -8 )
                  {
                    if ( v231 == -16 && !v256 )
                      v256 = v235;
                    LODWORD(v232) = v234 & (v265 + v232);
                    v235 = v264 + 112LL * (unsigned int)v232;
                    v231 = *(_QWORD *)v235;
                    if ( v223 == *(_QWORD *)v235 )
                      goto LABEL_366;
                    ++v265;
                  }
                  goto LABEL_383;
                }
              }
LABEL_366:
              *((_DWORD *)v76 + 18) = v249;
              if ( *(_QWORD *)v235 != -8 )
                --*((_DWORD *)v76 + 19);
              v233 = v235 + 48;
              *(_QWORD *)v235 = v223;
              *(_QWORD *)(v235 + 8) = 0;
              *(_QWORD *)(v235 + 16) = v235 + 48;
              *(_QWORD *)(v235 + 24) = v235 + 48;
              *(_QWORD *)(v235 + 32) = 8;
              *(_DWORD *)(v235 + 40) = 0;
            }
LABEL_332:
            if ( v230 != v235 + 8 )
            {
              v320 = v76;
              sub_16CCD50(v235 + 8, v230, v233, v232, v231, v234);
              v76 = v320;
            }
            for ( v218 += 14; v221 != v218; v218 += 14 )
            {
              if ( *v218 != (void *)-16LL && *v218 != (void *)-8LL )
                break;
            }
            if ( v218 == (void **)(*(_QWORD *)(v222 + 24) + 112LL * *(unsigned int *)(v222 + 40)) )
              goto LABEL_102;
            v219 = v347;
            v220 = *v218;
            if ( !v347 )
              goto LABEL_327;
LABEL_340:
            v237 = (v219 - 1) & (((unsigned int)v220 >> 9) ^ ((unsigned int)v220 >> 4));
            v238 = (_QWORD *)(v345 + ((unsigned __int64)v237 << 6));
            v239 = (void *)v238[3];
            if ( v239 != v220 )
            {
              v266 = 1;
              while ( v239 != (void *)-8LL )
              {
                v267 = v266 + 1;
                v237 = (v219 - 1) & (v237 + v266);
                v238 = (_QWORD *)(v345 + ((unsigned __int64)v237 << 6));
                v239 = (void *)v238[3];
                if ( v239 == v220 )
                  goto LABEL_341;
                v266 = v267;
              }
              goto LABEL_327;
            }
LABEL_341:
            if ( v238 == (_QWORD *)(v345 + (v219 << 6)) )
              goto LABEL_327;
            v336 = 6;
            v337 = 0;
            v338 = v238[7];
            v223 = v338;
            if ( v338 != -8 && v338 != 0 && v338 != -16 )
            {
              v305 = v220;
              v321 = v76;
              sub_1649AC0((unsigned __int64 *)&v336, v238[5] & 0xFFFFFFFFFFFFFFF8LL);
              v223 = v338;
              v220 = v305;
              v76 = v321;
            }
            if ( !v223 )
              goto LABEL_327;
            if ( v223 == -8 || v223 == -16 )
              goto LABEL_328;
            v306 = v220;
            v322 = v76;
            sub_1649B30(&v336);
            v224 = *(_DWORD *)(v222 + 40);
            v220 = v306;
            v76 = v322;
            if ( v224 )
              goto LABEL_329;
          }
          v246 = 1;
          v247 = 0;
          while ( v245 != -8 )
          {
            if ( v245 == -16 && !v247 )
              v247 = v227;
            v243 = v241 & (v246 + v243);
            v227 = (_QWORD *)(v242 + 112LL * v243);
            v245 = *v227;
            if ( (void *)*v227 == v307 )
              goto LABEL_375;
            ++v246;
          }
        }
        if ( v247 )
          v227 = v247;
        goto LABEL_375;
      }
    }
  }
LABEL_102:
  if ( *a4 )
    sub_1400330(*a4, **(_QWORD **)(a1 + 400), *(_QWORD *)(a1 + 160));
  if ( (_DWORD)v342 )
  {
    v293 = (char *)(8LL * (unsigned int)v342);
    v84 = 0;
    while ( 1 )
    {
      v85 = *(_QWORD *)&v84[(_QWORD)src];
      v337 = 2;
      v338 = 0;
      v339 = v85;
      if ( v85 != 0 && v85 != -8 && v85 != -16 )
        sub_164C220((__int64)&v337);
      v336 = (__int64)&unk_49E6B50;
      i = &v344;
      if ( !v347 )
        break;
      v86 = v339;
      v98 = (v347 - 1) & (((unsigned int)v339 >> 9) ^ ((unsigned int)v339 >> 4));
      v87 = (_QWORD *)(v345 + ((unsigned __int64)v98 << 6));
      v99 = v87[3];
      if ( v339 == v99 )
        goto LABEL_136;
      v209 = 1;
      v210 = 0;
      while ( v99 != -8 )
      {
        if ( !v210 && v99 == -16 )
          v210 = v87;
        v98 = (v347 - 1) & (v209 + v98);
        v87 = (_QWORD *)(v345 + ((unsigned __int64)v98 << 6));
        v99 = v87[3];
        if ( v339 == v99 )
          goto LABEL_136;
        ++v209;
      }
      if ( v210 )
        v87 = v210;
      v344 = (__int64 *)((char *)v344 + 1);
      v88 = v346 + 1;
      if ( 4 * ((int)v346 + 1) >= 3 * v347 )
        goto LABEL_111;
      if ( v347 - HIDWORD(v346) - v88 <= v347 >> 3 )
      {
        sub_12E48B0((__int64)&v344, v347);
        if ( v347 )
        {
          v86 = v339;
          v211 = 1;
          v212 = 0;
          v213 = (v347 - 1) & (((unsigned int)v339 >> 9) ^ ((unsigned int)v339 >> 4));
          v87 = (_QWORD *)(v345 + ((unsigned __int64)v213 << 6));
          v214 = v87[3];
          if ( v214 != v339 )
          {
            while ( v214 != -8 )
            {
              if ( !v212 && v214 == -16 )
                v212 = v87;
              v213 = (v347 - 1) & (v211 + v213);
              v87 = (_QWORD *)(v345 + ((unsigned __int64)v213 << 6));
              v214 = v87[3];
              if ( v339 == v214 )
                goto LABEL_113;
              ++v211;
            }
LABEL_317:
            if ( v212 )
              v87 = v212;
          }
LABEL_113:
          v88 = v346 + 1;
          goto LABEL_114;
        }
LABEL_112:
        v86 = v339;
        v87 = 0;
        goto LABEL_113;
      }
LABEL_114:
      LODWORD(v346) = v88;
      v89 = v87[3];
      if ( v89 == -8 )
      {
        if ( v86 != -8 )
          goto LABEL_119;
      }
      else
      {
        --HIDWORD(v346);
        if ( v86 != v89 )
        {
          if ( v89 && v89 != -16 )
          {
            sub_1649B30(v87 + 1);
            v86 = v339;
          }
LABEL_119:
          v87[3] = v86;
          if ( v86 != 0 && v86 != -8 && v86 != -16 )
            sub_1649AC0(v87 + 1, v337 & 0xFFFFFFFFFFFFFFF8LL);
          v86 = v339;
        }
      }
      v90 = i;
      v87[5] = 6;
      v87[6] = 0;
      v87[4] = v90;
      v87[7] = 0;
LABEL_136:
      v336 = (__int64)&unk_49EE2B0;
      if ( v86 != -8 && v86 != 0 && v86 != -16 )
        sub_1649B30(&v337);
      v100 = v87[7];
      v101 = *(_QWORD *)(a1 + 160);
      v102 = *(_DWORD *)(v101 + 24);
      if ( v102 )
      {
        v103 = v102 - 1;
        v104 = *(_QWORD *)(v101 + 8);
        v105 = *(_QWORD *)&v84[(_QWORD)src];
        v106 = v103 & (((unsigned int)v105 >> 9) ^ ((unsigned int)v105 >> 4));
        v107 = (__int64 *)(v104 + 16LL * v106);
        v108 = *v107;
        if ( v105 == *v107 )
        {
LABEL_141:
          v109 = v107[1];
          if ( v109 )
            sub_1400330(v109, v87[7], *(_QWORD *)(a1 + 160));
        }
        else
        {
          v177 = 1;
          while ( v108 != -8 )
          {
            v178 = v177 + 1;
            v106 = v103 & (v177 + v106);
            v107 = (__int64 *)(v104 + 16LL * v106);
            v108 = *v107;
            if ( v105 == *v107 )
              goto LABEL_141;
            v177 = v178;
          }
        }
      }
      v110 = sub_157EBA0(v100);
      v303 = sub_15F4DF0(v110, 0);
      v111 = sub_157F280(v303);
      v113 = v112;
      v114 = v111;
      if ( v111 != v112 )
      {
        while ( 1 )
        {
          v115 = 0x17FFFFFFE8LL;
          v116 = *(unsigned int *)(v114 + 56);
          v117 = *(_BYTE *)(v114 + 23) & 0x40;
          v118 = *(_DWORD *)(v114 + 20) & 0xFFFFFFF;
          v119 = v118;
          if ( v118 )
          {
            v120 = 24LL * (unsigned int)v116 + 8;
            v121 = 0;
            do
            {
              v122 = v114 - 24LL * v118;
              if ( (_BYTE)v117 )
                v122 = *(_QWORD *)(v114 - 8);
              if ( *(_QWORD *)&v84[(_QWORD)src] == *(_QWORD *)(v122 + v120) )
              {
                v115 = 24 * v121;
                goto LABEL_151;
              }
              ++v121;
              v120 += 8;
            }
            while ( v118 != (_DWORD)v121 );
            v115 = 0x17FFFFFFE8LL;
          }
LABEL_151:
          if ( (_BYTE)v117 )
            v123 = *(_QWORD *)(v114 - 8);
          else
            v123 = v114 - 24LL * v118;
          v124 = *(_QWORD *)(v123 + v115);
          v125 = v347;
          if ( v347 )
          {
            v117 = (v347 - 1) & (((unsigned int)v124 >> 9) ^ ((unsigned int)v124 >> 4));
            v123 = v345 + (v117 << 6);
            v126 = *(_QWORD *)(v123 + 24);
            if ( v126 == v124 )
            {
LABEL_155:
              v125 = v345 + ((unsigned __int64)v347 << 6);
              if ( v123 != v125 )
                v124 = *(_QWORD *)(v123 + 56);
            }
            else
            {
              v123 = 1;
              while ( v126 != -8 )
              {
                v117 = (v347 - 1) & ((_DWORD)v123 + (_DWORD)v117);
                v287 = v123 + 1;
                v123 = v345 + ((unsigned __int64)(unsigned int)v117 << 6);
                v126 = *(_QWORD *)(v123 + 24);
                if ( v124 == v126 )
                  goto LABEL_155;
                v123 = v287;
              }
            }
          }
          if ( (_DWORD)v116 == v118 )
          {
            sub_15F55D0(v114, v123, v118, v125, v116, v117);
            v119 = *(_DWORD *)(v114 + 20) & 0xFFFFFFF;
          }
          v127 = (v119 + 1) & 0xFFFFFFF;
          v128 = v127 - 1;
          v129 = v127 | *(_DWORD *)(v114 + 20) & 0xF0000000;
          *(_DWORD *)(v114 + 20) = v129;
          if ( (v129 & 0x40000000) != 0 )
            v130 = *(_QWORD *)(v114 - 8);
          else
            v130 = v114 - 24 * v127;
          v131 = (_QWORD *)(v130 + 24LL * v128);
          if ( *v131 )
          {
            v132 = v131[1];
            v133 = v131[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v133 = v132;
            if ( v132 )
              *(_QWORD *)(v132 + 16) = *(_QWORD *)(v132 + 16) & 3LL | v133;
          }
          *v131 = v124;
          if ( v124 )
          {
            v134 = *(_QWORD *)(v124 + 8);
            v131[1] = v134;
            if ( v134 )
              *(_QWORD *)(v134 + 16) = (unsigned __int64)(v131 + 1) | *(_QWORD *)(v134 + 16) & 3LL;
            v131[2] = (v124 + 8) | v131[2] & 3LL;
            *(_QWORD *)(v124 + 8) = v131;
          }
          v135 = *(_DWORD *)(v114 + 20) & 0xFFFFFFF;
          v136 = (*(_BYTE *)(v114 + 23) & 0x40) != 0 ? *(_QWORD *)(v114 - 8) : v114 - 24 * v135;
          *(_QWORD *)(v136 + 8LL * (unsigned int)(v135 - 1) + 24LL * *(unsigned int *)(v114 + 56) + 8) = v100;
          v137 = *(_QWORD *)(v114 + 32);
          if ( !v137 )
            break;
          v114 = 0;
          if ( *(_BYTE *)(v137 - 8) == 77 )
            v114 = v137 - 24;
          if ( v113 == v114 )
            goto LABEL_174;
        }
LABEL_506:
        BUG();
      }
LABEL_174:
      v138 = (__int64 *)sub_157F7B0(v100);
      if ( v138 )
      {
        v145 = sub_157EE30(v303);
        LOWORD(v338) = 257;
        v146 = *v138;
        if ( v145 )
          v145 -= 24;
        v147 = sub_1648B60(64);
        v148 = v147;
        if ( v147 )
        {
          sub_15F1EA0(v147, v146, 53, 0, 0, v145);
          *(_DWORD *)(v148 + 56) = 0;
          sub_164B780(v148, &v336);
          sub_1648880(v148, *(_DWORD *)(v148 + 56), 1);
        }
        v149 = *(_QWORD *)(v303 + 8);
        if ( v149 )
        {
          while ( 1 )
          {
            v150 = sub_1648700(v149);
            if ( (unsigned __int8)(*((_BYTE *)v150 + 16) - 25) <= 9u )
              break;
            v149 = *(_QWORD *)(v149 + 8);
            if ( !v149 )
              goto LABEL_175;
          }
LABEL_219:
          v158 = v150[5];
          v159 = sub_157F7B0(v158);
          sub_164D160(v159, v148, a6, a7, a8, a9, v160, v161, a12, a13);
          v166 = *(_DWORD *)(v148 + 20) & 0xFFFFFFF;
          if ( v166 == *(_DWORD *)(v148 + 56) )
          {
            sub_15F55D0(v148, v148, v162, v163, v164, v165);
            v166 = *(_DWORD *)(v148 + 20) & 0xFFFFFFF;
          }
          v167 = (v166 + 1) & 0xFFFFFFF;
          v168 = v167 | *(_DWORD *)(v148 + 20) & 0xF0000000;
          *(_DWORD *)(v148 + 20) = v168;
          if ( (v168 & 0x40000000) != 0 )
            v151 = *(_QWORD *)(v148 - 8);
          else
            v151 = v148 - 24 * v167;
          v152 = (__int64 *)(v151 + 24LL * (unsigned int)(v167 - 1));
          if ( *v152 )
          {
            v153 = v152[1];
            v154 = v152[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v154 = v153;
            if ( v153 )
              *(_QWORD *)(v153 + 16) = *(_QWORD *)(v153 + 16) & 3LL | v154;
          }
          *v152 = v159;
          if ( v159 )
          {
            v155 = *(_QWORD *)(v159 + 8);
            v152[1] = v155;
            if ( v155 )
              *(_QWORD *)(v155 + 16) = (unsigned __int64)(v152 + 1) | *(_QWORD *)(v155 + 16) & 3LL;
            v152[2] = (v159 + 8) | v152[2] & 3;
            *(_QWORD *)(v159 + 8) = v152;
          }
          v156 = *(_DWORD *)(v148 + 20) & 0xFFFFFFF;
          if ( (*(_BYTE *)(v148 + 23) & 0x40) != 0 )
            v157 = *(_QWORD *)(v148 - 8);
          else
            v157 = v148 - 24 * v156;
          *(_QWORD *)(v157 + 8LL * (unsigned int)(v156 - 1) + 24LL * *(unsigned int *)(v148 + 56) + 8) = v158;
          while ( 1 )
          {
            v149 = *(_QWORD *)(v149 + 8);
            if ( !v149 )
              break;
            v150 = sub_1648700(v149);
            if ( (unsigned __int8)(*((_BYTE *)v150 + 16) - 25) <= 9u )
              goto LABEL_219;
          }
        }
      }
LABEL_175:
      v84 += 8;
      if ( v84 == v293 )
        goto LABEL_176;
    }
    v344 = (__int64 *)((char *)v344 + 1);
LABEL_111:
    sub_12E48B0((__int64)&v344, 2 * v347);
    if ( v347 )
    {
      v86 = v339;
      v272 = 1;
      v212 = 0;
      v273 = (v347 - 1) & (((unsigned int)v339 >> 9) ^ ((unsigned int)v339 >> 4));
      v87 = (_QWORD *)(v345 + ((unsigned __int64)v273 << 6));
      v274 = v87[3];
      if ( v339 != v274 )
      {
        while ( v274 != -8 )
        {
          if ( !v212 && v274 == -16 )
            v212 = v87;
          v273 = (v347 - 1) & (v272 + v273);
          v87 = (_QWORD *)(v345 + ((unsigned __int64)v273 << 6));
          v274 = v87[3];
          if ( v339 == v274 )
            goto LABEL_113;
          ++v272;
        }
        goto LABEL_317;
      }
      goto LABEL_113;
    }
    goto LABEL_112;
  }
LABEL_176:
  v139 = *(__int64 **)(a1 + 400);
  v140 = (__int64)(*(_QWORD *)(a1 + 408) - (_QWORD)v139) >> 3;
  if ( (_DWORD)v140 )
  {
    v315 = 0;
    v304 = 8LL * (unsigned int)(v140 - 1);
    while ( 1 )
    {
      v141 = *(_QWORD *)&v315[(_QWORD)v139];
      v142 = *(_QWORD *)(v141 + 48);
      v143 = v141 + 40;
      if ( v143 != v142 )
        break;
LABEL_188:
      if ( v315 == (char *)v304 )
        goto LABEL_224;
      v315 += 8;
    }
    while ( v142 )
    {
      sub_1B75040(&v336, &v344, 3, 0, 0);
      sub_1B79630(&v336, v142 - 24);
      sub_1B75110(&v336);
      if ( *(_BYTE *)(v142 - 8) == 78
        && (v144 = *(_QWORD *)(v142 - 48), !*(_BYTE *)(v144 + 16))
        && (*(_BYTE *)(v144 + 33) & 0x20) != 0
        && *(_DWORD *)(v144 + 36) == 4 )
      {
        sub_14CE830(*(_QWORD *)(a1 + 176), v142 - 24);
        v142 = *(_QWORD *)(v142 + 8);
        if ( v143 == v142 )
        {
LABEL_187:
          v139 = *(__int64 **)(a1 + 400);
          goto LABEL_188;
        }
      }
      else
      {
        v142 = *(_QWORD *)(v142 + 8);
        if ( v143 == v142 )
          goto LABEL_187;
      }
    }
    sub_1B75040(&v336, &v344, 3, 0, 0);
    sub_1B79630(&v336, 0);
    sub_1B75110(&v336);
    goto LABEL_506;
  }
LABEL_224:
  v169 = (_QWORD *)sub_157EBA0(*(_QWORD *)(a1 + 320));
  sub_19C0AD0(a1, a2, a3, *v139, **(_QWORD **)(a1 + 376), v169, a5);
  sub_14045C0(*(_QWORD *)(a1 + 168), (__int64)v169, (__int64)a4);
  if ( v169 )
  {
    sub_15F2000((__int64)v169);
    sub_1648B90((__int64)v169);
  }
  v172 = *(_BYTE **)(a1 + 200);
  if ( v172 == *(_BYTE **)(a1 + 208) )
  {
    sub_13FD960(a1 + 192, v172, &v331);
  }
  else
  {
    if ( v172 )
    {
      *(_QWORD *)v172 = v331;
      v172 = *(_BYTE **)(a1 + 200);
    }
    *(_QWORD *)(a1 + 200) = v172 + 8;
  }
  v336 = 6;
  v337 = 0;
  *(_BYTE *)(a1 + 289) = 1;
  v338 = a2;
  if ( a2 != 0 && a2 != -8 && a2 != -16 )
    sub_164C220((__int64)&v336);
  sub_19C3980(a1, (__int64)a4, a2, a3, 0, a6, a7, a8, a9, v170, v171, a12, a13);
  v175 = *(_QWORD *)(a1 + 200);
  if ( v175 == *(_QWORD *)(a1 + 192) )
    goto LABEL_273;
  v176 = v338;
  if ( *(_QWORD **)(v175 - 8) == v331 )
  {
    if ( !v338 )
      goto LABEL_238;
    if ( *(_BYTE *)(v338 + 16) > 0x10u )
    {
      sub_19C3980(a1, (__int64)v331, v338, a3, 1, a6, a7, a8, a9, v173, v174, a12, a13);
LABEL_273:
      v176 = v338;
    }
  }
  if ( v176 != 0 && v176 != -8 && v176 != -16 )
    sub_1649B30(&v336);
LABEL_238:
  if ( v350 )
  {
    if ( v349 )
    {
      v268 = v348;
      v269 = &v348[2 * v349];
      do
      {
        if ( *v268 != -4 && *v268 != -8 )
        {
          v270 = v268[1];
          if ( v270 )
            sub_161E7C0((__int64)(v268 + 1), v270);
        }
        v268 += 2;
      }
      while ( v269 != v268 );
    }
    j___libc_free_0(v348);
  }
  if ( v347 )
  {
    v198 = (_QWORD *)v345;
    v333[0] = 2;
    v333[1] = 0;
    v199 = -8;
    v200 = (_QWORD *)(v345 + ((unsigned __int64)v347 << 6));
    v334 = -8;
    v332 = (char *)&unk_49E6B50;
    v335 = 0;
    v337 = 2;
    v338 = 0;
    v339 = -16;
    v336 = (__int64)&unk_49E6B50;
    i = 0;
    while ( 1 )
    {
      v201 = v198[3];
      if ( v199 != v201 )
      {
        v199 = v339;
        if ( v201 != v339 )
        {
          v202 = v198[7];
          if ( v202 != -8 && v202 != 0 && v202 != -16 )
          {
            sub_1649B30(v198 + 5);
            v201 = v198[3];
          }
          v199 = v201;
        }
      }
      *v198 = &unk_49EE2B0;
      if ( v199 != 0 && v199 != -8 && v199 != -16 )
        sub_1649B30(v198 + 1);
      v198 += 8;
      if ( v200 == v198 )
        break;
      v199 = v334;
    }
    v336 = (__int64)&unk_49EE2B0;
    if ( v339 != 0 && v339 != -8 && v339 != -16 )
      sub_1649B30(&v337);
    v332 = (char *)&unk_49EE2B0;
    if ( v334 != 0 && v334 != -8 && v334 != -16 )
      sub_1649B30(v333);
  }
  j___libc_free_0(v345);
  if ( src != v343 )
    _libc_free((unsigned __int64)src);
}
