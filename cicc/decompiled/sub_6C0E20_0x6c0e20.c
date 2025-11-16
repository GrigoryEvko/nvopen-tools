// Function: sub_6C0E20
// Address: 0x6c0e20
//
__int64 __fastcall sub_6C0E20(__int64 a1, __m128i *a2, _BYTE *a3, __m128i *a4)
{
  unsigned int *v6; // rdx
  _BOOL8 v7; // rcx
  _QWORD *v8; // r12
  bool v9; // zf
  __int64 *v10; // rbx
  unsigned __int8 v11; // al
  char v12; // al
  __int64 v13; // rax
  int v14; // r10d
  __int64 v15; // r11
  __int64 jj; // r15
  int v18; // r8d
  int v19; // r10d
  __int64 v20; // r14
  void **v21; // rbx
  __int64 v22; // r11
  unsigned int v23; // eax
  __int64 v24; // rax
  char kk; // dl
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  int v30; // r10d
  __int64 v31; // rdi
  unsigned __int8 v32; // r15
  __m128i v33; // xmm3
  __m128i v34; // xmm4
  __m128i v35; // xmm0
  __m128i v36; // xmm1
  __m128i v37; // xmm5
  __m128i v38; // xmm6
  __m128i v39; // xmm7
  __m128i v40; // xmm0
  unsigned int v41; // eax
  __m128i v42; // xmm2
  __m128i v43; // xmm1
  char v44; // dl
  char v45; // cl
  unsigned int v46; // eax
  int v47; // r15d
  __m128i v48; // xmm4
  __m128i v49; // xmm5
  __int64 v50; // rax
  __m128i v51; // xmm6
  __m128i v52; // xmm7
  char v53; // al
  char v54; // al
  __int64 v55; // rdi
  __int64 v56; // rax
  void *v57; // rcx
  __int64 v58; // rax
  char mm; // dl
  __int64 v60; // rdx
  char v61; // al
  void *v62; // rdi
  __int64 v63; // rax
  int v64; // eax
  char v65; // al
  __int64 v66; // rax
  __int64 v67; // rdx
  __int64 v68; // rcx
  __int64 v69; // r8
  __int64 v70; // r9
  __int64 v71; // r8
  __int64 v72; // r9
  __int64 v73; // rax
  char ii; // dl
  __int64 v75; // rdi
  __int64 v76; // rdx
  __int64 v77; // rcx
  __int64 v78; // r8
  __int64 v79; // r9
  unsigned __int16 v80; // ax
  _QWORD *v81; // r14
  __int64 v82; // r15
  __int64 v83; // rdx
  __int64 v84; // rax
  __int64 v85; // r9
  char v86; // si
  int v87; // r10d
  __int64 v88; // rax
  _BOOL4 v89; // edx
  __int64 v90; // rax
  __int64 v91; // rax
  __int64 v92; // r15
  __int64 v93; // rax
  int v94; // r14d
  unsigned __int16 v95; // ax
  __int64 v96; // r14
  char n; // al
  __int64 v98; // rax
  __int64 v99; // r15
  __int64 v100; // rsi
  __int64 v101; // rax
  __int64 v102; // rdi
  __int64 v103; // rax
  __m128i v104; // xmm7
  __m128i v105; // xmm4
  __m128i v106; // xmm5
  __m128i v107; // xmm6
  __m128i v108; // xmm0
  __m128i v109; // xmm7
  __m128i v110; // xmm4
  __m128i v111; // xmm5
  __m128i v112; // xmm6
  __m128i v113; // xmm7
  __m128i v114; // xmm4
  __m128i v115; // xmm5
  __int64 v116; // rdx
  __int64 v117; // rcx
  __int64 v118; // r8
  __int64 v119; // r9
  __int64 v120; // rax
  char v121; // al
  __int64 v122; // rax
  int v123; // eax
  __int64 v124; // rax
  __int64 v125; // rax
  __int64 v126; // rax
  __int64 v127; // rax
  char v128; // dl
  __int64 v129; // rax
  __int64 v130; // rax
  __int64 v131; // rax
  __int64 v132; // rax
  __int64 v133; // rax
  __int64 v134; // rax
  __int64 v135; // rax
  __int64 v136; // rax
  __int64 v137; // rax
  __int64 v138; // rax
  __int64 v139; // rax
  __int64 v140; // rax
  __int64 v141; // rax
  __int64 v142; // rax
  __int64 v143; // rax
  __int64 v144; // rbx
  int v145; // eax
  __int64 v146; // rax
  __int64 v147; // rax
  __int64 v148; // rax
  __int64 *v149; // rax
  __int64 v150; // rax
  __int64 v151; // rbx
  __int64 v152; // rdx
  __int64 v153; // rcx
  __int64 v154; // r8
  __int64 v155; // r9
  __int64 v156; // rax
  __int64 v157; // rdx
  __int64 v158; // rcx
  __int64 v159; // r8
  __int64 v160; // r9
  unsigned __int16 v161; // r15
  __int64 j; // rbx
  __int64 v163; // r15
  unsigned int v164; // r14d
  char m; // al
  unsigned int v166; // r14d
  unsigned __int16 v167; // bx
  __int32 v168; // edx
  __int16 v169; // ax
  char v170; // dl
  __int64 v171; // rax
  __int64 v172; // rax
  __int64 v173; // rdx
  __int64 v174; // rcx
  __int64 v175; // r8
  __int64 v176; // r9
  __int8 v177; // al
  FILE *v178; // r15
  FILE *v179; // r14
  __int64 v180; // rax
  __int64 v181; // rax
  __int64 v182; // rax
  char v183; // dl
  __int64 v184; // rax
  __int64 v185; // r15
  char v186; // dl
  __int64 v187; // rax
  __int64 v188; // rcx
  __int64 v189; // r8
  __int64 v190; // r9
  __int64 v191; // rcx
  __int64 v192; // r8
  __int64 v193; // r9
  __int64 i; // rax
  unsigned __int8 v195; // al
  __int64 v196; // rdx
  __int64 v197; // rsi
  __m128i *v198; // rdi
  __int64 v199; // rsi
  __int64 v200; // rax
  char k; // dl
  __int64 v202; // rax
  __int64 v203; // rdi
  unsigned __int8 v204; // al
  __int64 v205; // rax
  __int64 v206; // r15
  int v207; // eax
  int v208; // eax
  int v209; // eax
  __int64 v210; // rax
  int v211; // eax
  int v212; // [rsp+10h] [rbp-770h]
  int v213; // [rsp+14h] [rbp-76Ch]
  _BOOL4 v214; // [rsp+28h] [rbp-758h]
  int v215; // [rsp+2Ch] [rbp-754h]
  __int64 v216; // [rsp+30h] [rbp-750h]
  int v217; // [rsp+38h] [rbp-748h]
  __int64 v218; // [rsp+38h] [rbp-748h]
  __int64 v219; // [rsp+38h] [rbp-748h]
  __int64 v220; // [rsp+38h] [rbp-748h]
  __int64 v221; // [rsp+38h] [rbp-748h]
  __int64 v222; // [rsp+38h] [rbp-748h]
  __int64 v223; // [rsp+38h] [rbp-748h]
  __int64 v224; // [rsp+38h] [rbp-748h]
  __int64 v225; // [rsp+38h] [rbp-748h]
  __int64 v226; // [rsp+38h] [rbp-748h]
  __int64 v227; // [rsp+38h] [rbp-748h]
  __int64 v228; // [rsp+38h] [rbp-748h]
  __int64 v229; // [rsp+38h] [rbp-748h]
  __int64 v230; // [rsp+38h] [rbp-748h]
  __int64 v231; // [rsp+38h] [rbp-748h]
  __int64 v232; // [rsp+38h] [rbp-748h]
  int v233; // [rsp+38h] [rbp-748h]
  int v234; // [rsp+40h] [rbp-740h]
  bool v235; // [rsp+47h] [rbp-739h]
  unsigned int v236; // [rsp+48h] [rbp-738h]
  int v237; // [rsp+48h] [rbp-738h]
  int v238; // [rsp+48h] [rbp-738h]
  int v239; // [rsp+48h] [rbp-738h]
  int v240; // [rsp+48h] [rbp-738h]
  int v241; // [rsp+48h] [rbp-738h]
  int v242; // [rsp+48h] [rbp-738h]
  int v243; // [rsp+48h] [rbp-738h]
  int v244; // [rsp+48h] [rbp-738h]
  int v245; // [rsp+48h] [rbp-738h]
  int v246; // [rsp+48h] [rbp-738h]
  int v247; // [rsp+48h] [rbp-738h]
  int v248; // [rsp+48h] [rbp-738h]
  int v249; // [rsp+48h] [rbp-738h]
  int v250; // [rsp+48h] [rbp-738h]
  int v251; // [rsp+48h] [rbp-738h]
  __int64 v252; // [rsp+48h] [rbp-738h]
  __int64 v253; // [rsp+48h] [rbp-738h]
  __int64 v254; // [rsp+58h] [rbp-728h]
  __int64 v255; // [rsp+58h] [rbp-728h]
  __int64 v256; // [rsp+58h] [rbp-728h]
  bool v257; // [rsp+60h] [rbp-720h]
  __int64 v258; // [rsp+60h] [rbp-720h]
  unsigned int v259; // [rsp+68h] [rbp-718h]
  __int64 v260; // [rsp+68h] [rbp-718h]
  __int64 v261; // [rsp+68h] [rbp-718h]
  bool v262; // [rsp+68h] [rbp-718h]
  __m128i *v263; // [rsp+70h] [rbp-710h]
  bool v264; // [rsp+70h] [rbp-710h]
  unsigned int v266; // [rsp+88h] [rbp-6F8h] BYREF
  int v267; // [rsp+8Ch] [rbp-6F4h] BYREF
  int v268; // [rsp+90h] [rbp-6F0h] BYREF
  int v269; // [rsp+94h] [rbp-6ECh] BYREF
  __int64 v270; // [rsp+98h] [rbp-6E8h] BYREF
  __int64 v271; // [rsp+A0h] [rbp-6E0h] BYREF
  __int64 v272; // [rsp+A8h] [rbp-6D8h] BYREF
  __int64 v273; // [rsp+B0h] [rbp-6D0h] BYREF
  __int64 v274; // [rsp+B8h] [rbp-6C8h] BYREF
  __int64 v275; // [rsp+C0h] [rbp-6C0h] BYREF
  __int64 v276; // [rsp+C8h] [rbp-6B8h] BYREF
  __int64 v277; // [rsp+D0h] [rbp-6B0h] BYREF
  __int64 v278; // [rsp+D8h] [rbp-6A8h] BYREF
  void *v279; // [rsp+E0h] [rbp-6A0h] BYREF
  __int64 v280; // [rsp+E8h] [rbp-698h]
  __int64 v281; // [rsp+F0h] [rbp-690h]
  __int64 v282; // [rsp+F8h] [rbp-688h]
  __int64 v283; // [rsp+100h] [rbp-680h]
  __int64 v284; // [rsp+108h] [rbp-678h]
  __int128 v285; // [rsp+110h] [rbp-670h]
  _QWORD v286[2]; // [rsp+120h] [rbp-660h] BYREF
  _DWORD v287[40]; // [rsp+130h] [rbp-650h] BYREF
  _BYTE v288[352]; // [rsp+1D0h] [rbp-5B0h] BYREF
  _BYTE v289[352]; // [rsp+330h] [rbp-450h] BYREF
  __m128i v290; // [rsp+490h] [rbp-2F0h] BYREF
  __m128i v291; // [rsp+4A0h] [rbp-2E0h] BYREF
  __int64 v292; // [rsp+4B0h] [rbp-2D0h]
  __int64 v293; // [rsp+520h] [rbp-260h]
  char v294; // [rsp+5CDh] [rbp-1B3h]
  __m128i v295; // [rsp+5F0h] [rbp-190h] BYREF
  __m128i v296; // [rsp+600h] [rbp-180h]
  __m128i v297; // [rsp+610h] [rbp-170h]
  __m128i v298; // [rsp+620h] [rbp-160h]
  __m128i v299; // [rsp+630h] [rbp-150h]
  __m128i v300; // [rsp+640h] [rbp-140h]
  __m128i v301; // [rsp+650h] [rbp-130h]
  __m128i v302; // [rsp+660h] [rbp-120h]
  __m128i v303; // [rsp+670h] [rbp-110h]
  __m128i v304; // [rsp+680h] [rbp-100h]
  __m128i v305; // [rsp+690h] [rbp-F0h]
  __m128i v306; // [rsp+6A0h] [rbp-E0h]
  __m128i v307; // [rsp+6B0h] [rbp-D0h]
  __m128i v308; // [rsp+6C0h] [rbp-C0h]
  __m128i v309; // [rsp+6D0h] [rbp-B0h]
  __m128i v310; // [rsp+6E0h] [rbp-A0h]
  __m128i v311; // [rsp+6F0h] [rbp-90h]
  __m128i v312; // [rsp+700h] [rbp-80h]
  __m128i v313; // [rsp+710h] [rbp-70h]
  __m128i v314; // [rsp+720h] [rbp-60h]
  __m128i v315; // [rsp+730h] [rbp-50h]
  __m128i v316; // [rsp+740h] [rbp-40h]

  v263 = a2;
  v271 = 0;
  v6 = (unsigned int *)qword_4D03C50;
  v267 = 0;
  v268 = 0;
  v7 = (*(_BYTE *)(qword_4D03C50 + 20LL) & 0x20) != 0;
  v235 = (*(_BYTE *)(qword_4D03C50 + 20LL) & 0x20) != 0;
  *(_BYTE *)(qword_4D03C50 + 20LL) &= ~0x20u;
  if ( a3 )
  {
    a1 = (__int64)a3;
    v8 = v289;
    a2 = (__m128i *)v289;
    sub_6F8CA0(a3, v289, v288, &v276, &v266, &v277);
    v263 = (__m128i *)v288;
    if ( a3[56] )
      goto LABEL_26;
  }
  else
  {
    v8 = (_QWORD *)a1;
    v276 = *(_QWORD *)&dword_4F063F8;
    v266 = dword_4F06650[0];
  }
  v9 = (*((_BYTE *)v8 + 18) & 1) == 0;
  v273 = *(_QWORD *)((char *)v8 + 68);
  if ( v9
    || !v263[4].m128i_i32[1]
    || (v10 = (__int64 *)((char *)v263[4].m128i_i64 + 4),
        a2 = (__m128i *)&v273,
        a1 = (__int64)v263[4].m128i_i64 + 4,
        (int)sub_7294D0((char *)v263[4].m128i_i64 + 4, &v273) >= 0) )
  {
    v10 = &v273;
  }
  v275 = *v10;
  if ( dword_4F077C4 == 2 )
  {
    v234 = unk_4D047D8;
    if ( !unk_4D047D8 )
    {
      v214 = 0;
      v215 = 0;
      goto LABEL_7;
    }
    v65 = *((_BYTE *)v8 + 18);
    if ( (v65 & 1) == 0 )
    {
      v6 = (unsigned int *)*((unsigned __int16 *)v8 + 9);
      LOWORD(v6) = (unsigned __int16)v6 & 0x8040;
      if ( (_WORD)v6 == 0x8000 )
      {
        v215 = dword_4F077BC;
        if ( dword_4F077BC )
        {
          v214 = 0;
          v215 = 0;
          if ( qword_4F077A8 > 0x9DCFu )
            v234 = 1;
          else
            v234 = ((*((_BYTE *)v8 + 19) >> 3) ^ 1) & 1;
        }
        else
        {
          v214 = 0;
          v234 = 1;
        }
        goto LABEL_7;
      }
      v6 = (unsigned int *)*((unsigned __int8 *)v8 + 16);
      if ( (_BYTE)v6 == 6 || (_BYTE)v6 == 3 )
      {
        v6 = (unsigned int *)v8[17];
LABEL_105:
        if ( *((char *)v8 + 19) < 0 )
        {
          v215 = 0;
          v234 = 0;
          v214 = (v65 & 0x40) != 0;
        }
        else
        {
          v214 = 0;
          v234 = 0;
          v215 = ((*((_BYTE *)v6 + 81) >> 4) ^ 1) & 1;
        }
        goto LABEL_7;
      }
      if ( (_BYTE)v6 == 1 )
      {
        v6 = (unsigned int *)v8[18];
        if ( *((_BYTE *)v6 + 24) == 20 )
        {
          v6 = (unsigned int *)**((_QWORD **)v6 + 7);
          goto LABEL_105;
        }
      }
    }
  }
  v214 = 0;
  v215 = 0;
  v234 = 0;
LABEL_7:
  if ( !unk_4D041F8 )
    goto LABEL_8;
  if ( !*(_BYTE *)(qword_4D03C50 + 16LL) )
  {
LABEL_29:
    sub_6E68E0(59, v8);
    v18 = 0;
    v212 = 0;
    v259 = 0;
    v236 = 0;
    LODWORD(v254) = 0;
    goto LABEL_30;
  }
  a1 = (__int64)v8;
  v91 = sub_6EB5C0(v8);
  v92 = v91;
  if ( !v91 )
    goto LABEL_8;
  if ( (*(_BYTE *)(v91 + 202) & 4) != 0
    && dword_4F077C0
    && (qword_4F077A8 <= 0x9E33u || (*(_BYTE *)(qword_4D03C50 + 19LL) & 4) != 0) )
  {
    v93 = *(_QWORD *)(v91 + 256);
    v269 = 0;
    v92 = *(_QWORD *)(v93 + 8);
    if ( !v92 )
      goto LABEL_8;
  }
  else
  {
    v269 = 0;
  }
  if ( !*(_BYTE *)(v92 + 174) && *(_WORD *)(v92 + 176) )
  {
    a2 = (__m128i *)&v269;
    a1 = v92;
    v94 = sub_7176C0(v92, &v269);
    v95 = *(_WORD *)(v92 + 176);
    if ( v95 > 0x617Eu )
    {
      if ( v95 == 25767 )
      {
        v269 = 1;
        goto LABEL_344;
      }
    }
    else if ( v95 > 0x617Cu )
    {
      if ( !unk_4F04C50 )
        goto LABEL_408;
      v180 = *(_QWORD *)(unk_4F04C50 + 32LL);
      if ( *(char *)(v180 + 192) >= 0 )
        goto LABEL_408;
      for ( i = *(_QWORD *)(v180 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      if ( (*(_BYTE *)(*(_QWORD *)(i + 168) + 16LL) & 1) == 0 )
      {
LABEL_408:
        a2 = (__m128i *)v8;
        a1 = 1778;
        sub_6E68E0(1778, v8);
      }
    }
    if ( !v269 )
    {
      if ( v94 )
        goto LABEL_9;
      goto LABEL_8;
    }
LABEL_344:
    v151 = sub_6EB5C0(v8);
    v156 = sub_724DC0(v8, a2, v152, v153, v154, v155);
    v161 = *(_WORD *)(v151 + 176);
    v272 = v156;
    if ( !a3 )
    {
      v278 = *(_QWORD *)&dword_4F063F8;
      sub_7B8B50(v8, a2, v157, v158);
      ++*(_BYTE *)(qword_4F061C8 + 36LL);
      ++*(_QWORD *)(qword_4D03C50 + 40LL);
    }
    if ( v161 > 0x619Fu )
    {
      if ( v161 == 25766 )
      {
        v287[0] = 0;
        v185 = sub_726700(27);
        sub_6E70E0(v185, a4);
        sub_6E1E00(3, &v290, 0, 0);
        sub_69ED20((__int64)&v295, 0, 0, 1);
        sub_6F69D0(&v295, 0);
        sub_6E2B30(&v295, 0);
        if ( !v296.m128i_i8[0] )
          goto LABEL_427;
        v186 = *(_BYTE *)(v295.m128i_i64[0] + 140);
        if ( v186 == 12 )
        {
          v187 = v295.m128i_i64[0];
          do
          {
            v187 = *(_QWORD *)(v187 + 160);
            v186 = *(_BYTE *)(v187 + 140);
          }
          while ( v186 == 12 );
        }
        if ( v186 )
        {
          if ( (unsigned int)sub_8D3350(v295.m128i_i64[0]) )
          {
            if ( v296.m128i_i8[0] == 1 )
              sub_6F4D20(&v295, 1, 1);
            if ( !(unsigned int)sub_6E9820(&v295) )
              *(_BYTE *)(v185 + 64) |= 1u;
          }
          else
          {
            sub_6E68E0(41, &v295);
            v287[0] = 1;
          }
        }
        else
        {
LABEL_427:
          sub_6E6870(&v295);
          v287[0] = 1;
        }
        *(_QWORD *)(v185 + 56) = sub_6F6F40(&v295, 0);
        sub_6B06D0(v185, a4, v287, v188, v189, v190);
        sub_6B06D0(v185, a4, v287, v191, v192, v193);
        if ( !v287[0] )
        {
          sub_6E6B60(a4, 0);
          goto LABEL_368;
        }
      }
      else
      {
        if ( v161 != 25767 )
        {
          v264 = *(_BYTE *)(qword_4D03C50 + 16LL) <= 3u;
          goto LABEL_351;
        }
        sub_6F69D0(v8, 28);
        if ( a3 )
          sub_6F8800(*((_QWORD *)a3 + 2), a3, &v295);
        else
          sub_69ED20((__int64)&v295, 0, 0, 1);
        sub_6F69D0(&v295, 0);
        if ( !v296.m128i_i8[0] )
          goto LABEL_384;
        v170 = *(_BYTE *)(v295.m128i_i64[0] + 140);
        if ( v170 == 12 )
        {
          v171 = v295.m128i_i64[0];
          do
          {
            v171 = *(_QWORD *)(v171 + 160);
            v170 = *(_BYTE *)(v171 + 140);
          }
          while ( v170 == 12 );
        }
        if ( v170 )
        {
          if ( (unsigned int)sub_8D2E70(v295.m128i_i64[0]) || (unsigned int)sub_8D3D40(v295.m128i_i64[0]) )
          {
            v206 = sub_6F6F40(v8, 0);
            *(_QWORD *)(v206 + 16) = sub_6F6F40(&v295, 0);
            v207 = sub_732700(v295.m128i_i32[0], v295.m128i_i32[0], 0, 0, 0, 0, 0, 0);
            sub_701D00(
              v206,
              v207,
              0,
              0,
              0,
              0,
              0,
              0,
              0,
              0,
              0,
              (__int64)&dword_4F077C8,
              (__int64)v8 + 68,
              (__int64)&dword_4F077C8,
              (__int64)a4,
              0,
              (__int64)&v290);
            goto LABEL_368;
          }
          sub_6E6930(3163, &v295, v295.m128i_i64[0]);
        }
        else
        {
LABEL_384:
          sub_6E6870(&v295);
        }
      }
      sub_6E6260(a4);
    }
    else
    {
      if ( v161 > 0x617Bu )
      {
        switch ( v161 )
        {
          case 0x617Cu:
          case 0x619Fu:
            sub_6AC240((__int64)a4, (__int64)v8, v157, v158, v159, v160);
            goto LABEL_368;
          case 0x617Fu:
            sub_6AC910((__int64)a4, (__int64)v8, v157, v158, v159, v160);
            goto LABEL_368;
          case 0x6180u:
            sub_6AC740((__int64)a4, (__int64)v8, v157, v158, v159, v160);
            goto LABEL_368;
          case 0x6181u:
          case 0x619Eu:
            goto LABEL_367;
          case 0x6182u:
            sub_6ACB40((__int64)a4, (__int64)v8, 1, v158, v159, v160);
            goto LABEL_368;
          default:
            goto LABEL_350;
        }
      }
      if ( v161 != 16714 )
      {
LABEL_350:
        v264 = *(_BYTE *)(qword_4D03C50 + 16LL) <= 3u;
        if ( v161 == 4461 )
        {
          v204 = 5;
          if ( !(unk_4F068B4 | v264) )
            v204 = (unk_4F04C50 == 0) + 4;
          sub_6E2140(v204, v287, 0, 0, a3);
          v205 = qword_4D03C50;
          v262 = (*(_BYTE *)(qword_4D03C50 + 18LL) & 8) != 0;
          *(_BYTE *)(qword_4D03C50 + 18LL) |= 8u;
          if ( !(unk_4F068B4 | v264) )
          {
            if ( unk_4F04C50 )
              *(_BYTE *)(v205 + 17) |= 2u;
            v264 = 0;
          }
LABEL_352:
          if ( a3 )
            sub_6F8800(*((_QWORD *)a3 + 2), a3, &v290);
          else
            sub_69ED20((__int64)&v290, 0, 0, 1);
          sub_6F6C80(&v290);
          if ( v161 == 4461
            && unk_4F07708
            && unk_4F04C50
            && (dword_4F077C0
             || dword_4F077BC && ((unsigned __int64)(qword_4F077A8 - 40000LL) <= 0x63 || qword_4F077A8 > 0x9D6Bu))
            && v291.m128i_i8[0] == 1
            && *(_BYTE *)(v293 + 24) == 3 )
          {
            v203 = *(_QWORD *)(v293 + 56);
            if ( (*(_BYTE *)(v203 + 89) & 1) == 0 )
            {
              v258 = *(_QWORD *)(v293 + 56);
              if ( (unsigned int)sub_6EA1E0(v203) )
              {
                if ( sub_6EA380(v258, 0, 0, 1) )
                  sub_6F69D0(&v290, 0);
              }
            }
          }
          if ( v291.m128i_i16[0] == 513 )
            sub_6F4D20(&v290, 0, 1);
          *(_BYTE *)(qword_4D03C50 + 18LL) = (8 * v262) | *(_BYTE *)(qword_4D03C50 + 18LL) & 0xF7;
          v55 = *(_QWORD *)(v151 + 152);
          for ( j = sub_73D790(v55); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
            ;
          if ( dword_4F04C44 != -1
            || (v202 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v202 + 6) & 6) != 0)
            || *(_BYTE *)(v202 + 4) == 12 )
          {
            if ( *((_BYTE *)v8 + 16) == 3
              || (v55 = v290.m128i_i64[0], (unsigned int)sub_8D3D40(v290.m128i_i64[0]))
              || v291.m128i_i8[0] == 2 && v294 == 12 )
            {
              sub_6F40C0(&v290);
              sub_6F69D0(v8, 0);
              v199 = 32;
              sub_6E5A30(v8[11], 32, 4);
              if ( !v291.m128i_i8[0] )
                goto LABEL_452;
              v200 = v290.m128i_i64[0];
              for ( k = *(_BYTE *)(v290.m128i_i64[0] + 140); k == 12; k = *(_BYTE *)(v200 + 140) )
                v200 = *(_QWORD *)(v200 + 160);
              if ( k )
              {
                v208 = sub_6F6F40(&v290, 0);
                v199 = (__int64)&v295;
                v198 = (__m128i *)v8;
                sub_7022F0(
                  (_DWORD)v8,
                  (unsigned int)&v295,
                  v208,
                  0,
                  0,
                  0,
                  0,
                  0,
                  (__int64)&dword_4F077C8,
                  (__int64)v8 + 68,
                  (__int64)&dword_4F077C8,
                  (__int64)a4,
                  0,
                  0);
                if ( v291.m128i_i8[0] == 2 || v161 == 4432 || v264 )
                {
                  v198 = a4;
                  sub_6F4B70(a4);
                }
              }
              else
              {
LABEL_452:
                v198 = a4;
                sub_6E6260(a4);
              }
              goto LABEL_446;
            }
          }
          if ( v161 == 4432 )
          {
            v163 = v290.m128i_i64[0];
            v164 = *(unsigned __int8 *)(j + 160);
            for ( m = *(_BYTE *)(v290.m128i_i64[0] + 140); m == 12; m = *(_BYTE *)(v163 + 140) )
              v163 = *(_QWORD *)(v163 + 160);
            switch ( m )
            {
              case 0:
              case 15:
              case 19:
                v209 = -1;
                goto LABEL_507;
              case 1:
                v209 = 0;
                goto LABEL_507;
              case 2:
                if ( !dword_4F077BC || qword_4F077A8 <= 0x76BFu )
                  goto LABEL_521;
                if ( (unsigned int)sub_8D29A0(v163) )
                {
                  v209 = 4;
                }
                else if ( dword_4F077BC
                       && qword_4F077A8 > 0x76BFu
                       && *(_BYTE *)(v163 + 140) == 2
                       && (*(_BYTE *)(v163 + 161) & 8) != 0 )
                {
                  v209 = 3;
                }
                else
                {
LABEL_521:
                  v209 = 1;
                }
                goto LABEL_507;
              case 3:
                v209 = 8;
                goto LABEL_507;
              case 4:
              case 5:
                v209 = 9;
                goto LABEL_507;
              case 6:
                v55 = v163;
                if ( !(unsigned int)sub_8D2E30(v163) )
                  goto LABEL_512;
                goto LABEL_506;
              case 7:
                if ( !dword_4F077BC )
                  goto LABEL_506;
                v209 = 10;
                if ( qword_4F077A8 <= 0x76BFu )
                  goto LABEL_506;
                goto LABEL_507;
              case 8:
                if ( !dword_4F077BC )
                  goto LABEL_506;
                v209 = 14;
                if ( qword_4F077A8 <= 0x76BFu )
                  goto LABEL_506;
                goto LABEL_507;
              case 9:
              case 10:
                v209 = 12;
                goto LABEL_507;
              case 11:
                v209 = 13;
                goto LABEL_507;
              case 13:
                if ( qword_4F077A8 <= 0x76BFu )
                {
LABEL_506:
                  v209 = 5;
                }
                else
                {
                  v210 = sub_8D4870(v163);
                  v209 = (unsigned int)sub_8D2310(v210) == 0 ? 7 : 12;
                }
LABEL_507:
                v197 = v209;
                v196 = v164;
                goto LABEL_445;
              default:
                goto LABEL_512;
            }
          }
          if ( v161 != 4461 )
            goto LABEL_512;
          if ( sub_694910(&v290) )
          {
            v195 = 1;
          }
          else if ( v291.m128i_i8[0] != 2 || (v195 = 1, v294 == 6) )
          {
            if ( unk_4F04C50 && (!unk_4F068B4 || (unsigned int)sub_6E4B50()) && !v264 )
            {
              sub_6F69D0(v8, 0);
              sub_6E5A30(v8[11], 32, 4);
              sub_6FE880(&v290, 1);
              v211 = sub_6F6F40(&v290, 0);
              v199 = (__int64)&v295;
              v198 = (__m128i *)v8;
              sub_7022F0(
                (_DWORD)v8,
                (unsigned int)&v295,
                v211,
                0,
                0,
                0,
                0,
                0,
                (__int64)&dword_4F077C8,
                (__int64)v8 + 68,
                (__int64)&dword_4F077C8,
                (__int64)a4,
                0,
                0);
LABEL_446:
              sub_6E2B30(v198, v199);
              goto LABEL_368;
            }
            v195 = 0;
          }
          v196 = *(unsigned __int8 *)(j + 160);
          v197 = v195;
LABEL_445:
          sub_72BAF0(v272, v197, v196);
          v198 = (__m128i *)v272;
          v199 = (__int64)a4;
          sub_6E6A50(v272, a4);
          goto LABEL_446;
        }
LABEL_351:
        sub_6E2140(5, v287, 0, 0, a3);
        v262 = (*(_BYTE *)(qword_4D03C50 + 18LL) & 8) != 0;
        goto LABEL_352;
      }
LABEL_367:
      sub_6ACB40((__int64)a4, (__int64)v8, 0, v158, v159, v160);
    }
LABEL_368:
    if ( a3 )
    {
      v166 = *(_DWORD *)(*(_QWORD *)a3 + 44LL);
      v167 = *(_WORD *)(*(_QWORD *)a3 + 48LL);
      v278 = *(_QWORD *)(*(_QWORD *)a3 + 28LL);
    }
    else
    {
      v166 = dword_4F063F8;
      v167 = word_4F063FC[0];
      sub_7BE280(28, 18, 0, 0);
      --*(_BYTE *)(qword_4F061C8 + 36LL);
      --*(_QWORD *)(qword_4D03C50 + 40LL);
    }
    v168 = *((_DWORD *)v8 + 17);
    v169 = *((_WORD *)v8 + 36);
    a4[4].m128i_i32[3] = v166;
    a4[5].m128i_i16[0] = v167;
    a4[4].m128i_i16[4] = v169;
    a4[4].m128i_i32[1] = v168;
    *(_QWORD *)dword_4F07508 = *(__int64 *)((char *)a4[4].m128i_i64 + 4);
    unk_4F061D8 = *(__int64 *)((char *)&a4[4].m128i_i64[1] + 4);
    sub_6E3280(a4, &v278);
    sub_724E30(&v272);
    return sub_6E1990(v271);
  }
LABEL_8:
  v11 = *(_BYTE *)(qword_4D03C50 + 16LL);
  if ( v11 <= 3u )
  {
    v6 = &word_4D04898;
    a2 = (__m128i *)word_4D04898;
    if ( !word_4D04898 || !v11 )
      goto LABEL_29;
  }
LABEL_9:
  v257 = a3 != 0;
  if ( *((_BYTE *)v8 + 16) == 1 )
  {
    v66 = v8[18];
    if ( *(_BYTE *)(v66 + 24) == 1 && (unsigned __int8)(*(_BYTE *)(v66 + 56) - 22) <= 1u )
    {
      if ( !a3 )
      {
        sub_7B8B50(a1, a2, v6, v7);
        v257 = 0;
        v18 = 0;
        LODWORD(v254) = 0;
        v212 = 0;
        v259 = 0;
        v274 = *(_QWORD *)&dword_4F063F8;
        v213 = 0;
        v236 = 1;
LABEL_33:
        v19 = v18;
        jj = 0;
        v20 = 0;
        v21 = 0;
        v22 = 0;
        goto LABEL_34;
      }
      v18 = 0;
      v212 = 0;
      v259 = 0;
      v274 = v277;
      v236 = 1;
      LODWORD(v254) = 0;
LABEL_30:
      v257 = a3 != 0;
      if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 1) != 0 && a3 )
        goto LABEL_26;
      v213 = v18;
      goto LABEL_33;
    }
  }
  if ( dword_4F077C4 == 2 && (unsigned int)sub_8D3A70(*v8) )
  {
    v96 = *v8;
    for ( n = *(_BYTE *)(*v8 + 140LL); n == 12; n = *(_BYTE *)(v96 + 140) )
      v96 = *(_QWORD *)(v96 + 160);
    if ( (*(_BYTE *)(v96 + 177) & 0x20) == 0 )
      goto LABEL_270;
    if ( n != 9 || (*(_BYTE *)(*(_QWORD *)(v96 + 168) + 109LL) & 0x20) == 0 )
      goto LABEL_223;
    v182 = sub_72F130(v96);
    v183 = *(_BYTE *)(v96 + 177);
    if ( (v183 & 0x20) != 0 )
    {
      *(_BYTE *)(v96 + 177) = v183 & 0xDF;
      if ( v182 && !(unsigned int)sub_8DBE70(*(_QWORD *)(v182 + 152)) )
      {
        *(_BYTE *)(v96 + 177) |= 0x20u;
LABEL_270:
        if ( (unsigned int)sub_8D23B0(v96) && (unsigned int)sub_8D3A70(v96) )
          sub_8AD220(v96, 0);
        *v263 = _mm_loadu_si128((const __m128i *)v8);
        v263[1] = _mm_loadu_si128((const __m128i *)v8 + 1);
        v263[2] = _mm_loadu_si128((const __m128i *)v8 + 2);
        v263[3] = _mm_loadu_si128((const __m128i *)v8 + 3);
        v263[4] = _mm_loadu_si128((const __m128i *)v8 + 4);
        v263[5] = _mm_loadu_si128((const __m128i *)v8 + 5);
        v263[6] = _mm_loadu_si128((const __m128i *)v8 + 6);
        v263[7] = _mm_loadu_si128((const __m128i *)v8 + 7);
        v263[8] = _mm_loadu_si128((const __m128i *)v8 + 8);
        v121 = *((_BYTE *)v8 + 16);
        if ( v121 == 2 )
        {
          v263[9] = _mm_loadu_si128((const __m128i *)v8 + 9);
          v263[10] = _mm_loadu_si128((const __m128i *)v8 + 10);
          v263[11] = _mm_loadu_si128((const __m128i *)v8 + 11);
          v263[12] = _mm_loadu_si128((const __m128i *)v8 + 12);
          v263[13] = _mm_loadu_si128((const __m128i *)v8 + 13);
          v263[14] = _mm_loadu_si128((const __m128i *)v8 + 14);
          v263[15] = _mm_loadu_si128((const __m128i *)v8 + 15);
          v263[16] = _mm_loadu_si128((const __m128i *)v8 + 16);
          v263[17] = _mm_loadu_si128((const __m128i *)v8 + 17);
          v263[18] = _mm_loadu_si128((const __m128i *)v8 + 18);
          v263[19] = _mm_loadu_si128((const __m128i *)v8 + 19);
          v263[20] = _mm_loadu_si128((const __m128i *)v8 + 20);
          v263[21] = _mm_loadu_si128((const __m128i *)v8 + 21);
        }
        else if ( v121 == 5 || v121 == 1 )
        {
          v263[9].m128i_i64[0] = v8[18];
        }
        v254 = sub_7D3790(42, v96);
        if ( v254 )
        {
          sub_6E7190(v254, 0, v8);
          sub_82F1E0(v263, 0, v8);
          v18 = 1;
          v212 = 1;
        }
        else
        {
          v212 = 0;
          v18 = 1;
        }
        v259 = 0;
        v236 = 0;
        goto LABEL_30;
      }
      *(_BYTE *)(v96 + 177) |= 0x20u;
    }
    else if ( v182 && !(unsigned int)sub_8DBE70(*(_QWORD *)(v182 + 152)) )
    {
      goto LABEL_270;
    }
LABEL_223:
    sub_6F40C0(v8);
    v18 = 0;
    v212 = 0;
    v259 = 1;
    v236 = 0;
    LODWORD(v254) = 0;
    goto LABEL_30;
  }
  v259 = 0;
  if ( *((_WORD *)v8 + 8) == 772 && (*((_BYTE *)v8 + 18) & 1) == 0 )
  {
    v81 = (_QWORD *)v8[17];
    v256 = v81[8];
    if ( dword_4F077BC
      && (v259 = qword_4F077B4) == 0
      && qword_4D03C50
      && (*(_BYTE *)(qword_4D03C50 + 17LL) & 0x40) != 0
      && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0 )
    {
      v82 = sub_724D80(12);
      sub_7249B0(v82, 3);
      *(_QWORD *)(v82 + 128) = *(_QWORD *)&dword_4D03B80;
      *(_BYTE *)(v82 + 177) = ((*((_BYTE *)v8 + 18) & 0x40) != 0) | *(_BYTE *)(v82 + 177) & 0xFE;
      *(_BYTE *)(v82 + 200) = *(_BYTE *)(*v81 + 72LL);
      sub_877D50(v82, *v81);
      v83 = v81[8];
      if ( (*((_BYTE *)v81 + 81) & 0x10) != 0 )
      {
        sub_877E20(0, v82, v83);
      }
      else if ( v83 )
      {
        sub_877E90(0, v82);
      }
      sub_6E6A50(v82, v8);
    }
    else
    {
      v259 = sub_830D50(v81, v81, &v273, *((_BYTE *)v8 + 19) & 1, v263);
      if ( v259 )
      {
        v85 = v8[11];
        v86 = *((_BYTE *)v8 + 18);
        v295.m128i_i64[0] = *(_QWORD *)((char *)v8 + 76);
        v259 = (unsigned int)*(char *)(v256 + 177) >> 31;
        sub_6EAB60((_DWORD)v81, (v86 & 0x40) != 0, 0, (unsigned int)&v273, (unsigned int)&v295, v85, (__int64)v8);
        if ( !a3
          && (*((_BYTE *)v8 + 20) & 4) == 0
          && (unk_4D04A10 & 1) != 0
          && dword_4F04C64 != -1
          && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 2) != 0 )
        {
          sub_6E4620(v8);
        }
      }
      else
      {
        sub_6E6840(v8);
      }
      sub_82F1E0(v263, 1, v8);
    }
  }
  sub_6F69D0(v8, 28);
  v12 = *((_BYTE *)v8 + 16);
  if ( v12 == 6 )
  {
    v254 = v8[17];
    v295.m128i_i64[0] = *(_QWORD *)((char *)v8 + 76);
    if ( dword_4F077C4 == 2 && (v234 & 1) != 0 )
    {
      v259 = 0;
      v15 = 0;
      jj = 0;
      v14 = 1;
      goto LABEL_233;
    }
    sub_886080(v254);
    sub_656900(v254);
    if ( dword_4F077C4 == 2 )
      goto LABEL_116;
    if ( unk_4F07778 <= 199900 )
    {
      v102 = 4;
    }
    else
    {
      if ( dword_4D04964 )
      {
LABEL_116:
        if ( (unsigned int)sub_6E5430(v254, 28, v67, v68, v69, v70) )
          sub_6851A0(0x14u, (_DWORD *)v8 + 17, *(_QWORD *)(*(_QWORD *)v254 + 8LL));
        goto LABEL_118;
      }
      v102 = 5;
    }
    sub_6E5DE0(v102, 223, (char *)v8 + 68, *(_QWORD *)(*(_QWORD *)v254 + 8LL));
LABEL_118:
    sub_6EAB60(v254, 0, 0, v254 + 48, (unsigned int)&v295, v8[11], (__int64)v8);
    if ( *((_BYTE *)v8 + 16) )
    {
      v73 = *v8;
      for ( ii = *(_BYTE *)(*v8 + 140LL); ii == 12; ii = *(_BYTE *)(v73 + 140) )
        v73 = *(_QWORD *)(v73 + 160);
      if ( ii )
        sub_6F5FA0(v8, 0, 0, 1, v71, v72);
    }
    v20 = *(_QWORD *)(v254 + 88);
    for ( jj = *(_QWORD *)(v20 + 152); *(_BYTE *)(jj + 140) == 12; jj = *(_QWORD *)(jj + 160) )
      ;
    if ( dword_4F077C4 != 2 )
    {
      v75 = v8[11];
      sub_6E5A30(v75, 32, 4);
      v22 = 0;
      v19 = 0;
      v259 = 0;
      LODWORD(v254) = 0;
      goto LABEL_127;
    }
    v259 = 0;
    v15 = 0;
    goto LABEL_162;
  }
  if ( v12 == 3 )
  {
    v98 = v8[17];
    v254 = v98;
    if ( (*(_BYTE *)(v98 + 81) & 0x10) != 0 )
    {
      if ( *(_BYTE *)(v98 + 80) == 20 )
      {
        if ( dword_4F04C44 != -1
          || (v181 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v181 + 6) & 6) != 0)
          || *(_BYTE *)(v181 + 4) == 12 )
        {
          if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v254 + 88) + 176LL) + 174LL) == 3 )
          {
            jj = 0;
            sub_6F40C0(v8);
            v14 = 0;
            v15 = 0;
            v259 = 1;
            goto LABEL_233;
          }
        }
      }
      if ( !a3 )
      {
        v259 = 0;
        v15 = 0;
        jj = 0;
        v14 = 1;
        goto LABEL_233;
      }
      if ( (*(_BYTE *)(v254 + 83) & 0x20) != 0 )
      {
        v99 = *(_QWORD *)(v254 + 64);
        sub_878710(v254, &v295);
        if ( (v296.m128i_i8[1] & 0x40) == 0 )
        {
          v296.m128i_i8[0] &= ~0x80u;
          v296.m128i_i64[1] = 0;
        }
        v100 = v99;
        jj = 0;
        v101 = sub_7D2AC0(&v295, v100, 0);
        v14 = 1;
        v15 = 0;
        v259 = 0;
        LODWORD(v254) = v101;
        v8[17] = v101;
        goto LABEL_233;
      }
    }
    v259 = 0;
    v14 = 1;
    v15 = 0;
    jj = 0;
LABEL_233:
    v21 = 0;
    v20 = 0;
    v220 = v15;
    v239 = v14;
    sub_6E5A30(v8[11], 32, 4);
    v19 = v239;
    v22 = v220;
    goto LABEL_153;
  }
  if ( dword_4F077C4 == 2 )
  {
    if ( dword_4F04C44 != -1
      || (v13 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v13 + 6) & 6) != 0)
      || *(_BYTE *)(v13 + 4) == 12 )
    {
      if ( (unsigned int)sub_8D3D70(*v8)
        || (unsigned int)qword_4F077B4 | dword_4F077BC && (unsigned int)sub_8DBE70(*v8) && !unk_4D047A8 )
      {
        sub_6F40C0(v8);
        v14 = qword_4F077B4 | dword_4F077BC;
        if ( (unsigned int)qword_4F077B4 | dword_4F077BC )
        {
          if ( v259 )
          {
            if ( unk_4D047A8 )
            {
              v14 = 0;
              v15 = 0;
              jj = 0;
              LODWORD(v254) = 0;
            }
            else
            {
              jj = 0;
              sub_6F4B70(v8);
              *((_BYTE *)v8 + 18) &= ~1u;
              v15 = 0;
              LODWORD(v254) = 0;
              v14 = 0;
              v184 = *(_QWORD *)&dword_4D03B80;
              *v8 = *(_QWORD *)&dword_4D03B80;
              v8[34] = v184;
            }
            goto LABEL_233;
          }
          v14 = 0;
        }
        v259 = 1;
        v15 = 0;
        jj = 0;
        LODWORD(v254) = 0;
        goto LABEL_233;
      }
    }
  }
  sub_6FA3A0(v8);
  if ( (*((_BYTE *)v8 + 18) & 1) != 0 && (unsigned int)sub_8D3D10(*v8) )
  {
    v84 = sub_8D4870(*v8);
    v15 = 0;
    v259 = 0;
    jj = v84;
    goto LABEL_88;
  }
  if ( dword_4F077C4 == 2 )
  {
    if ( dword_4F04C44 != -1
      || (v63 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v63 + 6) & 6) != 0)
      || *(_BYTE *)(v63 + 4) == 12 )
    {
      if ( (unsigned int)sub_8D2EF0(*v8) )
      {
        v172 = sub_8D46C0(*v8);
        if ( (unsigned int)sub_8D3D40(v172) )
        {
          v259 = 1;
          goto LABEL_87;
        }
      }
    }
  }
  v259 = sub_6E9610(v8);
  if ( !v259 )
  {
LABEL_87:
    v15 = 0;
    jj = 0;
    goto LABEL_88;
  }
  v260 = v8[1];
  jj = sub_8D46C0(*v8);
  v103 = sub_6EB5C0(v8);
  v15 = v260;
  v20 = v103;
  if ( (*((_BYTE *)v8 + 18) & 8) != 0 || *((_BYTE *)v8 + 16) != 2 )
  {
    if ( dword_4F077C4 != 2 )
    {
      v75 = v8[11];
      sub_6E5A30(v75, 32, 4);
      v22 = v260;
      if ( !v20 )
      {
        LODWORD(v254) = 0;
        v21 = 0;
        v19 = 0;
        v259 = 0;
        goto LABEL_153;
      }
      v259 = 0;
      v19 = 0;
      LODWORD(v254) = 0;
      goto LABEL_127;
    }
    v259 = 0;
    goto LABEL_90;
  }
  v259 = 0;
LABEL_88:
  if ( dword_4F077C4 != 2 )
  {
    v21 = 0;
    v20 = 0;
    v253 = v15;
    sub_6E5A30(v8[11], 32, 4);
    v22 = v253;
    v19 = 0;
    LODWORD(v254) = 0;
    goto LABEL_153;
  }
  v20 = 0;
LABEL_90:
  if ( dword_4F04C44 != -1
    || (v120 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v120 + 6) & 6) != 0)
    || *(_BYTE *)(v120 + 4) == 12 )
  {
    if ( v20 )
      goto LABEL_162;
    if ( jj )
    {
      v255 = v15;
      v64 = sub_8DBE70(jj);
      v15 = v255;
      v14 = v64;
      if ( v64 )
      {
        v259 = 1;
        v14 = 0;
        jj = 0;
      }
      LODWORD(v254) = 0;
      goto LABEL_233;
    }
  }
  if ( !v20 )
  {
    LODWORD(v254) = 0;
    v14 = 0;
    goto LABEL_233;
  }
LABEL_162:
  v254 = *(_QWORD *)(v20 + 216);
  if ( v254 )
  {
LABEL_163:
    v87 = 1;
    v254 = *(_QWORD *)v20;
    goto LABEL_164;
  }
  v87 = 0;
  if ( !jj )
    goto LABEL_164;
  if ( *(_BYTE *)(jj + 140) == 12 )
  {
    v147 = jj;
    do
      v147 = *(_QWORD *)(v147 + 160);
    while ( *(_BYTE *)(v147 + 140) == 12 );
    if ( (*(_BYTE *)(*(_QWORD *)(v147 + 168) + 20LL) & 2) != 0 )
      goto LABEL_163;
    v148 = jj;
    do
      v148 = *(_QWORD *)(v148 + 160);
    while ( *(_BYTE *)(v148 + 140) == 12 );
    v149 = *(__int64 **)(v148 + 168);
  }
  else
  {
    v149 = *(__int64 **)(jj + 168);
    if ( (*((_BYTE *)v149 + 20) & 2) != 0 )
      goto LABEL_163;
  }
  v150 = *v149;
  if ( !v150 )
  {
    LODWORD(v254) = 0;
    v87 = 0;
    goto LABEL_164;
  }
  v87 = 0;
  if ( (*(_BYTE *)(v150 + 35) & 1) != 0 )
    goto LABEL_163;
LABEL_164:
  v75 = v8[11];
  v218 = v15;
  v237 = v87;
  sub_6E5A30(v75, 32, 4);
  v19 = v237;
  v22 = v218;
LABEL_127:
  v21 = 0;
  if ( *(_BYTE *)(v20 + 174) )
    goto LABEL_153;
  v80 = *(_WORD *)(v20 + 176);
  if ( !v80 )
    goto LABEL_153;
  v280 = 0;
  v279 = sub_6BBC40;
  v281 = 0;
  v282 = 0;
  v283 = 0x100000000LL;
  v284 = 0;
  v285 = 0u;
  memset(v286, 0, 12);
  if ( v80 <= 0x3D23u )
  {
    if ( v80 > 0x3CDAu )
    {
      switch ( v80 )
      {
        case 0x3CDBu:
        case 0x3D05u:
          LODWORD(v283) = 1;
          LODWORD(v284) = 1;
          v279 = sub_68BF70;
          goto LABEL_186;
        case 0x3CDDu:
          goto LABEL_185;
        case 0x3CDEu:
          v228 = v22;
          v247 = v19;
          LODWORD(v284) = 1;
          v137 = sub_72CBE0(v75, 0x100000000LL, (unsigned __int16)(v80 - 15579), v77, v78, v79);
          jj = v281;
          v19 = v247;
          v280 = v137;
          v22 = v228;
          v283 = 0x200000002LL;
          v123 = HIDWORD(v284);
          goto LABEL_284;
        case 0x3CE5u:
        case 0x3CE6u:
          v225 = v22;
          v244 = v19;
          if ( v80 == 15590 )
          {
            v281 = sub_7D3810(1);
            v127 = sub_727560();
            v19 = v244;
            v128 = 11;
          }
          else
          {
            v281 = sub_7D3810(2);
            v127 = sub_727560();
            v19 = v244;
            v128 = 12;
          }
          v22 = v225;
          v282 = v127;
          *(_BYTE *)(v127 + 32) = v128;
          jj = v281;
          v279 = 0;
          v123 = HIDWORD(v284);
          goto LABEL_284;
        case 0x3CEAu:
        case 0x3CEFu:
LABEL_303:
          LODWORD(v283) = 1;
          v279 = sub_68C540;
          goto LABEL_186;
        case 0x3D1Bu:
        case 0x3D1Cu:
        case 0x3D1Du:
        case 0x3D1Eu:
        case 0x3D1Fu:
        case 0x3D20u:
        case 0x3D21u:
        case 0x3D22u:
        case 0x3D23u:
          LODWORD(v283) = 1;
          LODWORD(v286[0]) = 1;
          v279 = sub_68E940;
          goto LABEL_186;
        default:
          goto LABEL_152;
      }
    }
    if ( v80 > 0x10B1u )
    {
      if ( v80 != 4452 )
      {
        switch ( v80 )
        {
          case 0x11EBu:
            goto LABEL_320;
          case 0x11ECu:
          case 0x11EDu:
          case 0x11EEu:
          case 0x11EFu:
          case 0x11F0u:
          case 0x11F1u:
          case 0x11F2u:
          case 0x11F3u:
          case 0x11F4u:
          case 0x11F5u:
          case 0x11F6u:
          case 0x11F7u:
          case 0x11F8u:
          case 0x11F9u:
          case 0x11FAu:
          case 0x11FBu:
          case 0x11FCu:
          case 0x11FDu:
          case 0x11FEu:
          case 0x11FFu:
          case 0x1200u:
          case 0x1226u:
          case 0x1227u:
          case 0x1228u:
          case 0x1229u:
          case 0x122Au:
          case 0x122Bu:
          case 0x122Cu:
          case 0x122Du:
          case 0x122Eu:
          case 0x122Fu:
          case 0x1230u:
          case 0x1231u:
          case 0x1232u:
          case 0x1233u:
          case 0x1234u:
          case 0x1235u:
          case 0x1236u:
          case 0x1237u:
          case 0x1238u:
          case 0x1239u:
          case 0x123Au:
          case 0x123Bu:
          case 0x123Cu:
          case 0x123Du:
          case 0x123Eu:
          case 0x123Fu:
          case 0x1240u:
          case 0x1241u:
          case 0x1242u:
          case 0x1243u:
          case 0x1244u:
          case 0x1245u:
          case 0x1246u:
          case 0x1247u:
          case 0x1248u:
          case 0x1249u:
          case 0x124Au:
          case 0x124Bu:
          case 0x124Cu:
          case 0x124Du:
          case 0x124Eu:
          case 0x124Fu:
          case 0x1250u:
          case 0x1251u:
          case 0x1252u:
          case 0x1253u:
          case 0x1254u:
          case 0x1255u:
          case 0x1256u:
          case 0x1257u:
          case 0x1258u:
          case 0x1259u:
          case 0x125Au:
          case 0x125Bu:
          case 0x125Cu:
          case 0x125Du:
          case 0x125Eu:
          case 0x125Fu:
          case 0x1260u:
          case 0x1261u:
          case 0x1262u:
          case 0x1263u:
          case 0x1264u:
          case 0x1265u:
          case 0x1266u:
          case 0x1267u:
          case 0x1268u:
          case 0x1269u:
          case 0x126Au:
          case 0x126Bu:
          case 0x126Cu:
          case 0x126Du:
          case 0x126Eu:
          case 0x126Fu:
          case 0x1270u:
          case 0x1271u:
          case 0x1272u:
          case 0x1273u:
          case 0x1274u:
          case 0x1275u:
          case 0x1276u:
          case 0x1277u:
          case 0x1278u:
          case 0x1279u:
          case 0x127Au:
          case 0x127Bu:
          case 0x127Cu:
          case 0x127Du:
          case 0x127Eu:
          case 0x127Fu:
          case 0x1280u:
          case 0x1281u:
            goto LABEL_152;
          case 0x1201u:
          case 0x1202u:
          case 0x1204u:
          case 0x1205u:
          case 0x1207u:
          case 0x1208u:
          case 0x1209u:
          case 0x120Bu:
          case 0x120Cu:
          case 0x120Du:
          case 0x120Eu:
          case 0x120Fu:
          case 0x1212u:
          case 0x1213u:
          case 0x1214u:
          case 0x1219u:
          case 0x121Au:
          case 0x121Cu:
          case 0x121Du:
          case 0x121Eu:
          case 0x121Fu:
          case 0x1220u:
          case 0x1221u:
          case 0x1223u:
          case 0x1224u:
          case 0x1225u:
            LODWORD(v283) = 1;
            HIDWORD(v285) = 1;
            v279 = sub_68E940;
            goto LABEL_186;
          case 0x1203u:
          case 0x1206u:
          case 0x120Au:
          case 0x1211u:
          case 0x1215u:
          case 0x1216u:
          case 0x1217u:
          case 0x1218u:
          case 0x121Bu:
          case 0x1222u:
            LODWORD(v283) = 2;
            HIDWORD(v285) = 1;
            v279 = sub_68E940;
            goto LABEL_186;
          case 0x1210u:
            LODWORD(v283) = 3;
            HIDWORD(v285) = 1;
            v279 = sub_68E940;
            goto LABEL_186;
          case 0x1282u:
            goto LABEL_303;
          default:
            goto LABEL_153;
        }
        goto LABEL_153;
      }
LABEL_320:
      LODWORD(v283) = 2;
      v279 = sub_68C540;
    }
    else
    {
      if ( v80 > 0x1047u )
      {
        switch ( v80 )
        {
          case 0x1048u:
          case 0x104Fu:
          case 0x1065u:
          case 0x106Bu:
          case 0x1073u:
          case 0x1079u:
          case 0x107Fu:
          case 0x1085u:
          case 0x1095u:
          case 0x109Bu:
          case 0x10A9u:
          case 0x10B1u:
            LODWORD(v283) = 3;
            DWORD1(v285) = 1;
            goto LABEL_186;
          case 0x1056u:
            v226 = v22;
            v245 = v19;
            v140 = sub_72C390();
            LODWORD(v283) = 6;
            v280 = v140;
            goto LABEL_305;
          case 0x105Cu:
            v232 = v22;
            v251 = v19;
            v143 = sub_72C390();
            jj = v281;
            LODWORD(v283) = 6;
            v280 = v143;
            v19 = v251;
            HIDWORD(v286[0]) = 1;
            v123 = HIDWORD(v284);
            DWORD1(v285) = 1;
            v22 = v232;
            goto LABEL_284;
          case 0x105Du:
            v226 = v22;
            v245 = v19;
            v142 = sub_72CBE0(v75, 0x100000000LL, v76, v77, v78, v79);
            LODWORD(v283) = 4;
            v280 = v142;
            goto LABEL_305;
          case 0x1063u:
            LODWORD(v283) = 3;
            DWORD1(v285) = 1;
            HIDWORD(v286[0]) = 1;
            goto LABEL_186;
          case 0x108Cu:
          case 0x10A2u:
            v226 = v22;
            v245 = v19;
            v129 = sub_72CBE0(v75, 0x100000000LL, v76, v77, v78, v79);
            LODWORD(v283) = 3;
            v280 = v129;
            goto LABEL_305;
          case 0x1092u:
            LODWORD(v283) = 2;
            HIDWORD(v286[0]) = 1;
            DWORD1(v285) = 1;
            goto LABEL_186;
          case 0x10A8u:
            v231 = v22;
            v250 = v19;
            v141 = sub_72CBE0(v75, 0x100000000LL, v76, v77, v78, v79);
            jj = v281;
            LODWORD(v283) = 3;
            v280 = v141;
            v19 = v250;
            DWORD1(v285) = 1;
            v123 = HIDWORD(v284);
            HIDWORD(v286[0]) = 1;
            v22 = v231;
            goto LABEL_284;
          default:
            goto LABEL_152;
        }
      }
      if ( v80 <= 0x75u )
      {
        if ( v80 <= 0x65u )
          goto LABEL_153;
        switch ( v80 )
        {
          case 'f':
          case 'g':
            v221 = v22;
            v240 = v19;
            HIDWORD(v284) = 1;
            v122 = sub_72C390();
            jj = v281;
            LODWORD(v283) = 5;
            v280 = v122;
            v19 = v240;
            v123 = HIDWORD(v284);
            v22 = v221;
            goto LABEL_284;
          case 'h':
          case 'i':
          case 'j':
          case 'n':
          case 'o':
          case 'p':
            HIDWORD(v284) = 1;
            LODWORD(v283) = 3;
            break;
          case 'q':
            v223 = v22;
            v242 = v19;
            HIDWORD(v284) = 1;
            v125 = sub_72CBE0(v75, 0x100000000LL, v76, v77, v78, v79);
            jj = v281;
            LODWORD(v283) = 2;
            v280 = v125;
            v19 = v242;
            v123 = HIDWORD(v284);
            v22 = v223;
            goto LABEL_284;
          case 's':
            HIDWORD(v284) = 1;
            LODWORD(v283) = 2;
            break;
          case 'u':
            v224 = v22;
            v243 = v19;
            HIDWORD(v284) = 1;
            v126 = sub_72CBE0(v75, 0x100000000LL, v76, v77, v78, v79);
            jj = v281;
            LODWORD(v283) = 3;
            v280 = v126;
            v19 = v243;
            v123 = HIDWORD(v284);
            v22 = v224;
            goto LABEL_284;
          default:
            goto LABEL_152;
        }
        LODWORD(v284) = 1;
        goto LABEL_186;
      }
      if ( v80 != 3484 )
      {
        if ( v80 > 0xD9Cu )
        {
          if ( v80 != 3489 )
          {
            v21 = 0;
            if ( (unsigned __int16)(v80 - 3495) > 1u )
              goto LABEL_153;
          }
LABEL_185:
          LODWORD(v284) = 1;
          LODWORD(v283) = 1;
          goto LABEL_186;
        }
        if ( v80 != 3431 )
        {
LABEL_152:
          v21 = 0;
          goto LABEL_153;
        }
      }
      LODWORD(v284) = 1;
      LODWORD(v283) = 2;
    }
LABEL_186:
    v21 = &v279;
    v20 = 0;
    jj = 0;
    goto LABEL_153;
  }
  if ( v80 > 0x6286u )
    goto LABEL_153;
  if ( v80 > 0x6240u )
  {
    switch ( v80 )
    {
      case 0x6241u:
        v222 = v22;
        v241 = v19;
        v135 = sub_72CBE0(v75, 0x100000000LL, v76, v77, v78, v79);
        LODWORD(v283) = 4;
        v280 = v135;
        goto LABEL_310;
      case 0x6242u:
        LODWORD(v283) = 3;
        DWORD1(v285) = 1;
        *(_QWORD *)((char *)v286 + 4) = 0x100000001LL;
        goto LABEL_186;
      case 0x6248u:
        v226 = v22;
        v245 = v19;
        v134 = sub_72CBE0(v75, 0x100000000LL, v76, v77, v78, v79);
        LODWORD(v283) = 4;
        v280 = v134;
        goto LABEL_308;
      case 0x6249u:
        v222 = v22;
        v241 = v19;
        HIDWORD(v286[0]) = 1;
        v133 = sub_72CBE0(v75, 0x100000000LL, v76, v77, v78, v79);
        LODWORD(v283) = 4;
        v280 = v133;
        DWORD1(v285) = 1;
        goto LABEL_294;
      case 0x624Fu:
      case 0x6257u:
      case 0x625Fu:
      case 0x6263u:
      case 0x6267u:
      case 0x626Bu:
      case 0x6273u:
        goto LABEL_292;
      case 0x6250u:
      case 0x6258u:
      case 0x6260u:
      case 0x6264u:
      case 0x6268u:
      case 0x626Cu:
      case 0x6274u:
        v222 = v22;
        v241 = v19;
        v124 = sub_72CBE0(v75, 0x100000000LL, v76, v77, v78, v79);
        LODWORD(v283) = 4;
        v280 = v124;
        DWORD1(v285) = 1;
        goto LABEL_294;
      case 0x627Bu:
        v222 = v22;
        v241 = v19;
        v132 = sub_72CBE0(v75, 0x100000000LL, v76, v77, v78, v79);
        LODWORD(v283) = 5;
        v280 = v132;
LABEL_310:
        *(_QWORD *)((char *)&v285 + 4) = 0x100000001LL;
        goto LABEL_294;
      case 0x627Cu:
        HIDWORD(v286[0]) = 1;
LABEL_292:
        LODWORD(v283) = 4;
        DWORD1(v285) = 1;
        LODWORD(v286[1]) = 1;
        goto LABEL_186;
      case 0x6280u:
        v226 = v22;
        v245 = v19;
        v131 = sub_72C390();
        LODWORD(v283) = 7;
        v280 = v131;
LABEL_308:
        LODWORD(v286[1]) = 1;
LABEL_305:
        jj = v281;
        v19 = v245;
        *(_QWORD *)((char *)&v285 + 4) = 0x100000001LL;
        v22 = v226;
        v123 = HIDWORD(v284);
        break;
      case 0x6281u:
        v227 = v22;
        v246 = v19;
        v130 = sub_72C390();
        jj = v281;
        LODWORD(v283) = 7;
        v280 = v130;
        v19 = v246;
        *(_QWORD *)((char *)v286 + 4) = 0x100000001LL;
        v22 = v227;
        DWORD1(v285) = 1;
        v123 = HIDWORD(v284);
        break;
      case 0x6286u:
        v222 = v22;
        v241 = v19;
        v136 = sub_72CBE0(v75, 0x100000000LL, v76, v77, v78, v79);
        LODWORD(v283) = 2;
        v280 = v136;
        DWORD1(v285) = 1;
LABEL_294:
        LODWORD(v286[1]) = 1;
        v123 = HIDWORD(v284);
        jj = v281;
        v19 = v241;
        v22 = v222;
        break;
      default:
        goto LABEL_152;
    }
LABEL_284:
    if ( v123 )
      LODWORD(v284) = 1;
    if ( jj )
    {
      LODWORD(v254) = jj;
      v20 = 0;
      v19 = 1;
      jj = 0;
      *((_BYTE *)v8 + 18) |= 0x40u;
      v21 = &v279;
      v234 = 0;
    }
    else
    {
      v20 = 0;
      v21 = &v279;
    }
    goto LABEL_153;
  }
  if ( v80 > 0x6136u )
  {
    if ( v80 != 24989 )
      goto LABEL_153;
    v279 = sub_68DF50;
    goto LABEL_186;
  }
  if ( v80 > 0x60D0u )
  {
    switch ( v80 )
    {
      case 0x60D1u:
      case 0x60D7u:
      case 0x60E3u:
      case 0x60E9u:
      case 0x60F1u:
      case 0x60F7u:
      case 0x60FDu:
      case 0x6105u:
      case 0x6111u:
      case 0x6117u:
      case 0x611Du:
      case 0x6123u:
      case 0x6129u:
      case 0x6136u:
        LODWORD(v283) = 2;
        LODWORD(v285) = 1;
        goto LABEL_186;
      case 0x60DDu:
        v229 = v22;
        v248 = v19;
        v138 = sub_72C390();
        jj = v281;
        LODWORD(v283) = 3;
        v280 = v138;
        v19 = v248;
        LODWORD(v285) = 1;
        v123 = HIDWORD(v284);
        v22 = v229;
        goto LABEL_284;
      case 0x610Bu:
        v230 = v22;
        v249 = v19;
        v139 = sub_72CBE0(v75, 0x100000000LL, v76, v77, v78, v79);
        jj = v281;
        LODWORD(v283) = 1;
        v280 = v139;
        v19 = v249;
        LODWORD(v285) = 1;
        v123 = HIDWORD(v284);
        v22 = v230;
        goto LABEL_284;
      case 0x6130u:
        LODWORD(v283) = 3;
        v21 = &v279;
        v20 = 0;
        jj = 0;
        LODWORD(v285) = 1;
        break;
      default:
        goto LABEL_152;
    }
    goto LABEL_153;
  }
  if ( v80 == 16688 )
  {
    v219 = v22;
    v238 = v19;
    v90 = sub_73D220();
    v19 = v238;
    v22 = v219;
    if ( !(_DWORD)qword_4F077B4
      || (v216 = v219,
          v233 = v238,
          v252 = *(_QWORD *)(*(_QWORD *)(v20 + 152) + 160LL),
          v144 = sub_73C570(v90, 1, -1),
          v145 = sub_8DA9C0(v252, v144),
          v19 = v233,
          v22 = v216,
          v145) )
    {
      v21 = 0;
      if ( HIDWORD(v284) )
        LODWORD(v284) = 1;
      goto LABEL_153;
    }
    v146 = sub_72D2E0(v144, 0);
    jj = v281;
    v19 = v233;
    v280 = v146;
    v22 = v216;
    v279 = sub_689FC0;
    v123 = HIDWORD(v284);
    goto LABEL_284;
  }
LABEL_153:
  if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 1) != 0 && a3 )
    goto LABEL_26;
  if ( !v22 )
    v22 = jj;
  if ( v21 )
  {
    v18 = 1;
    v213 = 0;
    v212 = 0;
    if ( !*v21 )
      v18 = v19;
    v236 = 0;
  }
  else
  {
    v213 = 0;
    v18 = v19;
    v212 = 0;
    v236 = 0;
  }
LABEL_34:
  v23 = 1;
  if ( *((_BYTE *)v8 + 16) )
  {
    v24 = *v8;
    for ( kk = *(_BYTE *)(*v8 + 140LL); kk == 12; kk = *(_BYTE *)(v24 + 140) )
      v24 = *(_QWORD *)(v24 + 160);
    v23 = kk == 0;
  }
  v217 = v19;
  sub_6C0910(v22, v20, v236, &v270, v18, v259, v23, 0, (__int64)a3, 0, 0, &v271, 0, 0, &v277);
  v30 = v217;
  if ( v257 && ((*(_BYTE *)(qword_4D03C50 + 19LL) & 1) != 0 || a3[56]) )
    goto LABEL_26;
  v31 = (__int64)dword_4F07508;
  *(_QWORD *)dword_4F07508 = v273;
  if ( v21 && *v21 )
  {
    v32 = sub_67D840(68);
    sub_67D850(68, 3, 0);
    v31 = 68;
    v20 = ((__int64 (__fastcall *)(_QWORD *, __int64, __int64 *, void **, __int64 *))*v21)(v8, v271, &v277, v21, &v270);
    sub_67D850(68, v32, 0);
    v30 = v217;
    if ( v20 )
    {
      jj = *(_QWORD *)(v20 + 152);
      goto LABEL_45;
    }
    sub_6E6000(68, v32, v26, v27, v28, v29);
LABEL_26:
    sub_6E6260(a4);
    return sub_6E1990(v271);
  }
LABEL_45:
  if ( v30 )
  {
    v33 = _mm_loadu_si128((const __m128i *)v8 + 1);
    v34 = _mm_loadu_si128((const __m128i *)v8 + 2);
    v35 = _mm_loadu_si128(xmmword_4F07340);
    v36 = _mm_loadu_si128(&xmmword_4F07340[1]);
    v295 = _mm_loadu_si128((const __m128i *)v8);
    v37 = _mm_loadu_si128((const __m128i *)v8 + 3);
    v296 = v33;
    v38 = _mm_loadu_si128((const __m128i *)v8 + 4);
    v39 = _mm_loadu_si128((const __m128i *)v8 + 5);
    v290 = v35;
    v292 = xmmword_4F07340[2].m128i_i64[0];
    v40 = _mm_loadu_si128((const __m128i *)v8 + 6);
    v41 = *((char *)v8 + 18);
    v291 = v36;
    v42 = _mm_loadu_si128((const __m128i *)v8 + 8);
    v43 = _mm_loadu_si128((const __m128i *)v8 + 7);
    v44 = *((_BYTE *)v8 + 16);
    v297 = v34;
    v45 = v41;
    v298 = v37;
    v46 = v41 >> 31;
    v299 = v38;
    v300 = v39;
    v301 = v40;
    v302 = v43;
    v303 = v42;
    if ( v44 == 2 )
    {
      v104 = _mm_loadu_si128((const __m128i *)v8 + 10);
      v105 = _mm_loadu_si128((const __m128i *)v8 + 11);
      v106 = _mm_loadu_si128((const __m128i *)v8 + 12);
      v304 = _mm_loadu_si128((const __m128i *)v8 + 9);
      v107 = _mm_loadu_si128((const __m128i *)v8 + 13);
      v108 = _mm_loadu_si128((const __m128i *)v8 + 21);
      v305 = v104;
      v109 = _mm_loadu_si128((const __m128i *)v8 + 14);
      v306 = v105;
      v110 = _mm_loadu_si128((const __m128i *)v8 + 15);
      v307 = v106;
      v111 = _mm_loadu_si128((const __m128i *)v8 + 16);
      v308 = v107;
      v112 = _mm_loadu_si128((const __m128i *)v8 + 17);
      v309 = v109;
      v113 = _mm_loadu_si128((const __m128i *)v8 + 18);
      v310 = v110;
      v114 = _mm_loadu_si128((const __m128i *)v8 + 19);
      v311 = v111;
      v115 = _mm_loadu_si128((const __m128i *)v8 + 20);
      v312 = v112;
      v313 = v113;
      v314 = v114;
      v315 = v115;
      v316 = v108;
    }
    else if ( v44 == 5 || v44 == 1 )
    {
      v304.m128i_i64[0] = v8[18];
    }
    v47 = 0;
    if ( (*((_BYTE *)v8 + 20) & 4) != 0 )
    {
      v48 = _mm_loadu_si128((const __m128i *)(v8 + 3));
      v47 = v30;
      v49 = _mm_loadu_si128((const __m128i *)(v8 + 5));
      v292 = v8[7];
      v290 = v48;
      v291 = v49;
    }
    if ( (v45 & 1) == 0 )
      v30 = v213;
    if ( (unsigned int)sub_84C4B0(
                         v254,
                         (*((_BYTE *)v8 + 19) & 8) != 0,
                         v8[13],
                         v30,
                         (_DWORD)v263,
                         (unsigned int)&v271,
                         v234,
                         0,
                         0,
                         v213,
                         0,
                         0,
                         v46,
                         (__int64)&v295,
                         (__int64)&v273,
                         v266,
                         (__int64)&v277,
                         (__int64)&v268,
                         (__int64)v8,
                         (__int64)&v270) )
    {
      if ( v47 )
      {
        v50 = v292;
        v51 = _mm_loadu_si128(&v290);
        v52 = _mm_loadu_si128(&v291);
        *((_BYTE *)v8 + 20) |= 4u;
        v8[7] = v50;
        *(__m128i *)(v8 + 3) = v51;
        *(__m128i *)(v8 + 5) = v52;
      }
    }
    else
    {
      sub_6E6260(v8);
    }
    if ( v236 )
    {
LABEL_57:
      *a4 = _mm_loadu_si128((const __m128i *)v8);
      a4[1] = _mm_loadu_si128((const __m128i *)v8 + 1);
      a4[2] = _mm_loadu_si128((const __m128i *)v8 + 2);
      a4[3] = _mm_loadu_si128((const __m128i *)v8 + 3);
      a4[4] = _mm_loadu_si128((const __m128i *)v8 + 4);
      a4[5] = _mm_loadu_si128((const __m128i *)v8 + 5);
      a4[6] = _mm_loadu_si128((const __m128i *)v8 + 6);
      a4[7] = _mm_loadu_si128((const __m128i *)v8 + 7);
      a4[8] = _mm_loadu_si128((const __m128i *)v8 + 8);
      v53 = *((_BYTE *)v8 + 16);
      switch ( v53 )
      {
        case 2:
          a4[9] = _mm_loadu_si128((const __m128i *)v8 + 9);
          a4[10] = _mm_loadu_si128((const __m128i *)v8 + 10);
          a4[11] = _mm_loadu_si128((const __m128i *)v8 + 11);
          a4[12] = _mm_loadu_si128((const __m128i *)v8 + 12);
          a4[13] = _mm_loadu_si128((const __m128i *)v8 + 13);
          a4[14] = _mm_loadu_si128((const __m128i *)v8 + 14);
          a4[15] = _mm_loadu_si128((const __m128i *)v8 + 15);
          a4[16] = _mm_loadu_si128((const __m128i *)v8 + 16);
          a4[17] = _mm_loadu_si128((const __m128i *)v8 + 17);
          a4[18] = _mm_loadu_si128((const __m128i *)v8 + 18);
          a4[19] = _mm_loadu_si128((const __m128i *)v8 + 19);
          a4[20] = _mm_loadu_si128((const __m128i *)v8 + 20);
          a4[21] = _mm_loadu_si128((const __m128i *)v8 + 21);
          break;
        case 5:
          a4[9].m128i_i64[0] = v8[18];
          break;
        case 1:
          a4[9].m128i_i64[0] = v8[18];
          break;
      }
      goto LABEL_61;
    }
    goto LABEL_172;
  }
  if ( v236 )
  {
    if ( v270 )
    {
      if ( (unsigned int)sub_6E5430(v31, v236, v26, v27, v28, v29) )
        sub_6851C0(0x8Cu, &v274);
      sub_6E6840(v8);
    }
    goto LABEL_57;
  }
  if ( jj && (*((_BYTE *)v8 + 18) & 1) != 0 )
  {
    v261 = sub_82EAF0(jj, 0, 0);
    sub_839CB0(v263, v20, jj, v261);
    if ( v295.m128i_i32[2] != 7 )
    {
      sub_82F0D0(&v295, (char *)v263[4].m128i_i64 + 4);
      sub_831410(jj, v263);
      goto LABEL_172;
    }
    if ( (unsigned int)sub_6E5430(v263, v20, v116, v117, v118, v119) )
    {
      if ( !(unsigned int)sub_6E5430(v263, v20, v173, v174, v175, v176) )
      {
LABEL_402:
        sub_6E6840(v263);
        goto LABEL_172;
      }
      v177 = v296.m128i_i8[4];
      v178 = (FILE *)((char *)v263[4].m128i_i64 + 4);
      if ( v296.m128i_i8[4] == 2 )
      {
        if ( v263[1].m128i_i8[1] != 1 )
          goto LABEL_399;
        if ( !(unsigned int)sub_6ED0A0(v263) )
        {
          if ( v20 )
            sub_6854C0(0xA49u, v178, *(_QWORD *)v20);
          else
            sub_6851C0(0xA4Bu, v178);
          goto LABEL_402;
        }
        v177 = v296.m128i_i8[4];
      }
      if ( v177 == 1 && (v263[1].m128i_i8[1] == 2 || (unsigned int)sub_6ED0A0(v263)) && !(unsigned int)sub_8D4D20(v261) )
      {
        if ( v20 )
          sub_6854C0(0xA4Au, v178, *(_QWORD *)v20);
        else
          sub_6851C0(0xA4Cu, v178);
        goto LABEL_402;
      }
LABEL_399:
      if ( v20 )
        v179 = (FILE *)sub_67E020(0x43Eu, v178, *(_QWORD *)v20);
      else
        v179 = (FILE *)sub_67D9D0(0x13Bu, v178);
      sub_82E460(v263->m128i_i64[0], v179);
      sub_685910((__int64)v179, v179);
      goto LABEL_402;
    }
  }
LABEL_172:
  v88 = sub_6EB5C0(v8);
  v89 = 0;
  if ( v88 && v212 && *(_BYTE *)(v88 + 174) == 5 )
    v89 = *(_BYTE *)(v88 + 176) == 42;
  v290.m128i_i64[0] = 0;
  sub_7022F0(
    (_DWORD)v8,
    (_DWORD)v263,
    v270,
    0,
    v215,
    v214,
    v268,
    v89,
    (__int64)&v275,
    (__int64)&v276,
    (__int64)&v277,
    (__int64)a4,
    (__int64)&v267,
    (__int64)&v290);
  if ( v290.m128i_i64[0] )
    sub_6E3AC0(v290.m128i_i64[0], &v276, v266, &v277);
LABEL_61:
  if ( v235 )
  {
    v54 = *(_BYTE *)(qword_4D03C50 + 20LL);
    if ( (v54 & 0x20) == 0 )
      *(_BYTE *)(qword_4D03C50 + 20LL) = v54 | 0x20;
  }
  v55 = (__int64)a4;
  a4[4].m128i_i32[1] = v275;
  a4[4].m128i_i16[4] = WORD2(v275);
  *(_QWORD *)dword_4F07508 = *(__int64 *)((char *)a4[4].m128i_i64 + 4);
  v56 = v277;
  *(__int64 *)((char *)&a4[4].m128i_i64[1] + 4) = v277;
  unk_4F061D8 = v56;
  sub_6E3280(a4, &v276);
  if ( !v267 )
  {
    v55 = 2;
    sub_6E26D0(2, a4);
  }
  if ( v21 )
  {
    v57 = v21[3];
    if ( !v57 || !a4[1].m128i_i8[0] )
      goto LABEL_75;
    v58 = a4->m128i_i64[0];
    for ( mm = *(_BYTE *)(a4->m128i_i64[0] + 140); mm == 12; mm = *(_BYTE *)(v58 + 140) )
      v58 = *(_QWORD *)(v58 + 160);
    if ( !mm )
      goto LABEL_75;
    v60 = *(_QWORD *)(a4[9].m128i_i64[0] + 72);
    v61 = *(_BYTE *)(v60 + 24);
    if ( v61 == 2 || v61 == 20 )
    {
      *(_QWORD *)(v60 + 64) = v57;
LABEL_75:
      v62 = v21[1];
      if ( v62 )
        sub_6FC3F0(v62, a4, 1);
      return sub_6E1990(v271);
    }
LABEL_512:
    sub_721090(v55);
  }
  return sub_6E1990(v271);
}
