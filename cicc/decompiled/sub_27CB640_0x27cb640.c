// Function: sub_27CB640
// Address: 0x27cb640
//
__int64 __fastcall sub_27CB640(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // r12
  __int64 v8; // r14
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  char **v14; // r13
  char **v15; // r12
  int v16; // ebx
  char *v17; // rdi
  __int64 v18; // r8
  __int64 v19; // r9
  unsigned __int64 *v20; // rax
  __int64 v21; // rsi
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rcx
  unsigned __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rax
  unsigned __int64 v27; // rcx
  __int64 v28; // r14
  __int64 *v29; // rdi
  unsigned int v30; // edx
  unsigned __int64 *v31; // rax
  __int64 *v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r12
  __int64 *v35; // r13
  unsigned __int64 v36; // rbx
  __int64 v37; // r12
  unsigned __int64 v38; // r13
  __int64 v39; // r15
  __int64 v40; // rsi
  _QWORD *v41; // rax
  _QWORD *v42; // rdx
  int v43; // r8d
  __m128i *v44; // rdx
  __m128i v45; // xmm4
  __m128i v46; // xmm5
  __m128i v47; // xmm6
  __m128i v48; // xmm7
  __m128i v49; // xmm3
  __int64 *v50; // r12
  _QWORD *v51; // rax
  __int64 *v52; // r15
  _QWORD *v53; // r12
  __int64 (__fastcall *v54)(__int64, __int64 *, __int64, const __m128i *); // rax
  __int64 v55; // rax
  __int64 *v56; // r13
  __int64 v57; // r10
  __int64 *v58; // rax
  __int64 v59; // r13
  __int64 v60; // rax
  __int64 v61; // rdi
  __int64 v62; // r14
  __int64 v63; // r13
  __int64 *v64; // r13
  __int64 v66; // rax
  int v67; // eax
  _QWORD *v68; // rax
  __m128i *v69; // rdx
  __int64 v70; // r13
  __m128i si128; // xmm0
  void *v72; // rdx
  __int64 v73; // rdi
  __int64 v74; // rax
  __m128i *v75; // rdx
  __m128i v76; // xmm0
  __int64 *v77; // r12
  __int64 v78; // rbx
  __int64 *v79; // rdi
  __int64 v80; // rsi
  _QWORD *v81; // r13
  __int64 v82; // rdx
  __int64 v83; // rcx
  __int64 v84; // r8
  _QWORD *v85; // rax
  __int64 v86; // r10
  unsigned __int32 v87; // r8d
  unsigned __int64 v88; // rdx
  unsigned __int32 v89; // r8d
  __int64 v90; // r10
  __int64 v91; // r10
  __int64 v92; // rax
  __int64 v93; // rcx
  __int64 *v94; // r14
  _QWORD *v95; // rax
  __int64 v96; // rcx
  unsigned __int64 v97; // rax
  _QWORD *v98; // rax
  __int64 *v99; // rax
  __int64 v100; // rcx
  __int64 *v101; // r9
  _QWORD *v102; // rax
  __int64 v103; // rcx
  unsigned __int64 v104; // rax
  _QWORD *v105; // rax
  __int64 *v106; // rax
  __int64 *v107; // r9
  __int64 *v108; // r14
  __int64 v109; // rax
  const __m128i *v110; // r14
  __m128i *v111; // rax
  __int64 v112; // rdx
  __int64 v113; // rcx
  __int64 v114; // r8
  __int8 v115; // r15
  __int64 *v116; // rax
  _QWORD *v117; // r15
  __int64 *v118; // rax
  _QWORD *v119; // r12
  __int8 v120; // bl
  _QWORD *v121; // rax
  _QWORD *v122; // r8
  __int64 v123; // rcx
  __int64 v124; // rcx
  __int8 v125; // r15
  unsigned __int64 v126; // rbx
  __int8 v127; // dl
  unsigned __int64 v128; // rax
  __m128i v129; // xmm6
  __m128i v130; // xmm7
  __int64 v131; // rax
  char v132; // bl
  __int64 *v133; // r12
  unsigned __int64 *v134; // rax
  __int64 *v135; // rax
  __m128i v136; // xmm5
  signed __int64 v137; // r14
  _QWORD *v138; // rax
  _QWORD *v139; // r9
  __int64 v140; // rdx
  _QWORD *v141; // rax
  __int64 v142; // rcx
  __int64 v143; // rdi
  __m128i *v144; // rax
  __m128i v145; // xmm0
  __int64 v146; // r12
  const char *v147; // rax
  size_t v148; // rdx
  _WORD *v149; // rdi
  char *v150; // rsi
  size_t v151; // r13
  unsigned __int64 v152; // rax
  __int64 v153; // rdi
  void *v154; // rax
  __int64 v155; // rax
  unsigned __int64 v156; // rdi
  unsigned __int64 v157; // rax
  __int64 v158; // rax
  _QWORD *v159; // rax
  _QWORD *v160; // rax
  __int64 v161; // rax
  _QWORD *v162; // rax
  _QWORD *v163; // rax
  __int64 *v164; // rax
  __int64 v165; // rsi
  unsigned __int64 v166; // rax
  unsigned __int64 v167; // rax
  __int64 v168; // rcx
  _QWORD *v169; // rax
  __int64 v170; // rcx
  __int64 v171; // rdx
  unsigned __int64 v172; // rax
  __int64 v173; // rcx
  _QWORD *v174; // r10
  __int64 v175; // r13
  __m128i *v176; // rax
  __m128i v177; // xmm0
  const char *v178; // rax
  size_t v179; // rdx
  __int64 v180; // r10
  unsigned __int8 *v181; // rsi
  _BYTE *v182; // rdi
  unsigned __int64 v183; // rax
  __int64 v184; // r10
  __m128i *v185; // rax
  __m128i v186; // xmm0
  __int64 v187; // rax
  __int64 v188; // rax
  __int64 v189; // rax
  _QWORD *v190; // [rsp+8h] [rbp-698h]
  __int64 v191; // [rsp+10h] [rbp-690h]
  _QWORD *v192; // [rsp+10h] [rbp-690h]
  _QWORD *v193; // [rsp+10h] [rbp-690h]
  __int64 *v194; // [rsp+10h] [rbp-690h]
  __int32 v195; // [rsp+18h] [rbp-688h]
  _QWORD *v196; // [rsp+18h] [rbp-688h]
  unsigned __int32 v197; // [rsp+18h] [rbp-688h]
  unsigned __int32 v198; // [rsp+18h] [rbp-688h]
  __int64 v199; // [rsp+18h] [rbp-688h]
  __int64 v200; // [rsp+20h] [rbp-680h]
  __int64 *v201; // [rsp+20h] [rbp-680h]
  __int64 v202; // [rsp+20h] [rbp-680h]
  __int64 v203; // [rsp+20h] [rbp-680h]
  __int64 v204; // [rsp+20h] [rbp-680h]
  unsigned __int64 v205; // [rsp+28h] [rbp-678h]
  __int64 v206; // [rsp+30h] [rbp-670h]
  __int64 v207; // [rsp+30h] [rbp-670h]
  __int64 v208; // [rsp+30h] [rbp-670h]
  __int64 v209; // [rsp+30h] [rbp-670h]
  size_t v210; // [rsp+30h] [rbp-670h]
  _QWORD *v211; // [rsp+30h] [rbp-670h]
  __int64 v212; // [rsp+38h] [rbp-668h]
  __int64 v213; // [rsp+38h] [rbp-668h]
  _QWORD *v214; // [rsp+38h] [rbp-668h]
  _QWORD *v215; // [rsp+40h] [rbp-660h]
  __int64 v216; // [rsp+48h] [rbp-658h]
  unsigned __int32 v217; // [rsp+48h] [rbp-658h]
  _QWORD *v218; // [rsp+48h] [rbp-658h]
  unsigned __int64 v219; // [rsp+50h] [rbp-650h]
  __int64 v220; // [rsp+60h] [rbp-640h]
  unsigned __int64 v221; // [rsp+70h] [rbp-630h]
  __int64 *v222; // [rsp+78h] [rbp-628h]
  __int64 v223; // [rsp+80h] [rbp-620h]
  __int64 v224; // [rsp+88h] [rbp-618h]
  char v228; // [rsp+B8h] [rbp-5E8h]
  __int64 (__fastcall *v229)(__int64, __int64 *, __int64, const __m128i *); // [rsp+B8h] [rbp-5E8h]
  _QWORD *v230; // [rsp+B8h] [rbp-5E8h]
  __int64 v231; // [rsp+D0h] [rbp-5D0h]
  unsigned int v232; // [rsp+D8h] [rbp-5C8h]
  __int8 v233; // [rsp+D8h] [rbp-5C8h]
  __int64 *v234; // [rsp+D8h] [rbp-5C8h]
  __int8 v235; // [rsp+D8h] [rbp-5C8h]
  _BYTE *v236; // [rsp+D8h] [rbp-5C8h]
  _BYTE *v237; // [rsp+D8h] [rbp-5C8h]
  char v238; // [rsp+E0h] [rbp-5C0h]
  __int64 *v239; // [rsp+E8h] [rbp-5B8h]
  char v240; // [rsp+F7h] [rbp-5A9h]
  __int64 v241; // [rsp+F8h] [rbp-5A8h]
  __int64 *v242; // [rsp+F8h] [rbp-5A8h]
  _BYTE *v243; // [rsp+F8h] [rbp-5A8h]
  __int64 *v244; // [rsp+108h] [rbp-598h] BYREF
  __int64 v245; // [rsp+110h] [rbp-590h] BYREF
  _QWORD *v246; // [rsp+118h] [rbp-588h] BYREF
  __int64 v247[2]; // [rsp+120h] [rbp-580h] BYREF
  __m128i v248; // [rsp+130h] [rbp-570h] BYREF
  __int64 v249; // [rsp+140h] [rbp-560h]
  __m128i v250; // [rsp+150h] [rbp-550h] BYREF
  __m128i v251; // [rsp+160h] [rbp-540h] BYREF
  __int128 v252; // [rsp+170h] [rbp-530h] BYREF
  __int128 v253; // [rsp+180h] [rbp-520h]
  _OWORD v254[3]; // [rsp+1A0h] [rbp-500h] BYREF
  __m128i v255; // [rsp+1D0h] [rbp-4D0h]
  __m128i v256; // [rsp+1E0h] [rbp-4C0h]
  __m128i v257; // [rsp+1F0h] [rbp-4B0h]
  __m128i v258; // [rsp+200h] [rbp-4A0h] BYREF
  __m128i v259; // [rsp+210h] [rbp-490h] BYREF
  __m128i v260; // [rsp+220h] [rbp-480h] BYREF
  __m128i v261; // [rsp+230h] [rbp-470h] BYREF
  __m128i v262; // [rsp+240h] [rbp-460h] BYREF
  __m128i v263; // [rsp+250h] [rbp-450h] BYREF
  char v264; // [rsp+260h] [rbp-440h]
  __int64 v265; // [rsp+270h] [rbp-430h] BYREF
  __int64 v266; // [rsp+278h] [rbp-428h]
  __int64 *v267; // [rsp+280h] [rbp-420h] BYREF
  unsigned int v268; // [rsp+288h] [rbp-418h]
  _BYTE *v269; // [rsp+2C0h] [rbp-3E0h] BYREF
  __int64 v270; // [rsp+2C8h] [rbp-3D8h]
  _BYTE v271[32]; // [rsp+2D0h] [rbp-3D0h] BYREF
  __int64 *v272; // [rsp+2F0h] [rbp-3B0h] BYREF
  __int64 v273; // [rsp+2F8h] [rbp-3A8h]
  _BYTE v274[128]; // [rsp+300h] [rbp-3A0h] BYREF
  __m128i v275; // [rsp+380h] [rbp-320h] BYREF
  __int64 v276; // [rsp+390h] [rbp-310h] BYREF
  _QWORD *v277; // [rsp+398h] [rbp-308h]
  char v278; // [rsp+3A0h] [rbp-300h] BYREF
  __int64 v279; // [rsp+3B0h] [rbp-2F0h]
  __int64 v280; // [rsp+3B8h] [rbp-2E8h]
  __int16 v281; // [rsp+3C0h] [rbp-2E0h]
  __int64 v282; // [rsp+3C8h] [rbp-2D8h]
  void **v283; // [rsp+3D0h] [rbp-2D0h]
  void **v284; // [rsp+3D8h] [rbp-2C8h]
  __int64 v285; // [rsp+3E0h] [rbp-2C0h]
  int v286; // [rsp+3E8h] [rbp-2B8h]
  __int16 v287; // [rsp+3ECh] [rbp-2B4h]
  char v288; // [rsp+3EEh] [rbp-2B2h]
  __int64 v289; // [rsp+3F0h] [rbp-2B0h]
  __int64 v290; // [rsp+3F8h] [rbp-2A8h]
  void *v291; // [rsp+400h] [rbp-2A0h] BYREF
  void *v292; // [rsp+408h] [rbp-298h] BYREF
  __int64 v293; // [rsp+460h] [rbp-240h] BYREF
  unsigned __int64 v294; // [rsp+468h] [rbp-238h]
  __int64 v295; // [rsp+470h] [rbp-230h] BYREF
  int v296; // [rsp+478h] [rbp-228h]
  char v297; // [rsp+47Ch] [rbp-224h]
  _QWORD v298[2]; // [rsp+480h] [rbp-220h] BYREF
  __int64 v299; // [rsp+490h] [rbp-210h]
  _QWORD *v300; // [rsp+498h] [rbp-208h]
  __int64 v301; // [rsp+4A0h] [rbp-200h]
  int v302; // [rsp+4A8h] [rbp-1F8h]
  char v303; // [rsp+4ACh] [rbp-1F4h]
  _QWORD v304[62]; // [rsp+4B0h] [rbp-1F0h] BYREF

  v6 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v7 = sub_BC1CD0(a4, &unk_4F875F0, a3);
  if ( *(_QWORD *)(v7 + 40) == *(_QWORD *)(v7 + 48) )
  {
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  v8 = v6 + 8;
  v220 = v6 + 8;
  v223 = v7 + 8;
  v9 = sub_BC1CD0(a4, &unk_4F881D0, a3) + 8;
  v10 = sub_BC1CD0(a4, &unk_4F8E5A8, a3);
  v247[1] = a4;
  v14 = *(char ***)(v7 + 40);
  v15 = *(char ***)(v7 + 48);
  v247[0] = a3;
  v231 = v10 + 8;
  if ( v15 == v14 )
  {
    v240 = 0;
  }
  else
  {
    v240 = 0;
    v16 = 0;
    do
    {
      v17 = *v14++;
      v16 |= sub_F6AC10(v17, v8, v223, v9, 0, 0, 0);
      v240 |= sub_11D2180((__int64)*(v14 - 1), v8, v223, v9, v18, v19);
    }
    while ( v15 != v14 );
    if ( (_BYTE)v16 )
    {
      v240 = qword_4FFD1A8;
      if ( !(_BYTE)qword_4FFD1A8 )
      {
        v301 = 2;
        v294 = (unsigned __int64)v298;
        v300 = v304;
        v295 = 0x100000002LL;
        v302 = 0;
        v303 = 1;
        v296 = 0;
        v297 = 1;
        v298[0] = &qword_4F82400;
        v293 = 1;
        if ( &qword_4F82400 == (__int64 *)&unk_4F8D9A8 )
        {
          HIDWORD(v295) = 0;
          v293 = 2;
        }
        HIDWORD(v301) = 1;
        v304[0] = &unk_4F8D9A8;
        v299 = 1;
        sub_BBE020(a4, a3, (__int64)&v293, v11);
        if ( !v303 )
          _libc_free((unsigned __int64)v300);
        if ( !v297 )
          _libc_free(v294);
        v240 = v16;
      }
    }
  }
  v20 = (unsigned __int64 *)&v267;
  v265 = 0;
  v266 = 1;
  do
  {
    *v20 = -4096;
    v20 += 2;
  }
  while ( v20 != (unsigned __int64 *)&v269 );
  v21 = (__int64)&v265;
  v269 = v271;
  v270 = 0x400000000LL;
  sub_F774D0(v223, (__int64)&v265, (__int64)&v269, v11, v12, v13);
  v26 = (unsigned int)v270;
  v244 = &v265;
  if ( !(_DWORD)v270 )
    goto LABEL_76;
  v239 = (__int64 *)v9;
  do
  {
    v27 = (unsigned __int64)v269;
    v28 = *(_QWORD *)&v269[8 * v26 - 8];
    if ( (v266 & 1) != 0 )
    {
      v29 = (__int64 *)&v267;
      v21 = 3;
      goto LABEL_13;
    }
    v21 = v268;
    v29 = v267;
    if ( v268 )
    {
      v21 = v268 - 1;
LABEL_13:
      v30 = v21 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
      v31 = (unsigned __int64 *)&v29[2 * v30];
      v24 = *v31;
      if ( v28 == *v31 )
      {
LABEL_14:
        *v31 = -8192;
        ++HIDWORD(v266);
        v27 = (unsigned __int64)v269;
        LODWORD(v266) = (2 * ((unsigned int)v266 >> 1) - 2) | v266 & 1;
      }
      else
      {
        v67 = 1;
        while ( v24 != -4096 )
        {
          v25 = (unsigned int)(v67 + 1);
          v30 = v21 & (v67 + v30);
          v31 = (unsigned __int64 *)&v29[2 * v30];
          v24 = *v31;
          if ( v28 == *v31 )
            goto LABEL_14;
          v67 = v25;
        }
      }
    }
    v26 = (unsigned int)(v270 - 1);
    v32 = (__int64 *)(v27 + 8 * v26 - 8);
    while ( 1 )
    {
      LODWORD(v270) = v26;
      if ( !(_DWORD)v26 )
        break;
      v33 = *v32;
      v21 = (unsigned int)(v26 - 1);
      --v32;
      if ( v33 )
        break;
      v26 = (unsigned int)v21;
    }
    v23 = (unsigned int)qword_4FFD448;
    v22 = (__int64)(*(_QWORD *)(v28 + 40) - *(_QWORD *)(v28 + 32)) >> 3;
    if ( v22 >= (unsigned int)qword_4FFD448 )
      continue;
    v34 = sub_D4B130(v28);
    if ( !v34 )
      goto LABEL_74;
    v35 = (__int64 *)sub_27CA6C0(v247);
    v36 = sub_FDD860(v35, **(_QWORD **)(v28 + 32));
    v21 = sub_D4B130(v28);
    v23 = sub_FDD860(v35, v21);
    if ( v23 && v36 )
    {
      v22 = v36 % v23;
      v221 = v36 / v23;
      if ( !(_BYTE)qword_4FFD1A8 && (unsigned int)qword_4FFD0C8 > v221 )
        goto LABEL_74;
      v228 = 1;
    }
    else
    {
      v228 = 0;
    }
    v222 = (__int64 *)sub_AA48A0(v34);
    v293 = (__int64)&v295;
    v294 = 0x1000000000LL;
    v25 = *(_QWORD *)(v28 + 32);
    v241 = *(_QWORD *)(v28 + 40);
    if ( v25 == v241 )
      goto LABEL_74;
    v238 = 0;
    v37 = *(_QWORD *)(v28 + 32);
    do
    {
      while ( 1 )
      {
        v38 = *(_QWORD *)(*(_QWORD *)v37 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v38 == *(_QWORD *)v37 + 48LL )
          goto LABEL_255;
        if ( !v38 )
          BUG();
        if ( (unsigned int)*(unsigned __int8 *)(v38 - 24) - 30 > 0xA )
LABEL_255:
          BUG();
        if ( *(_BYTE *)(v38 - 24) != 31 )
          goto LABEL_46;
        if ( (*(_DWORD *)(v38 - 20) & 0x7FFFFFF) == 1 )
          goto LABEL_46;
        v39 = *(_QWORD *)(v38 + 16);
        if ( v39 == sub_D47930(v28) )
          goto LABEL_46;
        v40 = *(_QWORD *)(v38 - 56);
        if ( *(_BYTE *)(v28 + 84) )
          break;
        if ( !sub_C8CA60(v28 + 56, v40) )
          goto LABEL_95;
LABEL_39:
        v43 = 0;
        if ( (_BYTE)qword_4FFD1A8 )
          goto LABEL_43;
LABEL_40:
        v232 = v43;
        LODWORD(v272) = sub_FF0300(v231, *(_QWORD *)(v38 + 16), v43);
        if ( v228 )
        {
          v21 = v221;
          if ( sub_F02E20((unsigned int *)&v272, v221) >= (unsigned __int64)(unsigned int)qword_4FFD0C8 )
          {
            if ( !v232 )
              goto LABEL_43;
            goto LABEL_96;
          }
        }
        else
        {
          v21 = 15;
          sub_F02DB0(&v275, 0xFu, 0x10u);
          v24 = v232;
          if ( v275.m128i_i32[0] <= (unsigned int)v272 )
          {
            if ( !v232 )
              goto LABEL_43;
            goto LABEL_96;
          }
        }
LABEL_46:
        v37 += 8;
        if ( v241 == v37 )
          goto LABEL_47;
      }
      v41 = *(_QWORD **)(v28 + 64);
      v42 = &v41[*(unsigned int *)(v28 + 76)];
      if ( v41 != v42 )
      {
        while ( v40 != *v41 )
        {
          if ( v42 == ++v41 )
            goto LABEL_95;
        }
        goto LABEL_39;
      }
LABEL_95:
      v43 = 1;
      if ( !(_BYTE)qword_4FFD1A8 )
        goto LABEL_40;
LABEL_96:
      v66 = sub_BD5C60(v38 - 24);
      v284 = &v292;
      v275.m128i_i64[0] = (__int64)&v276;
      v275.m128i_i64[1] = 0x200000000LL;
      v282 = v66;
      v287 = 512;
      v292 = &unk_49DA0B0;
      v281 = 0;
      v283 = &v291;
      v291 = &unk_49DA100;
      v285 = 0;
      v286 = 0;
      v288 = 7;
      v289 = 0;
      v290 = 0;
      v279 = 0;
      v280 = 0;
      sub_D5F1F0((__int64)&v275, v38 - 24);
      sub_F35C60(v38 - 24, v275.m128i_i64);
      sub_FF0720(v231, *(_QWORD *)(v38 + 16));
      nullsub_61();
      v291 = &unk_49DA100;
      nullsub_63();
      if ( (__int64 *)v275.m128i_i64[0] != &v276 )
        _libc_free(v275.m128i_u64[0]);
      v238 = 1;
LABEL_43:
      v275.m128i_i64[0] = 0;
      v275.m128i_i64[1] = (__int64)&v278;
      v276 = 8;
      LODWORD(v277) = 0;
      BYTE4(v277) = 1;
      if ( (*(_BYTE *)(v38 - 17) & 0x40) != 0 )
        v44 = *(__m128i **)(v38 - 32);
      else
        v44 = (__m128i *)(v38 - 24 - 32LL * (*(_DWORD *)(v38 - 20) & 0x7FFFFFF));
      v21 = (__int64)v239;
      sub_27CADD0(v28, v239, v44, (__int64)&v293, (__int64)&v275, v25);
      if ( BYTE4(v277) )
        goto LABEL_46;
      v37 += 8;
      _libc_free(v275.m128i_u64[1]);
    }
    while ( v241 != v37 );
LABEL_47:
    if ( !(_DWORD)v294 )
      goto LABEL_69;
    if ( (_BYTE)qword_4FFD288 )
    {
      v68 = sub_CB72A0();
      v69 = (__m128i *)v68[4];
      v70 = (__int64)v68;
      if ( v68[3] - (_QWORD)v69 <= 0x15u )
      {
        sub_CB6200((__int64)v68, "irce: looking at loop ", 0x16u);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_42BEB50);
        v69[1].m128i_i32[0] = 1869573152;
        v69[1].m128i_i16[2] = 8304;
        *v69 = si128;
        v68[4] += 22LL;
      }
      sub_D49BF0(v28, v70, 0, 1, 0);
      v72 = *(void **)(v70 + 32);
      if ( *(_QWORD *)(v70 + 24) - (_QWORD)v72 <= 0xEu )
      {
        v73 = sub_CB6200(v70, "irce: loop has ", 0xFu);
      }
      else
      {
        v73 = v70;
        qmemcpy(v72, "irce: loop has ", 15);
        *(_QWORD *)(v70 + 32) += 15LL;
      }
      v74 = sub_CB59D0(v73, (unsigned int)v294);
      v75 = *(__m128i **)(v74 + 32);
      if ( *(_QWORD *)(v74 + 24) - (_QWORD)v75 <= 0x19u )
      {
        sub_CB6200(v74, " inductive range checks: \n", 0x1Au);
      }
      else
      {
        v76 = _mm_load_si128((const __m128i *)&xmmword_42BEB60);
        qmemcpy(&v75[1], " checks: \n", 10);
        *v75 = v76;
        *(_QWORD *)(v74 + 32) += 26LL;
      }
      v77 = (__int64 *)v293;
      v78 = v293 + 32LL * (unsigned int)v294;
      if ( v293 != v78 )
      {
        do
        {
          v79 = v77;
          v77 += 4;
          sub_27CAA50(v79, v70);
        }
        while ( (__int64 *)v78 != v77 );
      }
    }
    v21 = (__int64)v239;
    v245 = 0;
    sub_29FED30(&v258, v239, v28, (unsigned __int8)qword_4FFCFE8, &v245);
    if ( !v264 )
      goto LABEL_69;
    v45 = _mm_load_si128(&v259);
    v46 = _mm_load_si128(&v260);
    v47 = _mm_load_si128(&v261);
    v48 = _mm_load_si128(&v262);
    v254[0] = _mm_load_si128(&v258);
    v49 = _mm_load_si128(&v263);
    v254[1] = v45;
    v254[2] = v46;
    v255 = v47;
    v256 = v48;
    v257 = v49;
    v50 = sub_DD8400((__int64)v239, v262.m128i_i64[0]);
    v21 = (__int64)sub_DD8400((__int64)v239, v255.m128i_i64[0]);
    v51 = sub_DCC810(v239, v21, (__int64)v50, 0, 0);
    v22 = (unsigned __int64)sub_27CA820;
    v52 = (__int64 *)v293;
    v53 = v51;
    v249 = 0;
    v272 = (__int64 *)v274;
    v273 = 0x400000000LL;
    v248 = 0;
    v233 = v257.m128i_i8[1];
    v54 = sub_27CACD0;
    if ( !v257.m128i_i8[1] )
      v54 = sub_27CA820;
    v229 = v54;
    v55 = 32LL * (unsigned int)v294;
    v56 = (__int64 *)(v293 + v55);
    if ( v293 == v293 + v55 )
      goto LABEL_70;
    v224 = v28;
    v242 = (__int64 *)(v293 + v55);
    while ( 2 )
    {
      v58 = (__int64 *)v53[4];
      v246 = v53;
      v59 = sub_D95540(*v58);
      if ( *(_BYTE *)(v59 + 8) != 12 )
        v59 = 0;
      v60 = sub_D95540(*v52);
      v61 = v52[2];
      v62 = v60;
      if ( *(_BYTE *)(v60 + 8) == 12 )
      {
        v57 = sub_D95540(v61);
        if ( *(_BYTE *)(v57 + 8) != 12 )
          v57 = 0;
        if ( !v59 )
          goto LABEL_59;
        v22 = *(_DWORD *)(v59 + 8) >> 8;
        if ( (unsigned int)v22 > *(_DWORD *)(v62 + 8) >> 8 || v246[5] != 2 )
          goto LABEL_59;
        v216 = v57;
        v80 = *(_QWORD *)v246[4];
        if ( v233 )
        {
          v81 = sub_DD2D10((__int64)v239, v80, v62);
          v21 = sub_D33D80(v246, (__int64)v239, v112, v113, v114);
          v85 = sub_DD2D10((__int64)v239, v21, v62);
        }
        else
        {
          v81 = sub_DC2CB0((__int64)v239, v80, v62);
          v21 = sub_D33D80(v246, (__int64)v239, v82, v83, v84);
          v85 = sub_DC2CB0((__int64)v239, v21, v62);
        }
        v86 = v216;
        if ( *((_WORD *)v85 + 12) || (v22 = v52[1], *(_WORD *)(v22 + 24)) || (_QWORD *)v22 != v85 )
        {
LABEL_59:
          v52 += 4;
          if ( v242 == v52 )
            goto LABEL_65;
LABEL_60:
          v233 = v257.m128i_i8[1];
          continue;
        }
        v87 = *(_DWORD *)(v62 + 8) >> 8;
        v206 = *v52;
        v275.m128i_i32[2] = v87;
        v217 = v87 - 1;
        v212 = 1LL << ((unsigned __int8)v87 - 1);
        if ( v87 <= 0x40 )
        {
          v191 = v86;
          v195 = v87;
          v88 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v87;
          if ( !v87 )
            v88 = 0;
          v275.m128i_i64[0] = ~v212 & v88;
          v215 = sub_DA26C0(v239, (__int64)&v275);
          v89 = v195;
          v90 = v191;
          if ( v275.m128i_i32[2] <= 0x40u )
          {
            v275.m128i_i32[2] = v195;
LABEL_132:
            v275.m128i_i64[0] = 0;
            goto LABEL_133;
          }
LABEL_231:
          v156 = v275.m128i_i64[0];
          if ( !v275.m128i_i64[0] )
          {
LABEL_216:
            v275.m128i_i32[2] = v89;
            if ( v89 <= 0x40 )
              goto LABEL_132;
LABEL_217:
            v199 = v90;
            sub_C43690((__int64)&v275, 0, 0);
            v90 = v199;
            if ( v275.m128i_i32[2] > 0x40u )
            {
              *(_QWORD *)(v275.m128i_i64[0] + 8LL * (v217 >> 6)) |= v212;
LABEL_134:
              v213 = v90;
              v190 = sub_DA26C0(v239, (__int64)&v275);
              v91 = v213;
              if ( v275.m128i_i32[2] > 0x40u && v275.m128i_i64[0] )
              {
                j_j___libc_free_0_0(v275.m128i_u64[0]);
                v91 = v213;
              }
              v200 = v91;
              v214 = sub_DCC810(v239, v206, (__int64)v81, 0, 0);
              v92 = sub_D95540((__int64)v214);
              v196 = sub_DA2C50((__int64)v239, v92, 0, 0);
              v250.m128i_i64[0] = (__int64)&v246;
              v250.m128i_i64[1] = (__int64)v239;
              v218 = (_QWORD *)v52[2];
              v192 = sub_DA2C50((__int64)v239, v62, 1, 0);
              if ( *(_DWORD *)(v200 + 8) >> 8 > *(_DWORD *)(v62 + 8) >> 8 )
              {
                if ( (_BYTE)qword_4FFCD48 )
                {
                  v174 = sub_CB72A0();
                  v175 = v246[6];
                  v176 = (__m128i *)v174[4];
                  if ( v174[3] - (_QWORD)v176 <= 0x11u )
                  {
                    v211 = v174;
                    sub_CB6200((__int64)v174, "irce: in function ", 0x12u);
                    v174 = v211;
                  }
                  else
                  {
                    v177 = _mm_load_si128((const __m128i *)&xmmword_42BEB80);
                    v176[1].m128i_i16[0] = 8302;
                    *v176 = v177;
                    v174[4] += 18LL;
                  }
                  v207 = (__int64)v174;
                  v178 = sub_BD5D20(*(_QWORD *)(**(_QWORD **)(v175 + 32) + 72LL));
                  v180 = v207;
                  v181 = (unsigned __int8 *)v178;
                  v182 = *(_BYTE **)(v207 + 32);
                  v183 = *(_QWORD *)(v207 + 24) - (_QWORD)v182;
                  if ( v183 < v179 )
                  {
                    sub_CB6200(v207, v181, v179);
                    v180 = v207;
                    v182 = *(_BYTE **)(v207 + 32);
                    v183 = *(_QWORD *)(v207 + 24) - (_QWORD)v182;
                  }
                  else if ( v179 )
                  {
                    v204 = v207;
                    v210 = v179;
                    memcpy(v182, v181, v179);
                    v180 = v204;
                    v189 = *(_QWORD *)(v204 + 24);
                    v182 = (_BYTE *)(*(_QWORD *)(v204 + 32) + v210);
                    *(_QWORD *)(v204 + 32) = v182;
                    v183 = v189 - (_QWORD)v182;
                  }
                  if ( v183 <= 4 )
                  {
                    v209 = v180;
                    sub_CB6200(v180, ", in ", 5u);
                    v180 = v209;
                  }
                  else
                  {
                    *(_DWORD *)v182 = 1852383276;
                    v182[4] = 32;
                    *(_QWORD *)(v180 + 32) += 5LL;
                  }
                  v208 = v180;
                  sub_D49BF0(v175, v180, 0, 1, 0);
                  v184 = v208;
                  v185 = *(__m128i **)(v208 + 32);
                  if ( *(_QWORD *)(v208 + 24) - (_QWORD)v185 <= 0x2Au )
                  {
                    sub_CB6200(v208, "there is range check with scaled boundary:\n", 0x2Bu);
                    v184 = v208;
                  }
                  else
                  {
                    v186 = _mm_load_si128((const __m128i *)&xmmword_4394040);
                    qmemcpy(&v185[2], " boundary:\n", 11);
                    *v185 = v186;
                    v185[1] = _mm_load_si128((const __m128i *)&xmmword_4394050);
                    *(_QWORD *)(v208 + 32) += 43LL;
                  }
                  sub_27CAA50(v52, v184);
                }
                v158 = sub_D95540((__int64)v218);
                v159 = sub_DC5000((__int64)v239, (__int64)v215, v158, 0);
                v160 = sub_DCC810(v239, (__int64)v159, (__int64)v218, 0, 0);
                v193 = sub_27CA920((__int64 **)&v250, (__int64)v160);
                v161 = sub_D95540((__int64)v218);
                v162 = sub_DC5000((__int64)v239, (__int64)v190, v161, 0);
                v163 = sub_DCC810(v239, (__int64)v218, (__int64)v162, 0, 0);
                v277 = sub_27CA920((__int64 **)&v250, (__int64)v163);
                v275.m128i_i64[0] = (__int64)&v276;
                v276 = (__int64)v193;
                v275.m128i_i64[1] = 0x200000002LL;
                v164 = sub_DC8BD0(v239, (__int64)&v275, 0, 0);
                v165 = (__int64)v164;
                if ( (__int64 *)v275.m128i_i64[0] != &v276 )
                {
                  v194 = v164;
                  _libc_free(v275.m128i_u64[0]);
                  v165 = (__int64)v194;
                }
                v192 = sub_DC5200((__int64)v239, v165, v62, 0);
                v218 = sub_DC5200((__int64)v239, (__int64)v218, v62, 0);
              }
              v276 = (__int64)sub_27CA920((__int64 **)&v250, (__int64)v218);
              v277 = v192;
              v275.m128i_i64[0] = (__int64)&v276;
              v275.m128i_i64[1] = 0x200000002LL;
              v94 = sub_DC8BD0(v239, (__int64)&v275, 0, 0);
              if ( (__int64 *)v275.m128i_i64[0] != &v276 )
                _libc_free(v275.m128i_u64[0]);
              if ( v233 )
              {
                v95 = sub_DCC810(v239, (__int64)v196, (__int64)v215, 0, 0);
                v97 = sub_DCDFA0(v239, (__int64)v214, (__int64)v95, v96);
                v98 = sub_DCC810(v239, (__int64)v196, v97, 4, 0);
              }
              else
              {
                v157 = sub_DCE160(v239, (__int64)v196, (__int64)v214, v93);
                v98 = sub_DCC810(v239, (__int64)v196, v157, 2, 0);
              }
              v276 = (__int64)v98;
              v275.m128i_i64[0] = (__int64)&v276;
              v277 = v94;
              v275.m128i_i64[1] = 0x200000002LL;
              v99 = sub_DC8BD0(v239, (__int64)&v275, 0, 0);
              v101 = v99;
              if ( (__int64 *)v275.m128i_i64[0] != &v276 )
              {
                v201 = v99;
                _libc_free(v275.m128i_u64[0]);
                v101 = v201;
              }
              if ( v233 )
              {
                v234 = v101;
                v102 = sub_DCC810(v239, (__int64)v218, (__int64)v215, 0, 0);
                v104 = sub_DCDFA0(v239, (__int64)v214, (__int64)v102, v103);
                v105 = sub_DCC810(v239, (__int64)v218, v104, 4, 0);
              }
              else
              {
                v234 = v101;
                v166 = sub_DCE160(v239, (__int64)v218, (__int64)v214, v100);
                v105 = sub_DCC810(v239, (__int64)v218, v166, 2, 0);
              }
              v276 = (__int64)v105;
              v277 = v94;
              v275.m128i_i64[0] = (__int64)&v276;
              v275.m128i_i64[1] = 0x200000002LL;
              v106 = sub_DC8BD0(v239, (__int64)&v275, 0, 0);
              v107 = v234;
              v108 = v106;
              if ( (__int64 *)v275.m128i_i64[0] != &v276 )
              {
                _libc_free(v275.m128i_u64[0]);
                v107 = v234;
              }
              *(_QWORD *)&v252 = v107;
              v21 = (__int64)v239;
              *((_QWORD *)&v252 + 1) = v108;
              LOBYTE(v253) = 1;
              v229((__int64)&v275, v239, (__int64)&v248, (const __m128i *)&v252);
              if ( (_BYTE)v276 )
              {
                v109 = (unsigned int)v273;
                v23 = HIDWORD(v273);
                v110 = (const __m128i *)v52;
                v22 = (unsigned __int64)v272;
                v24 = (unsigned int)v273 + 1LL;
                if ( v24 > HIDWORD(v273) )
                {
                  if ( v272 > v52 || &v272[4 * (unsigned int)v273] <= v52 )
                  {
                    v21 = (__int64)v274;
                    v110 = (const __m128i *)v52;
                    sub_C8D5F0((__int64)&v272, v274, (unsigned int)v273 + 1LL, 0x20u, v24, v25);
                    v22 = (unsigned __int64)v272;
                    v109 = (unsigned int)v273;
                  }
                  else
                  {
                    v21 = (__int64)v274;
                    v137 = (char *)v52 - (char *)v272;
                    sub_C8D5F0((__int64)&v272, v274, (unsigned int)v273 + 1LL, 0x20u, v24, v25);
                    v22 = (unsigned __int64)v272;
                    v109 = (unsigned int)v273;
                    v110 = (const __m128i *)((char *)v272 + v137);
                  }
                }
                v111 = (__m128i *)(v22 + 32 * v109);
                *v111 = _mm_loadu_si128(v110);
                v111[1] = _mm_loadu_si128(v110 + 1);
                LODWORD(v273) = v273 + 1;
                if ( (_BYTE)v249 )
                {
                  v248 = _mm_load_si128(&v275);
                }
                else
                {
                  v136 = _mm_load_si128(&v275);
                  LOBYTE(v249) = 1;
                  v248 = v136;
                }
              }
              goto LABEL_59;
            }
LABEL_133:
            v275.m128i_i64[0] |= v212;
            goto LABEL_134;
          }
LABEL_215:
          v198 = v89;
          v203 = v90;
          j_j___libc_free_0_0(v156);
          v89 = v198;
          v90 = v203;
          goto LABEL_216;
        }
        v197 = v87;
        v202 = v86;
        sub_C43690((__int64)&v275, -1, 1);
        v155 = ~v212;
        if ( v275.m128i_i32[2] <= 0x40u )
        {
          v275.m128i_i64[0] &= v155;
          v215 = sub_DA26C0(v239, (__int64)&v275);
          v90 = v202;
          v89 = v197;
          if ( v275.m128i_i32[2] > 0x40u )
            goto LABEL_231;
        }
        else
        {
          *(_QWORD *)(v275.m128i_i64[0] + 8LL * (v217 >> 6)) &= v155;
          v215 = sub_DA26C0(v239, (__int64)&v275);
          v90 = v202;
          v89 = v197;
          if ( v275.m128i_i32[2] > 0x40u )
          {
            v156 = v275.m128i_i64[0];
            if ( v275.m128i_i64[0] )
              goto LABEL_215;
          }
        }
        v275.m128i_i32[2] = v89;
        goto LABEL_217;
      }
      break;
    }
    sub_D95540(v61);
    v52 += 4;
    if ( v242 != v52 )
      goto LABEL_60;
LABEL_65:
    if ( !(_BYTE)v249 )
      goto LABEL_118;
    v63 = sub_D95540(v248.m128i_i64[0]);
    if ( (_BYTE)qword_4FFCF08 || v63 == v257.m128i_i64[1] )
    {
      v22 = *(_DWORD *)(v63 + 8) >> 8;
      if ( (unsigned int)v22 >= *(_DWORD *)(v257.m128i_i64[1] + 8) >> 8 )
      {
        v115 = v257.m128i_i8[1];
        v250 = 0;
        v235 = v257.m128i_i8[1];
        v251 = 0;
        v116 = sub_DD8400((__int64)v239, v255.m128i_i64[1]);
        if ( v115 )
        {
          v117 = sub_DD2D10((__int64)v239, (__int64)v116, v63);
          v135 = sub_DD8400((__int64)v239, v256.m128i_i64[1]);
          v119 = sub_DD2D10((__int64)v239, (__int64)v135, v63);
        }
        else
        {
          v117 = sub_DC2CB0((__int64)v239, (__int64)v116, v63);
          v118 = sub_DD8400((__int64)v239, v256.m128i_i64[1]);
          v119 = sub_DC2CB0((__int64)v239, (__int64)v118, v63);
        }
        v120 = v257.m128i_i8[0];
        v121 = sub_DA2C50((__int64)v239, v63, 1, 0);
        if ( v120 )
        {
          v243 = v117;
          v122 = sub_DCC810(v239, (__int64)v119, (__int64)v121, 0, 0);
        }
        else
        {
          v277 = v121;
          v230 = v121;
          v275.m128i_i64[0] = (__int64)&v276;
          v276 = (__int64)v119;
          v275.m128i_i64[1] = 0x200000002LL;
          v138 = sub_DC7EB0(v239, (__int64)&v275, 0, 0);
          v139 = v230;
          v243 = v138;
          if ( (__int64 *)v275.m128i_i64[0] != &v276 )
          {
            _libc_free(v275.m128i_u64[0]);
            v139 = v230;
          }
          v275.m128i_i64[0] = (__int64)&v276;
          v276 = (__int64)v117;
          v277 = v139;
          v275.m128i_i64[1] = 0x200000002LL;
          v119 = sub_DC7EB0(v239, (__int64)&v275, 0, 0);
          if ( (__int64 *)v275.m128i_i64[0] != &v276 )
            _libc_free(v275.m128i_u64[0]);
          v122 = v117;
        }
        if ( v235 )
        {
          v236 = v122;
          v205 = v205 & 0xFFFFFF0000000000LL | 0x29;
          if ( (unsigned __int8)sub_DC3A60((__int64)v239, v205, v248.m128i_i64[0], v243) )
          {
            v219 = v219 & 0xFFFFFF0000000000LL | 0x28;
            if ( (unsigned __int8)sub_DC3A60((__int64)v239, v219, v236, (_BYTE *)v248.m128i_i64[1]) )
              goto LABEL_173;
            v171 = v248.m128i_i64[1];
            v125 = 0;
            v126 = 0;
LABEL_234:
            v172 = sub_DCE160(v239, (__int64)v119, v171, v124);
            v128 = sub_DCDFA0(v239, (__int64)v243, v172, v173);
            v127 = 1;
LABEL_174:
            v250.m128i_i64[0] = v126;
            v251.m128i_i8[8] = v127;
            v250.m128i_i8[8] = v125;
            v129 = _mm_load_si128(&v250);
            v251.m128i_i64[0] = v128;
            v130 = _mm_load_si128(&v251);
            v252 = (__int128)v129;
            v253 = (__int128)v130;
            v131 = sub_D95540(v248.m128i_i64[0]);
            v21 = v224;
            sub_2A00470(
              (unsigned int)&v275,
              v224,
              v223,
              (unsigned int)sub_27CA6E0,
              (unsigned int)&v244,
              (unsigned int)v254,
              (__int64)v239,
              v220,
              v131,
              v252,
              v253);
            v132 = sub_2A02F40(&v275);
            if ( v132 )
            {
              if ( (_BYTE)qword_4FFD368 )
              {
                v143 = sub_C5F790((__int64)&v275, v224);
                v144 = *(__m128i **)(v143 + 32);
                if ( *(_QWORD *)(v143 + 24) - (_QWORD)v144 <= 0x11u )
                {
                  v21 = (__int64)"irce: in function ";
                  sub_CB6200(v143, "irce: in function ", 0x12u);
                }
                else
                {
                  v145 = _mm_load_si128((const __m128i *)&xmmword_42BEB80);
                  v144[1].m128i_i16[0] = 8302;
                  *v144 = v145;
                  *(_QWORD *)(v143 + 32) += 18LL;
                }
                v146 = sub_C5F790(v143, v21);
                v147 = sub_BD5D20(*(_QWORD *)(**(_QWORD **)(v224 + 32) + 72LL));
                v149 = *(_WORD **)(v146 + 32);
                v150 = (char *)v147;
                v151 = v148;
                v152 = *(_QWORD *)(v146 + 24) - (_QWORD)v149;
                if ( v152 < v148 )
                {
                  v188 = sub_CB6200(v146, (unsigned __int8 *)v150, v148);
                  v149 = *(_WORD **)(v188 + 32);
                  v146 = v188;
                  v152 = *(_QWORD *)(v188 + 24) - (_QWORD)v149;
                }
                else if ( v148 )
                {
                  memcpy(v149, v150, v148);
                  v187 = *(_QWORD *)(v146 + 24);
                  v149 = (_WORD *)(v151 + *(_QWORD *)(v146 + 32));
                  *(_QWORD *)(v146 + 32) = v149;
                  v152 = v187 - (_QWORD)v149;
                }
                if ( v152 <= 1 )
                {
                  v150 = ": ";
                  v149 = (_WORD *)v146;
                  sub_CB6200(v146, (unsigned __int8 *)": ", 2u);
                }
                else
                {
                  *v149 = 8250;
                  *(_QWORD *)(v146 + 32) += 2LL;
                }
                v153 = sub_C5F790((__int64)v149, (__int64)v150);
                v154 = *(void **)(v153 + 32);
                if ( *(_QWORD *)(v153 + 24) - (_QWORD)v154 <= 0xBu )
                {
                  v150 = "constrained ";
                  sub_CB6200(v153, "constrained ", 0xCu);
                }
                else
                {
                  qmemcpy(v154, "constrained ", 12);
                  *(_QWORD *)(v153 + 32) += 12LL;
                }
                v21 = sub_C5F790(v153, (__int64)v150);
                sub_D49BF0(v224, v21, 0, 1, 0);
              }
              v133 = v272;
              v64 = &v272[4 * (unsigned int)v273];
              if ( v272 != v64 )
              {
                do
                {
                  v22 = sub_ACD6D0(v222);
                  v134 = (unsigned __int64 *)v133[3];
                  if ( *v134 )
                  {
                    v21 = v134[2];
                    v23 = v134[1];
                    *(_QWORD *)v21 = v23;
                    if ( v23 )
                    {
                      v21 = v134[2];
                      *(_QWORD *)(v23 + 16) = v21;
                    }
                  }
                  *v134 = v22;
                  if ( v22 )
                  {
                    v23 = *(_QWORD *)(v22 + 16);
                    v21 = v22 + 16;
                    v134[1] = v23;
                    if ( v23 )
                      *(_QWORD *)(v23 + 16) = v134 + 1;
                    v134[2] = v21;
                    *(_QWORD *)(v22 + 16) = v134;
                  }
                  v133 += 4;
                }
                while ( v64 != v133 );
                v64 = v272;
              }
              v238 = v132;
            }
            else
            {
LABEL_118:
              v64 = v272;
            }
            if ( v64 != (__int64 *)v274 )
              goto LABEL_120;
LABEL_69:
            v56 = (__int64 *)v293;
            goto LABEL_70;
          }
          v167 = sub_DCE160(v239, (__int64)v119, v248.m128i_i64[0], v123);
          v126 = sub_DCDFA0(v239, (__int64)v243, v167, v168);
          v219 = v219 & 0xFFFFFF0000000000LL | 0x28;
          if ( !(unsigned __int8)sub_DC3A60((__int64)v239, v219, v236, (_BYTE *)v248.m128i_i64[1]) )
          {
            v171 = v248.m128i_i64[1];
            v125 = 1;
            goto LABEL_234;
          }
LABEL_227:
          v125 = 1;
          v127 = 0;
          v128 = 0;
          goto LABEL_174;
        }
        v237 = v122;
        v205 = v205 & 0xFFFFFF0000000000LL | 0x25;
        if ( (unsigned __int8)sub_DC3A60((__int64)v239, v205, v248.m128i_i64[0], v243) )
        {
          v219 = v219 & 0xFFFFFF0000000000LL | 0x24;
          if ( (unsigned __int8)sub_DC3A60((__int64)v239, v219, v237, (_BYTE *)v248.m128i_i64[1]) )
          {
LABEL_173:
            v125 = 0;
            v126 = 0;
            v127 = 0;
            v128 = 0;
            goto LABEL_174;
          }
          v140 = v248.m128i_i64[1];
          v125 = 0;
          v126 = 0;
        }
        else
        {
          v169 = sub_DCEE80(v239, (__int64)v119, v248.m128i_i64[0], 0);
          v126 = sub_DCE050(v239, (__int64)v243, (__int64)v169, v170);
          v219 = v219 & 0xFFFFFF0000000000LL | 0x24;
          if ( (unsigned __int8)sub_DC3A60((__int64)v239, v219, v237, (_BYTE *)v248.m128i_i64[1]) )
            goto LABEL_227;
          v140 = v248.m128i_i64[1];
          v125 = 1;
        }
        v141 = sub_DCEE80(v239, (__int64)v119, v140, 0);
        v128 = sub_DCE050(v239, (__int64)v243, (__int64)v141, v142);
        v127 = 1;
        goto LABEL_174;
      }
    }
    v238 = 0;
    v64 = v272;
    if ( v272 == (__int64 *)v274 )
      goto LABEL_69;
LABEL_120:
    _libc_free((unsigned __int64)v64);
    v56 = (__int64 *)v293;
LABEL_70:
    if ( v56 != &v295 )
      _libc_free((unsigned __int64)v56);
    if ( v238 && (v240 = qword_4FFD1A8) == 0 )
    {
      v301 = 2;
      v294 = (unsigned __int64)v298;
      v300 = v304;
      v295 = 0x100000002LL;
      v302 = 0;
      v303 = 1;
      v296 = 0;
      v297 = 1;
      v298[0] = &qword_4F82400;
      v293 = 1;
      if ( &qword_4F82400 == (__int64 *)&unk_4F8D9A8 )
      {
        HIDWORD(v295) = 0;
        v293 = 2;
      }
      v21 = a3;
      HIDWORD(v301) = 1;
      v304[0] = &unk_4F8D9A8;
      v299 = 1;
      sub_BBE020(a4, a3, (__int64)&v293, v23);
      if ( !v303 )
        _libc_free((unsigned __int64)v300);
      if ( !v297 )
        _libc_free(v294);
      v26 = (unsigned int)v270;
      v240 = v238;
    }
    else
    {
LABEL_74:
      v26 = (unsigned int)v270;
    }
  }
  while ( (_DWORD)v26 );
LABEL_76:
  if ( v240 )
  {
    sub_22D0390(a1, v21, v22, v23, v24, v25);
  }
  else
  {
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  if ( v269 != v271 )
    _libc_free((unsigned __int64)v269);
  if ( (v266 & 1) == 0 )
    sub_C7D6A0((__int64)v267, 16LL * v268, 8);
  return a1;
}
