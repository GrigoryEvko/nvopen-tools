// Function: sub_26ADC90
// Address: 0x26adc90
//
unsigned __int64 *__fastcall sub_26ADC90(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v4; // r12
  char *v5; // rax
  char *v6; // rax
  __int64 v7; // rsi
  __int64 (__fastcall *v8)(_QWORD, __int64); // rax
  unsigned int *v9; // rax
  _QWORD *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  float v13; // xmm0_4
  unsigned __int64 v14; // rbx
  int v15; // r13d
  __int64 v16; // r12
  __int64 v17; // r14
  __int64 v18; // rdi
  signed __int64 v19; // rax
  int v20; // edx
  __int64 *v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 *v25; // rax
  _BYTE *v26; // rax
  __int64 v27; // rbx
  __int64 v28; // r13
  __int64 v29; // rax
  __int64 v30; // rdx
  unsigned __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rbx
  unsigned __int64 v34; // rax
  int v35; // edx
  __int64 v36; // rax
  unsigned __int64 v37; // rax
  int v38; // eax
  __int64 *v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // rsi
  __int64 v42; // r8
  __int64 v43; // r9
  __int64 *v44; // rax
  unsigned __int64 v45; // r13
  unsigned __int64 v46; // r12
  __int64 v47; // r15
  __int64 *v48; // rbx
  __int64 *v49; // r14
  __int64 v50; // rdi
  __int64 v51; // rax
  __int64 v52; // rax
  unsigned int v53; // eax
  __int64 v54; // rdx
  char v55; // al
  unsigned __int64 v56; // rdi
  unsigned __int64 v57; // rdi
  __int64 *v58; // rbx
  __int64 *v59; // r12
  __int64 v60; // rsi
  __int64 v61; // rdi
  __int64 *v62; // r12
  _BYTE *v63; // rbx
  _BYTE *v64; // r12
  unsigned __int64 v65; // r13
  unsigned __int64 v66; // rdi
  char v68; // dl
  __int64 v69; // rax
  _BYTE *v70; // rsi
  __int64 v71; // rax
  unsigned int v72; // eax
  __int64 v73; // r12
  __int64 v74; // rax
  __int64 v75; // r8
  char *v76; // rax
  __int64 v77; // rdx
  __int64 v78; // rbx
  __int64 v79; // rdx
  __int64 v80; // r8
  __int64 v81; // r9
  __m128i v82; // xmm1
  __m128i v83; // xmm3
  __int64 v84; // rcx
  __int64 *v85; // r12
  __int64 v86; // rax
  __int64 v87; // r9
  __int64 v88; // rdx
  unsigned int v89; // eax
  unsigned int v90; // ecx
  _QWORD *v91; // rdi
  unsigned int v92; // edx
  __int64 v93; // rax
  __int64 v94; // r8
  __int64 v95; // r12
  __int64 v96; // r13
  __int64 v97; // r13
  __int64 v98; // r14
  __int64 v99; // r12
  __int64 v100; // rax
  unsigned __int64 v101; // rdx
  _QWORD *v102; // rcx
  unsigned __int64 i; // rax
  bool v104; // zf
  unsigned __int64 *v105; // rbx
  unsigned __int64 *v106; // r12
  __int64 *v107; // rax
  __int64 *v108; // r15
  unsigned __int64 v109; // rax
  __int64 v110; // r13
  unsigned int j; // ebx
  __int64 *v112; // r15
  int v113; // r14d
  unsigned __int64 v114; // rbx
  __int64 v115; // rsi
  signed __int64 v116; // rax
  __int64 *v117; // rdx
  __int64 v118; // rcx
  __int64 v119; // r8
  __int64 v120; // r9
  bool v121; // sf
  bool v122; // of
  __int64 *v123; // r12
  char v124; // di
  __int64 *v125; // r13
  __int64 v126; // rsi
  __int64 *v127; // rax
  __int64 v128; // rax
  __int64 *v129; // r8
  int v130; // r12d
  __int64 v131; // r9
  __int64 v132; // r15
  size_t v133; // r13
  char *v134; // r12
  __int64 v135; // rdx
  unsigned __int64 *v136; // rcx
  unsigned __int64 v137; // rdi
  unsigned __int64 v138; // rsi
  int v139; // eax
  __int64 v140; // rdx
  unsigned __int64 *v141; // r13
  _QWORD *v142; // rdi
  __int64 *v143; // rax
  __int64 *v144; // rbx
  __int64 *v145; // r13
  __int64 v146; // rdi
  unsigned int v147; // ecx
  __int64 v148; // rsi
  __int64 *v149; // rbx
  __int64 v150; // rsi
  __int64 v151; // rdi
  unsigned __int64 v152; // rcx
  __int64 *v153; // rax
  __int64 *v154; // rbx
  __int64 v155; // r12
  __int64 v156; // rax
  __int64 v157; // r8
  __int64 v158; // r14
  __int64 v159; // rcx
  __int64 v160; // rbx
  __int64 v161; // r8
  __int64 v162; // r9
  __m128i v163; // xmm4
  __m128i v164; // xmm6
  __int64 v165; // rdx
  __int64 v166; // rax
  __int64 v167; // rax
  _QWORD *v168; // rdi
  char *v169; // r12
  __int64 v170; // rax
  __int64 v171; // rax
  unsigned __int64 v172; // [rsp+8h] [rbp-818h]
  __int64 v173; // [rsp+10h] [rbp-810h]
  __int64 v174; // [rsp+18h] [rbp-808h]
  unsigned int v175; // [rsp+2Ch] [rbp-7F4h]
  __int64 *v177; // [rsp+48h] [rbp-7D8h]
  unsigned __int64 v178; // [rsp+50h] [rbp-7D0h]
  __int64 *v181; // [rsp+78h] [rbp-7A8h]
  __int64 v182; // [rsp+88h] [rbp-798h]
  __int64 v183; // [rsp+90h] [rbp-790h]
  __int64 v185; // [rsp+A0h] [rbp-780h]
  __int64 v186; // [rsp+A8h] [rbp-778h]
  __int64 v187; // [rsp+A8h] [rbp-778h]
  __int64 v188; // [rsp+B0h] [rbp-770h]
  char v189; // [rsp+BAh] [rbp-766h]
  char v190; // [rsp+BBh] [rbp-765h]
  unsigned int v191; // [rsp+BCh] [rbp-764h]
  __int64 v192; // [rsp+C0h] [rbp-760h]
  void *src; // [rsp+C8h] [rbp-758h]
  int srca; // [rsp+C8h] [rbp-758h]
  void *srcb; // [rsp+C8h] [rbp-758h]
  __int64 *srcc; // [rsp+C8h] [rbp-758h]
  void *srcd; // [rsp+C8h] [rbp-758h]
  unsigned int v198; // [rsp+D4h] [rbp-74Ch] BYREF
  __int64 v199; // [rsp+D8h] [rbp-748h] BYREF
  _BYTE *v200; // [rsp+E0h] [rbp-740h] BYREF
  _BYTE *v201; // [rsp+E8h] [rbp-738h]
  _BYTE *v202; // [rsp+F0h] [rbp-730h]
  __int64 *v203; // [rsp+100h] [rbp-720h] BYREF
  __int64 v204; // [rsp+108h] [rbp-718h]
  _BYTE v205[64]; // [rsp+110h] [rbp-710h] BYREF
  __int64 v206[4]; // [rsp+150h] [rbp-6D0h] BYREF
  unsigned __int64 v207[6]; // [rsp+170h] [rbp-6B0h] BYREF
  unsigned __int64 v208[2]; // [rsp+1A0h] [rbp-680h] BYREF
  __int64 v209; // [rsp+1B0h] [rbp-670h] BYREF
  __int64 *v210; // [rsp+1C0h] [rbp-660h] BYREF
  __int64 v211; // [rsp+1D0h] [rbp-650h] BYREF
  __int64 v212; // [rsp+1F0h] [rbp-630h] BYREF
  __int64 *v213; // [rsp+1F8h] [rbp-628h]
  __int64 v214; // [rsp+200h] [rbp-620h]
  int v215; // [rsp+208h] [rbp-618h]
  char v216; // [rsp+20Ch] [rbp-614h]
  char v217; // [rsp+210h] [rbp-610h] BYREF
  unsigned __int64 v218[2]; // [rsp+250h] [rbp-5D0h] BYREF
  char v219; // [rsp+260h] [rbp-5C0h] BYREF
  _BYTE *v220; // [rsp+268h] [rbp-5B8h]
  __int64 v221; // [rsp+270h] [rbp-5B0h]
  _BYTE v222[56]; // [rsp+278h] [rbp-5A8h] BYREF
  __int64 v223; // [rsp+2B0h] [rbp-570h]
  __int64 v224; // [rsp+2B8h] [rbp-568h]
  char v225; // [rsp+2C0h] [rbp-560h]
  int v226; // [rsp+2C4h] [rbp-55Ch]
  int v227; // [rsp+2C8h] [rbp-558h]
  char v228[8]; // [rsp+2D0h] [rbp-550h] BYREF
  __int64 v229; // [rsp+2D8h] [rbp-548h]
  unsigned int v230; // [rsp+2E8h] [rbp-538h]
  unsigned __int64 v231; // [rsp+2F0h] [rbp-530h]
  unsigned __int64 v232; // [rsp+2F8h] [rbp-528h]
  __int64 v233; // [rsp+308h] [rbp-518h]
  __int64 k; // [rsp+310h] [rbp-510h]
  __int64 *v235; // [rsp+318h] [rbp-508h]
  unsigned int v236; // [rsp+320h] [rbp-500h]
  char v237; // [rsp+328h] [rbp-4F8h] BYREF
  __int64 *v238; // [rsp+348h] [rbp-4D8h]
  unsigned int v239; // [rsp+350h] [rbp-4D0h]
  __int64 v240; // [rsp+358h] [rbp-4C8h] BYREF
  _QWORD v241[13]; // [rsp+370h] [rbp-4B0h] BYREF
  char v242; // [rsp+3D8h] [rbp-448h] BYREF
  _QWORD v243[2]; // [rsp+418h] [rbp-408h] BYREF
  char v244; // [rsp+428h] [rbp-3F8h] BYREF
  char v245; // [rsp+488h] [rbp-398h] BYREF
  void *v246; // [rsp+490h] [rbp-390h] BYREF
  int v247; // [rsp+498h] [rbp-388h]
  char v248; // [rsp+49Ch] [rbp-384h]
  __int64 v249; // [rsp+4A0h] [rbp-380h]
  __m128i v250; // [rsp+4A8h] [rbp-378h]
  __int64 v251; // [rsp+4B8h] [rbp-368h]
  __m128i v252; // [rsp+4C0h] [rbp-360h]
  __m128i v253; // [rsp+4D0h] [rbp-350h]
  _BYTE *v254; // [rsp+4E0h] [rbp-340h] BYREF
  __int64 v255; // [rsp+4E8h] [rbp-338h]
  _BYTE v256[320]; // [rsp+4F0h] [rbp-330h] BYREF
  char v257; // [rsp+630h] [rbp-1F0h]
  int v258; // [rsp+634h] [rbp-1ECh]
  __int64 v259; // [rsp+638h] [rbp-1E8h]
  _QWORD *v260; // [rsp+640h] [rbp-1E0h] BYREF
  __int64 v261; // [rsp+648h] [rbp-1D8h]
  _QWORD v262[8]; // [rsp+650h] [rbp-1D0h] BYREF
  _QWORD v263[50]; // [rsp+690h] [rbp-190h] BYREF

  v4 = *(_QWORD *)(a3 + 80);
  v224 = a3;
  if ( v4 )
    v4 -= 24;
  v218[0] = (unsigned __int64)&v219;
  v218[1] = 0x100000000LL;
  v220 = v222;
  v221 = 0x600000000LL;
  v227 = *(_DWORD *)(a3 + 92);
  v223 = 0;
  v225 = 0;
  v226 = 0;
  sub_B1F440((__int64)v218);
  sub_D51D90((__int64)v228, (__int64)v218);
  v5 = &v242;
  memset(v241, 0, 96);
  v241[12] = 1;
  do
  {
    *(_QWORD *)v5 = -4096;
    v5 += 16;
  }
  while ( v5 != (char *)v243 );
  v6 = &v244;
  v243[0] = 0;
  v243[1] = 1;
  do
  {
    *(_QWORD *)v6 = -4096;
    v6 += 24;
    *((_DWORD *)v6 - 4) = 0x7FFFFFFF;
  }
  while ( v6 != &v245 );
  v7 = a3;
  sub_FF9360(v241, a3, (__int64)v228, 0, 0, 0);
  v8 = *(__int64 (__fastcall **)(_QWORD, __int64))(a2 + 56);
  if ( v8 )
  {
    v7 = a3;
    v177 = 0;
    v181 = (__int64 *)v8(*(_QWORD *)(a2 + 64), a3);
  }
  else
  {
    v153 = (__int64 *)sub_22077B0(8u);
    v181 = v153;
    v154 = v153;
    if ( v153 )
    {
      v7 = a3;
      sub_FE7FB0(v153, (const char *)a3, (__int64)v241, (__int64)v228);
      v177 = v154;
    }
    else
    {
      v177 = 0;
    }
  }
  v9 = *(unsigned int **)(*(_QWORD *)(a2 + 88) + 8LL);
  if ( !v9 || (v7 = *v9, (_DWORD)v7) )
  {
    *a1 = 0;
    goto LABEL_52;
  }
  v10 = (_QWORD *)sub_22077B0(0x1B0u);
  v178 = (unsigned __int64)v10;
  if ( v10 )
  {
    *v10 = v10 + 2;
    v10[1] = 0x400000000LL;
  }
  v12 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, _QWORD *))(a2 + 40))(*(_QWORD *)(a2 + 48), a3, v11, v10);
  v13 = 0.0;
  v14 = 0;
  if ( *(_QWORD *)(a3 + 80) != a3 + 72 )
  {
    src = (void *)v4;
    v15 = 0;
    v16 = *(_QWORD *)(a3 + 80);
    v17 = v12;
    do
    {
      v18 = v16 - 24;
      if ( !v16 )
        v18 = 0;
      v19 = sub_26ABAF0(v18, v17);
      if ( v20 == 1 )
        v15 = 1;
      v122 = __OFADD__(v19, v14);
      v14 += v19;
      if ( v122 )
      {
        v14 = 0x8000000000000000LL;
        if ( v19 > 0 )
          v14 = 0x7FFFFFFFFFFFFFFFLL;
      }
      v16 = *(_QWORD *)(v16 + 8);
    }
    while ( v16 != a3 + 72 );
    v175 = v15;
    v4 = (__int64)src;
    v174 = 0;
    if ( v15 )
      goto LABEL_22;
    v13 = (float)(int)v14;
  }
  v175 = 0;
  v174 = (unsigned int)(int)(float)(v13 * *(float *)&qword_4FF5AE8);
LABEL_22:
  v172 = v175;
  sub_F02DB0(&v198, (int)(float)((float)(int)qword_4FF5A08 * *(float *)&qword_4FF5928), qword_4FF5A08);
  v213 = (__int64 *)&v217;
  v199 = v4;
  v200 = 0;
  v201 = 0;
  v202 = 0;
  v212 = 0;
  v214 = 8;
  v215 = 0;
  v216 = 1;
  sub_9319A0((__int64)&v200, 0, &v199);
  v7 = v199;
  if ( !v216 )
    goto LABEL_216;
  v25 = v213;
  v22 = HIDWORD(v214);
  v21 = &v213[HIDWORD(v214)];
  if ( v213 != v21 )
  {
    while ( v199 != *v25 )
    {
      if ( v21 == ++v25 )
        goto LABEL_219;
    }
    goto LABEL_27;
  }
LABEL_219:
  if ( HIDWORD(v214) < (unsigned int)v214 )
  {
    ++HIDWORD(v214);
    *v21 = v199;
    ++v212;
  }
  else
  {
LABEL_216:
    sub_C8CC70((__int64)&v212, v199, (__int64)v21, v22, v23, v24);
  }
LABEL_27:
  v189 = 0;
  while ( 1 )
  {
    v26 = v201;
    if ( v201 == v200 )
      break;
    while ( 1 )
    {
      v27 = *((_QWORD *)v26 - 1);
      v201 = v26 - 8;
      v28 = *(_QWORD *)(a2 + 88);
      v29 = sub_FDD2C0(v181, v27, 0);
      v261 = v30;
      v260 = (_QWORD *)v29;
      if ( !(_BYTE)v30 )
        break;
      v7 = (__int64)v260;
      if ( !sub_D84450(v28, (unsigned __int64)v260) )
        break;
      v26 = v201;
      if ( v201 == v200 )
        goto LABEL_141;
    }
    v7 = v27;
    v185 = v27;
    v260 = (_QWORD *)sub_FDD2C0(v181, v27, 0);
    v31 = 0;
    v261 = v32;
    if ( (_BYTE)v32 )
      v31 = (unsigned __int64)v260;
    if ( (unsigned int)qword_4FF5A08 <= v31 )
    {
      v33 = *(_QWORD *)(v27 + 48);
      v183 = v185 + 48;
      v34 = v33 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v185 + 48 == (v33 & 0xFFFFFFFFFFFFFFF8LL) )
      {
        v188 = 0;
      }
      else
      {
        if ( !v34 )
          BUG();
        v35 = *(unsigned __int8 *)(v34 - 24);
        v36 = v34 - 24;
        if ( (unsigned int)(v35 - 30) >= 0xB )
          v36 = 0;
        v188 = v36;
      }
      v191 = 0;
      v37 = v33 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v183 == (v33 & 0xFFFFFFFFFFFFFFF8LL) )
        goto LABEL_50;
LABEL_39:
      if ( !v37 )
LABEL_138:
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v37 - 24) - 30 <= 0xA )
      {
        v38 = sub_B46E30(v37 - 24);
        goto LABEL_42;
      }
LABEL_50:
      v38 = 0;
LABEL_42:
      v7 = v191;
      if ( v38 == v191 )
        continue;
      v41 = sub_B46EC0(v188, v191);
      if ( !v216 )
        goto LABEL_100;
      v44 = v213;
      v40 = HIDWORD(v214);
      v39 = &v213[HIDWORD(v214)];
      if ( v213 != v39 )
      {
        while ( v41 != *v44 )
        {
          if ( v39 == ++v44 )
            goto LABEL_121;
        }
        ++v191;
        goto LABEL_49;
      }
LABEL_121:
      if ( HIDWORD(v214) < (unsigned int)v214 )
      {
        ++HIDWORD(v214);
        *v39 = v41;
        ++v212;
LABEL_101:
        v69 = sub_B46EC0(v188, v191);
        v70 = v201;
        v260 = (_QWORD *)v69;
        if ( v201 == v202 )
        {
          sub_F38A10((__int64)&v200, v201, &v260);
        }
        else
        {
          if ( v201 )
          {
            *(_QWORD *)v201 = v69;
            v70 = v201;
          }
          v201 = v70 + 8;
        }
        v71 = sub_B46EC0(v188, v191);
        v72 = sub_FF0430((__int64)v241, v185, v71);
        if ( v72 > v198 )
        {
          ++v191;
          v33 = *(_QWORD *)(v185 + 48);
          goto LABEL_49;
        }
        v203 = (__int64 *)v205;
        v204 = 0x800000000LL;
        v86 = sub_B46EC0(v188, v191);
        if ( v86 )
        {
          v88 = (unsigned int)(*(_DWORD *)(v86 + 44) + 1);
          v89 = *(_DWORD *)(v86 + 44) + 1;
        }
        else
        {
          v89 = 0;
          v88 = 0;
        }
        if ( v89 < (unsigned int)v221 && *(_QWORD *)&v220[8 * v88] )
        {
          v90 = 8;
          v262[0] = *(_QWORD *)&v220[8 * v88];
          v260 = v262;
          v91 = v262;
          v92 = 1;
          v261 = 0x800000001LL;
          v93 = 0;
          while ( 1 )
          {
            v94 = v93 + 1;
            v95 = v91[v92 - 1];
            LODWORD(v261) = v92 - 1;
            v96 = *(_QWORD *)v95;
            if ( v93 + 1 > (unsigned __int64)v90 )
            {
              sub_C8D5F0((__int64)&v203, v205, v93 + 1, 8u, v94, v87);
              v93 = (unsigned int)v204;
            }
            v203[v93] = v96;
            v97 = *(_QWORD *)(v95 + 24);
            v98 = *(unsigned int *)(v95 + 32);
            v99 = 8 * v98;
            v100 = (unsigned int)v261;
            LODWORD(v204) = v204 + 1;
            v101 = v98 + (unsigned int)v261;
            if ( v101 > HIDWORD(v261) )
            {
              sub_C8D5F0((__int64)&v260, v262, v101, 8u, v94, v87);
              v100 = (unsigned int)v261;
            }
            v91 = v260;
            v102 = &v260[v100];
            if ( v99 )
            {
              for ( i = 0; i != v99; i += 8LL )
                v102[i / 8] = *(_QWORD *)(v97 + i);
              v91 = v260;
              LODWORD(v100) = v261;
            }
            LODWORD(v261) = v98 + v100;
            v92 = v98 + v100;
            if ( !((_DWORD)v98 + (_DWORD)v100) )
              break;
            v93 = (unsigned int)v204;
            v90 = HIDWORD(v204);
          }
          if ( v91 != v262 )
            _libc_free((unsigned __int64)v91);
          v107 = v203;
        }
        else
        {
          v107 = (__int64 *)v205;
        }
        v190 = sub_AA5590(*v107, 1);
        if ( v190 )
        {
          v108 = v203;
          v85 = &v203[(unsigned int)v204];
          if ( v203 != v85 )
          {
            v173 = (__int64)v203;
            v182 = (__int64)v203;
            v192 = 0;
            do
            {
              v186 = *(_QWORD *)v182;
              v109 = *(_QWORD *)(*(_QWORD *)v182 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
              if ( v109 != *(_QWORD *)v182 + 48LL )
              {
                if ( !v109 )
                  goto LABEL_138;
                v110 = v109 - 24;
                if ( (unsigned int)*(unsigned __int8 *)(v109 - 24) - 30 <= 0xA )
                {
                  srca = sub_B46E30(v110);
                  if ( srca )
                  {
                    for ( j = 0; j != srca; ++j )
                    {
                      v206[0] = sub_B46EC0(v110, j);
                      if ( v85 == sub_26AB430(v108, (__int64)v85, v206) )
                      {
                        if ( v192 )
                        {
                          v73 = *a4;
                          v74 = sub_B2BE50(*a4);
                          if ( !sub_B6EA50(v74) )
                          {
                            v166 = sub_B2BE50(v73);
                            v167 = sub_B6F970(v166);
                            if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v167 + 48LL))(v167) )
                              goto LABEL_117;
                          }
                          v75 = *(_QWORD *)(v206[0] + 56);
                          if ( v75 )
                            v75 -= 24;
                          sub_B176B0((__int64)&v260, (__int64)"partial-inlining", (__int64)"MultiExitRegion", 15, v75);
                          sub_B18290((__int64)&v260, "Region dominated by ", 0x14u);
                          v76 = (char *)sub_BD5D20(*v203);
                          sub_B16430((__int64)v208, "Block", 5u, v76, v77);
                          v78 = sub_2445430((__int64)&v260, (__int64)v208);
                          sub_B18290(v78, " has more than one region exit edge.", 0x24u);
                          v247 = *(_DWORD *)(v78 + 8);
                          v248 = *(_BYTE *)(v78 + 12);
                          v249 = *(_QWORD *)(v78 + 16);
                          v82 = _mm_loadu_si128((const __m128i *)(v78 + 24));
                          v246 = &unk_49D9D40;
                          v250 = v82;
                          v251 = *(_QWORD *)(v78 + 40);
                          v252 = _mm_loadu_si128((const __m128i *)(v78 + 48));
                          v83 = _mm_loadu_si128((const __m128i *)(v78 + 64));
                          v254 = v256;
                          v255 = 0x400000000LL;
                          v253 = v83;
                          v84 = *(unsigned int *)(v78 + 88);
                          if ( (_DWORD)v84 )
                            sub_26ACA40((__int64)&v254, v78 + 80, v79, v84, v80, v81);
                          v257 = *(_BYTE *)(v78 + 416);
                          v258 = *(_DWORD *)(v78 + 420);
                          v259 = *(_QWORD *)(v78 + 424);
                          v246 = &unk_49D9DB0;
                          if ( v210 != &v211 )
                            j_j___libc_free_0((unsigned __int64)v210);
                          if ( (__int64 *)v208[0] != &v209 )
                            j_j___libc_free_0(v208[0]);
LABEL_116:
                          v260 = &unk_49D9D40;
                          sub_23FD590((__int64)v263);
                          sub_1049740(a4, (__int64)&v246);
                          v246 = &unk_49D9D40;
                          sub_23FD590((__int64)&v254);
                          goto LABEL_117;
                        }
                        v192 = v186;
                      }
                    }
                  }
                }
              }
              v182 += 8;
            }
            while ( v85 != (__int64 *)v182 );
            if ( v192 )
            {
              v112 = (__int64 *)v173;
              v113 = 0;
              v114 = 0;
              do
              {
                srcb = (void *)*v112;
                v115 = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(a2 + 40))(
                         *(_QWORD *)(a2 + 48),
                         *(_QWORD *)(*v112 + 72));
                v116 = sub_26ABAF0((__int64)srcb, v115);
                if ( (_DWORD)v117 == 1 )
                  v113 = 1;
                v122 = __OFADD__(v116, v114);
                v114 += v116;
                if ( v122 )
                {
                  v114 = 0x8000000000000000LL;
                  if ( v116 > 0 )
                    v114 = 0x7FFFFFFFFFFFFFFFLL;
                }
                ++v112;
              }
              while ( v85 != v112 );
              if ( !(_BYTE)qword_4FF5BC8 )
              {
                v122 = __OFSUB__(v113, v175);
                v121 = (int)(v113 - v175) < 0;
                if ( v113 == v175 )
                {
                  v122 = __OFSUB__(v114, v174);
                  v121 = (__int64)(v114 - v174) < 0;
                }
                if ( v121 != v122 )
                {
                  v155 = *a4;
                  v156 = sub_B2BE50(*a4);
                  if ( !sub_B6EA50(v156) )
                  {
                    v170 = sub_B2BE50(v155);
                    v171 = sub_B6F970(v170);
                    if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v171 + 48LL))(v171) )
                      goto LABEL_117;
                  }
                  v157 = *(_QWORD *)(sub_B46EC0(v188, v191) + 56);
                  if ( v157 )
                    v157 -= 24;
                  sub_B178C0((__int64)&v260, (__int64)"partial-inlining", (__int64)"TooCostly", 9, v157);
                  sub_B16080((__int64)v208, "Callee", 6, (unsigned __int8 *)a3);
                  v158 = sub_26AC990((__int64)&v260, (__int64)v208);
                  sub_B18290(v158, " inline cost-savings smaller than ", 0x22u);
                  v172 = v175 | v172 & 0xFFFFFFFF00000000LL;
                  sub_B16D50((__int64)v206, "Cost", 4, v174, v172);
                  v160 = sub_B826F0(v158, (__int64)v206);
                  v247 = *(_DWORD *)(v160 + 8);
                  v248 = *(_BYTE *)(v160 + 12);
                  v249 = *(_QWORD *)(v160 + 16);
                  v163 = _mm_loadu_si128((const __m128i *)(v160 + 24));
                  v246 = &unk_49D9D40;
                  v250 = v163;
                  v251 = *(_QWORD *)(v160 + 40);
                  v252 = _mm_loadu_si128((const __m128i *)(v160 + 48));
                  v164 = _mm_loadu_si128((const __m128i *)(v160 + 64));
                  v254 = v256;
                  v255 = 0x400000000LL;
                  v253 = v164;
                  v165 = *(unsigned int *)(v160 + 88);
                  if ( (_DWORD)v165 )
                    sub_26ACA40((__int64)&v254, v160 + 80, v165, v159, v161, v162);
                  v257 = *(_BYTE *)(v160 + 416);
                  v258 = *(_DWORD *)(v160 + 420);
                  v259 = *(_QWORD *)(v160 + 424);
                  v246 = &unk_49D9DE8;
                  sub_2240A30(v207);
                  sub_2240A30((unsigned __int64 *)v206);
                  sub_2240A30((unsigned __int64 *)&v210);
                  sub_2240A30(v208);
                  goto LABEL_116;
                }
              }
              v123 = v203;
              v124 = v216;
              v125 = &v203[(unsigned int)v204];
              if ( v125 != v203 )
              {
                while ( 2 )
                {
                  v126 = *v123;
                  if ( v124 )
                  {
                    v127 = v213;
                    v118 = HIDWORD(v214);
                    v117 = &v213[HIDWORD(v214)];
                    if ( v213 != v117 )
                    {
                      while ( v126 != *v127 )
                      {
                        if ( v117 == ++v127 )
                          goto LABEL_214;
                      }
LABEL_184:
                      if ( v125 == ++v123 )
                        goto LABEL_185;
                      continue;
                    }
LABEL_214:
                    if ( HIDWORD(v214) < (unsigned int)v214 )
                    {
                      v118 = (unsigned int)++HIDWORD(v214);
                      *v117 = v126;
                      v124 = v216;
                      ++v212;
                      goto LABEL_184;
                    }
                  }
                  break;
                }
                sub_C8CC70((__int64)&v212, v126, (__int64)v117, v118, v119, v120);
                v124 = v216;
                goto LABEL_184;
              }
LABEL_185:
              v128 = sub_AA56F0(v192);
              v129 = v203;
              v130 = v204;
              v131 = v128;
              v132 = *v203;
              v133 = 8LL * (unsigned int)v204;
              v260 = v262;
              v261 = 0x800000000LL;
              if ( (unsigned int)v204 > 8uLL )
              {
                v187 = v128;
                srcc = v203;
                sub_C8D5F0((__int64)&v260, v262, (unsigned int)v204, 8u, (__int64)v203, v128);
                v129 = srcc;
                v131 = v187;
                v168 = &v260[(unsigned int)v261];
              }
              else
              {
                if ( !v133 )
                {
LABEL_187:
                  v263[0] = v132;
                  LODWORD(v261) = v133 + v130;
                  v134 = (char *)&v260;
                  v135 = *(unsigned int *)(v178 + 8);
                  v136 = *(unsigned __int64 **)v178;
                  v263[1] = v192;
                  v137 = *(unsigned int *)(v178 + 12);
                  v263[2] = v131;
                  v138 = v135 + 1;
                  v139 = v135;
                  if ( v135 + 1 > v137 )
                  {
                    if ( v136 > (unsigned __int64 *)&v260 || &v260 >= (_QWORD **)&v136[13 * v135] )
                    {
                      v134 = (char *)&v260;
                      sub_26AB4F0(v178, v138, v135, (__int64)v136, (__int64)v129, v131);
                      v135 = *(unsigned int *)(v178 + 8);
                      v136 = *(unsigned __int64 **)v178;
                      v139 = *(_DWORD *)(v178 + 8);
                    }
                    else
                    {
                      v169 = (char *)((char *)&v260 - (char *)v136);
                      sub_26AB4F0(v178, v138, v135, (__int64)v136, (__int64)v129, v131);
                      v136 = *(unsigned __int64 **)v178;
                      v135 = *(unsigned int *)(v178 + 8);
                      v134 = &v169[*(_QWORD *)v178];
                      v139 = *(_DWORD *)(v178 + 8);
                    }
                  }
                  v140 = 13 * v135;
                  v141 = &v136[v140];
                  if ( v141 )
                  {
                    *v141 = (unsigned __int64)(v141 + 2);
                    v141[1] = 0x800000000LL;
                    if ( *((_DWORD *)v134 + 2) )
                      sub_26AB6C0((__int64)&v136[v140], (__int64)v134, v140, (__int64)v136, (__int64)v129, v131);
                    v141[10] = *((_QWORD *)v134 + 10);
                    v141[11] = *((_QWORD *)v134 + 11);
                    v141[12] = *((_QWORD *)v134 + 12);
                    v139 = *(_DWORD *)(v178 + 8);
                  }
                  v142 = v260;
                  *(_DWORD *)(v178 + 8) = v139 + 1;
                  if ( v142 != v262 )
                    _libc_free((unsigned __int64)v142);
                  if ( v203 != (__int64 *)v205 )
                    _libc_free((unsigned __int64)v203);
                  ++v191;
                  v189 = v190;
                  v33 = *(_QWORD *)(v185 + 48);
                  goto LABEL_49;
                }
                v168 = v262;
              }
              srcd = (void *)v131;
              memcpy(v168, v129, v133);
              LODWORD(v133) = v261;
              v131 = (__int64)srcd;
              goto LABEL_187;
            }
            v85 = v108;
          }
        }
        else
        {
LABEL_117:
          v85 = v203;
        }
        if ( v85 != (__int64 *)v205 )
          _libc_free((unsigned __int64)v85);
      }
      else
      {
LABEL_100:
        sub_C8CC70((__int64)&v212, v41, (__int64)v39, v40, v42, v43);
        if ( v68 )
          goto LABEL_101;
      }
      ++v191;
      v33 = *(_QWORD *)(v185 + 48);
LABEL_49:
      v37 = v33 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v183 != (v33 & 0xFFFFFFFFFFFFFFF8LL) )
        goto LABEL_39;
      goto LABEL_50;
    }
  }
LABEL_141:
  if ( v189 )
  {
    v152 = v178;
    v104 = v216 == 0;
    v178 = 0;
    *a1 = v152;
    if ( !v104 )
      goto LABEL_143;
LABEL_210:
    _libc_free((unsigned __int64)v213);
    goto LABEL_143;
  }
  v104 = v216 == 0;
  *a1 = 0;
  if ( v104 )
    goto LABEL_210;
LABEL_143:
  if ( v200 )
  {
    v7 = v202 - v200;
    j_j___libc_free_0((unsigned __int64)v200);
  }
  if ( v178 )
  {
    v105 = *(unsigned __int64 **)v178;
    v106 = (unsigned __int64 *)(*(_QWORD *)v178 + 104LL * *(unsigned int *)(v178 + 8));
    if ( *(unsigned __int64 **)v178 != v106 )
    {
      do
      {
        v106 -= 13;
        if ( (unsigned __int64 *)*v106 != v106 + 2 )
          _libc_free(*v106);
      }
      while ( v105 != v106 );
      v106 = *(unsigned __int64 **)v178;
    }
    if ( v106 != (unsigned __int64 *)(v178 + 16) )
      _libc_free((unsigned __int64)v106);
    v7 = 432;
    j_j___libc_free_0(v178);
  }
LABEL_52:
  if ( v177 )
  {
    sub_FDC110(v177);
    v7 = 8;
    j_j___libc_free_0((unsigned __int64)v177);
  }
  sub_D77880((__int64)v241);
  sub_D786F0((__int64)v228);
  v45 = v232;
  v46 = v231;
  if ( v231 != v232 )
  {
    do
    {
      v47 = *(_QWORD *)v46;
      v48 = *(__int64 **)(*(_QWORD *)v46 + 8LL);
      v49 = *(__int64 **)(*(_QWORD *)v46 + 16LL);
      if ( v48 == v49 )
      {
        *(_BYTE *)(v47 + 152) = 1;
      }
      else
      {
        do
        {
          v50 = *v48++;
          sub_D47BB0(v50, v7);
        }
        while ( v49 != v48 );
        *(_BYTE *)(v47 + 152) = 1;
        v51 = *(_QWORD *)(v47 + 8);
        if ( v51 != *(_QWORD *)(v47 + 16) )
          *(_QWORD *)(v47 + 16) = v51;
      }
      v52 = *(_QWORD *)(v47 + 32);
      if ( v52 != *(_QWORD *)(v47 + 40) )
        *(_QWORD *)(v47 + 40) = v52;
      ++*(_QWORD *)(v47 + 56);
      if ( *(_BYTE *)(v47 + 84) )
      {
        *(_QWORD *)v47 = 0;
      }
      else
      {
        v53 = 4 * (*(_DWORD *)(v47 + 76) - *(_DWORD *)(v47 + 80));
        v54 = *(unsigned int *)(v47 + 72);
        if ( v53 < 0x20 )
          v53 = 32;
        if ( (unsigned int)v54 > v53 )
        {
          sub_C8C990(v47 + 56, v7);
        }
        else
        {
          v7 = 0xFFFFFFFFLL;
          memset(*(void **)(v47 + 64), -1, 8 * v54);
        }
        v55 = *(_BYTE *)(v47 + 84);
        *(_QWORD *)v47 = 0;
        if ( !v55 )
          _libc_free(*(_QWORD *)(v47 + 64));
      }
      v56 = *(_QWORD *)(v47 + 32);
      if ( v56 )
      {
        v7 = *(_QWORD *)(v47 + 48) - v56;
        j_j___libc_free_0(v56);
      }
      v57 = *(_QWORD *)(v47 + 8);
      if ( v57 )
      {
        v7 = *(_QWORD *)(v47 + 24) - v57;
        j_j___libc_free_0(v57);
      }
      v46 += 8LL;
    }
    while ( v45 != v46 );
    if ( v231 != v232 )
      v232 = v231;
  }
  v58 = v238;
  v59 = &v238[2 * v239];
  if ( v238 != v59 )
  {
    do
    {
      v60 = v58[1];
      v61 = *v58;
      v58 += 2;
      sub_C7D6A0(v61, v60, 16);
    }
    while ( v59 != v58 );
  }
  v239 = 0;
  if ( !v236 )
    goto LABEL_78;
  v143 = v235;
  v240 = 0;
  v144 = &v235[v236];
  v145 = v235 + 1;
  v233 = *v235;
  for ( k = v233 + 4096; v144 != v145; v143 = v235 )
  {
    v146 = *v145;
    v147 = (unsigned int)(v145 - v143) >> 7;
    v148 = 4096LL << v147;
    if ( v147 >= 0x1E )
      v148 = 0x40000000000LL;
    ++v145;
    sub_C7D6A0(v146, v148, 16);
  }
  v236 = 1;
  sub_C7D6A0(*v143, 4096, 16);
  v149 = v238;
  v62 = &v238[2 * v239];
  if ( v238 != v62 )
  {
    do
    {
      v150 = v149[1];
      v151 = *v149;
      v149 += 2;
      sub_C7D6A0(v151, v150, 16);
    }
    while ( v62 != v149 );
LABEL_78:
    v62 = v238;
  }
  if ( v62 != &v240 )
    _libc_free((unsigned __int64)v62);
  if ( v235 != (__int64 *)&v237 )
    _libc_free((unsigned __int64)v235);
  if ( v231 )
    j_j___libc_free_0(v231);
  sub_C7D6A0(v229, 16LL * v230, 8);
  v63 = v220;
  v64 = &v220[8 * (unsigned int)v221];
  if ( v220 != v64 )
  {
    do
    {
      v65 = *((_QWORD *)v64 - 1);
      v64 -= 8;
      if ( v65 )
      {
        v66 = *(_QWORD *)(v65 + 24);
        if ( v66 != v65 + 40 )
          _libc_free(v66);
        j_j___libc_free_0(v65);
      }
    }
    while ( v63 != v64 );
    v64 = v220;
  }
  if ( v64 != v222 )
    _libc_free((unsigned __int64)v64);
  if ( (char *)v218[0] != &v219 )
    _libc_free(v218[0]);
  return a1;
}
