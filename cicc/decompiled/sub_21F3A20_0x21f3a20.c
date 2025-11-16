// Function: sub_21F3A20
// Address: 0x21f3a20
//
void __fastcall sub_21F3A20(
        unsigned __int8 *a1,
        __int64 a2,
        unsigned __int64 a3,
        _QWORD *a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  unsigned __int64 v12; // rbx
  __int64 v13; // r12
  char v14; // al
  int v15; // r13d
  __int64 ***v16; // rax
  _QWORD *v17; // rax
  unsigned __int8 *v18; // rsi
  __int64 **v19; // rax
  _BYTE *v20; // rbx
  __int64 v21; // r15
  int v22; // r13d
  _QWORD *v23; // rax
  __int64 v24; // rax
  unsigned __int8 *v25; // rsi
  unsigned __int8 *v26; // rbx
  __int64 v27; // r14
  __int64 *v28; // r13
  __int64 v29; // rbx
  __int64 v30; // r11
  __int64 *v31; // rcx
  unsigned __int64 v32; // rdx
  int v33; // esi
  __int64 v34; // rax
  _QWORD *v35; // rax
  __int64 v36; // r11
  __int64 v37; // r12
  __int64 *v38; // rdx
  __int64 v39; // rsi
  __int64 v40; // rax
  int v41; // r8d
  __int64 v42; // r9
  char v43; // al
  int v44; // r13d
  __int64 *v45; // rbx
  __int64 v46; // rax
  __int64 v47; // rcx
  double v48; // xmm4_8
  double v49; // xmm5_8
  __int64 v50; // rsi
  unsigned __int8 *v51; // rsi
  unsigned __int8 **v52; // r13
  unsigned __int8 *v53; // rsi
  __int64 **v54; // rdx
  _QWORD *v55; // r12
  _QWORD *v56; // r13
  unsigned __int64 v57; // rdx
  _QWORD *v58; // rax
  _BOOL4 v59; // r14d
  __int64 v60; // rax
  __int64 v61; // r14
  __int64 v62; // r12
  int v63; // r13d
  unsigned int v64; // esi
  unsigned int v65; // ecx
  unsigned int v66; // eax
  int v67; // edx
  char v68; // al
  __int64 v69; // rsi
  unsigned __int8 *v70; // rsi
  unsigned int v71; // r12d
  _QWORD *v72; // rax
  __int64 *v73; // rax
  unsigned int v74; // r12d
  _QWORD *v75; // rax
  __int64 *v76; // rax
  unsigned __int8 *v77; // rax
  _BYTE *v78; // rsi
  __int64 *v79; // rdx
  __int64 ***v80; // r14
  __int64 v81; // rax
  _BYTE *v82; // rsi
  __int64 **v83; // rax
  __int64 v84; // rdx
  unsigned __int64 *v85; // rbx
  __int64 **v86; // rax
  unsigned __int64 v87; // rcx
  __int64 v88; // rsi
  unsigned __int8 *v89; // rsi
  _QWORD *v90; // rax
  _QWORD *v91; // rax
  unsigned __int8 *v92; // rsi
  unsigned __int8 *v93; // rax
  unsigned int v94; // ecx
  __int64 v95; // rax
  __int64 v96; // rax
  __int64 v97; // rbx
  int v98; // r8d
  int v99; // r9d
  __int64 v100; // rax
  __int64 *v101; // rax
  _BYTE *v102; // rdx
  __int64 v103; // rdx
  double v104; // xmm4_8
  double v105; // xmm5_8
  unsigned __int8 *v106; // rbx
  unsigned __int8 *v107; // r14
  __int64 v108; // rdx
  _QWORD *v109; // r14
  _QWORD *v110; // rbx
  unsigned __int64 v111; // rdx
  _QWORD *v112; // rax
  _BOOL4 v113; // r15d
  __int64 v114; // rax
  __int64 *v115; // rbx
  __int64 v116; // rax
  __int64 v117; // rcx
  __int64 v118; // rsi
  unsigned __int8 *v119; // rsi
  __int64 v120; // rax
  __int64 **v121; // r12
  __int64 *v122; // r14
  int v123; // edx
  int v124; // eax
  unsigned int v125; // eax
  int v126; // r13d
  unsigned int v127; // esi
  unsigned int v128; // ecx
  unsigned int v129; // eax
  int v130; // edx
  int v131; // eax
  _QWORD *v132; // rax
  unsigned __int8 *v133; // rsi
  __int64 **v134; // rax
  __int64 v135; // r13
  double v136; // xmm4_8
  double v137; // xmm5_8
  __int64 *v138; // rdi
  __int64 v139; // rax
  __int64 **v140; // rdx
  _QWORD *v141; // rax
  __int64 v142; // r14
  unsigned __int8 *v143; // rsi
  __int64 v144; // r12
  __int64 i; // r15
  __int64 v146; // r14
  __int64 v147; // rax
  __int64 v148; // rax
  __int64 v149; // rax
  __int64 v150; // rax
  _QWORD *v151; // rax
  __int64 *v152; // r15
  __int64 v153; // rax
  __int64 v154; // rsi
  __int64 v155; // rsi
  unsigned __int8 *v156; // rsi
  _QWORD *v157; // rax
  _QWORD *v158; // r15
  __int64 *v159; // r13
  __int64 v160; // rax
  __int64 v161; // rcx
  __int64 v162; // rsi
  unsigned __int8 *v163; // rsi
  _QWORD *v164; // rax
  __int64 v165; // rax
  __int64 v166; // rax
  __int64 **v167; // rax
  __int64 v168; // r14
  _QWORD *v169; // rax
  _QWORD *v170; // r15
  _QWORD *v171; // rcx
  __int64 **v172; // rax
  __int64 *v173; // rax
  __int64 v174; // rax
  __int64 v175; // rcx
  __int64 *v176; // r11
  __int64 **v177; // rax
  __int64 *v178; // r12
  __int64 v179; // rax
  __int64 v180; // rcx
  __int64 v181; // rsi
  unsigned __int8 *v182; // rsi
  __int64 *v183; // rax
  unsigned int v184; // eax
  __int64 v185; // rbx
  __int64 v186; // r12
  __int64 v187; // rax
  __int64 **v188; // rdx
  _QWORD *v189; // r15
  __int64 v190; // rsi
  __int64 v191; // rax
  __int64 v192; // rsi
  __int64 v193; // rdx
  unsigned __int8 *v194; // rsi
  unsigned __int8 *v195; // rsi
  _QWORD *v196; // rax
  __int64 v197; // rax
  __int64 v198; // r11
  _QWORD *v199; // rax
  _QWORD *v200; // r15
  __int64 *v201; // r13
  __int64 v202; // rax
  __int64 v203; // rcx
  __int64 v204; // rsi
  unsigned __int8 *v205; // rsi
  __int64 *v206; // r15
  __int64 v207; // rax
  __int64 v208; // rcx
  __int64 v209; // rsi
  unsigned __int8 *v210; // rsi
  __int64 *v211; // rbx
  __int64 v212; // rax
  __int64 v213; // rcx
  __int64 v214; // rsi
  unsigned __int8 *v215; // rsi
  _QWORD *v216; // rax
  __int64 v217; // rax
  __int64 v218; // rax
  __int64 **v219; // rax
  _QWORD *v220; // rax
  _QWORD *v221; // r15
  __int64 **v222; // rax
  __int64 *v223; // rax
  __int64 *v224; // rdi
  __int64 **v225; // rax
  __int64 *v226; // r12
  __int64 v227; // rax
  __int64 v228; // rcx
  __int64 v229; // rsi
  unsigned __int8 *v230; // rsi
  unsigned __int8 *v231; // rbx
  unsigned __int8 *v232; // r12
  __int64 v233; // rdx
  unsigned int v234; // [rsp+0h] [rbp-1F0h]
  __int64 *v235; // [rsp+10h] [rbp-1E0h]
  __int64 v236; // [rsp+20h] [rbp-1D0h]
  __int64 v238; // [rsp+30h] [rbp-1C0h]
  __int64 *v241; // [rsp+48h] [rbp-1A8h]
  unsigned int v242; // [rsp+48h] [rbp-1A8h]
  __int64 v243; // [rsp+50h] [rbp-1A0h]
  __int64 v244; // [rsp+50h] [rbp-1A0h]
  __int64 v245; // [rsp+50h] [rbp-1A0h]
  __int64 v246; // [rsp+50h] [rbp-1A0h]
  int v247; // [rsp+50h] [rbp-1A0h]
  __int64 v248; // [rsp+58h] [rbp-198h]
  unsigned int v249; // [rsp+58h] [rbp-198h]
  __int64 v250; // [rsp+58h] [rbp-198h]
  __int64 *v252; // [rsp+68h] [rbp-188h]
  int v253; // [rsp+68h] [rbp-188h]
  __int64 v254; // [rsp+68h] [rbp-188h]
  __int64 v255; // [rsp+70h] [rbp-180h]
  __int64 v256; // [rsp+70h] [rbp-180h]
  __int64 v257; // [rsp+70h] [rbp-180h]
  __int64 v258; // [rsp+70h] [rbp-180h]
  __int64 v259; // [rsp+70h] [rbp-180h]
  char v260; // [rsp+78h] [rbp-178h]
  int v261; // [rsp+78h] [rbp-178h]
  unsigned int v262; // [rsp+78h] [rbp-178h]
  __int64 *v263; // [rsp+78h] [rbp-178h]
  __int64 v264; // [rsp+78h] [rbp-178h]
  __int64 ***v265; // [rsp+88h] [rbp-168h] BYREF
  _QWORD *v266; // [rsp+90h] [rbp-160h] BYREF
  unsigned __int8 *v267; // [rsp+98h] [rbp-158h] BYREF
  __int64 *v268; // [rsp+A0h] [rbp-150h] BYREF
  _BYTE *v269; // [rsp+A8h] [rbp-148h]
  _BYTE *v270; // [rsp+B0h] [rbp-140h]
  unsigned __int8 *v271; // [rsp+C0h] [rbp-130h] BYREF
  unsigned __int8 *v272; // [rsp+C8h] [rbp-128h]
  unsigned __int8 *v273; // [rsp+D0h] [rbp-120h]
  unsigned __int8 *v274[2]; // [rsp+E0h] [rbp-110h] BYREF
  __int16 v275; // [rsp+F0h] [rbp-100h]
  unsigned __int8 *v276[2]; // [rsp+100h] [rbp-F0h] BYREF
  unsigned __int64 v277; // [rsp+110h] [rbp-E0h]
  _QWORD *v278; // [rsp+118h] [rbp-D8h]
  __int64 v279; // [rsp+120h] [rbp-D0h]
  int v280; // [rsp+128h] [rbp-C8h]
  __int64 v281; // [rsp+130h] [rbp-C0h]
  __int64 v282; // [rsp+138h] [rbp-B8h]
  unsigned __int8 *v283; // [rsp+150h] [rbp-A0h] BYREF
  __int64 **v284; // [rsp+158h] [rbp-98h]
  __int64 *v285; // [rsp+160h] [rbp-90h]
  __int64 v286; // [rsp+168h] [rbp-88h]
  __int64 v287; // [rsp+170h] [rbp-80h] BYREF
  int v288; // [rsp+178h] [rbp-78h]
  __int64 *v289; // [rsp+180h] [rbp-70h] BYREF
  __int64 v290; // [rsp+188h] [rbp-68h]
  _BYTE v291[32]; // [rsp+190h] [rbp-60h] BYREF
  __int64 v292; // [rsp+1B0h] [rbp-40h]
  int v293; // [rsp+1B8h] [rbp-38h]
  int v294; // [rsp+1BCh] [rbp-34h]

  v12 = a3 + 24;
  v13 = *(_QWORD *)a3;
  v14 = *(_BYTE *)(*(_QWORD *)a3 + 8LL);
  if ( v14 == 11 )
    goto LABEL_67;
  if ( (unsigned __int8)(v14 - 1) <= 5u )
  {
LABEL_3:
    v260 = 0;
    v15 = 4060;
    goto LABEL_4;
  }
  if ( v14 != 13 )
  {
    if ( v14 != 16 || (v61 = *(_QWORD *)(v13 + 24), (unsigned int)sub_1643030(v61) <= 7) )
    {
      v260 = 1;
      v15 = 4062;
      goto LABEL_4;
    }
    v62 = *(_QWORD *)(v13 + 32);
    v63 = 1 << (*(unsigned __int16 *)(a3 + 18) >> 1) >> 1;
    v64 = v62 * sub_15A9FE0((__int64)a1, v61);
    v65 = v64;
    v66 = v63;
    if ( v64 )
    {
      while ( 1 )
      {
        v67 = v66 % v65;
        v66 = v65;
        if ( !v67 )
          break;
        v65 = v67;
      }
    }
    else
    {
      v65 = v63;
    }
    if ( v64 == v65
      && (((_DWORD)v62 - 2) & 0xFFFFFFFD) == 0
      && (unsigned int)sub_1643030(v61) * (unsigned int)v62 <= 0x80 )
    {
      v68 = *(_BYTE *)(v61 + 8);
      if ( v68 != 11 )
      {
        if ( (unsigned __int8)(v68 - 1) <= 5u )
          goto LABEL_3;
        v260 = 0;
        v15 = 4062;
LABEL_4:
        v268 = 0;
        v269 = 0;
        v16 = *(__int64 ****)(a3 - 24);
        v270 = 0;
        v265 = v16;
        v17 = (_QWORD *)sub_16498A0(a3);
        v18 = *(unsigned __int8 **)(a3 + 48);
        v283 = 0;
        v286 = (__int64)v17;
        v19 = *(__int64 ***)(a3 + 40);
        v287 = 0;
        v288 = 0;
        v289 = 0;
        v290 = 0;
        v284 = v19;
        v285 = (__int64 *)v12;
        v276[0] = v18;
        if ( v18 )
        {
          sub_1623A60((__int64)v276, (__int64)v18, 2);
          if ( v283 )
            sub_161E7C0((__int64)&v283, (__int64)v283);
          v283 = v276[0];
          if ( v276[0] )
            sub_1623210((__int64)v276, v276[0], (__int64)&v283);
        }
        if ( v260 )
        {
          v71 = *(_DWORD *)(*(_QWORD *)a3 + 8LL);
          v72 = (_QWORD *)sub_15E0530(a2);
          v73 = (__int64 *)sub_1643330(v72);
          v276[0] = (unsigned __int8 *)sub_1646BA0(v73, v71 >> 8);
          sub_1278040((__int64)&v268, v269, v276);
          v74 = *((_DWORD *)*v265 + 2);
          v75 = (_QWORD *)sub_15E0530(a2);
          v76 = (__int64 *)sub_1643350(v75);
          v77 = (unsigned __int8 *)sub_1646BA0(v76, v74 >> 8);
          v78 = v269;
          v276[0] = v77;
          if ( v269 == v270 )
          {
            sub_1278040((__int64)&v268, v269, v276);
          }
          else
          {
            if ( v269 )
            {
              *(_QWORD *)v269 = v77;
              v78 = v269;
            }
            v269 = v78 + 8;
          }
          v79 = v268;
          v80 = v265;
          v274[0] = "bitCast";
          v275 = 259;
          if ( (__int64 **)v268[1] != *v265 )
          {
            if ( *((_BYTE *)v265 + 16) > 0x10u )
            {
              v84 = v268[1];
              LOWORD(v277) = 257;
              v80 = (__int64 ***)sub_15FDBD0(47, (__int64)v265, v84, (__int64)v276, 0);
              if ( v284 )
              {
                v85 = (unsigned __int64 *)v285;
                sub_157E9D0((__int64)(v284 + 5), (__int64)v80);
                v86 = v80[3];
                v87 = *v85;
                v80[4] = (__int64 **)v85;
                v87 &= 0xFFFFFFFFFFFFFFF8LL;
                v80[3] = (__int64 **)(v87 | (unsigned __int8)v86 & 7);
                *(_QWORD *)(v87 + 8) = v80 + 3;
                *v85 = *v85 & 7 | (unsigned __int64)(v80 + 3);
              }
              sub_164B780((__int64)v80, (__int64 *)v274);
              if ( v283 )
              {
                v271 = v283;
                sub_1623A60((__int64)&v271, (__int64)v283, 2);
                v88 = (__int64)v80[6];
                if ( v88 )
                  sub_161E7C0((__int64)(v80 + 6), v88);
                v89 = v271;
                v80[6] = (__int64 **)v271;
                if ( v89 )
                  sub_1623210((__int64)&v271, v89, (__int64)(v80 + 6));
              }
              v79 = v268;
            }
            else
            {
              v81 = sub_15A46C0(47, v265, (__int64 **)v268[1], 0);
              v79 = v268;
              v80 = (__int64 ***)v81;
            }
          }
          v265 = v80;
          v21 = sub_15E26F0(*(__int64 **)(a2 + 40), v15, v79, (v269 - (_BYTE *)v79) >> 3);
        }
        else
        {
          v276[0] = *(unsigned __int8 **)a3;
          sub_1278040((__int64)&v268, v270, v276);
          v82 = v269;
          v83 = *v265;
          v276[0] = (unsigned __int8 *)*v265;
          if ( v270 == v269 )
          {
            sub_1278040((__int64)&v268, v269, v276);
            v20 = v269;
          }
          else
          {
            if ( v269 )
            {
              *(_QWORD *)v269 = v83;
              v82 = v269;
            }
            v20 = v82 + 8;
            v269 = v82 + 8;
          }
          v21 = sub_15E26F0(*(__int64 **)(a2 + 40), v15, v268, (v20 - (_BYTE *)v268) >> 3);
        }
        v22 = 1 << (*(unsigned __int16 *)(a3 + 18) >> 1);
        v23 = (_QWORD *)sub_15E0530(a2);
        v24 = sub_1643350(v23);
        v266 = (_QWORD *)sub_159C470(v24, (unsigned int)(v22 >> 1), 0);
        v271 = 0;
        v272 = 0;
        v273 = 0;
        sub_1287830((__int64)&v271, 0, &v265);
        v25 = v272;
        if ( v272 == v273 )
        {
          sub_1287830((__int64)&v271, v272, &v266);
          v26 = v272;
        }
        else
        {
          if ( v272 )
          {
            *(_QWORD *)v272 = v266;
            v25 = v272;
          }
          v26 = v25 + 8;
          v272 = v25 + 8;
        }
        v27 = v290;
        v28 = v289;
        v274[0] = "LDG";
        v252 = (__int64 *)v271;
        v29 = (v26 - v271) >> 3;
        v275 = 259;
        v30 = *(_QWORD *)(v21 + 24);
        v31 = &v289[7 * v290];
        LOWORD(v277) = 257;
        if ( v289 == v31 )
        {
          v245 = v30;
          v90 = sub_1648AB0(72, (int)v29 + 1, 16 * (int)v290);
          v41 = v29 + 1;
          v36 = v245;
          v42 = v29;
          v37 = (__int64)v90;
          if ( v90 )
          {
            v248 = (__int64)v90;
LABEL_24:
            v244 = v36;
            sub_15F1EA0(v37, **(_QWORD **)(v36 + 16), 54, v37 - 24 * v42 - 24, v41, 0);
            *(_QWORD *)(v37 + 56) = 0;
            sub_15F5B40(v37, v244, v21, v252, v29, (__int64)v276, v28, v27);
            goto LABEL_25;
          }
        }
        else
        {
          v32 = (unsigned __int64)v289;
          v33 = 0;
          do
          {
            v34 = *(_QWORD *)(v32 + 40) - *(_QWORD *)(v32 + 32);
            v32 += 56LL;
            v33 += v34 >> 3;
          }
          while ( v31 != (__int64 *)v32 );
          v241 = &v289[7 * v290];
          v243 = v30;
          v35 = sub_1648AB0(72, (int)v29 + 1 + v33, 16 * (int)v290);
          v36 = v243;
          v37 = (__int64)v35;
          if ( v35 )
          {
            v248 = (__int64)v35;
            v38 = v28;
            LODWORD(v39) = 0;
            do
            {
              v40 = v38[5] - v38[4];
              v38 += 7;
              v39 = (unsigned int)(v40 >> 3) + (unsigned int)v39;
            }
            while ( v241 != v38 );
            v41 = v39 + v29 + 1;
            v42 = v29 + v39;
            goto LABEL_24;
          }
        }
        v248 = 0;
        v37 = 0;
LABEL_25:
        v43 = *(_BYTE *)(*(_QWORD *)v37 + 8LL);
        if ( v43 == 16 )
          v43 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)v37 + 16LL) + 8LL);
        if ( (unsigned __int8)(v43 - 1) <= 5u || *(_BYTE *)(v37 + 16) == 76 )
        {
          v44 = v288;
          if ( v287 )
            sub_1625C10(v37, 3, v287);
          sub_15F2440(v37, v44);
        }
        if ( v284 )
        {
          v45 = v285;
          sub_157E9D0((__int64)(v284 + 5), v37);
          v46 = *(_QWORD *)(v37 + 24);
          v47 = *v45;
          *(_QWORD *)(v37 + 32) = v45;
          v47 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v37 + 24) = v47 | v46 & 7;
          *(_QWORD *)(v47 + 8) = v37 + 24;
          *v45 = *v45 & 7 | (v37 + 24);
        }
        sub_164B780(v248, (__int64 *)v274);
        if ( v283 )
        {
          v276[0] = v283;
          sub_1623A60((__int64)v276, (__int64)v283, 2);
          v50 = *(_QWORD *)(v37 + 48);
          if ( v50 )
            sub_161E7C0(v37 + 48, v50);
          v51 = v276[0];
          *(unsigned __int8 **)(v37 + 48) = v276[0];
          if ( v51 )
            sub_1623210((__int64)v276, v51, v37 + 48);
        }
        if ( *(_BYTE *)(v37 + 16) <= 0x17u )
        {
LABEL_43:
          if ( v260 )
          {
            v54 = *(__int64 ***)a3;
            if ( *(_QWORD *)a3 != *(_QWORD *)v37 )
            {
              v274[0] = "bitCast";
              v275 = 259;
              if ( v54 != *(__int64 ***)v37 )
              {
                if ( *(_BYTE *)(v37 + 16) > 0x10u )
                {
                  LOWORD(v277) = 257;
                  v37 = sub_15FDBD0(47, v37, (__int64)v54, (__int64)v276, 0);
                  if ( v284 )
                  {
                    v115 = v285;
                    sub_157E9D0((__int64)(v284 + 5), v37);
                    v116 = *(_QWORD *)(v37 + 24);
                    v117 = *v115;
                    *(_QWORD *)(v37 + 32) = v115;
                    v117 &= 0xFFFFFFFFFFFFFFF8LL;
                    *(_QWORD *)(v37 + 24) = v117 | v116 & 7;
                    *(_QWORD *)(v117 + 8) = v37 + 24;
                    *v115 = *v115 & 7 | (v37 + 24);
                  }
                  sub_164B780(v37, (__int64 *)v274);
                  if ( v283 )
                  {
                    v267 = v283;
                    sub_1623A60((__int64)&v267, (__int64)v283, 2);
                    v118 = *(_QWORD *)(v37 + 48);
                    if ( v118 )
                      sub_161E7C0(v37 + 48, v118);
                    v119 = v267;
                    *(_QWORD *)(v37 + 48) = v267;
                    if ( v119 )
                      sub_1623210((__int64)&v267, v119, v37 + 48);
                  }
                }
                else
                {
                  v37 = sub_15A46C0(47, (__int64 ***)v37, v54, 0);
                }
              }
            }
          }
          sub_164D160(a3, v37, a5, a6, a7, a8, v48, v49, a11, a12);
          v55 = (_QWORD *)a4[2];
          v56 = a4 + 1;
          if ( !v55 )
          {
            v55 = a4 + 1;
            if ( (_QWORD *)a4[3] == v56 )
            {
              v59 = 1;
              goto LABEL_57;
            }
            goto LABEL_69;
          }
          while ( 1 )
          {
            v57 = v55[4];
            v58 = (_QWORD *)v55[3];
            if ( a3 < v57 )
              v58 = (_QWORD *)v55[2];
            if ( !v58 )
              break;
            v55 = v58;
          }
          if ( a3 < v57 )
          {
            if ( (_QWORD *)a4[3] != v55 )
            {
LABEL_69:
              if ( a3 <= *(_QWORD *)(sub_220EF80(v55) + 32) )
                goto LABEL_58;
              v59 = 1;
              if ( v56 == v55 )
                goto LABEL_57;
              goto LABEL_71;
            }
          }
          else if ( v57 >= a3 )
          {
LABEL_58:
            if ( v271 )
              j_j___libc_free_0(v271, v273 - v271);
            if ( v283 )
              sub_161E7C0((__int64)&v283, (__int64)v283);
            if ( v268 )
              j_j___libc_free_0(v268, v270 - (_BYTE *)v268);
            return;
          }
          v59 = 1;
          if ( v56 == v55 )
          {
LABEL_57:
            v60 = sub_22077B0(40);
            *(_QWORD *)(v60 + 32) = a3;
            sub_220F040(v59, v60, v55, v56);
            ++a4[5];
            goto LABEL_58;
          }
LABEL_71:
          v59 = a3 < v55[4];
          goto LABEL_57;
        }
        v52 = (unsigned __int8 **)(v37 + 48);
        v53 = *(unsigned __int8 **)(a3 + 48);
        v276[0] = v53;
        if ( v53 )
        {
          sub_1623A60((__int64)v276, (__int64)v53, 2);
          if ( v52 == v276 )
          {
            if ( v276[0] )
              sub_161E7C0(v37 + 48, (__int64)v276[0]);
            goto LABEL_43;
          }
          v69 = *(_QWORD *)(v37 + 48);
          if ( !v69 )
            goto LABEL_88;
        }
        else
        {
          if ( v52 == v276 )
            goto LABEL_43;
          v69 = *(_QWORD *)(v37 + 48);
          if ( !v69 )
            goto LABEL_43;
        }
        sub_161E7C0(v37 + 48, v69);
LABEL_88:
        v70 = v276[0];
        *(unsigned __int8 **)(v37 + 48) = v276[0];
        if ( v70 )
          sub_1623210((__int64)v276, v70, v37 + 48);
        goto LABEL_43;
      }
LABEL_67:
      v260 = 0;
      v15 = 4061;
      goto LABEL_4;
    }
    v271 = 0;
    v272 = 0;
    v120 = *(_QWORD *)(a3 + 40);
    v121 = *(__int64 ***)a3;
    v273 = 0;
    v122 = v121[4];
    v236 = *(_QWORD *)(v120 + 56);
    v235 = v121[3];
    v246 = *(_QWORD *)(a3 - 24);
    v234 = (unsigned int)v122;
    v242 = (unsigned int)v122;
    v261 = 4;
    v123 = sub_1643030((__int64)v235);
    if ( (unsigned int)v122 <= 3 )
    {
      v124 = 2;
      if ( (unsigned int)v122 <= 2 )
        v124 = (int)v122;
      v261 = v124;
    }
    if ( (unsigned int)(v123 * v261) > 0x80 )
    {
      v125 = v261;
      do
        v125 >>= 1;
      while ( v125 * v123 > 0x80 );
      v261 = v125;
    }
    v126 = 1 << (*(unsigned __int16 *)(a3 + 18) >> 1) >> 1;
    v127 = v261 * sub_15A9FE0((__int64)a1, (__int64)v235);
    v128 = v127;
    v129 = v126;
    if ( v127 )
    {
      while ( 1 )
      {
        v130 = v129 % v128;
        v129 = v128;
        if ( !v130 )
          break;
        v128 = v130;
      }
    }
    else
    {
      v128 = v126;
    }
    v131 = 1;
    if ( v127 == v128 )
      v131 = v261;
    v262 = v131;
    v132 = (_QWORD *)sub_16498A0(a3);
    v133 = *(unsigned __int8 **)(a3 + 48);
    v283 = 0;
    v286 = (__int64)v132;
    v134 = *(__int64 ***)(a3 + 40);
    v287 = 0;
    v288 = 0;
    v289 = 0;
    v290 = 0;
    v284 = v134;
    v285 = (__int64 *)v12;
    v276[0] = v133;
    if ( v133 )
    {
      sub_1623A60((__int64)v276, (__int64)v133, 2);
      if ( v283 )
        sub_161E7C0((__int64)&v283, (__int64)v283);
      v283 = v276[0];
      if ( v276[0] )
        sub_1623210((__int64)v276, v276[0], (__int64)&v283);
    }
    v249 = 0;
    v135 = sub_1599EF0(v121);
    if ( v262 <= 1 )
      goto LABEL_242;
LABEL_176:
    v138 = sub_16463B0(v235, v262);
    v139 = **(_QWORD **)(a3 - 24);
    if ( *(_BYTE *)(v139 + 8) == 16 )
      v139 = **(_QWORD **)(v139 + 16);
    v140 = (__int64 **)sub_1646BA0(v138, *(_DWORD *)(v139 + 8) >> 8);
    v275 = 259;
    v274[0] = "vecBitCast";
    if ( v140 != *(__int64 ***)v246 )
    {
      if ( *(_BYTE *)(v246 + 16) > 0x10u )
      {
        LOWORD(v277) = 257;
        v246 = sub_15FDBD0(47, v246, (__int64)v140, (__int64)v276, 0);
        if ( v284 )
        {
          v211 = v285;
          sub_157E9D0((__int64)(v284 + 5), v246);
          v212 = *(_QWORD *)(v246 + 24);
          v213 = *v211;
          *(_QWORD *)(v246 + 32) = v211;
          v213 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v246 + 24) = v213 | v212 & 7;
          *(_QWORD *)(v213 + 8) = v246 + 24;
          *v211 = *v211 & 7 | (v246 + 24);
        }
        sub_164B780(v246, (__int64 *)v274);
        if ( v283 )
        {
          v268 = (__int64 *)v283;
          sub_1623A60((__int64)&v268, (__int64)v283, 2);
          v214 = *(_QWORD *)(v246 + 48);
          if ( v214 )
            sub_161E7C0(v246 + 48, v214);
          v215 = (unsigned __int8 *)v268;
          *(_QWORD *)(v246 + 48) = v268;
          if ( v215 )
            sub_1623210((__int64)&v268, v215, v246 + 48);
        }
      }
      else
      {
        v246 = sub_15A46C0(47, (__int64 ***)v246, v140, 0);
      }
    }
    if ( v242 < v249 + v262 )
      goto LABEL_241;
    v249 += v262;
LABEL_183:
    v141 = sub_1648A60(64, 1u);
    v142 = (__int64)v141;
    if ( v141 )
      sub_15F9100((__int64)v141, (_QWORD *)v246, "splitVec", a3);
    v266 = (_QWORD *)v142;
    v143 = v272;
    if ( v272 == v273 )
    {
      sub_14147F0((__int64)&v271, v272, &v266);
      v142 = (__int64)v266;
    }
    else
    {
      if ( v272 )
      {
        *(_QWORD *)v272 = v142;
        v143 = v272;
      }
      v272 = v143 + 8;
    }
    v144 = 0;
    for ( i = v142; ; i = (__int64)v266 )
    {
      v274[0] = "extractSplitVec";
      v275 = 259;
      v149 = sub_1643360((_QWORD *)v286);
      v150 = sub_159C470(v149, v144, 0);
      if ( *(_BYTE *)(i + 16) > 0x10u || *(_BYTE *)(v150 + 16) > 0x10u )
      {
        v255 = v150;
        LOWORD(v277) = 257;
        v151 = sub_1648A60(56, 2u);
        v146 = (__int64)v151;
        if ( v151 )
          sub_15FA320((__int64)v151, (_QWORD *)i, v255, (__int64)v276, 0);
        if ( v284 )
        {
          v152 = v285;
          sub_157E9D0((__int64)(v284 + 5), v146);
          v153 = *(_QWORD *)(v146 + 24);
          v154 = *v152;
          *(_QWORD *)(v146 + 32) = v152;
          v154 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v146 + 24) = v154 | v153 & 7;
          *(_QWORD *)(v154 + 8) = v146 + 24;
          *v152 = *v152 & 7 | (v146 + 24);
        }
        sub_164B780(v146, (__int64 *)v274);
        if ( v283 )
        {
          v268 = (__int64 *)v283;
          sub_1623A60((__int64)&v268, (__int64)v283, 2);
          v155 = *(_QWORD *)(v146 + 48);
          if ( v155 )
            sub_161E7C0(v146 + 48, v155);
          v156 = (unsigned __int8 *)v268;
          *(_QWORD *)(v146 + 48) = v268;
          if ( v156 )
            sub_1623210((__int64)&v268, v156, v146 + 48);
        }
      }
      else
      {
        v146 = sub_15A37D0((_BYTE *)i, v150, 0);
      }
      v274[0] = "insertSplitVec";
      v275 = 259;
      v147 = sub_1643360((_QWORD *)v286);
      v148 = sub_159C470(v147, v249 - v262 + (unsigned int)v144, 0);
      if ( *(_BYTE *)(v135 + 16) > 0x10u || *(_BYTE *)(v146 + 16) > 0x10u || *(_BYTE *)(v148 + 16) > 0x10u )
      {
        v256 = v148;
        LOWORD(v277) = 257;
        v157 = sub_1648A60(56, 3u);
        v158 = v157;
        if ( v157 )
          sub_15FA480((__int64)v157, (__int64 *)v135, v146, v256, (__int64)v276, 0);
        if ( v284 )
        {
          v159 = v285;
          sub_157E9D0((__int64)(v284 + 5), (__int64)v158);
          v160 = v158[3];
          v161 = *v159;
          v158[4] = v159;
          v161 &= 0xFFFFFFFFFFFFFFF8LL;
          v158[3] = v161 | v160 & 7;
          *(_QWORD *)(v161 + 8) = v158 + 3;
          *v159 = *v159 & 7 | (unsigned __int64)(v158 + 3);
        }
        sub_164B780((__int64)v158, (__int64 *)v274);
        if ( v283 )
        {
          v268 = (__int64 *)v283;
          sub_1623A60((__int64)&v268, (__int64)v283, 2);
          v162 = v158[6];
          if ( v162 )
            sub_161E7C0((__int64)(v158 + 6), v162);
          v163 = (unsigned __int8 *)v268;
          v158[6] = v268;
          if ( v163 )
            sub_1623210((__int64)&v268, v163, (__int64)(v158 + 6));
        }
        v135 = (__int64)v158;
        if ( v262 <= (unsigned int)++v144 )
        {
LABEL_217:
          if ( v242 > v249 )
          {
            v164 = (_QWORD *)sub_15E0530(v236);
            v165 = sub_1643350(v164);
            v166 = sub_159C470(v165, 1, 0);
            v274[0] = "splitVecGEP";
            v267 = (unsigned __int8 *)v166;
            v275 = 259;
            if ( *(_BYTE *)(v246 + 16) > 0x10u || *(_BYTE *)(v166 + 16) > 0x10u )
            {
              LOWORD(v277) = 257;
              v167 = *(__int64 ***)v246;
              if ( *(_BYTE *)(*(_QWORD *)v246 + 8LL) == 16 )
                v167 = (__int64 **)*v167[2];
              v168 = (__int64)v167[3];
              v169 = sub_1648A60(72, 2u);
              v170 = v169;
              if ( v169 )
              {
                v257 = (__int64)v169;
                v171 = v169 - 6;
                v172 = *(__int64 ***)v246;
                if ( *(_BYTE *)(*(_QWORD *)v246 + 8LL) == 16 )
                  v172 = (__int64 **)*v172[2];
                v238 = (__int64)v171;
                v253 = *((_DWORD *)v172 + 2) >> 8;
                v173 = (__int64 *)sub_15F9F50(v168, (__int64)&v267, 1);
                v174 = sub_1646BA0(v173, v253);
                v175 = v238;
                v176 = (__int64 *)v174;
                v177 = *(__int64 ***)v246;
                if ( *(_BYTE *)(*(_QWORD *)v246 + 8LL) == 16
                  || (v177 = *(__int64 ***)v267, *(_BYTE *)(*(_QWORD *)v267 + 8LL) == 16) )
                {
                  v183 = sub_16463B0(v176, (unsigned int)v177[4]);
                  v175 = v238;
                  v176 = v183;
                }
                sub_15F1EA0((__int64)v170, (__int64)v176, 32, v175, 2, 0);
                v170[7] = v168;
                v170[8] = sub_15F9F50(v168, (__int64)&v267, 1);
                sub_15F9CE0((__int64)v170, v246, (__int64 *)&v267, 1, (__int64)v276);
              }
              else
              {
                v257 = 0;
              }
              if ( v284 )
              {
                v178 = v285;
                sub_157E9D0((__int64)(v284 + 5), (__int64)v170);
                v179 = v170[3];
                v180 = *v178;
                v170[4] = v178;
                v180 &= 0xFFFFFFFFFFFFFFF8LL;
                v170[3] = v180 | v179 & 7;
                *(_QWORD *)(v180 + 8) = v170 + 3;
                *v178 = *v178 & 7 | (unsigned __int64)(v170 + 3);
              }
              sub_164B780(v257, (__int64 *)v274);
              if ( v283 )
              {
                v268 = (__int64 *)v283;
                sub_1623A60((__int64)&v268, (__int64)v283, 2);
                v181 = v170[6];
                if ( v181 )
                  sub_161E7C0((__int64)(v170 + 6), v181);
                v182 = (unsigned __int8 *)v268;
                v170[6] = v268;
                if ( v182 )
                  sub_1623210((__int64)&v268, v182, (__int64)(v170 + 6));
              }
              v246 = (__int64)v170;
            }
            else
            {
              BYTE4(v276[0]) = 0;
              v246 = sub_15A2E80(0, v246, (__int64 **)&v267, 1u, 0, (__int64)v276, 0);
            }
          }
          if ( v242 >= v249 + v262 )
          {
            v249 += v262;
            goto LABEL_183;
          }
LABEL_241:
          v184 = v262;
          v262 = 2;
          if ( v184 >> 1 == 1 )
          {
LABEL_242:
            if ( v234 > v249 )
            {
              v185 = v249;
              v186 = v246;
              do
              {
                v187 = **(_QWORD **)(a3 - 24);
                if ( *(_BYTE *)(v187 + 8) == 16 )
                  v187 = **(_QWORD **)(v187 + 16);
                v188 = (__int64 **)sub_1646BA0(v235, *(_DWORD *)(v187 + 8) >> 8);
                v275 = 259;
                v274[0] = "vecBitCast";
                if ( v188 != *(__int64 ***)v186 )
                {
                  if ( *(_BYTE *)(v186 + 16) > 0x10u )
                  {
                    LOWORD(v277) = 257;
                    v186 = sub_15FDBD0(47, v186, (__int64)v188, (__int64)v276, 0);
                    if ( v284 )
                    {
                      v206 = v285;
                      sub_157E9D0((__int64)(v284 + 5), v186);
                      v207 = *(_QWORD *)(v186 + 24);
                      v208 = *v206;
                      *(_QWORD *)(v186 + 32) = v206;
                      v208 &= 0xFFFFFFFFFFFFFFF8LL;
                      *(_QWORD *)(v186 + 24) = v208 | v207 & 7;
                      *(_QWORD *)(v208 + 8) = v186 + 24;
                      *v206 = *v206 & 7 | (v186 + 24);
                    }
                    sub_164B780(v186, (__int64 *)v274);
                    if ( v283 )
                    {
                      v268 = (__int64 *)v283;
                      sub_1623A60((__int64)&v268, (__int64)v283, 2);
                      v209 = *(_QWORD *)(v186 + 48);
                      if ( v209 )
                        sub_161E7C0(v186 + 48, v209);
                      v210 = (unsigned __int8 *)v268;
                      *(_QWORD *)(v186 + 48) = v268;
                      if ( v210 )
                        sub_1623210((__int64)&v268, v210, v186 + 48);
                    }
                  }
                  else
                  {
                    v186 = sub_15A46C0(47, (__int64 ***)v186, v188, 0);
                  }
                }
                v276[0] = "splitVec";
                LOWORD(v277) = 259;
                v189 = sub_1648A60(64, 1u);
                if ( v189 )
                  sub_15F9210((__int64)v189, *(_QWORD *)(*(_QWORD *)v186 + 24LL), v186, 0, 0, 0);
                if ( v284 )
                {
                  v263 = v285;
                  sub_157E9D0((__int64)(v284 + 5), (__int64)v189);
                  v190 = *v263;
                  v191 = v189[3] & 7LL;
                  v189[4] = v263;
                  v190 &= 0xFFFFFFFFFFFFFFF8LL;
                  v189[3] = v190 | v191;
                  *(_QWORD *)(v190 + 8) = v189 + 3;
                  *v263 = *v263 & 7 | (unsigned __int64)(v189 + 3);
                }
                sub_164B780((__int64)v189, (__int64 *)v276);
                if ( v283 )
                {
                  v274[0] = v283;
                  sub_1623A60((__int64)v274, (__int64)v283, 2);
                  v192 = v189[6];
                  v193 = (__int64)(v189 + 6);
                  if ( v192 )
                  {
                    sub_161E7C0((__int64)(v189 + 6), v192);
                    v193 = (__int64)(v189 + 6);
                  }
                  v194 = v274[0];
                  v189[6] = v274[0];
                  if ( v194 )
                    sub_1623210((__int64)v274, v194, v193);
                }
                v266 = v189;
                v195 = v272;
                if ( v272 == v273 )
                {
                  sub_14147F0((__int64)&v271, v272, &v266);
                }
                else
                {
                  if ( v272 )
                  {
                    *(_QWORD *)v272 = v189;
                    v195 = v272;
                  }
                  v272 = v195 + 8;
                }
                v196 = (_QWORD *)sub_15E0530(v236);
                v197 = sub_1643350(v196);
                v198 = sub_159C470(v197, v185, 0);
                v275 = 259;
                v274[0] = "insertSplitVec";
                if ( *(_BYTE *)(v135 + 16) > 0x10u || *((_BYTE *)v266 + 16) > 0x10u || *(_BYTE *)(v198 + 16) > 0x10u )
                {
                  v258 = (__int64)v266;
                  v264 = v198;
                  LOWORD(v277) = 257;
                  v199 = sub_1648A60(56, 3u);
                  v200 = v199;
                  if ( v199 )
                    sub_15FA480((__int64)v199, (__int64 *)v135, v258, v264, (__int64)v276, 0);
                  if ( v284 )
                  {
                    v201 = v285;
                    sub_157E9D0((__int64)(v284 + 5), (__int64)v200);
                    v202 = v200[3];
                    v203 = *v201;
                    v200[4] = v201;
                    v203 &= 0xFFFFFFFFFFFFFFF8LL;
                    v200[3] = v203 | v202 & 7;
                    *(_QWORD *)(v203 + 8) = v200 + 3;
                    *v201 = *v201 & 7 | (unsigned __int64)(v200 + 3);
                  }
                  sub_164B780((__int64)v200, (__int64 *)v274);
                  if ( v283 )
                  {
                    v268 = (__int64 *)v283;
                    sub_1623A60((__int64)&v268, (__int64)v283, 2);
                    v204 = v200[6];
                    if ( v204 )
                      sub_161E7C0((__int64)(v200 + 6), v204);
                    v205 = (unsigned __int8 *)v268;
                    v200[6] = v268;
                    if ( v205 )
                      sub_1623210((__int64)&v268, v205, (__int64)(v200 + 6));
                  }
                  v135 = (__int64)v200;
                }
                else
                {
                  v135 = sub_15A3890((__int64 *)v135, (__int64)v266, v198, 0);
                }
                if ( v242 > (int)v185 + 1 )
                {
                  v216 = (_QWORD *)sub_15E0530(v236);
                  v217 = sub_1643350(v216);
                  v218 = sub_159C470(v217, 1, 0);
                  v267 = (unsigned __int8 *)v218;
                  v274[0] = "splitVecGEP";
                  v275 = 259;
                  if ( *(_BYTE *)(v186 + 16) > 0x10u || *(_BYTE *)(v218 + 16) > 0x10u )
                  {
                    LOWORD(v277) = 257;
                    v219 = *(__int64 ***)v186;
                    if ( *(_BYTE *)(*(_QWORD *)v186 + 8LL) == 16 )
                      v219 = (__int64 **)*v219[2];
                    v259 = (__int64)v219[3];
                    v220 = sub_1648A60(72, 2u);
                    v221 = v220;
                    if ( v220 )
                    {
                      v254 = (__int64)v220;
                      v250 = (__int64)(v220 - 6);
                      v222 = *(__int64 ***)v186;
                      if ( *(_BYTE *)(*(_QWORD *)v186 + 8LL) == 16 )
                        v222 = (__int64 **)*v222[2];
                      v247 = *((_DWORD *)v222 + 2) >> 8;
                      v223 = (__int64 *)sub_15F9F50(v259, (__int64)&v267, 1);
                      v224 = (__int64 *)sub_1646BA0(v223, v247);
                      v225 = *(__int64 ***)v186;
                      if ( *(_BYTE *)(*(_QWORD *)v186 + 8LL) == 16
                        || (v225 = *(__int64 ***)v267, *(_BYTE *)(*(_QWORD *)v267 + 8LL) == 16) )
                      {
                        v224 = sub_16463B0(v224, (unsigned int)v225[4]);
                      }
                      sub_15F1EA0((__int64)v221, (__int64)v224, 32, v250, 2, 0);
                      v221[7] = v259;
                      v221[8] = sub_15F9F50(v259, (__int64)&v267, 1);
                      sub_15F9CE0((__int64)v221, v186, (__int64 *)&v267, 1, (__int64)v276);
                    }
                    else
                    {
                      v254 = 0;
                    }
                    if ( v284 )
                    {
                      v226 = v285;
                      sub_157E9D0((__int64)(v284 + 5), (__int64)v221);
                      v227 = v221[3];
                      v228 = *v226;
                      v221[4] = v226;
                      v228 &= 0xFFFFFFFFFFFFFFF8LL;
                      v221[3] = v228 | v227 & 7;
                      *(_QWORD *)(v228 + 8) = v221 + 3;
                      *v226 = *v226 & 7 | (unsigned __int64)(v221 + 3);
                    }
                    sub_164B780(v254, (__int64 *)v274);
                    if ( v283 )
                    {
                      v268 = (__int64 *)v283;
                      sub_1623A60((__int64)&v268, (__int64)v283, 2);
                      v229 = v221[6];
                      if ( v229 )
                        sub_161E7C0((__int64)(v221 + 6), v229);
                      v230 = (unsigned __int8 *)v268;
                      v221[6] = v268;
                      if ( v230 )
                        sub_1623210((__int64)&v268, v230, (__int64)(v221 + 6));
                    }
                    v186 = (__int64)v221;
                  }
                  else
                  {
                    BYTE4(v276[0]) = 0;
                    v186 = sub_15A2E80(0, v186, (__int64 **)&v267, 1u, 0, (__int64)v276, 0);
                  }
                }
                ++v185;
              }
              while ( v242 > (unsigned int)v185 );
            }
            sub_164D160(a3, v135, a5, a6, a7, a8, v136, v137, a11, a12);
            sub_15F20C0((_QWORD *)a3);
            if ( v283 )
              sub_161E7C0((__int64)&v283, (__int64)v283);
            v231 = v271;
            v232 = v272;
            if ( v271 != v272 )
            {
              do
              {
                v233 = *(_QWORD *)v231;
                v231 += 8;
                sub_21F3A20(a1, a2, v233, a4);
              }
              while ( v232 != v231 );
              v232 = v271;
            }
            if ( v232 )
              j_j___libc_free_0(v232, v273 - v232);
            return;
          }
          goto LABEL_176;
        }
      }
      else
      {
        ++v144;
        v135 = sub_15A3890((__int64 *)v135, v146, v148, 0);
        if ( v262 <= (unsigned int)v144 )
          goto LABEL_217;
      }
    }
  }
  v284 = *(__int64 ***)a3;
  v285 = &v287;
  v283 = a1;
  v286 = 0x400000000LL;
  v289 = (__int64 *)v291;
  v290 = 0x400000000LL;
  v91 = (_QWORD *)sub_16498A0(a3);
  v92 = *(unsigned __int8 **)(a3 + 48);
  v277 = v12;
  v278 = v91;
  v93 = *(unsigned __int8 **)(a3 + 40);
  v276[0] = 0;
  v279 = 0;
  v280 = 0;
  v281 = 0;
  v282 = 0;
  v276[1] = v93;
  v274[0] = v92;
  if ( v92 )
  {
    sub_1623A60((__int64)v274, (__int64)v92, 2);
    if ( v276[0] )
      sub_161E7C0((__int64)v276, (__int64)v276[0]);
    v276[0] = v274[0];
    if ( v274[0] )
      sub_1623210((__int64)v274, v274[0], (__int64)v276);
  }
  v271 = 0;
  v272 = 0;
  v273 = 0;
  if ( !sub_15F32D0(a3) )
  {
    v94 = *(unsigned __int16 *)(a3 + 18);
    if ( (v94 & 1) == 0 )
    {
      v95 = *(_QWORD *)(a3 - 24);
      v294 = 0;
      v292 = v95;
      v293 = 1 << (v94 >> 1) >> 1;
      v96 = sub_1643350(v278);
      v97 = sub_159C470(v96, 0, 0);
      v100 = (unsigned int)v290;
      if ( (unsigned int)v290 >= HIDWORD(v290) )
      {
        sub_16CD150((__int64)&v289, v291, 0, 8, v98, v99);
        v100 = (unsigned int)v290;
      }
      v289[v100] = v97;
      LODWORD(v290) = v290 + 1;
      v267 = (unsigned __int8 *)sub_1599EF0(*(__int64 ***)a3);
      v101 = (__int64 *)sub_1649960(a3);
      v269 = v102;
      v103 = *(_QWORD *)a3;
      v268 = v101;
      v275 = 261;
      v274[0] = (unsigned __int8 *)&v268;
      sub_21F2C80((__int64)&v283, (__int64)v276, v103, (__int64 **)&v267, (__int64)v274, (__int64)&v271);
      sub_164D160(a3, (__int64)v267, a5, a6, a7, a8, v104, v105, a11, a12);
      v106 = v272;
      v107 = v271;
      LODWORD(v286) = 0;
      LODWORD(v290) = 0;
      while ( v106 != v107 )
      {
        v108 = *(_QWORD *)v107;
        v107 += 8;
        sub_21F3A20(a1, a2, v108, a4);
      }
      v109 = (_QWORD *)a4[2];
      v110 = a4 + 1;
      if ( !v109 )
      {
        v109 = a4 + 1;
        if ( v110 == (_QWORD *)a4[3] )
        {
          v113 = 1;
LABEL_139:
          v114 = sub_22077B0(40);
          *(_QWORD *)(v114 + 32) = a3;
          sub_220F040(v113, v114, v109, v110);
          ++a4[5];
          goto LABEL_140;
        }
        goto LABEL_321;
      }
      while ( 1 )
      {
        v111 = v109[4];
        v112 = (_QWORD *)v109[3];
        if ( a3 < v111 )
          v112 = (_QWORD *)v109[2];
        if ( !v112 )
          break;
        v109 = v112;
      }
      if ( a3 < v111 )
      {
        if ( (_QWORD *)a4[3] != v109 )
        {
LABEL_321:
          if ( *(_QWORD *)(sub_220EF80(v109) + 32) >= a3 )
            goto LABEL_140;
        }
LABEL_137:
        v113 = 1;
        if ( v110 != v109 )
          v113 = a3 < v109[4];
        goto LABEL_139;
      }
      if ( v111 < a3 )
        goto LABEL_137;
    }
  }
LABEL_140:
  if ( v271 )
    j_j___libc_free_0(v271, v273 - v271);
  if ( v276[0] )
    sub_161E7C0((__int64)v276, (__int64)v276[0]);
  if ( v289 != (__int64 *)v291 )
    _libc_free((unsigned __int64)v289);
  if ( v285 != &v287 )
    _libc_free((unsigned __int64)v285);
}
