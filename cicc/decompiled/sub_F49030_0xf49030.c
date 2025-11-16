// Function: sub_F49030
// Address: 0xf49030
//
__int64 __fastcall sub_F49030(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int8 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v8; // r12
  unsigned __int64 *v9; // r15
  __int64 v10; // r14
  __int64 v12; // rax
  __int64 v13; // r9
  _QWORD *v14; // rax
  __int64 v15; // rsi
  __int64 v16; // r8
  __int64 v17; // r12
  __int64 *v18; // r10
  __int64 v19; // r9
  __int64 v20; // r15
  __int64 v21; // r13
  __int64 v22; // rbx
  __int64 v23; // rdx
  unsigned int v24; // edx
  __int64 v25; // rdx
  __int64 *i; // rax
  __int64 v27; // rsi
  __int64 v28; // rbx
  unsigned __int64 v29; // rax
  int v30; // ecx
  unsigned __int64 v31; // r12
  unsigned __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // r12
  __int64 v35; // rcx
  unsigned int v36; // r8d
  _QWORD *v37; // rax
  __int64 v38; // rdi
  __int64 v39; // r13
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // r12
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rax
  unsigned __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // r12
  __int64 v50; // rdx
  __int64 v51; // r13
  char v52; // cl
  __int64 v53; // rax
  int v54; // ecx
  int v55; // edx
  __int64 v56; // rax
  __int64 v57; // r15
  _QWORD *v58; // rdx
  __int64 v59; // r8
  __int64 v60; // r9
  __int64 v61; // rsi
  __int64 v62; // r12
  __int64 v63; // rax
  __int64 v64; // rdi
  unsigned int v65; // esi
  _QWORD *v66; // r14
  bool v67; // bl
  _QWORD *v68; // r13
  _QWORD *m; // rbx
  __int64 v70; // rdx
  __int64 v71; // rsi
  unsigned int v72; // ecx
  _QWORD *v73; // rax
  __int64 v74; // r9
  __int64 v75; // r12
  __int64 v76; // r8
  __int64 v77; // r9
  __int64 n; // rbx
  _QWORD *v79; // r11
  _QWORD *v80; // rdx
  _QWORD *v81; // r14
  __int64 v82; // r13
  __int64 v83; // rdi
  _QWORD *v84; // r12
  _QWORD *v85; // rax
  __int64 v86; // rax
  int v87; // edx
  int v88; // ecx
  __int64 v89; // r12
  __int64 v90; // r13
  __int64 v91; // rdi
  __int64 *v92; // r12
  unsigned int v93; // eax
  __int64 *v94; // rdx
  __int64 v95; // rcx
  __int64 v96; // rbx
  __int64 *v97; // rax
  __int64 v98; // r8
  __int64 v99; // r12
  __int64 v100; // r10
  __int64 v101; // r14
  __int64 *v102; // rax
  __int64 *v103; // rdx
  __int64 *v104; // rdi
  __int64 v105; // rsi
  __int64 v106; // rbx
  __int64 v107; // r14
  __int64 v108; // rdx
  __int64 v109; // rsi
  unsigned __int64 v110; // r15
  __int64 v111; // rsi
  __int64 v112; // rbx
  __int64 v113; // r8
  __int64 v114; // r9
  unsigned __int64 v115; // rax
  unsigned __int64 v116; // r15
  __int64 v117; // rax
  __int64 result; // rax
  char v119; // dl
  unsigned __int64 v120; // rax
  __int64 v121; // r13
  int v122; // eax
  int v123; // ebx
  __int64 v124; // rax
  unsigned __int64 v125; // rdx
  int v126; // r14d
  __int64 *v127; // rbx
  unsigned int v128; // r12d
  __int64 v129; // r15
  __int64 v130; // rax
  unsigned __int64 v131; // rdx
  __int64 v132; // rax
  __int64 v133; // rsi
  __int64 v134; // r13
  __int64 v135; // r11
  unsigned int v136; // r14d
  __int64 v137; // rsi
  __int64 v138; // rdi
  unsigned int v139; // ecx
  _QWORD *v140; // rdx
  __int64 v141; // r8
  __int64 v142; // r12
  __int64 v143; // rbx
  __int64 v144; // rbx
  __int64 v145; // rdx
  __int64 v146; // rdx
  __int64 v147; // rax
  __int64 v148; // rbx
  __int64 v149; // rdx
  __int64 **v150; // rax
  unsigned __int64 v151; // r13
  __int64 **v152; // r12
  __int64 *v153; // rcx
  __int64 *v154; // rdx
  __int64 v155; // rax
  __int64 **v156; // rax
  __int64 **v157; // rdx
  _BOOL8 v158; // rdi
  int v159; // edx
  int v160; // r10d
  __int64 v161; // r13
  __int64 v162; // rbx
  __int64 **v163; // rdi
  _QWORD *v164; // r12
  __int64 v165; // r15
  __int64 v166; // rsi
  _QWORD *v167; // rdi
  __int64 v168; // rax
  __int64 v169; // rbx
  __int64 v170; // r14
  __int64 **v171; // r13
  unsigned __int64 v172; // r15
  __int64 **v173; // rax
  __int64 *v174; // rcx
  __int64 *v175; // rdx
  __int64 v176; // rax
  __int64 **v177; // rax
  __int64 **v178; // rdx
  _BOOL8 v179; // rdi
  __int64 j; // r15
  __int64 v181; // r14
  __int64 v182; // r12
  __int64 v183; // r13
  int v184; // ebx
  __int64 k; // r15
  __int64 v186; // rax
  int v187; // eax
  int v188; // r10d
  __int64 v189; // rax
  int v190; // eax
  int v191; // r9d
  __int64 v192; // [rsp+0h] [rbp-430h]
  __int64 v193; // [rsp+8h] [rbp-428h]
  _BYTE *v194; // [rsp+10h] [rbp-420h]
  _BYTE *v195; // [rsp+18h] [rbp-418h]
  __int64 v196; // [rsp+20h] [rbp-410h]
  __int64 v197; // [rsp+28h] [rbp-408h]
  __int64 v198; // [rsp+30h] [rbp-400h]
  __int64 v199; // [rsp+38h] [rbp-3F8h]
  __int64 v200; // [rsp+40h] [rbp-3F0h]
  __int64 v201; // [rsp+48h] [rbp-3E8h]
  int v202; // [rsp+50h] [rbp-3E0h]
  __int64 v204; // [rsp+58h] [rbp-3D8h]
  __int64 v205; // [rsp+60h] [rbp-3D0h]
  __int64 v206; // [rsp+68h] [rbp-3C8h]
  __int64 v207; // [rsp+70h] [rbp-3C0h]
  int v208; // [rsp+70h] [rbp-3C0h]
  __int64 v209; // [rsp+70h] [rbp-3C0h]
  __int64 v210; // [rsp+78h] [rbp-3B8h]
  __int64 v211; // [rsp+78h] [rbp-3B8h]
  int v212; // [rsp+80h] [rbp-3B0h]
  __int64 v213; // [rsp+80h] [rbp-3B0h]
  __int64 v214; // [rsp+88h] [rbp-3A8h]
  __int64 v215; // [rsp+88h] [rbp-3A8h]
  __int64 *v216; // [rsp+90h] [rbp-3A0h]
  __int64 v217; // [rsp+90h] [rbp-3A0h]
  int v218; // [rsp+90h] [rbp-3A0h]
  __int64 **v219; // [rsp+90h] [rbp-3A0h]
  __int64 **v220; // [rsp+90h] [rbp-3A0h]
  __int64 v221; // [rsp+98h] [rbp-398h]
  __int64 v222; // [rsp+98h] [rbp-398h]
  __int64 v223; // [rsp+98h] [rbp-398h]
  unsigned __int64 *v224; // [rsp+98h] [rbp-398h]
  __int64 v225; // [rsp+98h] [rbp-398h]
  __int64 **v226; // [rsp+98h] [rbp-398h]
  unsigned __int64 *v227; // [rsp+98h] [rbp-398h]
  unsigned __int64 *v228; // [rsp+98h] [rbp-398h]
  __int64 **v229; // [rsp+98h] [rbp-398h]
  __int64 v230; // [rsp+A0h] [rbp-390h]
  __int64 v231; // [rsp+A0h] [rbp-390h]
  __int64 v232; // [rsp+A0h] [rbp-390h]
  __int64 *v233; // [rsp+A0h] [rbp-390h]
  __int64 *v234; // [rsp+A0h] [rbp-390h]
  __int64 v235; // [rsp+A0h] [rbp-390h]
  __int64 v236; // [rsp+A0h] [rbp-390h]
  __int64 v237; // [rsp+A0h] [rbp-390h]
  __int64 v238; // [rsp+A0h] [rbp-390h]
  __int64 v239; // [rsp+A0h] [rbp-390h]
  __int64 v240; // [rsp+A0h] [rbp-390h]
  __int64 v241; // [rsp+A0h] [rbp-390h]
  __int64 v242; // [rsp+A8h] [rbp-388h]
  _QWORD *v243; // [rsp+A8h] [rbp-388h]
  unsigned __int64 *v244; // [rsp+A8h] [rbp-388h]
  _QWORD *v245; // [rsp+A8h] [rbp-388h]
  __int64 v246; // [rsp+A8h] [rbp-388h]
  __int64 v247; // [rsp+A8h] [rbp-388h]
  __int64 v248; // [rsp+A8h] [rbp-388h]
  __int64 v249; // [rsp+A8h] [rbp-388h]
  __int64 v250; // [rsp+A8h] [rbp-388h]
  unsigned __int64 *v251; // [rsp+A8h] [rbp-388h]
  __int64 v252; // [rsp+A8h] [rbp-388h]
  __int64 v253; // [rsp+B8h] [rbp-378h] BYREF
  __int64 *v254; // [rsp+C0h] [rbp-370h] BYREF
  __int64 *v255; // [rsp+C8h] [rbp-368h]
  __int64 v256; // [rsp+D0h] [rbp-360h]
  _QWORD v257[3]; // [rsp+E0h] [rbp-350h] BYREF
  unsigned __int8 v258; // [rsp+F8h] [rbp-338h]
  __int64 v259; // [rsp+100h] [rbp-330h]
  __int64 v260; // [rsp+108h] [rbp-328h]
  char v261; // [rsp+110h] [rbp-320h]
  _BYTE v262[32]; // [rsp+120h] [rbp-310h] BYREF
  _QWORD *v263; // [rsp+140h] [rbp-2F0h]
  _BYTE *v264; // [rsp+160h] [rbp-2D0h] BYREF
  __int64 v265; // [rsp+168h] [rbp-2C8h]
  _BYTE v266[64]; // [rsp+170h] [rbp-2C0h] BYREF
  _BYTE *v267; // [rsp+1B0h] [rbp-280h] BYREF
  __int64 v268; // [rsp+1B8h] [rbp-278h]
  _BYTE v269[128]; // [rsp+1C0h] [rbp-270h] BYREF
  __int64 *v270; // [rsp+240h] [rbp-1F0h] BYREF
  __int64 v271; // [rsp+248h] [rbp-1E8h]
  _QWORD v272[16]; // [rsp+250h] [rbp-1E0h] BYREF
  __int64 *v273; // [rsp+2D0h] [rbp-160h] BYREF
  __int64 v274; // [rsp+2D8h] [rbp-158h]
  _BYTE v275[128]; // [rsp+2E0h] [rbp-150h] BYREF
  __int64 v276; // [rsp+360h] [rbp-D0h] BYREF
  __int64 *v277; // [rsp+368h] [rbp-C8h] BYREF
  __int64 v278; // [rsp+370h] [rbp-C0h]
  __int64 **v279; // [rsp+378h] [rbp-B8h]
  __int64 **v280; // [rsp+380h] [rbp-B0h] BYREF
  __int64 v281; // [rsp+388h] [rbp-A8h]
  __int64 v282; // [rsp+390h] [rbp-A0h]
  __int64 v283; // [rsp+398h] [rbp-98h]
  __int16 v284; // [rsp+3A0h] [rbp-90h]

  v9 = (unsigned __int64 *)&v276;
  v10 = a4;
  v200 = a1;
  v199 = a2;
  v259 = a7;
  v257[0] = a1;
  v260 = a8;
  v12 = *(_QWORD *)(a1 + 120);
  v257[1] = a2;
  v198 = a6;
  v257[2] = a4;
  v258 = a5;
  v276 = v12;
  v261 = sub_A73ED0(&v276, 72);
  if ( a3 )
  {
    v13 = *(_QWORD *)(a2 + 80);
    v201 = *(_QWORD *)(a3 + 40);
  }
  else
  {
    v13 = *(_QWORD *)(a2 + 80);
    if ( !v13 )
      BUG();
    a3 = *(_QWORD *)(v13 + 32);
    v201 = v13 - 24;
    if ( a3 )
      a3 -= 24;
  }
  v194 = v266;
  v264 = v266;
  v265 = 0x800000000LL;
  v206 = a2 + 72;
  v14 = &v264;
  if ( a2 + 72 == v13 )
    goto LABEL_22;
  v15 = v8;
  v16 = v10;
  v17 = v13;
  v18 = &v276;
  v19 = v15;
  v20 = a3;
  do
  {
    if ( !v17 )
      goto LABEL_339;
    v21 = *(_QWORD *)(v17 + 32);
    v22 = v17 + 24;
    if ( v21 != v17 + 24 )
    {
      while ( 1 )
      {
        if ( !v21 )
          goto LABEL_336;
        if ( *(_BYTE *)(v21 - 24) != 85 )
          goto LABEL_8;
        v23 = *(_QWORD *)(v21 - 56);
        if ( !v23
          || *(_BYTE *)v23
          || *(_QWORD *)(v23 + 24) != *(_QWORD *)(v21 + 56)
          || (*(_BYTE *)(v23 + 33) & 0x20) == 0 )
        {
          goto LABEL_8;
        }
        v24 = *(_DWORD *)(v23 + 36);
        if ( v24 > 0x45 )
        {
          if ( v24 == 71 )
            goto LABEL_17;
LABEL_8:
          v21 = *(_QWORD *)(v21 + 8);
          if ( v22 == v21 )
            break;
        }
        else
        {
          if ( v24 <= 0x43 )
            goto LABEL_8;
LABEL_17:
          v25 = (unsigned int)v265;
          if ( (unsigned __int64)(unsigned int)v265 + 1 > HIDWORD(v265) )
          {
            v216 = v18;
            v221 = v19;
            v231 = v16;
            v243 = v14;
            sub_C8D5F0((__int64)v14, v266, (unsigned int)v265 + 1LL, 8u, v16, v19);
            v25 = (unsigned int)v265;
            v18 = v216;
            v19 = v221;
            v16 = v231;
            v14 = v243;
          }
          *(_QWORD *)&v264[8 * v25] = v21 - 24;
          LODWORD(v265) = v265 + 1;
          v21 = *(_QWORD *)(v21 + 8);
          if ( v22 == v21 )
            break;
        }
      }
    }
    v17 = *(_QWORD *)(v17 + 8);
  }
  while ( v17 != v206 );
  a3 = v20;
  v10 = v16;
  v8 = v19;
  v9 = (unsigned __int64 *)v18;
LABEL_22:
  v254 = 0;
  v255 = 0;
  v256 = 0;
  sub_F47010((__int64)v257, v201, a3 + 24, 0, &v254);
  for ( i = v255; v254 != v255; i = v255 )
  {
    v27 = *(i - 1);
    LOWORD(v8) = 1;
    v255 = i - 1;
    sub_F47010((__int64)v257, v27, *(_QWORD *)(v27 + 56), v8, &v254);
  }
  v195 = v269;
  v267 = v269;
  v268 = 0x1000000000LL;
  v28 = *(_QWORD *)(v199 + 80);
  v230 = a1 + 72;
  if ( v28 != v206 )
  {
    do
    {
      v33 = *(unsigned int *)(v10 + 24);
      v34 = v28 - 24;
      if ( !v28 )
        v34 = 0;
      if ( (_DWORD)v33 )
      {
        v35 = *(_QWORD *)(v10 + 8);
        v36 = (v33 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
        v37 = (_QWORD *)(v35 + ((unsigned __int64)v36 << 6));
        v38 = v37[3];
        if ( v34 == v38 )
        {
LABEL_36:
          if ( v37 != (_QWORD *)(v35 + (v33 << 6)) )
          {
            v276 = 6;
            v277 = 0;
            v278 = v37[7];
            v39 = v278;
            if ( v278 != 0 && v278 != -4096 && v278 != -8192 )
            {
              sub_BD6050(v9, v37[5] & 0xFFFFFFFFFFFFFFF8LL);
              v39 = v278;
            }
            if ( v39 )
            {
              if ( v39 != -4096 && v39 != -8192 )
                sub_BD60C0(v9);
              sub_AA4AC0(v39, v230);
              v40 = sub_AA5930(v34);
              v242 = v41;
              v42 = v40;
              if ( v40 != v41 )
              {
                while ( *(_BYTE *)sub_F46C80(v10, v42)[2] == 84 )
                {
                  v45 = (unsigned int)v268;
                  v46 = (unsigned int)v268 + 1LL;
                  if ( v46 > HIDWORD(v268) )
                  {
                    sub_C8D5F0((__int64)&v267, v269, v46, 8u, v43, v44);
                    v45 = (unsigned int)v268;
                  }
                  *(_QWORD *)&v267[8 * v45] = v42;
                  LODWORD(v268) = v268 + 1;
                  if ( !v42 )
                    BUG();
                  v47 = *(_QWORD *)(v42 + 32);
                  if ( !v47 )
                    goto LABEL_336;
                  v42 = 0;
                  if ( *(_BYTE *)(v47 - 24) == 84 )
                    v42 = v47 - 24;
                  if ( v242 == v42 )
                    break;
                }
              }
              v29 = *(_QWORD *)(v39 + 48) & 0xFFFFFFFFFFFFFFF8LL;
              if ( v29 == v39 + 48 )
              {
                v31 = 0;
              }
              else
              {
                if ( !v29 )
                  goto LABEL_336;
                v30 = *(unsigned __int8 *)(v29 - 24);
                v31 = 0;
                v32 = v29 - 24;
                if ( (unsigned int)(v30 - 30) < 0xB )
                  v31 = v32;
              }
              sub_FC75A0(v9, v10, a5 ^ 1u, 0, 0, 0);
              sub_FCD280(v9, v31);
              sub_FC7680(v9);
            }
          }
        }
        else
        {
          v190 = 1;
          while ( v38 != -4096 )
          {
            v191 = v190 + 1;
            v36 = (v33 - 1) & (v190 + v36);
            v37 = (_QWORD *)(v35 + ((unsigned __int64)v36 << 6));
            v38 = v37[3];
            if ( v34 == v38 )
              goto LABEL_36;
            v190 = v191;
          }
        }
      }
      v28 = *(_QWORD *)(v28 + 8);
    }
    while ( v206 != v28 );
    v202 = v268;
    if ( (_DWORD)v268 )
    {
      v212 = 0;
      v204 = 0;
      do
      {
        v132 = *(_QWORD *)&v267[8 * v204];
        v211 = *(_QWORD *)(v132 + 40);
        v208 = *(_DWORD *)(v132 + 4) & 0x7FFFFFF;
        v205 = sub_F46C80(v10, v211)[2];
        if ( v212 != (_DWORD)v268 )
        {
          v48 = v204;
          do
          {
            v133 = *(_QWORD *)&v267[8 * v48];
            if ( *(_QWORD *)(v133 + 40) != v211 )
              break;
            v134 = sub_F46C80(v10, v133)[2];
            v218 = v208;
            if ( v208 )
            {
              v135 = v10;
              v136 = 0;
              while ( 1 )
              {
                v147 = *(unsigned int *)(v135 + 24);
                v249 = 8LL * v136;
                if ( (_DWORD)v147 )
                {
                  v137 = *(_QWORD *)(v135 + 8);
                  v138 = *(_QWORD *)(*(_QWORD *)(v134 - 8) + 32LL * *(unsigned int *)(v134 + 72) + 8LL * v136);
                  v139 = (v147 - 1) & (((unsigned int)v138 >> 9) ^ ((unsigned int)v138 >> 4));
                  v140 = (_QWORD *)(v137 + ((unsigned __int64)v139 << 6));
                  v141 = v140[3];
                  if ( v138 == v141 )
                  {
LABEL_220:
                    if ( v140 != (_QWORD *)(v137 + (v147 << 6)) )
                    {
                      v276 = 6;
                      v277 = 0;
                      v278 = v140[7];
                      v142 = v278;
                      if ( v278 != -4096 && v278 != 0 && v278 != -8192 )
                      {
                        v235 = v135;
                        sub_BD6050(v9, v140[5] & 0xFFFFFFFFFFFFFFF8LL);
                        v142 = v278;
                        v135 = v235;
                      }
                      if ( v142 )
                      {
                        if ( v142 != -4096 && v142 != -8192 )
                        {
                          v236 = v135;
                          sub_BD60C0(v9);
                          v135 = v236;
                        }
                        v143 = 32LL * v136;
                        v225 = v135;
                        v237 = *(_QWORD *)(*(_QWORD *)(v134 - 8) + v143);
                        sub_FC75A0(v9, v135, a5 ^ 1u, 0, 0, 0);
                        v238 = sub_FCD360(v9, v237);
                        sub_FC7680(v9);
                        v144 = *(_QWORD *)(v134 - 8) + v143;
                        v135 = v225;
                        if ( *(_QWORD *)v144 )
                        {
                          v145 = *(_QWORD *)(v144 + 8);
                          **(_QWORD **)(v144 + 16) = v145;
                          if ( v145 )
                            *(_QWORD *)(v145 + 16) = *(_QWORD *)(v144 + 16);
                        }
                        *(_QWORD *)v144 = v238;
                        if ( v238 )
                        {
                          v146 = *(_QWORD *)(v238 + 16);
                          *(_QWORD *)(v144 + 8) = v146;
                          if ( v146 )
                            *(_QWORD *)(v146 + 16) = v144 + 8;
                          *(_QWORD *)(v144 + 16) = v238 + 16;
                          *(_QWORD *)(v238 + 16) = v144;
                        }
                        ++v136;
                        *(_QWORD *)(*(_QWORD *)(v134 - 8) + 32LL * *(unsigned int *)(v134 + 72) + v249) = v142;
                        goto LABEL_236;
                      }
                    }
                  }
                  else
                  {
                    v159 = 1;
                    while ( v141 != -4096 )
                    {
                      v160 = v159 + 1;
                      v139 = (v147 - 1) & (v159 + v139);
                      v140 = (_QWORD *)(v137 + ((unsigned __int64)v139 << 6));
                      v141 = v140[3];
                      if ( v138 == v141 )
                        goto LABEL_220;
                      v159 = v160;
                    }
                  }
                }
                v250 = v135;
                sub_B48BF0(v134, v136, 0);
                --v218;
                v135 = v250;
LABEL_236:
                if ( v218 == v136 )
                {
                  v10 = v135;
                  break;
                }
              }
            }
            v48 = (unsigned int)++v212;
          }
          while ( v212 != (_DWORD)v268 );
          v204 = v48;
        }
        v49 = *(_QWORD *)(v205 + 56);
        v50 = *(_QWORD *)(v205 + 16);
        v51 = v49 - 24;
        if ( !v49 )
          v51 = 0;
        if ( v50 )
        {
          do
          {
            v52 = **(_BYTE **)(v50 + 24);
            v53 = v50;
            v50 = *(_QWORD *)(v50 + 8);
            if ( (unsigned __int8)(v52 - 30) <= 0xAu )
            {
              v54 = 0;
              while ( 1 )
              {
                v53 = *(_QWORD *)(v53 + 8);
                if ( !v53 )
                  break;
                while ( (unsigned __int8)(**(_BYTE **)(v53 + 24) - 30) <= 0xAu )
                {
                  v53 = *(_QWORD *)(v53 + 8);
                  ++v54;
                  if ( !v53 )
                    goto LABEL_67;
                }
              }
LABEL_67:
              v55 = v54 + 1;
              goto LABEL_68;
            }
          }
          while ( v50 );
          if ( (*(_DWORD *)(v51 + 4) & 0x7FFFFFF) == 0 )
            goto LABEL_69;
        }
        else
        {
          v55 = 0;
LABEL_68:
          if ( v55 == (*(_DWORD *)(v51 + 4) & 0x7FFFFFF) )
            goto LABEL_69;
        }
        LODWORD(v277) = 0;
        v279 = &v277;
        v280 = &v277;
        v278 = 0;
        v281 = 0;
        v148 = *(_QWORD *)(v205 + 16);
        if ( v148 )
        {
          while ( 1 )
          {
            v149 = *(_QWORD *)(v148 + 24);
            if ( (unsigned __int8)(*(_BYTE *)v149 - 30) <= 0xAu )
              break;
            v148 = *(_QWORD *)(v148 + 8);
            if ( !v148 )
              goto LABEL_278;
          }
          v239 = v51;
          v150 = 0;
LABEL_244:
          v151 = *(_QWORD *)(v149 + 40);
          v152 = &v277;
          if ( !v150 )
            goto LABEL_251;
          do
          {
            while ( 1 )
            {
              v153 = v150[2];
              v154 = v150[3];
              if ( (unsigned __int64)v150[4] >= v151 )
                break;
              v150 = (__int64 **)v150[3];
              if ( !v154 )
                goto LABEL_249;
            }
            v152 = v150;
            v150 = (__int64 **)v150[2];
          }
          while ( v153 );
LABEL_249:
          if ( v152 == &v277 || (unsigned __int64)v152[4] > v151 )
          {
LABEL_251:
            v226 = v152;
            v155 = sub_22077B0(48);
            *(_QWORD *)(v155 + 32) = v151;
            v152 = (__int64 **)v155;
            *(_DWORD *)(v155 + 40) = 0;
            v156 = (__int64 **)sub_F469A0(v9, v226, (unsigned __int64 *)(v155 + 32));
            if ( v157 )
            {
              v158 = &v277 == v157 || v156 || v151 < (unsigned __int64)v157[4];
              sub_220F040(v158, v152, v157, &v277);
              ++v281;
            }
            else
            {
              v229 = v156;
              j_j___libc_free_0(v152, 48);
              v152 = v229;
            }
          }
          --*((_DWORD *)v152 + 10);
          while ( 1 )
          {
            v148 = *(_QWORD *)(v148 + 8);
            if ( !v148 )
              break;
            v149 = *(_QWORD *)(v148 + 24);
            if ( (unsigned __int8)(*(_BYTE *)v149 - 30) <= 0xAu )
            {
              v150 = (__int64 **)v278;
              goto LABEL_244;
            }
          }
          v51 = v239;
        }
LABEL_278:
        if ( (*(_DWORD *)(v51 + 4) & 0x7FFFFFF) != 0 )
        {
          v227 = v9;
          v169 = 0;
          v209 = v10;
          v170 = v51;
          v240 = 8LL * (*(_DWORD *)(v51 + 4) & 0x7FFFFFF);
          do
          {
            v171 = &v277;
            v172 = *(_QWORD *)(*(_QWORD *)(v170 - 8) + 32LL * *(unsigned int *)(v170 + 72) + v169);
            v173 = (__int64 **)v278;
            if ( !v278 )
              goto LABEL_287;
            do
            {
              while ( 1 )
              {
                v174 = v173[2];
                v175 = v173[3];
                if ( (unsigned __int64)v173[4] >= v172 )
                  break;
                v173 = (__int64 **)v173[3];
                if ( !v175 )
                  goto LABEL_285;
              }
              v171 = v173;
              v173 = (__int64 **)v173[2];
            }
            while ( v174 );
LABEL_285:
            if ( v171 == &v277 || (unsigned __int64)v171[4] > v172 )
            {
LABEL_287:
              v219 = v171;
              v176 = sub_22077B0(48);
              *(_QWORD *)(v176 + 32) = v172;
              v171 = (__int64 **)v176;
              *(_DWORD *)(v176 + 40) = 0;
              v177 = (__int64 **)sub_F469A0(v227, v219, (unsigned __int64 *)(v176 + 32));
              if ( v178 )
              {
                v179 = &v277 == v178 || v177 || v172 < (unsigned __int64)v178[4];
                sub_220F040(v179, v171, v178, &v277);
                ++v281;
              }
              else
              {
                v220 = v177;
                j_j___libc_free_0(v171, 48);
                v171 = v220;
              }
            }
            ++*((_DWORD *)v171 + 10);
            v169 += 8;
          }
          while ( v240 != v169 );
          v10 = v209;
          v9 = v227;
        }
        v241 = v10;
        v228 = v9;
        for ( j = *(_QWORD *)(v205 + 56); ; j = *(_QWORD *)(j + 8) )
        {
          if ( !j )
            goto LABEL_336;
          if ( *(_BYTE *)(j - 24) != 84 )
            break;
          v181 = (__int64)v279;
          v182 = j - 24;
          v183 = j;
          if ( v279 != &v277 )
          {
            do
            {
              v184 = *(_DWORD *)(v181 + 40);
              for ( k = *(_QWORD *)(v181 + 32); v184; --v184 )
              {
                while ( (*(_DWORD *)(v183 - 20) & 0x7FFFFFF) == 0 )
                {
LABEL_307:
                  sub_B48BF0(v182, 0xFFFFFFFF, 0);
                  if ( !--v184 )
                    goto LABEL_304;
                }
                v186 = 0;
                while ( k != *(_QWORD *)(*(_QWORD *)(v183 - 32) + 32LL * *(unsigned int *)(v183 + 48) + 8 * v186) )
                {
                  if ( (*(_DWORD *)(v183 - 20) & 0x7FFFFFF) == (_DWORD)++v186 )
                    goto LABEL_307;
                }
                sub_B48BF0(v182, v186, 0);
              }
LABEL_304:
              v181 = sub_220EEE0(v181);
            }
            while ( (__int64 **)v181 != &v277 );
            j = v183;
          }
        }
        v10 = v241;
        v9 = v228;
        sub_F45560(v278);
        v49 = *(_QWORD *)(v205 + 56);
LABEL_69:
        if ( !v49 )
          BUG();
        if ( (*(_DWORD *)(v49 - 20) & 0x7FFFFFF) == 0 && *(_BYTE *)(v49 - 24) == 84 )
        {
          v251 = v9;
          v161 = *(_QWORD *)(v49 + 8);
          v162 = *(_QWORD *)(v211 + 56);
          while ( 1 )
          {
            v163 = *(__int64 ***)(v49 - 16);
            v164 = (_QWORD *)(v49 - 24);
            v165 = sub_ACADE0(v163);
            sub_BD84D0((__int64)v164, v165);
            v166 = v162 - 24;
            if ( !v162 )
              v166 = 0;
            v167 = sub_F46C80(v10, v166);
            v168 = v167[2];
            if ( v165 != v168 )
            {
              if ( v168 != 0 && v168 != -4096 && v168 != -8192 )
                sub_BD60C0(v167);
              v167[2] = v165;
              if ( v165 != 0 && v165 != -4096 && v165 != -8192 )
                sub_BD73F0((__int64)v167);
            }
            sub_B43D60(v164);
            if ( *(_BYTE *)(v161 - 24) != 84 )
              break;
            v49 = v161;
            v162 = *(_QWORD *)(v162 + 8);
            v161 = *(_QWORD *)(v161 + 8);
          }
          v9 = v251;
        }
      }
      while ( v202 != v212 );
    }
  }
  v253 = *(_QWORD *)(v200 + 120);
  v56 = sub_A74610(&v253);
  sub_A751C0((__int64)v262, **(_QWORD **)(*(_QWORD *)(v199 + 24) + 16LL), v56, 3);
  sub_B2D550(v200, (__int64)v262);
  v217 = sub_B2BEC0(v200);
  v214 = *(_QWORD *)(v199 + 80);
  if ( v206 != v214 )
  {
    v232 = v10;
    v244 = v9;
    while ( v214 )
    {
      v222 = v214 + 24;
      v57 = *(_QWORD *)(v214 + 32);
      if ( v214 + 24 != v57 )
      {
        while ( 1 )
        {
          v62 = v57 - 24;
          if ( !v57 )
            v62 = 0;
          v63 = *(unsigned int *)(v232 + 24);
          if ( !(_DWORD)v63 )
            goto LABEL_82;
          v64 = *(_QWORD *)(v232 + 8);
          v59 = (unsigned int)(v63 - 1);
          v65 = v59 & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
          v58 = (_QWORD *)(v64 + ((unsigned __int64)v65 << 6));
          v60 = v58[3];
          if ( v62 != v60 )
          {
            v87 = 1;
            while ( v60 != -4096 )
            {
              v88 = v87 + 1;
              v65 = v59 & (v87 + v65);
              v58 = (_QWORD *)(v64 + ((unsigned __int64)v65 << 6));
              v60 = v58[3];
              if ( v62 == v60 )
                goto LABEL_87;
              v87 = v88;
            }
            goto LABEL_82;
          }
LABEL_87:
          if ( v58 == (_QWORD *)(v64 + (v63 << 6)) )
            goto LABEL_82;
          v276 = 6;
          v277 = 0;
          v278 = v58[7];
          v66 = (_QWORD *)v278;
          if ( v278 != 0 && v278 != -4096 && v278 != -8192 )
          {
            sub_BD6050(v244, v58[5] & 0xFFFFFFFFFFFFFFF8LL);
            v66 = (_QWORD *)v278;
          }
          if ( !v66 )
            goto LABEL_82;
          v67 = v66 + 512 != 0 && v66 + 1024 != 0;
          if ( *(_BYTE *)v66 > 0x1Cu )
          {
            if ( v67 )
              sub_BD60C0(v244);
            v277 = 0;
            v276 = v217;
            v278 = 0;
            v279 = 0;
            v280 = 0;
            v281 = 0;
            v282 = 0;
            v283 = 0;
            v284 = 257;
            v61 = sub_1020E10(v66, v244, v58, 257, v59, v60, v192, v193, v194, v195, v196, v197, v198, v199, v200);
            if ( v61 )
            {
              sub_BD84D0((__int64)v66, v61);
              if ( (unsigned __int8)sub_F50EE0(v66, 0) )
              {
                sub_B43D60(v66);
              }
              else
              {
                v84 = sub_F46C80(v232, v62);
                v85 = (_QWORD *)v84[2];
                if ( v85 != v66 )
                {
                  if ( v85 + 512 != 0 && v85 != 0 && v85 != (_QWORD *)-8192LL )
                    sub_BD60C0(v84);
                  v84[2] = v66;
                  if ( v67 )
                    sub_BD73F0((__int64)v84);
                }
              }
            }
            goto LABEL_82;
          }
          if ( v67 )
          {
            sub_BD60C0(v244);
            v57 = *(_QWORD *)(v57 + 8);
            if ( v222 == v57 )
              break;
          }
          else
          {
LABEL_82:
            v57 = *(_QWORD *)(v57 + 8);
            if ( v222 == v57 )
              break;
          }
        }
      }
      v214 = *(_QWORD *)(v214 + 8);
      if ( v206 == v214 )
      {
        v10 = v232;
        v9 = v244;
        goto LABEL_97;
      }
    }
LABEL_339:
    BUG();
  }
LABEL_97:
  v68 = v264;
  *(_QWORD *)(v200 + 120) = v253;
  for ( m = &v68[(unsigned int)v265]; m != v68; ++v68 )
  {
    v70 = *(unsigned int *)(v10 + 24);
    if ( (_DWORD)v70 )
    {
      v71 = *(_QWORD *)(v10 + 8);
      v72 = (v70 - 1) & (((unsigned int)*v68 >> 9) ^ ((unsigned int)*v68 >> 4));
      v73 = (_QWORD *)(v71 + ((unsigned __int64)v72 << 6));
      v74 = v73[3];
      if ( *v68 == v74 )
      {
LABEL_100:
        if ( v73 != (_QWORD *)(v71 + (v70 << 6)) )
        {
          v276 = 6;
          v277 = 0;
          v278 = v73[7];
          v75 = v278;
          if ( v278 != -4096 && v278 != 0 && v278 != -8192 )
          {
            sub_BD6050(v9, v73[5] & 0xFFFFFFFFFFFFFFF8LL);
            v75 = v278;
          }
          if ( v75 )
          {
            if ( v75 != -4096 && v75 != -8192 )
              sub_BD60C0(v9);
            sub_FC75A0(v9, v10, a5 ^ 1u, 0, 0, 0);
            sub_FCD280(v9, v75);
            sub_FC7680(v9);
          }
        }
      }
      else
      {
        v187 = 1;
        while ( v74 != -4096 )
        {
          v188 = v187 + 1;
          v189 = ((_DWORD)v70 - 1) & (v72 + v187);
          v72 = v189;
          v73 = (_QWORD *)(v71 + (v189 << 6));
          v74 = v73[3];
          if ( *v68 == v74 )
            goto LABEL_100;
          v187 = v188;
        }
      }
    }
  }
  v207 = sub_F46C80(v10, v201)[2];
  v210 = v207 + 24;
  v213 = v200 + 72;
  v215 = v207 + 24;
  if ( v207 + 24 != v200 + 72 )
  {
    v223 = v10;
    while ( 1 )
    {
      for ( n = *(_QWORD *)(v215 + 32); v215 + 24 != n; n = *(_QWORD *)(n + 8) )
      {
        if ( !n )
          BUG();
        v83 = *(_QWORD *)(n + 40);
        if ( v83 )
        {
          v79 = (_QWORD *)sub_B14240(v83);
          v81 = v80;
        }
        else
        {
          v81 = &qword_4F81430[1];
          v79 = &qword_4F81430[1];
        }
        v245 = v79;
        v82 = sub_B43CA0(n - 24);
        sub_FC75A0(v9, v223, a5 ^ 1u, 0, 0, 0);
        sub_FCD310(v9, v82, v245, v81);
        sub_FC7680(v9);
      }
      v86 = *(_QWORD *)(v215 + 8);
      v215 = v86;
      if ( v213 == v86 )
        break;
      if ( !v86 )
        goto LABEL_339;
    }
    v10 = v223;
    v89 = v207 + 24;
    v90 = v86;
    do
    {
      v91 = v89 - 24;
      if ( !v89 )
        v91 = 0;
      sub_F5CD10(v91, 0, 0, 0);
      v89 = *(_QWORD *)(v89 + 8);
    }
    while ( v90 != v89 );
  }
  v92 = v272;
  v276 = 0;
  v277 = (__int64 *)&v280;
  v278 = 16;
  LODWORD(v279) = 0;
  BYTE4(v279) = 1;
  v270 = v272;
  v272[0] = v207;
  v271 = 0x1000000001LL;
  v93 = 1;
LABEL_137:
  if ( v93 )
  {
    do
    {
      v94 = v270;
      v95 = v93;
      v96 = v270[v93 - 1];
      LODWORD(v271) = v93 - 1;
      if ( BYTE4(v279) )
      {
        v97 = v277;
        v95 = HIDWORD(v278);
        v94 = &v277[HIDWORD(v278)];
        if ( v277 != v94 )
        {
          while ( v96 != *v97 )
          {
            if ( v94 == ++v97 )
              goto LABEL_205;
          }
          goto LABEL_143;
        }
LABEL_205:
        if ( HIDWORD(v278) < (unsigned int)v278 )
        {
          ++HIDWORD(v278);
          *v94 = v96;
          ++v276;
LABEL_192:
          v120 = *(_QWORD *)(v96 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v120 == v96 + 48 )
            goto LABEL_330;
          if ( !v120 )
            BUG();
          v121 = v120 - 24;
          if ( (unsigned int)*(unsigned __int8 *)(v120 - 24) - 30 > 0xA )
          {
LABEL_330:
            v76 = 0;
            v123 = 0;
            v121 = 0;
          }
          else
          {
            v122 = sub_B46E30(v121);
            v76 = v122;
            v123 = v122;
          }
          v124 = (unsigned int)v271;
          v125 = (unsigned int)v271 + v76;
          if ( v125 > HIDWORD(v271) )
          {
            v252 = v76;
            sub_C8D5F0((__int64)&v270, v92, v125, 8u, v76, v77);
            v124 = (unsigned int)v271;
            v76 = v252;
          }
          if ( v123 )
          {
            v248 = v10;
            v126 = v123;
            v127 = &v270[v124];
            v234 = v92;
            v128 = 0;
            v224 = v9;
            v129 = v76;
            do
            {
              if ( v127 )
                *v127 = sub_B46EC0(v121, v128);
              ++v128;
              ++v127;
            }
            while ( v128 != v126 );
            v76 = v129;
            v10 = v248;
            v92 = v234;
            v9 = v224;
            LODWORD(v124) = v271;
          }
          LODWORD(v271) = v124 + v76;
          v93 = v124 + v76;
          goto LABEL_137;
        }
      }
      sub_C8CC70((__int64)v9, v96, (__int64)v94, v95, v76, v77);
      if ( v119 )
        goto LABEL_192;
LABEL_143:
      v93 = v271;
    }
    while ( (_DWORD)v271 );
  }
  v98 = v207 + 24;
  v273 = (__int64 *)v275;
  v274 = 0x1000000000LL;
  if ( v210 != v213 )
  {
    v246 = v10;
    v233 = v92;
    v99 = v207 + 24;
    while ( 1 )
    {
      v100 = v99 - 24;
      if ( !v99 )
        v100 = 0;
      v101 = v100;
      if ( BYTE4(v279) )
      {
        v102 = v277;
        v103 = &v277[HIDWORD(v278)];
        if ( v277 == v103 )
          goto LABEL_208;
        while ( v100 != *v102 )
        {
          if ( v103 == ++v102 )
            goto LABEL_208;
        }
LABEL_153:
        v99 = *(_QWORD *)(v99 + 8);
        if ( v213 == v99 )
          goto LABEL_154;
      }
      else
      {
        if ( sub_C8CA60((__int64)v9, v100) )
          goto LABEL_153;
LABEL_208:
        v130 = (unsigned int)v274;
        v131 = (unsigned int)v274 + 1LL;
        if ( v131 > HIDWORD(v274) )
        {
          sub_C8D5F0((__int64)&v273, v275, v131, 8u, v98, 0);
          v130 = (unsigned int)v274;
        }
        v273[v130] = v101;
        LODWORD(v274) = v274 + 1;
        v99 = *(_QWORD *)(v99 + 8);
        if ( v213 == v99 )
        {
LABEL_154:
          v10 = v246;
          v92 = v233;
          v104 = v273;
          v105 = (unsigned int)v274;
          goto LABEL_155;
        }
      }
    }
  }
  v105 = 0;
  v104 = (__int64 *)v275;
LABEL_155:
  sub_F344A0(v104, v105, 0, 0);
  if ( v273 != (__int64 *)v275 )
    _libc_free(v273, v105);
  if ( v270 != v92 )
    _libc_free(v270, v105);
  if ( !BYTE4(v279) )
    _libc_free(v277, v105);
  if ( v210 != v213 )
  {
    v247 = v10;
    v106 = v207 + 24;
    while ( 1 )
    {
      v110 = *(_QWORD *)(v106 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v106 + 24 == v110 || !v110 || (unsigned int)*(unsigned __int8 *)(v110 - 24) - 30 > 0xA )
        goto LABEL_336;
      if ( *(_BYTE *)(v110 - 24) == 31
        && (*(_DWORD *)(v110 - 20) & 0x7FFFFFF) != 3
        && (v107 = *(_QWORD *)(v110 - 56), sub_AA54C0(v107)) )
      {
        sub_B43D60((_QWORD *)(v110 - 24));
        sub_BD84D0(v107, v106 - 24);
        v108 = v197;
        v109 = v193;
        LOWORD(v108) = 0;
        LOWORD(v109) = 1;
        v197 = v108;
        v193 = v109;
        sub_AA80F0(
          v106 - 24,
          (unsigned __int64 *)(v106 + 24),
          0,
          v107,
          *(__int64 **)(v107 + 56),
          v109,
          (__int64 *)(v107 + 48),
          v108);
        sub_AA5450((_QWORD *)v107);
        if ( v106 == v213 )
          goto LABEL_172;
      }
      else
      {
        v106 = *(_QWORD *)(v106 + 8);
        if ( v106 == v213 )
        {
LABEL_172:
          v10 = v247;
          break;
        }
      }
      if ( !v106 )
LABEL_334:
        BUG();
    }
  }
  v111 = v201;
  v112 = sub_F46C80(v10, v201)[2] + 24LL;
  if ( v112 != v213 )
  {
    while ( 1 )
    {
      v115 = *(_QWORD *)(v112 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v115 == v112 + 24 )
        break;
      if ( !v115 )
        break;
      v116 = v115 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v115 - 24) - 30 > 0xA )
        break;
      if ( *(_BYTE *)(v115 - 24) == 30 )
      {
        v117 = *(unsigned int *)(v198 + 8);
        if ( v117 + 1 > (unsigned __int64)*(unsigned int *)(v198 + 12) )
        {
          v111 = v198 + 16;
          sub_C8D5F0(v198, (const void *)(v198 + 16), v117 + 1, 8u, v113, v114);
          v117 = *(unsigned int *)(v198 + 8);
        }
        *(_QWORD *)(*(_QWORD *)v198 + 8 * v117) = v116;
        ++*(_DWORD *)(v198 + 8);
      }
      v112 = *(_QWORD *)(v112 + 8);
      if ( v213 == v112 )
        goto LABEL_184;
      if ( !v112 )
        goto LABEL_334;
    }
LABEL_336:
    BUG();
  }
LABEL_184:
  result = sub_F45730(v263, v111);
  if ( v267 != v195 )
    result = _libc_free(v267, v111);
  if ( v254 )
  {
    v111 = v256 - (_QWORD)v254;
    result = j_j___libc_free_0(v254, v256 - (_QWORD)v254);
  }
  if ( v264 != v194 )
    return _libc_free(v264, v111);
  return result;
}
