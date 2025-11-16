// Function: sub_3171DA0
// Address: 0x3171da0
//
_QWORD *__fastcall sub_3171DA0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  _QWORD *v8; // r15
  _QWORD *i; // rbx
  _QWORD *result; // rax
  _QWORD *v11; // r12
  char v12; // al
  _QWORD *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r13
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r14
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rax
  bool v26; // r13
  __int64 v27; // rax
  __int64 *v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r9
  __int64 v31; // rdi
  __int64 v32; // r14
  __int64 v33; // rax
  __int64 v34; // r13
  __int64 v35; // r8
  __int64 v36; // rsi
  __int64 *v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdx
  unsigned __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 **v43; // r13
  unsigned __int64 v44; // rcx
  __int64 v45; // r8
  _QWORD *v46; // r9
  __int64 v47; // rax
  __int64 v48; // rax
  unsigned int v49; // ebx
  __int64 v50; // r14
  unsigned __int64 v51; // r12
  __int64 v52; // rdx
  _QWORD *v53; // rax
  unsigned __int64 v54; // rdi
  unsigned __int64 v55; // rax
  __int64 v56; // r14
  __int64 *v57; // rax
  __int64 v58; // r15
  __int64 v59; // r9
  int v60; // r11d
  int v61; // r10d
  __int64 v62; // r14
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // rsi
  __int64 v66; // rax
  _BYTE *v67; // rax
  _QWORD *v68; // r9
  _QWORD *v69; // rdi
  unsigned int v70; // eax
  __int64 v71; // rdx
  __int64 v72; // r14
  __int64 v73; // r12
  __int64 v74; // r15
  char v75; // al
  __int64 v76; // r14
  __int64 v77; // rax
  __int64 v78; // rsi
  _QWORD *v79; // rax
  char v80; // r14
  __int64 v81; // rax
  __int64 v82; // rax
  __int64 v83; // r9
  __int64 v84; // r8
  __int64 v85; // r14
  __int64 v86; // rbx
  __int64 v87; // rdx
  __int64 v88; // rax
  __int64 v89; // r12
  __int64 v90; // rdx
  __int64 v91; // rbx
  __int64 v92; // r15
  __int64 v93; // rdx
  __int64 v94; // rcx
  __int64 v95; // rax
  __int64 v96; // rdx
  unsigned int v97; // edi
  unsigned __int64 v98; // rax
  __int64 v99; // rax
  __int64 v100; // r15
  __int64 v101; // r13
  _QWORD *v102; // r14
  __int64 v103; // rbx
  _QWORD *v104; // r12
  __int64 v105; // rax
  unsigned int v106; // eax
  __int64 v107; // rbx
  bool v108; // cc
  unsigned __int64 v109; // rdi
  char v110; // r14
  int v111; // r14d
  char v112; // cl
  char *v113; // rax
  char *v114; // r8
  char *v115; // rdx
  char v116; // cl
  __int64 *v117; // rax
  __int64 *v118; // r14
  __int64 *v119; // rdx
  _QWORD *v120; // rdx
  _QWORD *v121; // rcx
  _QWORD *v122; // rax
  unsigned int v123; // eax
  __int64 v124; // rax
  __int64 v125; // rsi
  unsigned __int64 v126; // rdx
  unsigned __int64 v127; // r8
  __int64 v128; // rax
  __int64 v129; // rcx
  __int64 v130; // rsi
  __int64 v131; // rcx
  unsigned __int64 v132; // r14
  __int64 v133; // rbx
  unsigned int v134; // edx
  __int64 v135; // r15
  __int64 v136; // rsi
  __int64 v137; // rbx
  unsigned __int64 v138; // rdi
  int v139; // r14d
  __int64 v140; // rbx
  __int64 *v141; // r12
  __int64 v142; // rdx
  __int64 *v143; // r13
  __int64 *v144; // rax
  __int64 *v145; // rax
  _QWORD *v146; // r10
  __int64 **v147; // r11
  char *v148; // r15
  __int64 v149; // r13
  _QWORD *v150; // r8
  _QWORD *v151; // r9
  char *v152; // r12
  __int64 v153; // rbx
  __int64 v154; // rdx
  char *v155; // r14
  char *v156; // rax
  char *v157; // rcx
  char v158; // al
  char *v159; // rax
  _QWORD *v160; // rdi
  size_t v161; // rdx
  __int64 v162; // rax
  unsigned __int64 v163; // rdx
  _QWORD *v164; // r15
  __int64 v165; // rbx
  unsigned __int64 v166; // rdi
  __int64 v167; // rax
  __int64 v168; // rax
  int v169; // [rsp+4h] [rbp-3ACh]
  __int64 v170; // [rsp+8h] [rbp-3A8h]
  unsigned int v171; // [rsp+8h] [rbp-3A8h]
  __int64 v172; // [rsp+10h] [rbp-3A0h]
  int v173; // [rsp+10h] [rbp-3A0h]
  int v174; // [rsp+10h] [rbp-3A0h]
  unsigned __int64 v175; // [rsp+18h] [rbp-398h]
  _QWORD *v176; // [rsp+18h] [rbp-398h]
  __int64 **v177; // [rsp+18h] [rbp-398h]
  unsigned int v178; // [rsp+18h] [rbp-398h]
  __int64 **v179; // [rsp+18h] [rbp-398h]
  _QWORD *v180; // [rsp+20h] [rbp-390h]
  _QWORD *v181; // [rsp+20h] [rbp-390h]
  int v182; // [rsp+20h] [rbp-390h]
  char *v183; // [rsp+20h] [rbp-390h]
  _QWORD *v184; // [rsp+28h] [rbp-388h]
  __int64 v185; // [rsp+28h] [rbp-388h]
  unsigned int v186; // [rsp+28h] [rbp-388h]
  _QWORD *v187; // [rsp+28h] [rbp-388h]
  _QWORD *v188; // [rsp+28h] [rbp-388h]
  _QWORD *v189; // [rsp+28h] [rbp-388h]
  _QWORD *v190; // [rsp+30h] [rbp-380h]
  _QWORD *v191; // [rsp+30h] [rbp-380h]
  __int64 v192; // [rsp+30h] [rbp-380h]
  __int64 v193; // [rsp+30h] [rbp-380h]
  _QWORD *v194; // [rsp+30h] [rbp-380h]
  _QWORD *v195; // [rsp+30h] [rbp-380h]
  __int64 *v197; // [rsp+48h] [rbp-368h]
  __int64 v200; // [rsp+60h] [rbp-350h]
  unsigned int *destf; // [rsp+70h] [rbp-340h]
  _QWORD *dest; // [rsp+70h] [rbp-340h]
  char desta; // [rsp+70h] [rbp-340h]
  _QWORD *destb; // [rsp+70h] [rbp-340h]
  __int64 *destc; // [rsp+70h] [rbp-340h]
  _QWORD *destd; // [rsp+70h] [rbp-340h]
  _QWORD *deste; // [rsp+70h] [rbp-340h]
  __int64 v209; // [rsp+78h] [rbp-338h]
  _QWORD *v210; // [rsp+78h] [rbp-338h]
  _QWORD *v211; // [rsp+78h] [rbp-338h]
  _QWORD *v212; // [rsp+80h] [rbp-330h] BYREF
  __int64 v213; // [rsp+88h] [rbp-328h]
  _QWORD v214[6]; // [rsp+90h] [rbp-320h] BYREF
  __int64 *v215; // [rsp+C0h] [rbp-2F0h] BYREF
  unsigned __int64 v216; // [rsp+C8h] [rbp-2E8h]
  __int64 v217; // [rsp+D0h] [rbp-2E0h] BYREF
  __int64 v218; // [rsp+D8h] [rbp-2D8h]
  _QWORD *v219; // [rsp+E0h] [rbp-2D0h] BYREF
  __int64 v220; // [rsp+E8h] [rbp-2C8h]
  _QWORD v221[2]; // [rsp+F0h] [rbp-2C0h] BYREF
  __int16 v222; // [rsp+100h] [rbp-2B0h]
  __int64 v223; // [rsp+108h] [rbp-2A8h]
  void **v224; // [rsp+110h] [rbp-2A0h]
  _QWORD *v225; // [rsp+118h] [rbp-298h]
  __int64 v226; // [rsp+120h] [rbp-290h]
  int v227; // [rsp+128h] [rbp-288h]
  __int16 v228; // [rsp+12Ch] [rbp-284h]
  char v229; // [rsp+12Eh] [rbp-282h]
  __int64 v230; // [rsp+130h] [rbp-280h]
  __int64 v231; // [rsp+138h] [rbp-278h]
  void *v232; // [rsp+140h] [rbp-270h] BYREF
  _QWORD v233[16]; // [rsp+148h] [rbp-268h] BYREF
  int v234; // [rsp+1C8h] [rbp-1E8h]
  char v235; // [rsp+1CCh] [rbp-1E4h]
  char v236; // [rsp+1D0h] [rbp-1E0h] BYREF
  _QWORD *v237; // [rsp+210h] [rbp-1A0h]
  char v238; // [rsp+218h] [rbp-198h]
  _QWORD *v239; // [rsp+220h] [rbp-190h]
  unsigned int v240; // [rsp+228h] [rbp-188h]
  __int64 v241; // [rsp+230h] [rbp-180h]
  __int64 *v242; // [rsp+238h] [rbp-178h]
  __int64 *v243; // [rsp+240h] [rbp-170h]
  __int64 v244; // [rsp+248h] [rbp-168h]
  _QWORD *v245; // [rsp+250h] [rbp-160h]
  __int64 v246; // [rsp+258h] [rbp-158h]
  unsigned int v247; // [rsp+260h] [rbp-150h]
  __int64 v248; // [rsp+268h] [rbp-148h] BYREF
  __int64 *v249; // [rsp+270h] [rbp-140h]
  __int64 v250; // [rsp+278h] [rbp-138h]
  int v251; // [rsp+280h] [rbp-130h]
  char v252; // [rsp+284h] [rbp-12Ch]
  char v253; // [rsp+288h] [rbp-128h] BYREF
  __int64 v254; // [rsp+2A8h] [rbp-108h]
  char *v255; // [rsp+2B0h] [rbp-100h]
  __int64 v256; // [rsp+2B8h] [rbp-F8h]
  int v257; // [rsp+2C0h] [rbp-F0h]
  char v258; // [rsp+2C4h] [rbp-ECh]
  char v259; // [rsp+2C8h] [rbp-E8h] BYREF
  void *src; // [rsp+2D8h] [rbp-D8h]
  __int64 v261; // [rsp+2E0h] [rbp-D0h]
  char v262; // [rsp+2E8h] [rbp-C8h] BYREF
  _QWORD v263[2]; // [rsp+318h] [rbp-98h] BYREF
  __int64 v264; // [rsp+328h] [rbp-88h]
  int v265; // [rsp+330h] [rbp-80h]
  char v266; // [rsp+334h] [rbp-7Ch]
  char v267; // [rsp+338h] [rbp-78h] BYREF
  __int64 v268; // [rsp+348h] [rbp-68h] BYREF
  __int64 *v269; // [rsp+350h] [rbp-60h]
  __int64 v270; // [rsp+358h] [rbp-58h]
  int v271; // [rsp+360h] [rbp-50h]
  unsigned __int8 v272; // [rsp+364h] [rbp-4Ch]
  char v273; // [rsp+368h] [rbp-48h] BYREF
  char v274; // [rsp+378h] [rbp-38h]
  bool v275; // [rsp+379h] [rbp-37h]
  char v276; // [rsp+37Ah] [rbp-36h]
  char v277; // [rsp+37Bh] [rbp-35h]

  v8 = *(_QWORD **)(a5 + 80);
  v197 = (__int64 *)a6;
  v200 = a5 + 72;
  if ( (_QWORD *)(a5 + 72) == v8 )
  {
    i = 0;
  }
  else
  {
    if ( !v8 )
      BUG();
    while ( 1 )
    {
      i = (_QWORD *)v8[4];
      if ( i != v8 + 3 )
        break;
      v8 = (_QWORD *)v8[1];
      if ( (_QWORD *)(a5 + 72) == v8 )
        break;
      if ( !v8 )
        BUG();
    }
  }
LABEL_7:
  result = (_QWORD *)v200;
  if ( v8 != (_QWORD *)v200 )
  {
    v11 = (_QWORD *)v200;
    while ( 1 )
    {
      if ( !i )
        BUG();
      v209 = (__int64)(i - 3);
      v12 = *((_BYTE *)i - 24);
      if ( v12 != 85 )
      {
        if ( *(_QWORD *)a8 == v209 )
          goto LABEL_17;
        if ( v12 == 60 )
        {
          if ( !*(_DWORD *)(a8 + 128)
            || v209 == *(_QWORD *)(a8 + 336)
            || (*((_BYTE *)i - 17) & 0x20) != 0 && sub_B91C10(v209, 39) )
          {
            goto LABEL_17;
          }
          v26 = (unsigned int)(*(_DWORD *)(a8 + 280) - 1) > 2;
          v27 = sub_B43CC0(v209);
          v235 = 1;
          v215 = (__int64 *)v27;
          v31 = v27;
          v219 = v221;
          v220 = 0x800000000LL;
          v233[14] = &v236;
          v216 = 0;
          v241 = a7;
          v217 = 0;
          v242 = (__int64 *)a8;
          v218 = 0;
          v243 = v197;
          v249 = (__int64 *)&v253;
          v255 = &v259;
          v233[13] = 0;
          v233[15] = 8;
          v234 = 0;
          v240 = 1;
          v239 = 0;
          v244 = 0;
          v245 = 0;
          v246 = 0;
          v247 = 0;
          v248 = 0;
          v250 = 4;
          v251 = 0;
          v252 = 1;
          v254 = 0;
          v256 = 2;
          v257 = 0;
          v258 = 1;
          src = &v262;
          v261 = 0x600000000LL;
          v263[1] = &v267;
          v269 = (__int64 *)&v273;
          v275 = v26;
          v32 = *(_QWORD *)(a8 + 120);
          v33 = *(unsigned int *)(a8 + 128);
          v263[0] = 0;
          v264 = 2;
          v34 = v32 + 8 * v33;
          v265 = 0;
          v266 = 1;
          v268 = 0;
          v270 = 2;
          v271 = 0;
          v272 = 1;
          v274 = 0;
          v277 = 0;
          if ( v32 == v34 )
            goto LABEL_85;
          v35 = 1;
          while ( 1 )
          {
            v36 = *(_QWORD *)(*(_QWORD *)v32 + 40LL);
            if ( !(_BYTE)v35 )
              goto LABEL_143;
            v37 = v269;
            v29 = HIDWORD(v270);
            v28 = &v269[HIDWORD(v270)];
            if ( v269 != v28 )
            {
              while ( v36 != *v37 )
              {
                if ( v28 == ++v37 )
                  goto LABEL_150;
              }
              goto LABEL_83;
            }
LABEL_150:
            if ( HIDWORD(v270) < (unsigned int)v270 )
            {
              v29 = (unsigned int)++HIDWORD(v270);
              *v28 = v36;
              v35 = v272;
              ++v268;
            }
            else
            {
LABEL_143:
              sub_C8CC70((__int64)&v268, v36, (__int64)v28, v29, v35, v30);
              v35 = v272;
            }
LABEL_83:
            v32 += 8;
            if ( v34 == v32 )
            {
              v31 = (__int64)v215;
LABEL_85:
              v38 = sub_AE4570(v31, *(i - 2));
              v238 = 1;
              LODWORD(v213) = *(_DWORD *)(v38 + 8) >> 8;
              if ( (unsigned int)v213 > 0x40 )
                sub_C43690((__int64)&v212, 0, 0);
              else
                v212 = 0;
              if ( v240 > 0x40 && v239 )
                j_j___libc_free_0_0((unsigned __int64)v239);
              v43 = &v215;
              v216 = 0;
              v217 = 0;
              v239 = v212;
              v240 = v213;
              sub_3109010((__int64)&v215, v209, v39, v40, v41, v42);
              v47 = (unsigned int)v220;
              if ( (_DWORD)v220 )
              {
                dest = v11;
                v190 = i;
                v184 = v8;
                while ( 1 )
                {
                  v48 = (__int64)&v219[3 * v47 - 3];
                  v49 = *(_DWORD *)(v48 + 16);
                  v50 = *(_QWORD *)v48;
                  *(_DWORD *)(v48 + 16) = 0;
                  v51 = *(_QWORD *)(v48 + 8);
                  LODWORD(v220) = v220 - 1;
                  v52 = 3LL * (unsigned int)v220;
                  v53 = &v219[3 * (unsigned int)v220];
                  if ( *((_DWORD *)v53 + 4) > 0x40u )
                  {
                    v54 = v53[1];
                    if ( v54 )
                      j_j___libc_free_0_0(v54);
                  }
                  v55 = v50 & 0xFFFFFFFFFFFFFFF8LL;
                  v237 = (_QWORD *)(v50 & 0xFFFFFFFFFFFFFFF8LL);
                  v238 = (v50 >> 2) & 1;
                  if ( v238 )
                  {
                    if ( v240 > 0x40 && v239 )
                    {
                      j_j___libc_free_0_0((unsigned __int64)v239);
                      v55 = (unsigned __int64)v237;
                    }
                    v240 = v49;
                    v49 = 0;
                    v239 = (_QWORD *)v51;
                  }
                  v56 = *(_QWORD *)(v55 + 24);
                  if ( !v252 )
                    break;
                  v57 = v249;
                  v44 = HIDWORD(v250);
                  v52 = (__int64)&v249[HIDWORD(v250)];
                  if ( v249 == (__int64 *)v52 )
                  {
LABEL_144:
                    if ( HIDWORD(v250) >= (unsigned int)v250 )
                      break;
                    v44 = (unsigned int)++HIDWORD(v250);
                    *(_QWORD *)v52 = v56;
                    ++v248;
                  }
                  else
                  {
                    while ( v56 != *v57 )
                    {
                      if ( (__int64 *)v52 == ++v57 )
                        goto LABEL_144;
                    }
                  }
LABEL_101:
                  switch ( *(_BYTE *)v56 )
                  {
                    case 0x1E:
                    case 0x1F:
                    case 0x20:
                    case 0x21:
                    case 0x23:
                    case 0x24:
                    case 0x25:
                    case 0x26:
                    case 0x27:
                    case 0x29:
                    case 0x2A:
                    case 0x2B:
                    case 0x2C:
                    case 0x2D:
                    case 0x2E:
                    case 0x2F:
                    case 0x30:
                    case 0x31:
                    case 0x32:
                    case 0x33:
                    case 0x34:
                    case 0x35:
                    case 0x36:
                    case 0x37:
                    case 0x38:
                    case 0x39:
                    case 0x3A:
                    case 0x3B:
                    case 0x3C:
                    case 0x3D:
                    case 0x40:
                    case 0x41:
                    case 0x42:
                    case 0x43:
                    case 0x44:
                    case 0x45:
                    case 0x46:
                    case 0x47:
                    case 0x48:
                    case 0x49:
                    case 0x4A:
                    case 0x4B:
                    case 0x4D:
                    case 0x50:
                    case 0x51:
                    case 0x52:
                    case 0x53:
                    case 0x57:
                    case 0x58:
                    case 0x59:
                    case 0x5A:
                    case 0x5B:
                    case 0x5C:
                    case 0x5D:
                    case 0x5E:
                    case 0x5F:
                    case 0x60:
                      goto LABEL_106;
                    case 0x22:
                    case 0x28:
                      goto LABEL_130;
                    case 0x3E:
                      v58 = v56;
                      if ( !(unsigned __int8)sub_B19DB0(v241, *v242, v56) )
                        v274 = 1;
                      if ( *(_QWORD *)(v56 - 64) != *v237 )
                        goto LABEL_106;
                      v67 = *(_BYTE **)(v56 - 32);
                      if ( *v67 != 60 )
                        goto LABEL_140;
                      v68 = v214;
                      v172 = v56;
                      v212 = v214;
                      v69 = v214;
                      v170 = v56;
                      v175 = v51;
                      v214[0] = v67;
                      v213 = 0x400000001LL;
                      v70 = 1;
                      break;
                    case 0x3F:
                      if ( !*(_QWORD *)(v56 + 16) )
                        goto LABEL_105;
                      if ( !(unsigned __int8)sub_3108E30((__int64)&v215, v56) )
                      {
                        v238 = 0;
                        if ( v240 > 0x40 && v239 )
                          j_j___libc_free_0_0((unsigned __int64)v239);
                        v239 = 0;
                        v240 = 1;
                      }
LABEL_104:
                      sub_3109010((__int64)&v215, v56, v52, v44, v45, (__int64)v46);
LABEL_105:
                      sub_3170BD0((__int64)&v215, v56);
                      goto LABEL_106;
                    case 0x4C:
                      v217 = v56;
                      v58 = v56;
                      goto LABEL_108;
                    case 0x4E:
                    case 0x4F:
                    case 0x54:
                    case 0x56:
                      goto LABEL_104;
                    case 0x55:
                      v66 = *(_QWORD *)(v56 - 32);
                      if ( !v66 || *(_BYTE *)v66 || *(_QWORD *)(v66 + 24) != *(_QWORD *)(v56 + 80) )
                        goto LABEL_130;
                      v106 = *(_DWORD *)(v66 + 36);
                      if ( v106 > 0xF5 )
                        goto LABEL_225;
                      if ( v106 > 0xED )
                      {
                        switch ( v106 )
                        {
                          case 0xEEu:
                          case 0xF0u:
                          case 0xF1u:
                          case 0xF3u:
                          case 0xF5u:
                            if ( (unsigned __int8)sub_B19DB0(v241, *v242, v56) )
                              goto LABEL_106;
                            v274 = 1;
                            v58 = v217;
                            break;
                          default:
                            goto LABEL_225;
                        }
                      }
                      else
                      {
                        if ( v106 <= 0x47 )
                        {
                          if ( v106 > 0x44 )
                            goto LABEL_106;
                          if ( !v106 )
                          {
LABEL_130:
                            sub_31700B0((__int64)&v215, (unsigned __int8 *)v56);
                            v58 = v217;
                            goto LABEL_107;
                          }
                        }
LABEL_225:
                        sub_3170360((__int64)&v215, v56, (__int64 *)v52, v44, v45, (__int64)v46);
                        v58 = v217;
                      }
                      goto LABEL_107;
                    default:
                      BUG();
                  }
                  while ( 1 )
                  {
                    v71 = v70;
                    v72 = v69[v70 - 1];
                    LODWORD(v213) = v70 - 1;
                    v73 = *(_QWORD *)(v72 + 16);
                    if ( v73 )
                      break;
LABEL_365:
                    v70 = v213;
                    v69 = v212;
                    if ( !(_DWORD)v213 )
                    {
                      v46 = v214;
                      v51 = v175;
                      if ( v212 == v214 )
                      {
LABEL_106:
                        v58 = v217;
                      }
                      else
                      {
                        _libc_free((unsigned __int64)v212);
                        v58 = v217;
                      }
                      goto LABEL_107;
                    }
                  }
                  while ( 1 )
                  {
                    v74 = *(_QWORD *)(v73 + 24);
                    v75 = *(_BYTE *)v74;
                    if ( *(_BYTE *)v74 <= 0x1Cu )
                      break;
                    switch ( v75 )
                    {
                      case '=':
                        sub_3109010((__int64)&v215, *(_QWORD *)(v73 + 24), v71, v44, v45, (__int64)v68);
                        sub_3170BD0((__int64)&v215, v74);
                        break;
                      case '>':
                        v168 = *(_QWORD *)(v74 - 32);
                        if ( v72 != v168 || !v168 )
                          goto LABEL_138;
                        break;
                      case 'U':
                        v167 = *(_QWORD *)(v74 - 32);
                        if ( !v167
                          || *(_BYTE *)v167
                          || *(_QWORD *)(v167 + 24) != *(_QWORD *)(v74 + 80)
                          || (*(_BYTE *)(v167 + 33) & 0x20) == 0
                          || (unsigned int)(*(_DWORD *)(v167 + 36) - 210) > 1 )
                        {
                          goto LABEL_138;
                        }
                        break;
                      case 'N':
                        v162 = (unsigned int)v213;
                        v44 = HIDWORD(v213);
                        v163 = (unsigned int)v213 + 1LL;
                        if ( v163 > HIDWORD(v213) )
                        {
                          sub_C8D5F0((__int64)&v212, v214, v163, 8u, v45, (__int64)v68);
                          v162 = (unsigned int)v213;
                        }
                        v71 = (__int64)v212;
                        v212[v162] = v74;
                        LODWORD(v213) = v213 + 1;
                        break;
                      default:
                        goto LABEL_138;
                    }
                    v73 = *(_QWORD *)(v73 + 8);
                    if ( !v73 )
                      goto LABEL_365;
                  }
LABEL_138:
                  v46 = v214;
                  v58 = v172;
                  v56 = v170;
                  v51 = v175;
                  if ( v212 != v214 )
                    _libc_free((unsigned __int64)v212);
LABEL_140:
                  v217 = v56;
LABEL_107:
                  if ( v58 )
                  {
LABEL_108:
                    if ( !(unsigned __int8)sub_B19DB0(v241, *v242, v58) )
                      v274 = 1;
                  }
                  if ( v216 )
                  {
                    v97 = v49;
                    v98 = v51;
                    i = v190;
                    v11 = dest;
                    v8 = v184;
                    if ( v97 > 0x40 && v98 )
                      j_j___libc_free_0_0(v98);
                    goto LABEL_116;
                  }
                  if ( v49 > 0x40 && v51 )
                    j_j___libc_free_0_0(v51);
                  v47 = (unsigned int)v220;
                  if ( !(_DWORD)v220 )
                  {
                    v11 = dest;
                    i = v190;
                    v8 = v184;
                    goto LABEL_116;
                  }
                }
                sub_C8CC70((__int64)&v248, v56, v52, v44, v45, (__int64)v46);
                goto LABEL_101;
              }
LABEL_116:
              if ( v277 )
              {
LABEL_117:
                if ( v276 )
                {
                  desta = v274;
                  if ( (_DWORD)v246 )
                  {
                    v120 = v245;
                    v121 = &v245[4 * v247];
                    if ( v245 != v121 )
                    {
                      while ( 1 )
                      {
                        v122 = v120;
                        if ( *v120 != -8192 && *v120 != -4096 )
                          break;
                        v120 += 4;
                        if ( v121 == v120 )
                          goto LABEL_119;
                      }
                      while ( v122 != v121 )
                      {
                        if ( !*((_BYTE *)v122 + 24) )
                          sub_C64ED0("Unable to handle an alias with unknown offset created before CoroBegin.", 1u);
                        v122 += 4;
                        if ( v122 == v121 )
                          break;
                        while ( *v122 == -8192 || *v122 == -4096 )
                        {
                          v122 += 4;
                          if ( v121 == v122 )
                            goto LABEL_119;
                        }
                      }
                    }
                  }
LABEL_119:
                  sub_C7D6A0(0, 0, 8);
                  v59 = v247;
                  if ( v247 )
                  {
                    v186 = v247;
                    v192 = v247;
                    v99 = sub_C7D670(32LL * v247, 8);
                    v61 = v246;
                    v60 = HIDWORD(v246);
                    v181 = v8;
                    v177 = v43;
                    v59 = v186;
                    v100 = v99;
                    v101 = 0;
                    v187 = i;
                    v102 = v11;
                    v103 = v99 + 8;
                    v104 = v245 + 1;
                    do
                    {
                      v105 = *(v104 - 1);
                      *(_QWORD *)(v103 - 8) = v105;
                      if ( v105 != -8192 && v105 != -4096 )
                      {
                        *(_BYTE *)(v103 + 16) = 0;
                        if ( *((_BYTE *)v104 + 16) )
                        {
                          v123 = *((_DWORD *)v104 + 2);
                          *(_DWORD *)(v103 + 8) = v123;
                          if ( v123 > 0x40 )
                          {
                            v169 = v61;
                            v171 = v59;
                            v174 = v60;
                            sub_C43780(v103, (const void **)v104);
                            v61 = v169;
                            v59 = v171;
                            v60 = v174;
                          }
                          else
                          {
                            *(_QWORD *)v103 = *v104;
                          }
                          *(_BYTE *)(v103 + 16) = 1;
                        }
                      }
                      ++v101;
                      v103 += 32;
                      v104 += 4;
                    }
                    while ( v192 != v101 );
                    v11 = v102;
                    i = v187;
                    v62 = v100;
                    v43 = v177;
                    v8 = v181;
                  }
                  else
                  {
                    v60 = 0;
                    v61 = 0;
                    v62 = 0;
                  }
                  v63 = *(unsigned int *)(a2 + 8);
                  if ( (unsigned int)v63 >= *(_DWORD *)(a2 + 12) )
                  {
                    v178 = v59;
                    v173 = v61;
                    v182 = v60;
                    v193 = sub_C8D7D0(a2, a2 + 16, 0, 0x30u, (unsigned __int64 *)&v212, v59);
                    v124 = v193 + 48LL * *(unsigned int *)(a2 + 8);
                    if ( v124 )
                    {
                      *(_DWORD *)(v124 + 32) = v178;
                      v125 = 0;
                      *(_QWORD *)(v124 + 8) = 1;
                      *(_QWORD *)v124 = v209;
                      *(_DWORD *)(v124 + 24) = v173;
                      *(_DWORD *)(v124 + 28) = v182;
                      *(_BYTE *)(v124 + 40) = desta;
                      *(_QWORD *)(v124 + 16) = v62;
                      v62 = 0;
                    }
                    else
                    {
                      v125 = 32LL * v178;
                      if ( v178 )
                      {
                        deste = v8;
                        v164 = i;
                        v165 = v62;
                        do
                        {
                          if ( *(_QWORD *)v165 != -4096 && *(_QWORD *)v165 != -8192 )
                          {
                            if ( *(_BYTE *)(v165 + 24) )
                            {
                              v108 = *(_DWORD *)(v165 + 16) <= 0x40u;
                              *(_BYTE *)(v165 + 24) = 0;
                              if ( !v108 )
                              {
                                v166 = *(_QWORD *)(v165 + 8);
                                if ( v166 )
                                  j_j___libc_free_0_0(v166);
                              }
                            }
                          }
                          v165 += 32;
                        }
                        while ( v62 + v125 != v165 );
                        i = v164;
                        v8 = deste;
                      }
                    }
                    sub_C7D6A0(v62, v125, 8);
                    v126 = *(_QWORD *)a2;
                    v127 = *(_QWORD *)a2 + 48LL * *(unsigned int *)(a2 + 8);
                    if ( *(_QWORD *)a2 != v127 )
                    {
                      v128 = v193;
                      do
                      {
                        if ( v128 )
                        {
                          v129 = *(_QWORD *)v126;
                          *(_DWORD *)(v128 + 32) = 0;
                          *(_QWORD *)(v128 + 16) = 0;
                          *(_DWORD *)(v128 + 24) = 0;
                          *(_DWORD *)(v128 + 28) = 0;
                          *(_QWORD *)v128 = v129;
                          *(_QWORD *)(v128 + 8) = 1;
                          v130 = *(_QWORD *)(v126 + 16);
                          ++*(_QWORD *)(v126 + 8);
                          v131 = *(_QWORD *)(v128 + 16);
                          *(_QWORD *)(v128 + 16) = v130;
                          LODWORD(v130) = *(_DWORD *)(v126 + 24);
                          *(_QWORD *)(v126 + 16) = v131;
                          LODWORD(v131) = *(_DWORD *)(v128 + 24);
                          *(_DWORD *)(v128 + 24) = v130;
                          LODWORD(v130) = *(_DWORD *)(v126 + 28);
                          *(_DWORD *)(v126 + 24) = v131;
                          LODWORD(v131) = *(_DWORD *)(v128 + 28);
                          *(_DWORD *)(v128 + 28) = v130;
                          LODWORD(v130) = *(_DWORD *)(v126 + 32);
                          *(_DWORD *)(v126 + 28) = v131;
                          LODWORD(v131) = *(_DWORD *)(v128 + 32);
                          *(_DWORD *)(v128 + 32) = v130;
                          *(_DWORD *)(v126 + 32) = v131;
                          *(_BYTE *)(v128 + 40) = *(_BYTE *)(v126 + 40);
                        }
                        v126 += 48LL;
                        v128 += 48;
                      }
                      while ( v127 != v126 );
                      v127 = *(_QWORD *)a2;
                      if ( *(_QWORD *)a2 + 48LL * *(unsigned int *)(a2 + 8) != *(_QWORD *)a2 )
                      {
                        v211 = i;
                        v132 = *(_QWORD *)a2;
                        v133 = *(_QWORD *)a2 + 48LL * *(unsigned int *)(a2 + 8);
                        do
                        {
                          v134 = *(_DWORD *)(v133 - 16);
                          v133 -= 48;
                          if ( v134 )
                          {
                            destd = v8;
                            v135 = v133;
                            v136 = *(_QWORD *)(v133 + 16) + 32LL * v134;
                            v137 = *(_QWORD *)(v133 + 16);
                            do
                            {
                              if ( *(_QWORD *)v137 != -4096 && *(_QWORD *)v137 != -8192 )
                              {
                                if ( *(_BYTE *)(v137 + 24) )
                                {
                                  v108 = *(_DWORD *)(v137 + 16) <= 0x40u;
                                  *(_BYTE *)(v137 + 24) = 0;
                                  if ( !v108 )
                                  {
                                    v138 = *(_QWORD *)(v137 + 8);
                                    if ( v138 )
                                      j_j___libc_free_0_0(v138);
                                  }
                                }
                              }
                              v137 += 32;
                            }
                            while ( v136 != v137 );
                            v133 = v135;
                            v8 = destd;
                          }
                          sub_C7D6A0(*(_QWORD *)(v133 + 16), 32LL * *(unsigned int *)(v133 + 32), 8);
                        }
                        while ( v133 != v132 );
                        i = v211;
                        v127 = *(_QWORD *)a2;
                      }
                    }
                    v139 = (int)v212;
                    if ( a2 + 16 != v127 )
                      _libc_free(v127);
                    ++*(_DWORD *)(a2 + 8);
                    *(_QWORD *)a2 = v193;
                    *(_DWORD *)(a2 + 12) = v139;
                  }
                  else
                  {
                    v64 = *(_QWORD *)a2 + 48 * v63;
                    if ( v64 )
                    {
                      *(_QWORD *)(v64 + 8) = 1;
                      v65 = 0;
                      *(_DWORD *)(v64 + 24) = v61;
                      *(_QWORD *)v64 = v209;
                      *(_DWORD *)(v64 + 28) = v60;
                      *(_DWORD *)(v64 + 32) = v59;
                      *(_BYTE *)(v64 + 40) = desta;
                      *(_QWORD *)(v64 + 16) = v62;
                      v62 = 0;
                    }
                    else
                    {
                      v65 = 32LL * (unsigned int)v59;
                      if ( (_DWORD)v59 )
                      {
                        v210 = i;
                        v107 = v62;
                        do
                        {
                          if ( *(_QWORD *)v107 != -8192 && *(_QWORD *)v107 != -4096 )
                          {
                            if ( *(_BYTE *)(v107 + 24) )
                            {
                              v108 = *(_DWORD *)(v107 + 16) <= 0x40u;
                              *(_BYTE *)(v107 + 24) = 0;
                              if ( !v108 )
                              {
                                v109 = *(_QWORD *)(v107 + 8);
                                if ( v109 )
                                  j_j___libc_free_0_0(v109);
                              }
                            }
                          }
                          v107 += 32;
                        }
                        while ( v62 + v65 != v107 );
                        i = v210;
                      }
                    }
                    sub_C7D6A0(v62, v65, 8);
                    ++*(_DWORD *)(a2 + 8);
                  }
                  sub_C7D6A0(0, 0, 8);
                }
                sub_316FEB0((__int64)v43);
                goto LABEL_17;
              }
              v110 = 1;
              if ( !v275 || HIDWORD(v256) == v257 )
              {
                if ( !v217 )
                {
                  v116 = v252;
                  v117 = v249;
                  if ( v252 )
                    v118 = &v249[HIDWORD(v250)];
                  else
                    v118 = &v249[(unsigned int)v250];
                  v119 = v249;
                  if ( v249 != v118 )
                  {
                    while ( 1 )
                    {
                      destc = v119;
                      if ( (unsigned __int64)*v119 < 0xFFFFFFFFFFFFFFFELL )
                        break;
                      if ( v118 == ++v119 )
                        goto LABEL_263;
                    }
                    if ( v119 != v118 )
                    {
                      v194 = v11;
                      v188 = i;
                      v140 = *v119;
                      while ( 2 )
                      {
                        if ( v116 )
                          v141 = &v117[HIDWORD(v250)];
                        else
                          v141 = &v117[(unsigned int)v250];
                        while ( v141 != v117 )
                        {
                          v142 = *v117;
                          v143 = v117;
                          if ( (unsigned __int64)*v117 < 0xFFFFFFFFFFFFFFFELL )
                          {
                            while ( v143 != v141 )
                            {
                              if ( (unsigned __int8)sub_24F5180((__int64)v243, v140, v142) )
                              {
                                v11 = v194;
                                i = v188;
                                v110 = 1;
                                v43 = &v215;
                                goto LABEL_255;
                              }
                              v145 = v143 + 1;
                              if ( v143 + 1 == v141 )
                                goto LABEL_314;
                              while ( 1 )
                              {
                                v142 = *v145;
                                v143 = v145;
                                if ( (unsigned __int64)*v145 < 0xFFFFFFFFFFFFFFFELL )
                                  break;
                                if ( v141 == ++v145 )
                                  goto LABEL_314;
                              }
                            }
                            break;
                          }
                          ++v117;
                        }
LABEL_314:
                        v144 = destc + 1;
                        if ( destc + 1 != v118 )
                        {
                          while ( 1 )
                          {
                            v140 = *v144;
                            destc = v144;
                            if ( (unsigned __int64)*v144 < 0xFFFFFFFFFFFFFFFELL )
                              break;
                            if ( v118 == ++v144 )
                              goto LABEL_317;
                          }
                          if ( v144 != v118 )
                          {
                            v117 = v249;
                            v116 = v252;
                            continue;
                          }
                        }
                        break;
                      }
LABEL_317:
                      v11 = v194;
                      i = v188;
                      v43 = &v215;
                    }
                  }
LABEL_263:
                  v110 = 0;
                }
                goto LABEL_255;
              }
              if ( HIDWORD(v264) == v265 )
              {
LABEL_255:
                v276 = v110;
                v277 = 1;
                goto LABEL_117;
              }
              v111 = v261;
              v212 = v214;
              v213 = 0x600000000LL;
              if ( !(_DWORD)v261 )
              {
LABEL_245:
                v110 = 1;
                if ( !(unsigned __int8)sub_D0E9C0((__int64)&v212, (__int64)&v268, (__int64)v263, v241, 0, (__int64)v46) )
                {
                  v110 = 0;
                  if ( v217 )
                  {
                    v112 = v258;
                    v113 = v255;
                    if ( v258 )
                      v114 = &v255[8 * HIDWORD(v256)];
                    else
                      v114 = &v255[8 * (unsigned int)v256];
                    v115 = v255;
                    if ( v255 == v114 )
                      goto LABEL_252;
                    while ( *(_QWORD *)v115 >= 0xFFFFFFFFFFFFFFFELL )
                    {
                      v115 += 8;
                      if ( v114 == v115 )
                        goto LABEL_252;
                    }
                    if ( v114 == v115 )
                    {
LABEL_252:
                      v110 = 0;
                    }
                    else
                    {
                      v146 = v8;
                      v147 = &v215;
                      v148 = v114;
                      v149 = *(_QWORD *)v115;
                      v150 = v11;
                      v151 = i;
                      v152 = v115;
                      while ( 1 )
                      {
                        v153 = (__int64)(v112 ? &v113[8 * HIDWORD(v256)] : &v113[8 * (unsigned int)v256]);
                        if ( v113 != (char *)v153 )
                        {
                          while ( 1 )
                          {
                            v154 = *(_QWORD *)v113;
                            v155 = v113;
                            if ( *(_QWORD *)v113 < 0xFFFFFFFFFFFFFFFELL )
                              break;
                            v113 += 8;
                            if ( (char *)v153 == v113 )
                              goto LABEL_336;
                          }
                          if ( (char *)v153 != v113 )
                            break;
                        }
LABEL_336:
                        v156 = v152 + 8;
                        if ( v152 + 8 == v148 )
                          goto LABEL_339;
                        while ( 1 )
                        {
                          v149 = *(_QWORD *)v156;
                          v152 = v156;
                          if ( *(_QWORD *)v156 < 0xFFFFFFFFFFFFFFFELL )
                            break;
                          v156 += 8;
                          if ( v148 == v156 )
                            goto LABEL_339;
                        }
                        if ( v148 == v156 )
                        {
LABEL_339:
                          v11 = v150;
                          i = v151;
                          v8 = v146;
                          v43 = v147;
                          goto LABEL_252;
                        }
                        v113 = v255;
                        v112 = v258;
                      }
                      v157 = v152;
                      v11 = v150;
                      while ( 1 )
                      {
                        v179 = v147;
                        v183 = v157;
                        v189 = v146;
                        v195 = v151;
                        v158 = sub_24F9770(v243, *(_QWORD *)(v149 + 40), *(_QWORD *)(v154 + 40));
                        v151 = v195;
                        v146 = v189;
                        v157 = v183;
                        v147 = v179;
                        if ( v158 )
                          break;
                        v159 = v155 + 8;
                        if ( v155 + 8 != (char *)v153 )
                        {
                          while ( 1 )
                          {
                            v154 = *(_QWORD *)v159;
                            v155 = v159;
                            if ( *(_QWORD *)v159 < 0xFFFFFFFFFFFFFFFELL )
                              break;
                            v159 += 8;
                            if ( (char *)v153 == v159 )
                              goto LABEL_346;
                          }
                          if ( (char *)v153 != v159 )
                            continue;
                        }
LABEL_346:
                        v150 = v11;
                        v152 = v183;
                        goto LABEL_336;
                      }
                      i = v195;
                      v8 = v189;
                      v43 = v179;
                      v110 = 1;
                    }
                  }
                }
                if ( v212 != v214 )
                  _libc_free((unsigned __int64)v212);
                goto LABEL_255;
              }
              if ( (unsigned int)v261 <= 6 )
              {
                v160 = v214;
                v161 = 8LL * (unsigned int)v261;
                goto LABEL_355;
              }
              sub_C8D5F0((__int64)&v212, v214, (unsigned int)v261, 8u, v45, (__int64)v46);
              v160 = v212;
              v161 = 8LL * (unsigned int)v261;
              if ( v161 )
LABEL_355:
                memcpy(v160, src, v161);
              LODWORD(v213) = v111;
              goto LABEL_245;
            }
          }
        }
        goto LABEL_37;
      }
      v14 = *(i - 7);
      if ( !v14 )
        goto LABEL_33;
      if ( !*(_BYTE *)v14
        && *(_QWORD *)(v14 + 24) == i[7]
        && (*(_BYTE *)(v14 + 33) & 0x20) != 0
        && *(_DWORD *)(v14 + 36) == 48
        || !*(_BYTE *)v14
        && *(_QWORD *)(v14 + 24) == i[7]
        && (*(_BYTE *)(v14 + 33) & 0x20) != 0
        && *(_DWORD *)(v14 + 36) == 57 )
      {
        goto LABEL_17;
      }
      if ( !*(_BYTE *)v14 && *(_QWORD *)(v14 + 24) == i[7] && (*(_BYTE *)(v14 + 33) & 0x20) != 0 )
      {
        if ( *(_DWORD *)(v14 + 36) == 60 )
          goto LABEL_17;
        v15 = (__int64)(i - 3);
        if ( v209 == *(_QWORD *)a8 )
          goto LABEL_17;
      }
      else
      {
LABEL_33:
        v15 = (__int64)(i - 3);
        if ( v209 == *(_QWORD *)a8 )
          goto LABEL_17;
        if ( !v14 )
          goto LABEL_37;
        if ( *(_BYTE *)v14 )
          goto LABEL_158;
      }
      if ( *(_QWORD *)(v14 + 24) != i[7] || (*(_BYTE *)(v14 + 33) & 0x20) == 0 || *(_DWORD *)(v14 + 36) != 29 )
      {
LABEL_158:
        if ( !*(_BYTE *)v14
          && *(_QWORD *)(v14 + 24) == i[7]
          && (*(_BYTE *)(v14 + 33) & 0x20) != 0
          && *(_DWORD *)(v14 + 36) == 31 )
        {
          goto LABEL_17;
        }
LABEL_37:
        v16 = *(i - 1);
        if ( !v16 )
          goto LABEL_17;
        while ( 1 )
        {
          v22 = *(_QWORD *)(v16 + 24);
          v23 = i[2];
          if ( v12 == 85 )
          {
            v25 = *(i - 7);
            if ( v25 )
            {
              if ( !*(_BYTE *)v25
                && *(_QWORD *)(v25 + 24) == i[7]
                && (*(_BYTE *)(v25 + 33) & 0x20) != 0
                && (unsigned int)(*(_DWORD *)(v25 + 36) - 60) <= 2 )
              {
                v23 = sub_AA56F0(i[2]);
              }
            }
          }
          if ( *(_BYTE *)v22 == 84 )
          {
            if ( (*(_DWORD *)(v22 + 4) & 0x7FFFFFFu) > 1 )
              goto LABEL_46;
            v17 = *(_QWORD *)(v22 + 40);
          }
          else
          {
            v17 = *(_QWORD *)(v22 + 40);
            if ( *(_BYTE *)v22 == 85 )
            {
              v24 = *(_QWORD *)(v22 - 32);
              if ( v24 )
              {
                if ( !*(_BYTE *)v24
                  && *(_QWORD *)(v24 + 24) == *(_QWORD *)(v22 + 80)
                  && (*(_BYTE *)(v24 + 33) & 0x20) != 0
                  && *(_DWORD *)(v24 + 36) == 62
                  || !*(_BYTE *)v24
                  && *(_QWORD *)(v24 + 24) == *(_QWORD *)(v22 + 80)
                  && (*(_BYTE *)(v24 + 33) & 0x20) != 0
                  && *(_DWORD *)(v24 + 36) == 61 )
                {
                  v17 = sub_AA54C0(*(_QWORD *)(v22 + 40));
                }
              }
            }
          }
          if ( sub_24F96E0(v197, v23, v17) )
          {
            if ( *(_BYTE *)(*(i - 2) + 8LL) == 11 )
              sub_C64ED0("token definition is separated from the use by a suspend point", 1u);
            v215 = i - 3;
            v20 = sub_31711D0(a1, (__int64 *)&v215, v18, v19, a5, a6);
            v21 = *(unsigned int *)(v20 + 8);
            a6 = v21 + 1;
            if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(v20 + 12) )
            {
              destf = (unsigned int *)v20;
              sub_C8D5F0(v20, (const void *)(v20 + 16), v21 + 1, 8u, a5, a6);
              v20 = (__int64)destf;
              v21 = destf[2];
            }
            *(_QWORD *)(*(_QWORD *)v20 + 8 * v21) = v22;
            ++*(_DWORD *)(v20 + 8);
          }
LABEL_46:
          v16 = *(_QWORD *)(v16 + 8);
          if ( !v16 )
            goto LABEL_17;
          v12 = *((_BYTE *)i - 24);
        }
      }
      v215 = 0;
      v216 = (unsigned __int64)&v219;
      v217 = 8;
      LODWORD(v218) = 0;
      BYTE4(v218) = 1;
      v76 = *(i - 1);
      if ( v76 )
        break;
LABEL_178:
      v80 = sub_3170240(i[2], (__int64)&v215, (__int64 *)v14, v15, a5, a6);
      if ( BYTE4(v218) )
      {
        if ( v80 )
          goto LABEL_185;
      }
      else
      {
        _libc_free(v216);
        if ( v80 )
        {
LABEL_185:
          v223 = sub_BD5C60(v209);
          v224 = &v232;
          v225 = v233;
          v215 = &v217;
          v232 = &unk_49DA100;
          v216 = 0x200000000LL;
          v222 = 0;
          v228 = 512;
          v233[0] = &unk_49DA0B0;
          v226 = 0;
          v227 = 0;
          v229 = 7;
          v230 = 0;
          v231 = 0;
          v221[0] = 0;
          v221[1] = 0;
          sub_D5F1F0((__int64)&v215, v209);
          v82 = sub_24F4BB0(a8, (__int64 *)&v215, *(_QWORD *)(v209 - 32LL * (*((_DWORD *)i - 5) & 0x7FFFFFF)), 0);
          v84 = *(i - 1);
          v85 = v82;
          if ( v84 )
          {
            v176 = i;
            v86 = *(i - 1);
            v180 = v11;
            do
            {
              v89 = *(_QWORD *)(v86 + 24);
              if ( *(_BYTE *)v89 == 85
                && (v90 = *(_QWORD *)(v89 - 32)) != 0
                && !*(_BYTE *)v90
                && *(_QWORD *)(v90 + 24) == *(_QWORD *)(v89 + 80)
                && (*(_BYTE *)(v90 + 33) & 0x20) != 0
                && *(_DWORD *)(v90 + 36) == 31 )
              {
                sub_BD84D0(*(_QWORD *)(v86 + 24), v85);
              }
              else
              {
                sub_D5F1F0((__int64)&v215, *(_QWORD *)(v86 + 24));
                sub_24F4CC0(a8, (__int64 *)&v215, v85, 0);
              }
              v87 = *(unsigned int *)(a3 + 8);
              if ( v87 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
              {
                sub_C8D5F0(a3, (const void *)(a3 + 16), v87 + 1, 8u, v84, v83);
                v87 = *(unsigned int *)(a3 + 8);
              }
              *(_QWORD *)(*(_QWORD *)a3 + 8 * v87) = v89;
              v88 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
              *(_DWORD *)(a3 + 8) = v88;
              v86 = *(_QWORD *)(v86 + 8);
            }
            while ( v86 );
            v11 = v180;
            i = v176;
          }
          else
          {
            v88 = *(unsigned int *)(a3 + 8);
          }
          if ( v88 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
          {
            sub_C8D5F0(a3, (const void *)(a3 + 16), v88 + 1, 8u, v84, v83);
            v88 = *(unsigned int *)(a3 + 8);
          }
          *(_QWORD *)(*(_QWORD *)a3 + 8 * v88) = v209;
          ++*(_DWORD *)(a3 + 8);
          nullsub_61();
          v232 = &unk_49DA100;
          nullsub_63();
          if ( v215 != &v217 )
            _libc_free((unsigned __int64)v215);
          a5 = *(_QWORD *)(v85 + 16);
          if ( a5 )
          {
            destb = i;
            v91 = *(_QWORD *)(v85 + 16);
            v191 = v8;
            do
            {
              while ( 1 )
              {
                v92 = *(_QWORD *)(v91 + 24);
                if ( (unsigned __int8)sub_24F5180((__int64)v197, v85, v92) )
                  break;
                v91 = *(_QWORD *)(v91 + 8);
                if ( !v91 )
                  goto LABEL_210;
              }
              v215 = (__int64 *)v85;
              v95 = sub_31711D0(a1, (__int64 *)&v215, v93, v94, a5, a6);
              v96 = *(unsigned int *)(v95 + 8);
              if ( v96 + 1 > (unsigned __int64)*(unsigned int *)(v95 + 12) )
              {
                v185 = v95;
                sub_C8D5F0(v95, (const void *)(v95 + 16), v96 + 1, 8u, a5, a6);
                v95 = v185;
                v96 = *(unsigned int *)(v185 + 8);
              }
              *(_QWORD *)(*(_QWORD *)v95 + 8 * v96) = v92;
              ++*(_DWORD *)(v95 + 8);
              v91 = *(_QWORD *)(v91 + 8);
            }
            while ( v91 );
LABEL_210:
            i = destb;
            v8 = v191;
          }
          goto LABEL_17;
        }
      }
      v81 = *(unsigned int *)(a4 + 8);
      if ( v81 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
      {
        sub_C8D5F0(a4, (const void *)(a4 + 16), v81 + 1, 8u, a5, a6);
        v81 = *(unsigned int *)(a4 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a4 + 8 * v81) = v209;
      ++*(_DWORD *)(a4 + 8);
LABEL_17:
      for ( i = (_QWORD *)i[1]; ; i = (_QWORD *)v8[4] )
      {
        v13 = v8 - 3;
        if ( !v8 )
          v13 = 0;
        result = v13 + 6;
        if ( i != result )
          break;
        v8 = (_QWORD *)v8[1];
        if ( v11 == v8 )
          goto LABEL_7;
        if ( !v8 )
          BUG();
      }
      if ( v11 == v8 )
        return result;
    }
    while ( 1 )
    {
      v77 = *(_QWORD *)(v76 + 24);
      if ( *(_BYTE *)v77 != 85 )
        goto LABEL_165;
      v14 = *(_QWORD *)(v77 - 32);
      if ( !v14
        || *(_BYTE *)v14
        || *(_QWORD *)(v14 + 24) != *(_QWORD *)(v77 + 80)
        || (*(_BYTE *)(v14 + 33) & 0x20) == 0
        || *(_DWORD *)(v14 + 36) != 30 )
      {
        goto LABEL_165;
      }
      v78 = *(_QWORD *)(v77 + 40);
      if ( !BYTE4(v218) )
        goto LABEL_183;
      v79 = (_QWORD *)v216;
      v15 = HIDWORD(v217);
      v14 = v216 + 8LL * HIDWORD(v217);
      if ( v216 != v14 )
      {
        while ( v78 != *v79 )
        {
          if ( (_QWORD *)v14 == ++v79 )
            goto LABEL_176;
        }
        goto LABEL_165;
      }
LABEL_176:
      if ( HIDWORD(v217) < (unsigned int)v217 )
      {
        v15 = (unsigned int)++HIDWORD(v217);
        *(_QWORD *)v14 = v78;
        v215 = (__int64 *)((char *)v215 + 1);
        v76 = *(_QWORD *)(v76 + 8);
        if ( !v76 )
          goto LABEL_178;
      }
      else
      {
LABEL_183:
        sub_C8CC70((__int64)&v215, v78, v14, v15, a5, a6);
LABEL_165:
        v76 = *(_QWORD *)(v76 + 8);
        if ( !v76 )
          goto LABEL_178;
      }
    }
  }
  return result;
}
