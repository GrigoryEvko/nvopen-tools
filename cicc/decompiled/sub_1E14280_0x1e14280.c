// Function: sub_1E14280
// Address: 0x1e14280
//
void __fastcall sub_1E14280(__int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 v4; // rax
  _BYTE *v5; // rdi
  _BYTE *v6; // rax
  _BYTE *v7; // r14
  __int64 v8; // r13
  __int64 *v9; // rdi
  __int64 v10; // rax
  __int64 (*v11)(void); // rdx
  __int64 (*v12)(void); // rax
  int *v13; // r8
  int v14; // r9d
  __int64 v15; // rdx
  int *v16; // r13
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // r15
  __int64 v20; // rbx
  __int64 v21; // r13
  __int64 v22; // r12
  unsigned int v23; // r14d
  _DWORD *v24; // rax
  unsigned __int8 v25; // al
  char v26; // r12
  __int64 v27; // rbx
  __int64 v28; // rax
  char v29; // al
  __int64 v30; // rdx
  unsigned __int16 *v31; // rax
  int v32; // edx
  unsigned __int16 *v33; // rax
  int v34; // ebx
  bool v35; // zf
  unsigned __int16 *v36; // r12
  unsigned int v37; // r15d
  _DWORD *v38; // rcx
  _DWORD *v39; // rax
  int v40; // eax
  int *v41; // rdx
  __int64 v42; // r13
  __int64 v43; // r14
  char v44; // si
  char *v45; // rax
  char *v46; // rdi
  __int64 v47; // r14
  _BOOL8 v48; // r9
  unsigned int v49; // edx
  _DWORD *v50; // rax
  _DWORD *v51; // rsi
  _BOOL8 v52; // rdi
  _DWORD *v53; // rax
  _DWORD *v54; // rsi
  unsigned int v55; // eax
  __int64 v56; // rbx
  __int64 v57; // rdi
  __int64 v58; // rbx
  __int64 v59; // rdi
  __int64 v60; // rbx
  __int64 v61; // rdi
  __int64 v62; // rbx
  __int64 v63; // rdi
  __int64 v64; // r14
  unsigned int v65; // edx
  __int64 v66; // rax
  _BOOL4 v67; // r8d
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rbx
  unsigned int *v71; // r12
  unsigned int v72; // ecx
  int *i; // r14
  unsigned int v74; // edx
  int *v75; // rax
  _BOOL4 v76; // ebx
  __int64 v77; // rax
  int v78; // eax
  unsigned int v79; // edx
  int *v80; // rax
  _BOOL4 v81; // r14d
  __int64 v82; // rax
  char *v83; // rsi
  char *v84; // rdx
  int v85; // eax
  char *v86; // rdi
  char v87; // al
  char *v88; // rsi
  char *v89; // rdx
  int v90; // eax
  char *v91; // rdi
  char v92; // al
  __int64 v93; // rax
  __int64 v94; // rsi
  __int64 v95; // rax
  _DWORD *v96; // rdi
  int *v97; // r10
  char v98; // bl
  char *v99; // r13
  unsigned int v100; // ecx
  int *v101; // r12
  unsigned int v102; // edx
  int *v103; // rax
  char v104; // di
  _BOOL4 v105; // r15d
  __int64 v106; // rax
  __int64 v107; // rax
  __int64 v108; // rax
  int *v109; // r10
  __int64 v110; // rsi
  __int64 v111; // rcx
  __int64 v112; // rax
  int *v113; // r10
  __int64 v114; // rsi
  __int64 v115; // rcx
  __int64 v116; // rax
  __int64 v117; // rdx
  char *v118; // rax
  char *v119; // rsi
  __int64 v120; // rax
  int *v121; // r9
  __int64 v122; // rax
  char v123; // di
  _BOOL4 v124; // r12d
  __int64 v125; // rax
  __int64 v126; // rax
  int *v127; // r8
  __int64 v128; // rsi
  __int64 v129; // rcx
  __int64 v130; // rax
  char v131; // r12
  unsigned int v132; // r14d
  int *v133; // rax
  char v134; // si
  _BOOL4 v135; // r12d
  __int64 v136; // rax
  __int64 v137; // rax
  int *v138; // [rsp+8h] [rbp-5A8h]
  _QWORD *v139; // [rsp+58h] [rbp-558h]
  __int64 v140; // [rsp+78h] [rbp-538h]
  __int64 v141; // [rsp+80h] [rbp-530h]
  __int64 v143; // [rsp+90h] [rbp-520h]
  __int64 v144; // [rsp+A0h] [rbp-510h]
  __int64 v145; // [rsp+A8h] [rbp-508h]
  unsigned __int16 *v146; // [rsp+B0h] [rbp-500h]
  __int64 v147; // [rsp+B0h] [rbp-500h]
  __int64 v148; // [rsp+B0h] [rbp-500h]
  int *v149; // [rsp+B0h] [rbp-500h]
  int *v150; // [rsp+B0h] [rbp-500h]
  int *v151; // [rsp+B0h] [rbp-500h]
  _BOOL4 v152; // [rsp+B8h] [rbp-4F8h]
  unsigned __int16 v153; // [rsp+B8h] [rbp-4F8h]
  int *v154; // [rsp+B8h] [rbp-4F8h]
  int *v155; // [rsp+B8h] [rbp-4F8h]
  __int64 v156; // [rsp+C0h] [rbp-4F0h]
  int *v157; // [rsp+C8h] [rbp-4E8h]
  __int64 v158; // [rsp+C8h] [rbp-4E8h]
  __int64 v159; // [rsp+C8h] [rbp-4E8h]
  unsigned int v160; // [rsp+DCh] [rbp-4D4h] BYREF
  __int64 v161; // [rsp+E0h] [rbp-4D0h] BYREF
  unsigned int v162; // [rsp+E8h] [rbp-4C8h]
  __int64 v163; // [rsp+F0h] [rbp-4C0h]
  __int64 v164; // [rsp+F8h] [rbp-4B8h]
  __int64 v165; // [rsp+100h] [rbp-4B0h]
  _BYTE *v166; // [rsp+110h] [rbp-4A0h] BYREF
  __int64 v167; // [rsp+118h] [rbp-498h]
  _BYTE v168[32]; // [rsp+120h] [rbp-490h] BYREF
  _BYTE *v169; // [rsp+140h] [rbp-470h] BYREF
  __int64 v170; // [rsp+148h] [rbp-468h]
  _BYTE v171[32]; // [rsp+150h] [rbp-460h] BYREF
  void *v172; // [rsp+170h] [rbp-440h] BYREF
  __int64 v173; // [rsp+178h] [rbp-438h]
  _BYTE v174[32]; // [rsp+180h] [rbp-430h] BYREF
  __int64 v175; // [rsp+1A0h] [rbp-410h] BYREF
  int v176; // [rsp+1A8h] [rbp-408h] BYREF
  __int64 v177; // [rsp+1B0h] [rbp-400h]
  int *v178; // [rsp+1B8h] [rbp-3F8h]
  int *v179; // [rsp+1C0h] [rbp-3F0h]
  __int64 v180; // [rsp+1C8h] [rbp-3E8h]
  unsigned __int64 v181[2]; // [rsp+1D0h] [rbp-3E0h] BYREF
  _BYTE v182[40]; // [rsp+1E0h] [rbp-3D0h] BYREF
  int v183; // [rsp+208h] [rbp-3A8h] BYREF
  __int64 v184; // [rsp+210h] [rbp-3A0h]
  int *v185; // [rsp+218h] [rbp-398h]
  int *v186; // [rsp+220h] [rbp-390h]
  __int64 v187; // [rsp+228h] [rbp-388h]
  _BYTE *v188; // [rsp+230h] [rbp-380h] BYREF
  __int64 v189; // [rsp+238h] [rbp-378h]
  _BYTE v190[40]; // [rsp+240h] [rbp-370h] BYREF
  int v191; // [rsp+268h] [rbp-348h] BYREF
  __int64 v192; // [rsp+270h] [rbp-340h]
  int *v193; // [rsp+278h] [rbp-338h]
  int *v194; // [rsp+280h] [rbp-330h]
  _BOOL8 v195; // [rsp+288h] [rbp-328h]
  _BYTE *v196; // [rsp+290h] [rbp-320h] BYREF
  __int64 v197; // [rsp+298h] [rbp-318h]
  _BYTE v198[40]; // [rsp+2A0h] [rbp-310h] BYREF
  int v199; // [rsp+2C8h] [rbp-2E8h] BYREF
  __int64 v200; // [rsp+2D0h] [rbp-2E0h]
  int *v201; // [rsp+2D8h] [rbp-2D8h]
  int *v202; // [rsp+2E0h] [rbp-2D0h]
  _BOOL8 v203; // [rsp+2E8h] [rbp-2C8h]
  void *dest; // [rsp+2F0h] [rbp-2C0h] BYREF
  __int64 v205; // [rsp+2F8h] [rbp-2B8h]
  _BYTE v206[64]; // [rsp+300h] [rbp-2B0h] BYREF
  __int64 v207; // [rsp+340h] [rbp-270h] BYREF
  int v208; // [rsp+348h] [rbp-268h] BYREF
  int *v209; // [rsp+350h] [rbp-260h]
  int *v210; // [rsp+358h] [rbp-258h]
  int *v211; // [rsp+360h] [rbp-250h]
  __int64 v212; // [rsp+368h] [rbp-248h]
  _BYTE *v213; // [rsp+370h] [rbp-240h] BYREF
  __int64 v214; // [rsp+378h] [rbp-238h]
  _BYTE v215[128]; // [rsp+380h] [rbp-230h] BYREF
  _BYTE *v216; // [rsp+400h] [rbp-1B0h] BYREF
  __int64 v217; // [rsp+408h] [rbp-1A8h]
  _BYTE v218[136]; // [rsp+410h] [rbp-1A0h] BYREF
  int v219; // [rsp+498h] [rbp-118h] BYREF
  __int64 v220; // [rsp+4A0h] [rbp-110h]
  int *v221; // [rsp+4A8h] [rbp-108h]
  int *v222; // [rsp+4B0h] [rbp-100h]
  __int64 v223; // [rsp+4B8h] [rbp-F8h]
  _BYTE *v224; // [rsp+4C0h] [rbp-F0h] BYREF
  __int64 v225; // [rsp+4C8h] [rbp-E8h]
  _BYTE v226[136]; // [rsp+4D0h] [rbp-E0h] BYREF
  int v227; // [rsp+558h] [rbp-58h] BYREF
  __int64 v228; // [rsp+560h] [rbp-50h]
  int *v229; // [rsp+568h] [rbp-48h]
  int *v230; // [rsp+570h] [rbp-40h]
  __int64 v231; // [rsp+578h] [rbp-38h]

  v144 = a2;
  if ( !a2 )
    BUG();
  v4 = a2;
  if ( (*(_BYTE *)a2 & 4) == 0 && (*(_BYTE *)(a2 + 46) & 8) != 0 )
  {
    do
      v4 = *(_QWORD *)(v4 + 8);
    while ( (*(_BYTE *)(v4 + 46) & 8) != 0 );
  }
  v5 = *(_BYTE **)(v4 + 8);
  if ( a3 != v5 )
  {
    while ( 1 )
    {
      if ( !v5 )
        BUG();
      v6 = v5;
      if ( (*v5 & 4) == 0 && (v5[46] & 8) != 0 )
      {
        do
          v6 = (_BYTE *)*((_QWORD *)v6 + 1);
        while ( (v6[46] & 8) != 0 );
      }
      v7 = (_BYTE *)*((_QWORD *)v6 + 1);
      sub_1E163F0(v5);
      if ( a3 == v7 )
        break;
      v5 = v7;
    }
  }
  v8 = 0;
  v9 = *(__int64 **)(*(_QWORD *)(a1 + 56) + 16LL);
  v140 = *(_QWORD *)(a1 + 56);
  v10 = *v9;
  v11 = *(__int64 (**)(void))(*v9 + 40);
  if ( v11 != sub_1D00B00 )
  {
    v8 = v11();
    v10 = **(_QWORD **)(v140 + 16);
  }
  v143 = 0;
  v12 = *(__int64 (**)(void))(v10 + 112);
  if ( v12 != sub_1D00B10 )
    v143 = v12();
  v139 = sub_1E0B640(v140, *(_QWORD *)(v8 + 8) + 1024LL, (__int64 *)(a2 + 64), 0);
  sub_1DD6E10(a1, (__int64 *)a2, (__int64)v139);
  if ( a3 != (_BYTE *)a2 )
    sub_1E16410(v139);
  v15 = 0x2000000000LL;
  v219 = 0;
  v16 = &v219;
  v17 = (__int64)&v183;
  v178 = &v176;
  v179 = &v176;
  dest = v206;
  v213 = v215;
  v205 = 0x1000000000LL;
  v216 = v218;
  v210 = &v208;
  v211 = &v208;
  v172 = v174;
  v166 = v168;
  v173 = 0x800000000LL;
  v167 = 0x800000000LL;
  v214 = 0x2000000000LL;
  v217 = 0x2000000000LL;
  v220 = 0;
  v221 = &v219;
  v222 = &v219;
  v223 = 0;
  v176 = 0;
  v177 = 0;
  v180 = 0;
  v208 = 0;
  v209 = 0;
  v212 = 0;
  v181[0] = (unsigned __int64)v182;
  v181[1] = 0x800000000LL;
  v189 = 0x800000000LL;
  v197 = 0x800000000LL;
  v188 = v190;
  v201 = &v199;
  v202 = &v199;
  v193 = &v191;
  v194 = &v191;
  v169 = v171;
  v183 = 0;
  v184 = 0;
  v185 = &v183;
  v186 = &v183;
  v187 = 0;
  v191 = 0;
  v192 = 0;
  v195 = 0;
  v196 = v198;
  v199 = 0;
  v200 = 0;
  v203 = 0;
  v170 = 0x400000000LL;
  if ( a3 == (_BYTE *)a2 )
  {
    v225 = 0x2000000000LL;
    v224 = v226;
    v227 = 0;
    v228 = 0;
    v229 = &v227;
    v230 = &v227;
    v231 = 0;
  }
  else
  {
    do
    {
      v18 = *(unsigned int *)(v144 + 40);
      if ( !(_DWORD)v18 )
        goto LABEL_31;
      LODWORD(v13) = 5 * v18;
      v19 = v144;
      v20 = 0;
      v157 = v16;
      v21 = 40 * v18;
      do
      {
        v22 = v20 + *(_QWORD *)(v19 + 32);
        if ( *(_BYTE *)v22 )
          goto LABEL_29;
        if ( (*(_BYTE *)(v22 + 3) & 0x10) != 0 )
        {
          v95 = (unsigned int)v170;
          if ( (unsigned int)v170 >= HIDWORD(v170) )
          {
            sub_16CD150((__int64)&v169, v171, 0, 8, (int)v13, v14);
            v95 = (unsigned int)v170;
          }
          v15 = (__int64)v169;
          *(_QWORD *)&v169[8 * v95] = v22;
          LODWORD(v170) = v170 + 1;
          goto LABEL_29;
        }
        v23 = *(_DWORD *)(v22 + 8);
        LODWORD(v224) = v23;
        if ( !v23 )
          goto LABEL_29;
        if ( v223 )
        {
          v93 = v220;
          if ( !v220 )
            goto LABEL_182;
          v94 = (__int64)v157;
          do
          {
            while ( 1 )
            {
              v17 = *(_QWORD *)(v93 + 16);
              v15 = *(_QWORD *)(v93 + 24);
              if ( v23 <= *(_DWORD *)(v93 + 32) )
                break;
              v93 = *(_QWORD *)(v93 + 24);
              if ( !v15 )
                goto LABEL_180;
            }
            v94 = v93;
            v93 = *(_QWORD *)(v93 + 16);
          }
          while ( v17 );
LABEL_180:
          if ( (int *)v94 == v157 || v23 < *(_DWORD *)(v94 + 32) )
          {
LABEL_182:
            if ( (unsigned __int8)((unsigned __int64)sub_1DCB780(
                                                       (__int64)v181,
                                                       (unsigned int *)&v224,
                                                       v15,
                                                       v17,
                                                       (int)v13) >> 32) )
            {
              v116 = (unsigned int)v167;
              if ( (unsigned int)v167 >= HIDWORD(v167) )
              {
                sub_16CD150((__int64)&v166, v168, 0, 4, (int)v13, v14);
                v116 = (unsigned int)v167;
              }
              v17 = (unsigned int)v224;
              v117 = (__int64)v166;
              *(_DWORD *)&v166[4 * v116] = (_DWORD)v224;
              LODWORD(v167) = v167 + 1;
              if ( (*(_BYTE *)(v22 + 4) & 1) != 0 )
                sub_1DCB780((__int64)&v196, (unsigned int *)&v224, v117, v17, (int)v13);
            }
            v15 = (*(_BYTE *)(v22 + 3) >> 4) ^ 1u;
            if ( (((*(_BYTE *)(v22 + 3) & 0x40) != 0) & ((*(_BYTE *)(v22 + 3) >> 4) ^ 1)) != 0 )
              sub_1DCB780((__int64)&v188, (unsigned int *)&v224, v15, v17, (int)v13);
            goto LABEL_29;
          }
        }
        else
        {
          v24 = v216;
          v15 = (__int64)&v216[4 * (unsigned int)v217];
          if ( v216 == (_BYTE *)v15 )
            goto LABEL_182;
          while ( v23 != *v24 )
          {
            if ( (_DWORD *)v15 == ++v24 )
              goto LABEL_182;
          }
          if ( (_DWORD *)v15 == v24 )
            goto LABEL_182;
        }
        v25 = *(_BYTE *)(v22 + 3);
        *(_BYTE *)(v22 + 4) |= 2u;
        v26 = ((v25 & 0x40) != 0) & ((v25 >> 4) ^ 1);
        if ( !v26 )
          goto LABEL_29;
        if ( v212 )
        {
          v15 = (__int64)v209;
          if ( v209 )
          {
            while ( 1 )
            {
              v17 = *(unsigned int *)(v15 + 32);
              v122 = *(_QWORD *)(v15 + 24);
              v123 = 0;
              if ( v23 < (unsigned int)v17 )
              {
                v122 = *(_QWORD *)(v15 + 16);
                v123 = v26;
              }
              if ( !v122 )
                break;
              v15 = v122;
            }
            if ( !v123 )
            {
              if ( v23 <= (unsigned int)v17 )
                goto LABEL_29;
LABEL_265:
              v124 = 1;
              if ( (int *)v15 == &v208 )
              {
LABEL_266:
                v148 = v15;
                v125 = sub_22077B0(40);
                *(_DWORD *)(v125 + 32) = (_DWORD)v224;
                sub_220F040(v124, v125, v148, &v208);
                ++v212;
                goto LABEL_29;
              }
LABEL_292:
              v124 = v23 < *(_DWORD *)(v15 + 32);
              goto LABEL_266;
            }
            if ( v210 == (int *)v15 )
              goto LABEL_265;
          }
          else
          {
            v15 = (__int64)&v208;
            if ( v210 == &v208 )
            {
              v15 = (__int64)&v208;
              v124 = 1;
              goto LABEL_266;
            }
          }
          v150 = (int *)v15;
          if ( v23 <= *(_DWORD *)(sub_220EF80(v15) + 32) )
            goto LABEL_29;
          v15 = (__int64)v150;
          if ( !v150 )
            goto LABEL_29;
          v124 = 1;
          if ( v150 == &v208 )
            goto LABEL_266;
          goto LABEL_292;
        }
        v17 = (__int64)dest;
        v15 = (__int64)dest + 4 * (unsigned int)v205;
        if ( dest != (void *)v15 )
        {
          v96 = dest;
          while ( v23 != *v96 )
          {
            if ( (_DWORD *)v15 == ++v96 )
              goto LABEL_206;
          }
          if ( (_DWORD *)v15 != v96 )
            goto LABEL_29;
        }
LABEL_206:
        v97 = v209;
        if ( (unsigned int)v205 <= 0xFuLL )
        {
          if ( (unsigned int)v205 >= HIDWORD(v205) )
          {
            sub_16CD150((__int64)&dest, v206, 0, 4, (int)v13, v14);
            v23 = (unsigned int)v224;
            v15 = (__int64)dest + 4 * (unsigned int)v205;
          }
          *(_DWORD *)v15 = v23;
          LODWORD(v205) = v205 + 1;
          goto LABEL_29;
        }
        v145 = v20;
        v98 = ((v25 & 0x40) != 0) & ((v25 >> 4) ^ 1);
        v141 = v21;
        v99 = (char *)dest + 4 * (unsigned int)v205 - 4;
        v147 = v19;
        if ( v209 )
        {
LABEL_208:
          v100 = *(_DWORD *)v99;
          v101 = v97;
          v14 = 0;
          while ( 1 )
          {
            v102 = v101[8];
            v103 = (int *)*((_QWORD *)v101 + 3);
            v104 = 0;
            if ( v100 < v102 )
            {
              v103 = (int *)*((_QWORD *)v101 + 2);
              v104 = v98;
            }
            if ( !v103 )
              break;
            v101 = v103;
          }
          if ( v104 )
          {
            if ( v210 != v101 )
              goto LABEL_275;
          }
          else if ( v100 <= v102 )
          {
            goto LABEL_218;
          }
LABEL_215:
          v105 = 1;
          if ( v101 != &v208 )
            v105 = *(_DWORD *)v99 < (unsigned int)v101[8];
LABEL_217:
          v106 = sub_22077B0(40);
          *(_DWORD *)(v106 + 32) = *(_DWORD *)v99;
          sub_220F040(v105, v106, v101, &v208);
          ++v212;
          v97 = v209;
          goto LABEL_218;
        }
        while ( 1 )
        {
          v101 = &v208;
          if ( v210 == &v208 )
          {
            v105 = 1;
            goto LABEL_217;
          }
LABEL_275:
          v138 = v97;
          v130 = sub_220EF80(v101);
          v97 = v138;
          if ( *(_DWORD *)(v130 + 32) < *(_DWORD *)v99 )
            goto LABEL_215;
LABEL_218:
          v35 = (_DWORD)v205 == 1;
          v107 = (unsigned int)(v205 - 1);
          LODWORD(v205) = v205 - 1;
          if ( v35 )
            break;
          v99 = (char *)dest + 4 * v107 - 4;
          if ( v97 )
            goto LABEL_208;
        }
        v131 = v98;
        v19 = v147;
        v20 = v145;
        v21 = v141;
        if ( !v97 )
        {
          if ( v210 == &v208 )
          {
            v97 = &v208;
            v135 = 1;
            goto LABEL_287;
          }
          v132 = (unsigned int)v224;
          v97 = &v208;
          goto LABEL_298;
        }
        v132 = (unsigned int)v224;
        v17 = 0;
        while ( 1 )
        {
          v15 = (unsigned int)v97[8];
          v133 = (int *)*((_QWORD *)v97 + 3);
          v134 = 0;
          if ( (unsigned int)v224 < (unsigned int)v15 )
          {
            v133 = (int *)*((_QWORD *)v97 + 2);
            v134 = v131;
          }
          if ( !v133 )
            break;
          v97 = v133;
        }
        if ( v134 )
        {
          if ( v210 == v97 )
            goto LABEL_285;
LABEL_298:
          v151 = v97;
          v137 = sub_220EF80(v97);
          v97 = v151;
          if ( v132 > *(_DWORD *)(v137 + 32) )
            goto LABEL_285;
          goto LABEL_29;
        }
        if ( (unsigned int)v224 > (unsigned int)v15 )
        {
LABEL_285:
          v135 = 1;
          if ( v97 != &v208 )
            v135 = v132 < v97[8];
LABEL_287:
          v149 = v97;
          v136 = sub_22077B0(40);
          *(_DWORD *)(v136 + 32) = (_DWORD)v224;
          sub_220F040(v135, v136, v149, &v208);
          ++v212;
        }
LABEL_29:
        v20 += 40;
      }
      while ( v21 != v20 );
      v16 = v157;
LABEL_31:
      if ( (_DWORD)v170 )
      {
        v158 = 0;
        v156 = 8LL * (unsigned int)v170;
        do
        {
          v27 = *(_QWORD *)&v169[v158];
          LODWORD(v224) = *(_DWORD *)(v27 + 8);
          if ( !(_DWORD)v224 )
            goto LABEL_50;
          if ( (unsigned __int8)((unsigned __int64)sub_1E13E90(
                                                     (__int64)&v216,
                                                     (unsigned int *)&v224,
                                                     v15,
                                                     v17,
                                                     (int)v13) >> 32) )
          {
            v28 = (unsigned int)v214;
            if ( (unsigned int)v214 >= HIDWORD(v214) )
            {
              sub_16CD150((__int64)&v213, v215, 0, 4, (int)v13, v14);
              v28 = (unsigned int)v214;
            }
            v17 = (unsigned int)v224;
            *(_DWORD *)&v213[4 * v28] = (_DWORD)v224;
            LODWORD(v214) = v214 + 1;
            v30 = *(unsigned __int8 *)(v27 + 3);
            v29 = (unsigned __int8)v30 >> 4;
            LOBYTE(v30) = (unsigned __int8)v30 >> 6;
            if ( (v29 & 1 & (unsigned __int8)v30) == 0 )
            {
LABEL_38:
              if ( !v143 )
                BUG();
              v31 = (unsigned __int16 *)(*(_QWORD *)(v143 + 56)
                                       + 2LL * *(unsigned int *)(*(_QWORD *)(v143 + 8) + 24LL * (unsigned int)v224 + 4));
              v32 = *v31;
              v33 = v31 + 1;
              v34 = v32 + (unsigned __int16)v224;
              v35 = (_WORD)v32 == 0;
              v15 = 0;
              if ( !v35 )
                v15 = (__int64)v33;
LABEL_41:
              v36 = (unsigned __int16 *)v15;
              if ( !v15 )
                goto LABEL_50;
LABEL_42:
              v37 = (unsigned __int16)v34;
              if ( !v223 )
              {
                v38 = &v216[4 * (unsigned int)v217];
                if ( v216 != (_BYTE *)v38 )
                {
                  v39 = v216;
                  while ( (unsigned __int16)v34 != *v39 )
                  {
                    if ( v38 == ++v39 )
                      goto LABEL_121;
                  }
                  if ( v38 != v39 )
                    goto LABEL_48;
                }
LABEL_121:
                LODWORD(v13) = v220;
                if ( (unsigned int)v217 > 0x1FuLL )
                {
                  v153 = v34;
                  v70 = v220;
                  v146 = v36;
                  v71 = (unsigned int *)&v216[4 * (unsigned int)v217 - 4];
                  if ( v220 )
                  {
LABEL_123:
                    v72 = *v71;
                    for ( i = (int *)v70; ; i = v75 )
                    {
                      v74 = i[8];
                      v75 = (int *)*((_QWORD *)i + 3);
                      if ( v72 < v74 )
                        v75 = (int *)*((_QWORD *)i + 2);
                      if ( !v75 )
                        break;
                    }
                    if ( v72 < v74 )
                    {
                      if ( v221 != i )
                        goto LABEL_138;
                    }
                    else if ( v72 <= v74 )
                    {
                      goto LABEL_133;
                    }
LABEL_130:
                    v76 = 1;
                    if ( i != v16 )
                      v76 = *v71 < i[8];
                  }
                  else
                  {
                    while ( 1 )
                    {
                      i = v16;
                      if ( v221 == v16 )
                        break;
LABEL_138:
                      if ( *(_DWORD *)(sub_220EF80(i) + 32) < *v71 )
                        goto LABEL_130;
LABEL_133:
                      v35 = (_DWORD)v217 == 1;
                      v78 = v217 - 1;
                      LODWORD(v217) = v217 - 1;
                      if ( v35 )
                      {
                        v13 = (int *)v70;
                        v36 = v146;
                        v34 = v153;
                        if ( !v13 )
                        {
                          v13 = v16;
                          if ( v221 != v16 )
                            goto LABEL_192;
                          v81 = 1;
LABEL_149:
                          v154 = v13;
                          v82 = sub_22077B0(40);
                          *(_DWORD *)(v82 + 32) = v37;
                          sub_220F040(v81, v82, v154, v16);
                          ++v223;
                          v69 = (unsigned int)v214;
                          if ( (unsigned int)v214 >= HIDWORD(v214) )
                          {
LABEL_150:
                            sub_16CD150((__int64)&v213, v215, 0, 4, (int)v13, v14);
                            v69 = (unsigned int)v214;
                          }
LABEL_120:
                          *(_DWORD *)&v213[4 * v69] = v37;
                          LODWORD(v214) = v214 + 1;
LABEL_48:
                          v40 = *v36;
                          v15 = 0;
                          ++v36;
                          v17 = (unsigned int)(v40 + v34);
                          if ( !(_WORD)v40 )
                            goto LABEL_41;
                          v34 += v40;
                          if ( !v36 )
                            goto LABEL_50;
                          goto LABEL_42;
                        }
                        while ( 1 )
                        {
                          v79 = v13[8];
                          v80 = (int *)*((_QWORD *)v13 + 3);
                          if ( v37 < v79 )
                            v80 = (int *)*((_QWORD *)v13 + 2);
                          if ( !v80 )
                            break;
                          v13 = v80;
                        }
                        if ( v37 < v79 )
                        {
                          if ( v221 == v13 )
                            goto LABEL_148;
LABEL_192:
                          v155 = v13;
                          if ( v37 > *(_DWORD *)(sub_220EF80(v13) + 32) )
                          {
                            v13 = v155;
                            if ( v155 )
                            {
                              v81 = 1;
                              if ( v155 == v16 )
                                goto LABEL_149;
                              goto LABEL_195;
                            }
                          }
                        }
                        else if ( v37 > v79 )
                        {
LABEL_148:
                          v81 = 1;
                          if ( v13 == v16 )
                            goto LABEL_149;
LABEL_195:
                          v81 = v37 < v13[8];
                          goto LABEL_149;
                        }
LABEL_119:
                        v69 = (unsigned int)v214;
                        if ( (unsigned int)v214 >= HIDWORD(v214) )
                          goto LABEL_150;
                        goto LABEL_120;
                      }
                      v71 = (unsigned int *)&v216[4 * v78 - 4];
                      if ( v70 )
                        goto LABEL_123;
                    }
                    v76 = 1;
                  }
                  v77 = sub_22077B0(40);
                  *(_DWORD *)(v77 + 32) = *v71;
                  sub_220F040(v76, v77, i, v16);
                  ++v223;
                  v70 = v220;
                  goto LABEL_133;
                }
                if ( (unsigned int)v217 >= HIDWORD(v217) )
                {
                  sub_16CD150((__int64)&v216, v218, 0, 4, v220, v14);
                  v38 = &v216[4 * (unsigned int)v217];
                }
                *v38 = (unsigned __int16)v34;
                LODWORD(v217) = v217 + 1;
                goto LABEL_119;
              }
              v64 = v220;
              if ( !v220 )
              {
                v64 = (__int64)v16;
                if ( v221 == v16 )
                {
                  v67 = 1;
                  goto LABEL_118;
                }
LABEL_152:
                if ( (unsigned int)(unsigned __int16)v34 <= *(_DWORD *)(sub_220EF80(v64) + 32) || !v64 )
                  goto LABEL_48;
                v67 = 1;
                if ( (int *)v64 == v16 )
                  goto LABEL_118;
                goto LABEL_155;
              }
              while ( 1 )
              {
                v65 = *(_DWORD *)(v64 + 32);
                v66 = *(_QWORD *)(v64 + 24);
                if ( (unsigned __int16)v34 < v65 )
                  v66 = *(_QWORD *)(v64 + 16);
                if ( !v66 )
                  break;
                v64 = v66;
              }
              if ( (unsigned __int16)v34 < v65 )
              {
                if ( v221 != (int *)v64 )
                  goto LABEL_152;
              }
              else if ( (unsigned __int16)v34 <= v65 )
              {
                goto LABEL_48;
              }
              v67 = 1;
              if ( (int *)v64 != v16 )
LABEL_155:
                v67 = (unsigned int)(unsigned __int16)v34 < *(_DWORD *)(v64 + 32);
LABEL_118:
              v152 = v67;
              v68 = sub_22077B0(40);
              *(_DWORD *)(v68 + 32) = (unsigned __int16)v34;
              sub_220F040(v152, v68, v64, v16);
              ++v223;
              goto LABEL_119;
            }
            sub_1DCB780((__int64)&v172, (unsigned int *)&v224, v30, v17, (int)v13);
          }
          else
          {
            if ( v212 )
            {
              sub_1E13070(&v207, (unsigned int *)&v224);
            }
            else
            {
              v83 = (char *)dest;
              v84 = (char *)dest + 4 * (unsigned int)v205;
              v14 = v205;
              if ( dest != v84 )
              {
                v17 = (unsigned int)v224;
                while ( 1 )
                {
                  v85 = *(_DWORD *)v83;
                  v86 = v83;
                  v83 += 4;
                  if ( v85 == (_DWORD)v224 )
                    break;
                  if ( v84 == v83 )
                    goto LABEL_164;
                }
                if ( v84 != v83 )
                {
                  memmove(v86, v83, v84 - v83);
                  v14 = v205;
                }
                LODWORD(v205) = --v14;
              }
            }
LABEL_164:
            v15 = *(unsigned __int8 *)(v27 + 3);
            v87 = (unsigned __int8)v15 >> 4;
            LOBYTE(v15) = (unsigned __int8)v15 >> 6;
            if ( (v87 & 1 & (unsigned __int8)v15) != 0 )
              goto LABEL_50;
            if ( v180 )
            {
              sub_1E13070(&v175, (unsigned int *)&v224);
            }
            else
            {
              v88 = (char *)v172;
              v89 = (char *)v172 + 4 * (unsigned int)v173;
              LODWORD(v13) = v173;
              if ( v172 == v89 )
                goto LABEL_38;
              v17 = (unsigned int)v224;
              while ( 1 )
              {
                v90 = *(_DWORD *)v88;
                v91 = v88;
                v88 += 4;
                if ( v90 == (_DWORD)v224 )
                  break;
                if ( v89 == v88 )
                  goto LABEL_38;
              }
              if ( v89 != v88 )
              {
                memmove(v91, v88, v89 - v88);
                LODWORD(v13) = v173;
              }
              LODWORD(v13) = (_DWORD)v13 - 1;
              LODWORD(v173) = (_DWORD)v13;
            }
          }
          v15 = *(unsigned __int8 *)(v27 + 3);
          v92 = (unsigned __int8)v15 >> 4;
          LOBYTE(v15) = (unsigned __int8)v15 >> 6;
          if ( (v92 & 1 & (unsigned __int8)v15) == 0 )
            goto LABEL_38;
LABEL_50:
          v158 += 8;
        }
        while ( v156 != v158 );
      }
      LODWORD(v170) = 0;
      v144 = *(_QWORD *)(v144 + 8);
    }
    while ( a3 != (_BYTE *)v144 );
    v41 = &v227;
    v224 = v226;
    v225 = 0x2000000000LL;
    v227 = 0;
    v228 = 0;
    v229 = &v227;
    v230 = &v227;
    v231 = 0;
    if ( (_DWORD)v214 )
    {
      v42 = 4LL * (unsigned int)v214;
      v43 = 0;
      while ( 1 )
      {
        v160 = *(_DWORD *)&v213[v43];
        v44 = (unsigned __int64)sub_1E13E90((__int64)&v224, &v160, (__int64)v41, v17, (int)v13) >> 32;
        if ( v44 )
          break;
LABEL_62:
        v43 += 4;
        if ( v42 == v43 )
          goto LABEL_63;
      }
      if ( v180 )
      {
        v120 = v177;
        if ( v177 )
        {
          v121 = &v176;
          do
          {
            if ( *(_DWORD *)(v120 + 32) < v160 )
            {
              v120 = *(_QWORD *)(v120 + 24);
            }
            else
            {
              v121 = (int *)v120;
              v120 = *(_QWORD *)(v120 + 16);
            }
          }
          while ( v120 );
          if ( v121 != &v176 && v121[8] <= v160 )
            goto LABEL_61;
        }
      }
      else
      {
        v45 = (char *)v172;
        v46 = (char *)v172 + 4 * (unsigned int)v173;
        if ( v172 != v46 )
        {
          while ( *(_DWORD *)v45 != v160 )
          {
            v45 += 4;
            if ( v46 == v45 )
              goto LABEL_244;
          }
          if ( v46 != v45 )
            goto LABEL_61;
        }
      }
LABEL_244:
      if ( v212 )
      {
        v126 = (__int64)v209;
        if ( v209 )
        {
          v127 = &v208;
          do
          {
            while ( 1 )
            {
              v128 = *(_QWORD *)(v126 + 16);
              v129 = *(_QWORD *)(v126 + 24);
              if ( *(_DWORD *)(v126 + 32) >= v160 )
                break;
              v126 = *(_QWORD *)(v126 + 24);
              if ( !v129 )
                goto LABEL_272;
            }
            v127 = (int *)v126;
            v126 = *(_QWORD *)(v126 + 16);
          }
          while ( v128 );
LABEL_272:
          v44 = 0;
          if ( v127 != &v208 )
            v44 = v127[8] <= v160;
          goto LABEL_61;
        }
      }
      else
      {
        v118 = (char *)dest;
        v119 = (char *)dest + 4 * (unsigned int)v205;
        if ( dest != v119 )
        {
          while ( *(_DWORD *)v118 != v160 )
          {
            v118 += 4;
            if ( v119 == v118 )
              goto LABEL_293;
          }
          v44 = v118 != v119;
          goto LABEL_61;
        }
      }
LABEL_293:
      v44 = 0;
LABEL_61:
      v162 = v160;
      v161 = 805306368;
      v163 = 0;
      *(_DWORD *)((char *)&v161 + 3) = ((v44 & 1) << 6) | 0x30;
      v164 = 0;
      *(_DWORD *)((char *)&v161 + 2) = WORD1(v161) & 0xF00F;
      v165 = 0;
      LODWORD(v161) = v161 & 0xFFF000FF;
      sub_1E1A9C0(v139, v140, &v161);
      goto LABEL_62;
    }
  }
LABEL_63:
  if ( (_DWORD)v167 )
  {
    v159 = 4LL * (unsigned int)v167;
    v47 = 0;
    do
    {
      v48 = v195;
      v49 = *(_DWORD *)&v166[v47];
      if ( v195 )
      {
        v112 = v192;
        if ( v192 )
        {
          v113 = &v191;
          do
          {
            while ( 1 )
            {
              v114 = *(_QWORD *)(v112 + 16);
              v115 = *(_QWORD *)(v112 + 24);
              if ( v49 <= *(_DWORD *)(v112 + 32) )
                break;
              v112 = *(_QWORD *)(v112 + 24);
              if ( !v115 )
                goto LABEL_234;
            }
            v113 = (int *)v112;
            v112 = *(_QWORD *)(v112 + 16);
          }
          while ( v114 );
LABEL_234:
          v48 = 0;
          if ( v113 != &v191 )
            v48 = v49 >= v113[8];
        }
        else
        {
          v48 = 0;
        }
      }
      else
      {
        v50 = v188;
        v51 = &v188[4 * (unsigned int)v189];
        if ( v188 != (_BYTE *)v51 )
        {
          while ( v49 != *v50 )
          {
            if ( v51 == ++v50 )
              goto LABEL_71;
          }
          v48 = v51 != v50;
        }
      }
LABEL_71:
      v52 = v203;
      if ( v203 )
      {
        v108 = v200;
        if ( v200 )
        {
          v109 = &v199;
          do
          {
            while ( 1 )
            {
              v110 = *(_QWORD *)(v108 + 16);
              v111 = *(_QWORD *)(v108 + 24);
              if ( v49 <= *(_DWORD *)(v108 + 32) )
                break;
              v108 = *(_QWORD *)(v108 + 24);
              if ( !v111 )
                goto LABEL_227;
            }
            v109 = (int *)v108;
            v108 = *(_QWORD *)(v108 + 16);
          }
          while ( v110 );
LABEL_227:
          v52 = 0;
          if ( v109 != &v199 )
            v52 = v49 >= v109[8];
        }
        else
        {
          v52 = 0;
        }
      }
      else
      {
        v53 = v196;
        v54 = &v196[4 * (unsigned int)v197];
        if ( v196 != (_BYTE *)v54 )
        {
          while ( v49 != *v53 )
          {
            if ( v54 == ++v53 )
              goto LABEL_77;
          }
          v52 = v54 != v53;
        }
      }
LABEL_77:
      v161 = 0x20000000;
      v162 = v49;
      v163 = 0;
      v164 = 0;
      v55 = (8 * v48) | 0x20;
      v165 = 0;
      if ( !v52 )
        v55 = 8 * v48;
      v47 += 4;
      BYTE4(v161) = v55 >> 5;
      BYTE3(v161) = ((8 * v48) << 6) | 0x20;
      WORD1(v161) &= 0xF00Fu;
      LODWORD(v161) = v161 & 0xFFF000FF;
      sub_1E1A9C0(v139, v140, &v161);
    }
    while ( v159 != v47 );
  }
  sub_1E12EA0(v228);
  if ( v224 != v226 )
    _libc_free((unsigned __int64)v224);
  if ( v169 != v171 )
    _libc_free((unsigned __int64)v169);
  v56 = v200;
  while ( v56 )
  {
    sub_1E12EA0(*(_QWORD *)(v56 + 24));
    v57 = v56;
    v56 = *(_QWORD *)(v56 + 16);
    j_j___libc_free_0(v57, 40);
  }
  if ( v196 != v198 )
    _libc_free((unsigned __int64)v196);
  v58 = v192;
  while ( v58 )
  {
    sub_1E12EA0(*(_QWORD *)(v58 + 24));
    v59 = v58;
    v58 = *(_QWORD *)(v58 + 16);
    j_j___libc_free_0(v59, 40);
  }
  if ( v188 != v190 )
    _libc_free((unsigned __int64)v188);
  v60 = v184;
  while ( v60 )
  {
    sub_1E12EA0(*(_QWORD *)(v60 + 24));
    v61 = v60;
    v60 = *(_QWORD *)(v60 + 16);
    j_j___libc_free_0(v61, 40);
  }
  if ( (_BYTE *)v181[0] != v182 )
    _libc_free(v181[0]);
  if ( v166 != v168 )
    _libc_free((unsigned __int64)v166);
  sub_1E12EA0((__int64)v209);
  if ( dest != v206 )
    _libc_free((unsigned __int64)dest);
  v62 = v177;
  while ( v62 )
  {
    sub_1E12EA0(*(_QWORD *)(v62 + 24));
    v63 = v62;
    v62 = *(_QWORD *)(v62 + 16);
    j_j___libc_free_0(v63, 40);
  }
  if ( v172 != v174 )
    _libc_free((unsigned __int64)v172);
  sub_1E12EA0(v220);
  if ( v216 != v218 )
    _libc_free((unsigned __int64)v216);
  if ( v213 != v215 )
    _libc_free((unsigned __int64)v213);
}
