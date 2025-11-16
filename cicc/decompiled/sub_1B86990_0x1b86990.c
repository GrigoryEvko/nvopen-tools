// Function: sub_1B86990
// Address: 0x1b86990
//
char __fastcall sub_1B86990(__int64 a1, __int64 *a2, __int64 a3, __int64 *a4, int a5, __m128i a6, __m128i a7)
{
  __int64 v8; // r12
  unsigned int v9; // eax
  unsigned int v10; // r15d
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rsi
  int v14; // ebx
  __int64 v15; // rax
  __int64 v16; // r8
  __int64 v17; // rsi
  unsigned int v18; // ecx
  __int64 v19; // rbx
  __int64 v20; // rax
  __int64 v21; // r8
  char result; // al
  _QWORD *v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  int v26; // eax
  __int64 v27; // rax
  __int64 v28; // rax
  int v29; // eax
  _QWORD *v30; // rax
  unsigned int v31; // ecx
  unsigned int v32; // ecx
  bool v33; // zf
  __int64 v34; // r15
  __int64 v35; // rax
  __int64 v36; // r14
  _QWORD *v37; // rdx
  _QWORD *v38; // r12
  unsigned __int64 v39; // r13
  _QWORD *v40; // rbx
  __int64 v41; // rax
  __int64 v42; // rdi
  __int64 *v43; // r13
  __int64 v44; // r13
  unsigned __int8 v45; // al
  __int64 v46; // r13
  unsigned __int8 v47; // dl
  __int64 *v48; // rdi
  __int64 v49; // rdx
  __int64 v50; // rax
  __int64 *v51; // rsi
  __int64 v52; // rsi
  _QWORD *v53; // rax
  __int64 v54; // r14
  __int64 *v55; // r14
  __int64 v56; // r15
  __int64 v57; // rdx
  __int64 v58; // r8
  int v59; // eax
  __int64 v60; // r12
  __int64 v61; // r15
  __int64 *v62; // rbx
  __int64 v63; // r13
  __int64 *v64; // r14
  unsigned __int64 v65; // rdi
  unsigned __int64 v66; // rax
  char v67; // si
  unsigned __int64 v68; // rdi
  unsigned __int64 v69; // rax
  char v70; // dl
  unsigned __int64 v71; // rcx
  _QWORD *v72; // r13
  __int64 v73; // r13
  _QWORD *v74; // r13
  unsigned __int64 v75; // r12
  _QWORD *v76; // rbx
  _QWORD *v77; // r9
  unsigned __int64 v78; // rcx
  __int64 *v79; // r8
  unsigned __int8 v80; // al
  __int64 *v81; // r9
  unsigned __int8 v82; // cl
  unsigned int v83; // edx
  __int64 v84; // rcx
  __int64 v85; // r10
  unsigned __int64 v86; // rdi
  unsigned __int64 v87; // rsi
  unsigned __int64 v88; // r14
  unsigned __int64 v89; // r14
  __int64 *v90; // r8
  __int64 *v91; // r9
  __int64 **v92; // r8
  __int64 *v93; // r9
  __int64 v94; // r13
  unsigned __int8 v95; // dl
  __int64 *v96; // r14
  __int64 v97; // r8
  __int64 v98; // rax
  _QWORD *v99; // r9
  __int64 v100; // r10
  bool v101; // cc
  unsigned __int64 v102; // rax
  __int64 v103; // rax
  _QWORD *v104; // rdx
  _QWORD *v105; // rax
  _QWORD *v106; // rdx
  __int64 v107; // rdi
  _QWORD *v108; // rcx
  __int64 v109; // rax
  _QWORD *v110; // rdx
  int v111; // eax
  __int64 v112; // rdi
  unsigned __int64 v113; // rax
  int v114; // eax
  int v115; // eax
  unsigned __int64 v116; // rax
  unsigned int v117; // eax
  __int64 v118; // rax
  int v119; // eax
  int v120; // eax
  int v121; // eax
  __int64 v122; // rax
  __int64 v123; // r14
  __int64 *v124; // r13
  __int64 v125; // r14
  __int64 v126; // rdi
  __int64 *v127; // rbx
  __int64 v128; // rdx
  __int64 v129; // rax
  unsigned int v130; // edx
  __int64 *v131; // rax
  __int64 v132; // rsi
  bool v133; // al
  bool v134; // dl
  __int64 v135; // [rsp-8h] [rbp-138h]
  __int64 v136; // [rsp+8h] [rbp-128h]
  __int64 v137; // [rsp+10h] [rbp-120h]
  __int64 v138; // [rsp+10h] [rbp-120h]
  _QWORD *v139; // [rsp+10h] [rbp-120h]
  unsigned int v140; // [rsp+18h] [rbp-118h]
  _QWORD *v141; // [rsp+18h] [rbp-118h]
  _QWORD *v142; // [rsp+18h] [rbp-118h]
  __int64 v143; // [rsp+18h] [rbp-118h]
  __int64 v144; // [rsp+18h] [rbp-118h]
  __int64 v145; // [rsp+20h] [rbp-110h]
  unsigned __int64 v146; // [rsp+20h] [rbp-110h]
  __int64 v147; // [rsp+20h] [rbp-110h]
  __int64 v148; // [rsp+20h] [rbp-110h]
  _QWORD *v149; // [rsp+20h] [rbp-110h]
  __int64 v150; // [rsp+20h] [rbp-110h]
  __int64 v151; // [rsp+28h] [rbp-108h]
  __int64 v152; // [rsp+28h] [rbp-108h]
  __int64 v153; // [rsp+28h] [rbp-108h]
  unsigned __int64 *v154; // [rsp+28h] [rbp-108h]
  __int64 v155; // [rsp+28h] [rbp-108h]
  __int64 v156; // [rsp+28h] [rbp-108h]
  __int64 v157; // [rsp+30h] [rbp-100h]
  __int64 *v158; // [rsp+30h] [rbp-100h]
  _QWORD *v159; // [rsp+30h] [rbp-100h]
  _QWORD *v160; // [rsp+30h] [rbp-100h]
  unsigned __int64 v161; // [rsp+38h] [rbp-F8h]
  unsigned int v162; // [rsp+38h] [rbp-F8h]
  unsigned __int64 v163; // [rsp+38h] [rbp-F8h]
  unsigned int v164; // [rsp+38h] [rbp-F8h]
  unsigned int v165; // [rsp+38h] [rbp-F8h]
  unsigned __int64 *v166; // [rsp+38h] [rbp-F8h]
  __int64 v167; // [rsp+38h] [rbp-F8h]
  __int64 v168; // [rsp+40h] [rbp-F0h]
  __int64 v169; // [rsp+40h] [rbp-F0h]
  __int64 v170; // [rsp+40h] [rbp-F0h]
  __int64 v171; // [rsp+40h] [rbp-F0h]
  __int64 v172; // [rsp+40h] [rbp-F0h]
  unsigned int v173; // [rsp+40h] [rbp-F0h]
  unsigned int v174; // [rsp+40h] [rbp-F0h]
  _QWORD *v175; // [rsp+40h] [rbp-F0h]
  __int64 *v176; // [rsp+40h] [rbp-F0h]
  __int64 v177; // [rsp+40h] [rbp-F0h]
  _QWORD *v178; // [rsp+40h] [rbp-F0h]
  _QWORD *v179; // [rsp+40h] [rbp-F0h]
  __int64 v181; // [rsp+48h] [rbp-E8h]
  __int64 *v182; // [rsp+48h] [rbp-E8h]
  __int64 *v183; // [rsp+48h] [rbp-E8h]
  __int64 *v184; // [rsp+48h] [rbp-E8h]
  bool v185; // [rsp+48h] [rbp-E8h]
  __int64 v186; // [rsp+50h] [rbp-E0h]
  __int64 *v187; // [rsp+50h] [rbp-E0h]
  char v188; // [rsp+50h] [rbp-E0h]
  unsigned int v189; // [rsp+50h] [rbp-E0h]
  __int64 *v190; // [rsp+50h] [rbp-E0h]
  __int64 v191; // [rsp+50h] [rbp-E0h]
  __int64 v192; // [rsp+50h] [rbp-E0h]
  __int64 v194; // [rsp+58h] [rbp-D8h]
  char v195; // [rsp+58h] [rbp-D8h]
  char v196; // [rsp+58h] [rbp-D8h]
  char v197; // [rsp+58h] [rbp-D8h]
  char v198; // [rsp+58h] [rbp-D8h]
  __int64 v199; // [rsp+58h] [rbp-D8h]
  char v200; // [rsp+58h] [rbp-D8h]
  char v201; // [rsp+58h] [rbp-D8h]
  __int64 v202; // [rsp+58h] [rbp-D8h]
  unsigned int v203; // [rsp+58h] [rbp-D8h]
  __int64 *v204; // [rsp+58h] [rbp-D8h]
  __int64 *v205; // [rsp+58h] [rbp-D8h]
  __int64 *v206; // [rsp+58h] [rbp-D8h]
  unsigned int v207; // [rsp+58h] [rbp-D8h]
  char v208; // [rsp+58h] [rbp-D8h]
  __int64 v209; // [rsp+58h] [rbp-D8h]
  __int64 *v210; // [rsp+60h] [rbp-D0h] BYREF
  unsigned int v211; // [rsp+68h] [rbp-C8h]
  __int64 *v212; // [rsp+70h] [rbp-C0h] BYREF
  unsigned int v213; // [rsp+78h] [rbp-B8h]
  __int64 *v214; // [rsp+80h] [rbp-B0h] BYREF
  unsigned int v215; // [rsp+88h] [rbp-A8h]
  __int64 *v216; // [rsp+90h] [rbp-A0h] BYREF
  unsigned int v217; // [rsp+98h] [rbp-98h]
  unsigned __int64 v218; // [rsp+A0h] [rbp-90h] BYREF
  unsigned int v219; // [rsp+A8h] [rbp-88h]
  __int64 *v220; // [rsp+B0h] [rbp-80h] BYREF
  unsigned int v221; // [rsp+B8h] [rbp-78h]
  unsigned __int64 v222; // [rsp+C0h] [rbp-70h] BYREF
  __int64 *v223; // [rsp+C8h] [rbp-68h] BYREF
  unsigned int v224; // [rsp+D0h] [rbp-60h]
  __int64 *v225; // [rsp+E0h] [rbp-50h] BYREF
  __int64 v226; // [rsp+E8h] [rbp-48h]
  __int64 v227; // [rsp+F0h] [rbp-40h] BYREF
  __int64 v228; // [rsp+F8h] [rbp-38h]

  v8 = a1;
  v9 = sub_15A9570(*(_QWORD *)(a1 + 40), *a2);
  v211 = v9;
  v10 = v9;
  if ( v9 > 0x40 )
  {
    sub_16A4EF0((__int64)&v210, 0, 0);
    v213 = v10;
    sub_16A4EF0((__int64)&v212, 0, 0);
  }
  else
  {
    v213 = v9;
    v210 = 0;
    v212 = 0;
  }
  v186 = sub_164A410((__int64)a2, *(_QWORD *)(a1 + 40), (__int64)&v210);
  v11 = sub_164A410(a3, *(_QWORD *)(a1 + 40), (__int64)&v212);
  v12 = *(_QWORD *)(a1 + 40);
  v13 = *(_QWORD *)v186;
  v194 = v11;
  v14 = 1;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v13 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v25 = *(_QWORD *)(v13 + 32);
        v13 = *(_QWORD *)(v13 + 24);
        v14 *= (_DWORD)v25;
        continue;
      case 1:
        LODWORD(v15) = 16;
        break;
      case 2:
        LODWORD(v15) = 32;
        break;
      case 3:
      case 9:
        LODWORD(v15) = 64;
        break;
      case 4:
        LODWORD(v15) = 80;
        break;
      case 5:
      case 6:
        LODWORD(v15) = 128;
        break;
      case 7:
        LODWORD(v15) = sub_15A9520(v12, 0);
        v12 = *(_QWORD *)(v8 + 40);
        LODWORD(v15) = 8 * v15;
        break;
      case 0xB:
        LODWORD(v15) = *(_DWORD *)(v13 + 8) >> 8;
        break;
      case 0xD:
        v23 = (_QWORD *)sub_15A9930(v12, v13);
        v12 = *(_QWORD *)(v8 + 40);
        v15 = 8LL * *v23;
        break;
      case 0xE:
        v151 = *(_QWORD *)(v13 + 24);
        v168 = *(_QWORD *)(v13 + 32);
        v161 = (unsigned int)sub_15A9FE0(v12, v151);
        v24 = sub_127FA20(v12, v151);
        v12 = *(_QWORD *)(v8 + 40);
        v15 = 8 * v161 * v168 * ((v161 + ((unsigned __int64)(v24 + 7) >> 3) - 1) / v161);
        break;
      case 0xF:
        LODWORD(v15) = sub_15A9520(v12, *(_DWORD *)(v13 + 8) >> 8);
        v12 = *(_QWORD *)(v8 + 40);
        LODWORD(v15) = 8 * v15;
        break;
    }
    break;
  }
  v16 = 1;
  v17 = *(_QWORD *)v194;
  v18 = (v15 * v14 + 7) & 0xFFFFFFF8;
  v19 = v18;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v17 + 8) )
    {
      case 1:
        v20 = 16;
        goto LABEL_12;
      case 2:
        v20 = 32;
        goto LABEL_12;
      case 3:
      case 9:
        v20 = 64;
        goto LABEL_12;
      case 4:
        v20 = 80;
        goto LABEL_12;
      case 5:
      case 6:
        v20 = 128;
        goto LABEL_12;
      case 7:
        v164 = v18;
        v171 = v16;
        v29 = sub_15A9520(v12, 0);
        v16 = v171;
        v18 = v164;
        v20 = (unsigned int)(8 * v29);
        goto LABEL_12;
      case 0xB:
        v20 = *(_DWORD *)(v17 + 8) >> 8;
        goto LABEL_12;
      case 0xD:
        v165 = v18;
        v172 = v16;
        v30 = (_QWORD *)sub_15A9930(v12, v17);
        v16 = v172;
        v18 = v165;
        v20 = 8LL * *v30;
        goto LABEL_12;
      case 0xE:
        v140 = v18;
        v145 = v16;
        v152 = *(_QWORD *)(v17 + 24);
        v170 = *(_QWORD *)(v17 + 32);
        v163 = (unsigned int)sub_15A9FE0(v12, v152);
        v27 = sub_127FA20(v12, v152);
        v16 = v145;
        v18 = v140;
        v20 = 8 * v170 * v163 * ((v163 + ((unsigned __int64)(v27 + 7) >> 3) - 1) / v163);
        goto LABEL_12;
      case 0xF:
        v162 = v18;
        v169 = v16;
        v26 = sub_15A9520(v12, *(_DWORD *)(v17 + 8) >> 8);
        v16 = v169;
        v18 = v162;
        v20 = (unsigned int)(8 * v26);
LABEL_12:
        v21 = v20 * v16;
        result = 0;
        if ( ((v21 + 7) & 0xFFFFFFFFFFFFFFF8LL) != v19 )
          goto LABEL_13;
        v173 = v18;
        sub_16A5D70((__int64)&v225, (__int64 *)&v210, v18);
        v31 = v173;
        if ( v211 > 0x40 && v210 )
        {
          j_j___libc_free_0_0(v210);
          v31 = v173;
        }
        v174 = v31;
        v210 = v225;
        v211 = v226;
        sub_16A5D70((__int64)&v225, (__int64 *)&v212, v31);
        v32 = v174;
        if ( v213 > 0x40 && v212 )
        {
          j_j___libc_free_0_0(v212);
          v32 = v174;
        }
        v212 = v225;
        v213 = v226;
        sub_16A5D70((__int64)&v225, a4, v32);
        if ( *((_DWORD *)a4 + 2) > 0x40u && *a4 )
          j_j___libc_free_0_0(*a4);
        *a4 = (__int64)v225;
        *((_DWORD *)a4 + 2) = v226;
        LODWORD(v226) = v213;
        if ( v213 > 0x40 )
          sub_16A4FD0((__int64)&v225, (const void **)&v212);
        else
          v225 = v212;
        sub_16A7590((__int64)&v225, (__int64 *)&v210);
        v215 = v226;
        v214 = v225;
        if ( v194 == v186 )
        {
          if ( (unsigned int)v226 <= 0x40 )
          {
            result = *a4 == (_QWORD)v225;
            goto LABEL_13;
          }
          result = sub_16A5220((__int64)&v214, (const void **)a4);
          goto LABEL_65;
        }
        LODWORD(v226) = *((_DWORD *)a4 + 2);
        if ( (unsigned int)v226 > 0x40 )
          sub_16A4FD0((__int64)&v225, (const void **)a4);
        else
          v225 = (__int64 *)*a4;
        sub_16A7590((__int64)&v225, (__int64 *)&v214);
        v33 = *(_BYTE *)(v8 + 124) == 0;
        v217 = v226;
        v216 = v225;
        if ( !v33 )
        {
          v220 = (__int64 *)v8;
          v34 = sub_1B86660((__int64 *)&v220, v186);
          v35 = sub_1B86660((__int64 *)&v220, v194);
          v222 = v34;
          v36 = v35;
          v224 = v217;
          if ( v217 > 0x40 )
            sub_16A4FD0((__int64)&v223, (const void **)&v216);
          else
            v223 = v216;
          v37 = *(_QWORD **)(v8 + 176);
          v175 = (_QWORD *)(v8 + 168);
          if ( v37 )
          {
            v157 = v8;
            v38 = (_QWORD *)(v8 + 168);
            v39 = v222;
            v40 = v37;
            do
            {
              if ( v40[4] < v39 || v40[4] == v39 && (int)sub_16A9900((__int64)(v40 + 5), (unsigned __int64 *)&v223) < 0 )
              {
                v40 = (_QWORD *)v40[3];
              }
              else
              {
                v38 = v40;
                v40 = (_QWORD *)v40[2];
              }
            }
            while ( v40 );
            v71 = v39;
            v72 = v38;
            v8 = v157;
            if ( v72 != v175 && v71 >= v72[4] && (v71 != v72[4] || (int)sub_16A9900((__int64)&v223, v72 + 5) >= 0) )
            {
              v43 = (__int64 *)v72[7];
LABEL_157:
              if ( v224 > 0x40 && v223 )
                j_j___libc_free_0_0(v223);
              goto LABEL_70;
            }
          }
          v73 = *(_QWORD *)(v8 + 24);
          v228 = sub_145CF40(v73, (__int64)&v216);
          v225 = &v227;
          v227 = v34;
          v226 = 0x200000002LL;
          v43 = sub_147DD40(v73, (__int64 *)&v225, 0, 0, a6, a7);
          if ( v225 != &v227 )
            _libc_free((unsigned __int64)v225);
          if ( *(_QWORD *)(v8 + 176) )
          {
            v158 = v43;
            v74 = *(_QWORD **)(v8 + 176);
            v153 = v8;
            v75 = v222;
            v76 = v175;
            do
            {
              if ( v74[4] < v75 || v74[4] == v75 && (int)sub_16A9900((__int64)(v74 + 5), (unsigned __int64 *)&v223) < 0 )
              {
                v74 = (_QWORD *)v74[3];
              }
              else
              {
                v76 = v74;
                v74 = (_QWORD *)v74[2];
              }
            }
            while ( v74 );
            v77 = v76;
            v78 = v75;
            v43 = v158;
            v8 = v153;
            if ( v175 != v76 && v78 >= v76[4] )
            {
              if ( v78 != v76[4] )
                goto LABEL_165;
              v115 = sub_16A9900((__int64)&v223, v76 + 5);
              v77 = v76;
              if ( v115 >= 0 )
                goto LABEL_165;
            }
          }
          else
          {
            v77 = v175;
          }
          v159 = v77;
          v98 = sub_22077B0(64);
          v99 = v159;
          v100 = v98;
          *(_QWORD *)(v98 + 32) = v222;
          v166 = (unsigned __int64 *)(v98 + 40);
          v101 = v224 <= 0x40;
          *(_DWORD *)(v98 + 48) = v224;
          if ( v101 )
          {
            *(_QWORD *)(v98 + 40) = v223;
          }
          else
          {
            v155 = v98;
            sub_16A4FD0((__int64)v166, (const void **)&v223);
            v100 = v155;
            v99 = v159;
          }
          *(_QWORD *)(v100 + 56) = 0;
          v154 = (unsigned __int64 *)(v100 + 32);
          if ( v175 == v99 )
          {
            if ( *(_QWORD *)(v8 + 200) )
            {
              v99 = *(_QWORD **)(v8 + 192);
              v113 = *(_QWORD *)(v100 + 32);
              if ( v99[4] < v113 )
                goto LABEL_224;
              if ( v99[4] == v113 )
              {
                v143 = v100;
                v149 = *(_QWORD **)(v8 + 192);
                v114 = sub_16A9900((__int64)(v99 + 5), v166);
                v99 = v149;
                v100 = v143;
                if ( v114 < 0 )
                  goto LABEL_224;
              }
            }
          }
          else
          {
            v102 = *(_QWORD *)(v100 + 32);
            v146 = v102;
            if ( v102 >= v99[4] )
            {
              if ( v102 != v99[4] )
              {
                if ( v102 <= v99[4] )
                  goto LABEL_216;
                goto LABEL_200;
              }
              v136 = v100;
              v139 = v99;
              v144 = (__int64)(v99 + 5);
              v119 = sub_16A9900((__int64)v166, v99 + 5);
              v99 = v139;
              v100 = v136;
              if ( v119 >= 0 )
              {
                v120 = sub_16A9900(v144, v166);
                v99 = v139;
                v100 = v136;
                if ( v120 >= 0 )
                  goto LABEL_216;
LABEL_200:
                if ( *(_QWORD **)(v8 + 192) != v99 )
                {
                  v137 = v100;
                  v141 = v99;
                  v103 = sub_220EEE0(v99);
                  v99 = v141;
                  v100 = v137;
                  v104 = (_QWORD *)v103;
                  if ( v146 < *(_QWORD *)(v103 + 32)
                    || v146 == *(_QWORD *)(v103 + 32)
                    && (v150 = v103,
                        v121 = sub_16A9900((__int64)v166, (unsigned __int64 *)(v103 + 40)),
                        v104 = (_QWORD *)v150,
                        v99 = v141,
                        v100 = v137,
                        v121 < 0) )
                  {
                    if ( !v99[3] )
                    {
LABEL_205:
                      if ( v99 != v175 )
                      {
                        v116 = v99[4];
                        v107 = 1;
                        if ( *(_QWORD *)(v100 + 32) >= v116 )
                        {
                          v107 = 0;
                          if ( *(_QWORD *)(v100 + 32) == v116 )
                          {
                            v156 = v100;
                            v160 = v99;
                            v117 = sub_16A9900((__int64)v166, v99 + 5);
                            v99 = v160;
                            v100 = v156;
                            v107 = v117 >> 31;
                          }
                        }
                        goto LABEL_207;
                      }
LABEL_206:
                      v107 = 1;
LABEL_207:
                      v108 = v175;
                      v177 = v100;
                      sub_220F040(v107, v100, v99, v108);
                      ++*(_QWORD *)(v8 + 200);
                      v77 = (_QWORD *)v177;
LABEL_165:
                      v77[7] = v43;
                      goto LABEL_157;
                    }
                    v99 = v104;
                    v105 = v104;
LABEL_204:
                    if ( v105 )
                      goto LABEL_206;
                    goto LABEL_205;
                  }
                  goto LABEL_203;
                }
LABEL_224:
                v105 = 0;
                goto LABEL_204;
              }
            }
            if ( *(_QWORD **)(v8 + 184) == v99 )
            {
LABEL_213:
              v105 = v99;
              goto LABEL_204;
            }
            v138 = v100;
            v142 = v99;
            v109 = sub_220EF80(v99);
            v99 = v142;
            v100 = v138;
            v110 = (_QWORD *)v109;
            if ( v146 > *(_QWORD *)(v109 + 32)
              || v146 == *(_QWORD *)(v109 + 32)
              && (v148 = v109,
                  v111 = sub_16A9900(v109 + 40, v166),
                  v110 = (_QWORD *)v148,
                  v99 = v142,
                  v100 = v138,
                  v111 < 0) )
            {
              v105 = (_QWORD *)v110[3];
              if ( !v105 )
              {
                v99 = v110;
                goto LABEL_204;
              }
              goto LABEL_213;
            }
          }
LABEL_203:
          v147 = v100;
          v105 = sub_1B7DB70(v8 + 160, v154);
          v100 = v147;
          v99 = v106;
          if ( v106 )
            goto LABEL_204;
          v99 = v105;
LABEL_216:
          if ( *(_DWORD *)(v100 + 48) > 0x40u )
          {
            v112 = *(_QWORD *)(v100 + 40);
            if ( v112 )
            {
              v167 = v100;
              v178 = v99;
              j_j___libc_free_0_0(v112);
              v100 = v167;
              v99 = v178;
            }
          }
          v179 = v99;
          j_j___libc_free_0(v100, 64);
          v77 = v179;
          goto LABEL_165;
        }
        v34 = sub_146F1B0(*(_QWORD *)(v8 + 24), v186);
        v36 = sub_146F1B0(*(_QWORD *)(v8 + 24), v194);
        v41 = sub_145CF40(*(_QWORD *)(v8 + 24), (__int64)&v216);
        v42 = *(_QWORD *)(v8 + 24);
        v228 = v41;
        v225 = &v227;
        v227 = v34;
        v226 = 0x200000002LL;
        v43 = sub_147DD40(v42, (__int64 *)&v225, 0, 0, a6, a7);
        if ( v225 != &v227 )
          _libc_free((unsigned __int64)v225);
LABEL_70:
        if ( (__int64 *)v36 == v43
          || byte_4FB7A20
          && (v44 = sub_145CF40(*(_QWORD *)(v8 + 24), (__int64)&v216),
              v44 == sub_14806B0(*(_QWORD *)(v8 + 24), v36, v34, 0, 0)) )
        {
          result = 1;
          goto LABEL_125;
        }
        v219 = v217;
        if ( v217 > 0x40 )
          sub_16A4FD0((__int64)&v218, (const void **)&v216);
        else
          v218 = (unsigned __int64)v216;
        v45 = *(_BYTE *)(v186 + 16);
        if ( v45 <= 0x17u )
        {
          v46 = 0;
          if ( v45 == 5 && *(_WORD *)(v186 + 18) == 32 )
            v46 = v186;
        }
        else
        {
          v46 = 0;
          if ( v45 == 56 )
            v46 = v186;
        }
        v47 = *(_BYTE *)(v194 + 16);
        if ( v47 <= 0x17u )
        {
          if ( v47 != 5 || *(_WORD *)(v194 + 18) != 32 )
            goto LABEL_121;
        }
        else if ( v47 != 56 )
        {
          if ( v45 == 79 && v47 == 79 && a5 != 3 && *(_QWORD *)(v186 - 72) == *(_QWORD *)(v194 - 72) )
          {
            LODWORD(v226) = v219;
            if ( v219 > 0x40 )
              sub_16A4FD0((__int64)&v225, (const void **)&v218);
            else
              v225 = (__int64 *)v218;
            result = sub_1B86990(v8, *(_QWORD *)(v186 - 48), *(_QWORD *)(v194 - 48), &v225, (unsigned int)(a5 + 1));
            if ( result )
            {
              LODWORD(v223) = v219;
              if ( v219 > 0x40 )
                sub_16A4FD0((__int64)&v222, (const void **)&v218);
              else
                v222 = v218;
              result = sub_1B86990(v8, *(_QWORD *)(v186 - 24), *(_QWORD *)(v194 - 24), &v222, (unsigned int)(a5 + 1));
              if ( (unsigned int)v223 > 0x40 )
              {
                if ( v222 )
                {
                  v208 = result;
                  j_j___libc_free_0_0(v222);
                  result = v208;
                }
              }
            }
            if ( (unsigned int)v226 > 0x40 )
            {
              v48 = v225;
              if ( v225 )
                goto LABEL_88;
            }
            goto LABEL_122;
          }
LABEL_121:
          result = 0;
          goto LABEL_122;
        }
        if ( !v46 )
          goto LABEL_121;
        v49 = *(_DWORD *)(v46 + 20) & 0xFFFFFFF;
        v50 = *(_DWORD *)(v194 + 20) & 0xFFFFFFF;
        if ( (_DWORD)v49 != (_DWORD)v50 )
          goto LABEL_121;
        v51 = (*(_BYTE *)(v46 + 23) & 0x40) != 0 ? *(__int64 **)(v46 - 8) : (__int64 *)(v46 - 24LL * (unsigned int)v49);
        v52 = *v51;
        v53 = (*(_BYTE *)(v194 + 23) & 0x40) != 0 ? *(_QWORD **)(v194 - 8) : (_QWORD *)(v194 - 24 * v50);
        if ( v52 != *v53 )
          goto LABEL_121;
        if ( (*(_BYTE *)(v46 + 23) & 0x40) != 0 )
          v54 = *(_QWORD *)(v46 - 8);
        else
          v54 = v46 - 24 * v49;
        v55 = (__int64 *)(v54 + 24);
        v56 = sub_16348C0(v46) | 4;
        if ( (*(_BYTE *)(v194 + 23) & 0x40) != 0 )
          v57 = *(_QWORD *)(v194 - 8);
        else
          v57 = v194 - 24LL * (*(_DWORD *)(v194 + 20) & 0xFFFFFFF);
        v181 = v57;
        v187 = (__int64 *)(v57 + 24);
        v58 = sub_16348C0(v194) | 4;
        v59 = *(_DWORD *)(v46 + 20) & 0xFFFFFFF;
        if ( v59 != 2 )
        {
          v199 = v8;
          v60 = v56;
          v61 = v58;
          v62 = v55;
          v63 = v181 + 24LL * (unsigned int)(v59 - 3) + 48;
          v64 = v187;
          while ( *v62 == *v64 )
          {
            v68 = v60 & 0xFFFFFFFFFFFFFFF8LL;
            v69 = v60 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (v60 & 4) == 0 || !v68 )
              v69 = sub_1643D30(v68, *v62);
            v70 = *(_BYTE *)(v69 + 8);
            if ( ((v70 - 14) & 0xFD) != 0 )
            {
              v60 = 0;
              if ( v70 == 13 )
                v60 = v69;
            }
            else
            {
              v60 = *(_QWORD *)(v69 + 24) | 4LL;
            }
            v62 += 3;
            v65 = v61 & 0xFFFFFFFFFFFFFFF8LL;
            v66 = v61 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (v61 & 4) == 0 || !v65 )
              v66 = sub_1643D30(v65, *v64);
            v67 = *(_BYTE *)(v66 + 8);
            if ( ((v67 - 14) & 0xFD) != 0 )
            {
              v61 = 0;
              if ( v67 == 13 )
                v61 = v66;
            }
            else
            {
              v61 = *(_QWORD *)(v66 + 24) | 4LL;
            }
            v64 += 3;
            if ( v64 == (__int64 *)v63 )
            {
              v55 = v62;
              v56 = v60;
              v8 = v199;
              goto LABEL_169;
            }
          }
          goto LABEL_121;
        }
        v63 = (__int64)v187;
LABEL_169:
        v79 = (__int64 *)*v55;
        v80 = *(_BYTE *)(*v55 + 16);
        if ( v80 <= 0x17u )
          goto LABEL_121;
        v81 = *(__int64 **)v63;
        v82 = *(_BYTE *)(*(_QWORD *)v63 + 16LL);
        if ( v82 <= 0x17u || v80 != v82 || *v79 != *v81 )
          goto LABEL_121;
        v83 = v219 - 1;
        v84 = 1LL << ((unsigned __int8)v219 - 1);
        if ( v219 > 0x40 )
        {
          if ( (*(_QWORD *)(v218 + 8LL * (v83 >> 6)) & v84) == 0 )
            goto LABEL_175;
          v183 = *(__int64 **)v63;
          v189 = v219 - 1;
          v204 = (__int64 *)*v55;
          if ( v189 == (unsigned int)sub_16A58A0((__int64)&v218) )
            goto LABEL_121;
          sub_16A8F40((__int64 *)&v218);
          v81 = v183;
          v79 = v204;
        }
        else
        {
          if ( (v84 & v218) == 0 )
            goto LABEL_175;
          if ( v218 == 1LL << v83 )
            goto LABEL_121;
          v218 = (0xFFFFFFFFFFFFFFFFLL >> (63 - ((v219 - 1) & 0x3F))) & ~v218;
        }
        v190 = v81;
        v205 = v79;
        sub_16A7400((__int64)&v218);
        v79 = v190;
        v81 = v205;
LABEL_175:
        v85 = *(_QWORD *)(v8 + 40);
        v86 = v56 & 0xFFFFFFFFFFFFFFF8LL;
        v87 = v56 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v56 & 4) == 0 || !v86 )
        {
          v184 = v81;
          v191 = *(_QWORD *)(v8 + 40);
          v206 = v79;
          v118 = sub_1643D30(v86, *v55);
          v81 = v184;
          v85 = v191;
          v79 = v206;
          v87 = v118;
        }
        v176 = v81;
        v182 = v79;
        v202 = v85;
        v88 = (unsigned int)sub_15A9FE0(v85, v87);
        v89 = (v88 + ((unsigned __int64)(sub_127FA20(v202, v87) + 7) >> 3) - 1) / v88 * v88;
        if ( sub_16A6CD0((__int64)&v218, v89) )
          goto LABEL_121;
        v203 = sub_16431D0(*v182);
        sub_16A6B50((__int64)&v225, (unsigned __int64 **)&v218, v89);
        sub_16A5DD0((__int64)&v220, (__int64)&v225, v203);
        v90 = v182;
        v91 = v176;
        if ( (unsigned int)v226 > 0x40 && v225 )
        {
          j_j___libc_free_0_0(v225);
          v91 = v176;
          v90 = v182;
        }
        v188 = *((_BYTE *)v90 + 16);
        if ( (unsigned __int8)(v188 - 61) > 1u )
          goto LABEL_246;
        v92 = (*((_BYTE *)v90 + 23) & 0x40) != 0
            ? (__int64 **)*(v90 - 1)
            : (__int64 **)&v90[-3 * (*((_DWORD *)v90 + 5) & 0xFFFFFFF)];
        v93 = (*((_BYTE *)v91 + 23) & 0x40) != 0 ? (__int64 *)*(v91 - 1) : &v91[-3 * (*((_DWORD *)v91 + 5) & 0xFFFFFFF)];
        v94 = *v93;
        v95 = *(_BYTE *)(*v93 + 16);
        if ( v95 <= 0x17u )
          goto LABEL_246;
        v96 = *v92;
        result = 0;
        v97 = **v92;
        if ( v97 != *(_QWORD *)v94 )
          goto LABEL_188;
        if ( v95 != 35 )
          goto LABEL_249;
        v128 = (*(_BYTE *)(v94 + 23) & 0x40) != 0
             ? *(_QWORD *)(v94 - 8)
             : v94 - 24LL * (*(_DWORD *)(v94 + 20) & 0xFFFFFFF);
        v129 = *(_QWORD *)(v128 + 24);
        if ( *(_BYTE *)(v129 + 16) == 13
          && ((v130 = *(_DWORD *)(v129 + 32), v131 = *(__int64 **)(v129 + 24), v130 > 0x40)
            ? (v132 = *v131)
            : (v132 = (__int64)((_QWORD)v131 << (64 - (unsigned __int8)v130)) >> (64 - (unsigned __int8)v130)),
              v209 = v97,
              v133 = sub_13A39D0((__int64)&v220, v132),
              v97 = v209,
              !v133) )
        {
          if ( v188 == 62 )
            v134 = sub_15F2380(v94);
          else
            v134 = sub_15F2370(v94);
          v185 = v134;
          v207 = sub_16431D0(*v96);
          if ( v185 )
            goto LABEL_254;
        }
        else
        {
LABEL_249:
          v207 = sub_16431D0(v97);
        }
        if ( *((_BYTE *)v96 + 16) <= 0x17u )
          goto LABEL_246;
        sub_14AA4E0((__int64)&v225, v207);
        sub_14BB090((__int64)v96, (__int64)&v225, *(_QWORD *)(v8 + 40), 0, 0, (__int64)v96, *(_QWORD *)(v8 + 16), 0);
        sub_16A5C50((__int64)&v222, (const void **)&v225, v221);
        if ( v188 == 62 )
          return sub_1B88157(&v222, &v225, v135);
        if ( (int)sub_16A9900((__int64)&v222, (unsigned __int64 *)&v220) < 0 )
        {
          sub_135E100((__int64 *)&v222);
          sub_135E100(&v227);
          sub_135E100((__int64 *)&v225);
LABEL_246:
          result = 0;
          goto LABEL_188;
        }
        sub_135E100((__int64 *)&v222);
        sub_135E100(&v227);
        sub_135E100((__int64 *)&v225);
LABEL_254:
        v192 = sub_146F1B0(*(_QWORD *)(v8 + 24), (__int64)v96);
        v122 = sub_146F1B0(*(_QWORD *)(v8 + 24), v94);
        v123 = *(_QWORD *)(v8 + 24);
        v124 = (__int64 *)v122;
        sub_16A5A50((__int64)&v225, (__int64 *)&v220, v207);
        v125 = sub_145CF40(v123, (__int64)&v225);
        sub_135E100((__int64 *)&v225);
        v126 = *(_QWORD *)(v8 + 24);
        v228 = v125;
        v226 = 0x200000002LL;
        v227 = v192;
        v225 = &v227;
        v127 = sub_147DD40(v126, (__int64 *)&v225, 0, 0, a6, a7);
        if ( v225 != &v227 )
          _libc_free((unsigned __int64)v225);
        result = v124 == v127;
LABEL_188:
        if ( v221 > 0x40 )
        {
          v48 = v220;
          if ( v220 )
          {
LABEL_88:
            v198 = result;
            j_j___libc_free_0_0(v48);
            result = v198;
          }
        }
LABEL_122:
        if ( v219 > 0x40 && v218 )
        {
          v200 = result;
          j_j___libc_free_0_0(v218);
          result = v200;
        }
LABEL_125:
        if ( v217 > 0x40 && v216 )
        {
          v201 = result;
          j_j___libc_free_0_0(v216);
          result = v201;
        }
        if ( v215 > 0x40 )
        {
LABEL_65:
          if ( v214 )
          {
            v197 = result;
            j_j___libc_free_0_0(v214);
            result = v197;
          }
        }
LABEL_13:
        if ( v213 > 0x40 && v212 )
        {
          v195 = result;
          j_j___libc_free_0_0(v212);
          result = v195;
        }
        if ( v211 > 0x40 )
        {
          if ( v210 )
          {
            v196 = result;
            j_j___libc_free_0_0(v210);
            return v196;
          }
        }
        return result;
      case 0x10:
        v28 = *(_QWORD *)(v17 + 32);
        v17 = *(_QWORD *)(v17 + 24);
        v16 *= v28;
        continue;
      default:
        BUG();
    }
  }
}
