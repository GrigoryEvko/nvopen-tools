// Function: sub_359D220
// Address: 0x359d220
//
__int64 __fastcall sub_359D220(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        int a8,
        unsigned int a9,
        char a10)
{
  __int64 v11; // rdi
  __int64 result; // rax
  __int64 v13; // r10
  __int64 v14; // rsi
  int v15; // r8d
  __int64 v16; // r9
  int v17; // r8d
  char v18; // di
  unsigned int v19; // r11d
  int v20; // ebx
  unsigned int i; // eax
  __int64 v22; // rdi
  __int64 v23; // rax
  unsigned int v24; // edx
  int *v25; // rsi
  int v26; // ecx
  __int64 v27; // r12
  unsigned __int64 v28; // rax
  signed int v29; // eax
  __int64 v30; // r8
  __int64 v31; // rax
  __int64 v32; // rcx
  __int64 v33; // rdx
  int *v34; // rax
  __int64 v35; // r8
  __int64 v36; // rax
  unsigned int v37; // ecx
  int *v38; // rdi
  int v39; // esi
  int v40; // ebx
  unsigned int v41; // eax
  int v42; // ecx
  bool v43; // al
  bool v44; // zf
  int v45; // eax
  int v46; // r13d
  __int64 v47; // r9
  __int64 v48; // rsi
  int v49; // eax
  int v50; // edx
  unsigned int v51; // eax
  int v52; // edi
  int *v53; // rax
  __int32 v54; // eax
  __int64 v55; // rdi
  int v56; // eax
  __int64 v57; // r8
  int v58; // edx
  unsigned int v59; // eax
  int v60; // esi
  unsigned __int64 v61; // rax
  __int64 v62; // r8
  __int64 v63; // r9
  unsigned __int64 v64; // r14
  bool v65; // r12
  int v66; // ecx
  __int64 v67; // rsi
  unsigned int v68; // eax
  int v69; // ecx
  int v70; // edx
  int v71; // edi
  __int64 v72; // rdi
  int v73; // eax
  int v74; // edx
  unsigned int v75; // eax
  int v76; // esi
  __int64 v77; // r12
  __int64 *v78; // rax
  _QWORD *v79; // r14
  __int64 v80; // rdx
  __int64 v81; // r12
  int v82; // eax
  __int64 v83; // rdi
  int v84; // edx
  __int64 v85; // r8
  int v86; // edx
  unsigned int v87; // esi
  int v88; // r9d
  __int64 v89; // rdi
  int v90; // eax
  __int64 v91; // rsi
  int v92; // eax
  unsigned int v93; // ecx
  int v94; // r10d
  int v95; // eax
  __int64 v96; // rsi
  int v97; // eax
  unsigned int v98; // ecx
  __int64 v99; // rax
  __int64 v100; // rsi
  __int64 v101; // rcx
  __int64 v102; // rdx
  char v103; // al
  __int64 v104; // r14
  int *v105; // rax
  int v106; // ebx
  __int64 v107; // r9
  __int64 v108; // r14
  unsigned int v109; // esi
  __int64 v110; // r8
  int v111; // r11d
  __int64 *v112; // rcx
  unsigned int v113; // edx
  __int64 *v114; // rax
  __int64 v115; // rdi
  __int64 *v116; // rax
  int v117; // eax
  int v118; // edx
  int v119; // r10d
  __int64 v120; // rdi
  int v121; // ecx
  int v122; // ecx
  unsigned int v123; // edx
  int v124; // esi
  int v125; // ecx
  __int64 v126; // rsi
  __int64 v127; // rax
  int v128; // eax
  int v129; // r12d
  int v130; // eax
  int v131; // r8d
  int v132; // r9d
  __int64 v133; // rdi
  int v134; // r12d
  unsigned __int64 v135; // r14
  unsigned __int64 v136; // r13
  __int64 v137; // r14
  int v138; // ebx
  int v139; // eax
  unsigned int v140; // eax
  int v141; // r10d
  __int64 v142; // r9
  int v143; // r10d
  __int64 v144; // rbx
  int v145; // r10d
  int v146; // esi
  int v147; // r9d
  int v148; // edi
  int v149; // r10d
  __int64 v150; // [rsp+20h] [rbp-140h]
  __int64 v151; // [rsp+28h] [rbp-138h]
  unsigned int v152; // [rsp+30h] [rbp-130h]
  __int64 v153; // [rsp+38h] [rbp-128h]
  __int64 v154; // [rsp+40h] [rbp-120h]
  __int64 *v156; // [rsp+50h] [rbp-110h]
  int v157; // [rsp+58h] [rbp-108h]
  int v158; // [rsp+5Ch] [rbp-104h]
  int v161; // [rsp+70h] [rbp-F0h]
  char v162; // [rsp+74h] [rbp-ECh]
  __int64 v163; // [rsp+78h] [rbp-E8h]
  unsigned int v164; // [rsp+80h] [rbp-E0h]
  unsigned int v165; // [rsp+84h] [rbp-DCh]
  unsigned int v166; // [rsp+88h] [rbp-D8h]
  signed int v167; // [rsp+8Ch] [rbp-D4h]
  unsigned int v168; // [rsp+90h] [rbp-D0h]
  unsigned int v169; // [rsp+94h] [rbp-CCh]
  int v170; // [rsp+98h] [rbp-C8h]
  unsigned int v171; // [rsp+9Ch] [rbp-C4h]
  __int64 v172; // [rsp+A0h] [rbp-C0h]
  int v173; // [rsp+A0h] [rbp-C0h]
  int v174; // [rsp+A8h] [rbp-B8h]
  int v175; // [rsp+ACh] [rbp-B4h]
  int v176; // [rsp+ACh] [rbp-B4h]
  unsigned int v177; // [rsp+ACh] [rbp-B4h]
  __int64 j; // [rsp+B8h] [rbp-A8h]
  unsigned int v181; // [rsp+D0h] [rbp-90h]
  __int32 v182; // [rsp+D4h] [rbp-8Ch]
  unsigned int v183; // [rsp+E4h] [rbp-7Ch] BYREF
  unsigned int v184; // [rsp+E8h] [rbp-78h] BYREF
  int v185; // [rsp+ECh] [rbp-74h] BYREF
  __int64 v186; // [rsp+F0h] [rbp-70h] BYREF
  __int64 v187; // [rsp+F8h] [rbp-68h] BYREF
  __m128i v188; // [rsp+100h] [rbp-60h] BYREF
  __int64 v189; // [rsp+110h] [rbp-50h]
  __int64 v190; // [rsp+118h] [rbp-48h]
  __int64 v191; // [rsp+120h] [rbp-40h]

  if ( a8 == a9 )
  {
    v181 = a8 - 1;
    v164 = a8;
  }
  else
  {
    v181 = 2 * a8 - a9;
    v164 = a9 - 1;
  }
  v11 = a1[6];
  v186 = *(_QWORD *)(v11 + 56);
  result = sub_2E311E0(v11);
  v13 = v186;
  v151 = result;
  if ( result != v186 )
  {
    v150 = a6 + 32LL * (unsigned int)a8;
    v153 = a6 + 32LL * a9;
    v152 = v181 + 2;
    v156 = a1 + 11;
    do
    {
      v14 = *(_QWORD *)(v13 + 32);
      v15 = *(_DWORD *)(v13 + 40);
      v184 = 0;
      v16 = a1[6];
      v17 = v15 & 0xFFFFFF;
      v183 = *(_DWORD *)(v14 + 8);
      if ( v17 == 1 )
      {
        v182 = 0;
        v161 = 0;
      }
      else
      {
        v18 = 0;
        v19 = 0;
        v20 = 0;
        for ( i = 1; i != v17; i += 2 )
        {
          if ( v16 == *(_QWORD *)(v14 + 40LL * (i + 1) + 24) )
            v18 = 1;
          else
            v20 = *(_DWORD *)(v14 + 40LL * i + 8);
          if ( v16 == *(_QWORD *)(v14 + 40LL * (i + 1) + 24) )
            v19 = *(_DWORD *)(v14 + 40LL * i + 8);
        }
        v182 = v19;
        v161 = v20;
        if ( v18 )
          v184 = v19;
        else
          v182 = 0;
      }
      v185 = 0;
      v22 = *(_QWORD *)(v150 + 8);
      v23 = *(unsigned int *)(v150 + 24);
      if ( (_DWORD)v23 )
      {
        v24 = (v23 - 1) & (37 * v182);
        v25 = (int *)(v22 + 8LL * v24);
        v26 = *v25;
        if ( *v25 == v182 )
        {
LABEL_17:
          if ( v25 != (int *)(v22 + 8 * v23) )
            v182 = v25[1];
        }
        else
        {
          v146 = 1;
          while ( v26 != -1 )
          {
            v147 = v146 + 1;
            v24 = (v23 - 1) & (v146 + v24);
            v25 = (int *)(v22 + 8LL * v24);
            v26 = *v25;
            if ( *v25 == v182 )
              goto LABEL_17;
            v146 = v147;
          }
        }
      }
      v27 = *a1;
      v170 = sub_3598DB0(*a1, v13);
      v28 = sub_2EBEE10(a1[3], v184);
      v29 = sub_3598DB0(v27, v28);
      v30 = (__int64)(a1 + 11);
      v167 = v29;
      v31 = a1[12];
      LODWORD(v187) = v183;
      if ( !v31 )
        goto LABEL_26;
      do
      {
        while ( 1 )
        {
          v32 = *(_QWORD *)(v31 + 16);
          v33 = *(_QWORD *)(v31 + 24);
          if ( *(_DWORD *)(v31 + 32) >= v183 )
            break;
          v31 = *(_QWORD *)(v31 + 24);
          if ( !v33 )
            goto LABEL_24;
        }
        v30 = v31;
        v31 = *(_QWORD *)(v31 + 16);
      }
      while ( v32 );
LABEL_24:
      if ( v156 == (__int64 *)v30 || v183 < *(_DWORD *)(v30 + 32) )
      {
LABEL_26:
        v188.m128i_i64[0] = (__int64)&v187;
        v30 = sub_359C130(a1 + 10, v30, (unsigned int **)&v188);
      }
      v162 = 0;
      v168 = *(_DWORD *)(v30 + 36);
      if ( *(_DWORD *)(*a1 + 96) > (signed int)a9 )
      {
        if ( v168 )
          goto LABEL_35;
      }
      else
      {
        if ( *(_DWORD *)(v30 + 36) )
          goto LABEL_35;
        if ( *(_BYTE *)(v30 + 40) )
        {
          v168 = 1;
          goto LABEL_35;
        }
      }
      v34 = sub_2FFAE70(a6 + 32LL * v164, (int *)&v184);
      sub_3599870(a1, a2, a7, a9, 0, v186, v183, v161, *v34);
      v35 = *(_QWORD *)(v153 + 8);
      v36 = *(unsigned int *)(v153 + 24);
      if ( (_DWORD)v36 )
      {
        v37 = (v36 - 1) & (37 * v184);
        v38 = (int *)(v35 + 8LL * v37);
        v39 = *v38;
        if ( v184 == *v38 )
        {
LABEL_32:
          if ( v38 != (int *)(v35 + 8 * v36) )
          {
            v40 = v38[1];
            *sub_2FFAE70(v153, (int *)&v183) = v40;
          }
        }
        else
        {
          v148 = 1;
          while ( v39 != -1 )
          {
            v149 = v148 + 1;
            v37 = (v36 - 1) & (v148 + v37);
            v38 = (int *)(v35 + 8LL * v37);
            v39 = *v38;
            if ( v184 == *v38 )
              goto LABEL_32;
            v148 = v149;
          }
        }
      }
      v168 = 0;
      v162 = a10;
LABEL_35:
      v41 = v181 + 2;
      if ( v167 >= (int)v181 && a8 != a9 )
      {
        v41 = v152 - v167;
        if ( (int)(v152 - v167) <= 0 )
          v41 = 1;
      }
      if ( v168 <= v41 )
        v41 = v168;
      v171 = v41;
      if ( v167 == -1 )
      {
        v42 = v170;
        v43 = a8 != a9 && v170 >= -1;
        if ( v43 )
          goto LABEL_171;
      }
      else
      {
        v42 = v167;
        if ( v170 >= v167 && a8 != a9 )
        {
LABEL_171:
          v158 = v42 == 0 && v171 == 1;
          v43 = v167 != -1;
          goto LABEL_46;
        }
        v43 = a8 == a9 && v170 > v167;
        if ( v43 )
        {
          v158 = v170 - v167;
          goto LABEL_46;
        }
        v43 = 1;
      }
      v158 = 0;
LABEL_46:
      if ( v171 )
      {
        v44 = !v43 || v170 <= v167;
        v45 = 0;
        if ( !v44 )
          v45 = v170 - v167;
        v169 = v164;
        v165 = v164 - v45;
        v154 = a6 + 32LL * (v164 - v45);
        v46 = 0;
        v157 = v158 + v42;
        v166 = v181 - v158;
        for ( j = a6 + 32LL * a9; ; j = a6 + 32LL * (a9 - v174) )
        {
          if ( v181 < v46 || v170 >= a8 || v181 < v46 + v157 )
          {
            v185 = v161;
            v174 = v46 + 1;
          }
          else
          {
            v55 = a6 + 32LL * v166;
            v56 = *(_DWORD *)(v55 + 24);
            v57 = *(_QWORD *)(v55 + 8);
            if ( v56 )
            {
              v58 = v56 - 1;
              v59 = (v56 - 1) & (37 * v184);
              v60 = *(_DWORD *)(v57 + 8LL * v59);
              if ( v184 == v60 )
              {
LABEL_66:
                v185 = *sub_2FFAE70(v55, (int *)&v184);
                v174 = v46 + 1;
                goto LABEL_67;
              }
              v132 = 1;
              while ( v60 != -1 )
              {
                v59 = v58 & (v132 + v59);
                v60 = *(_DWORD *)(v57 + 8LL * v59);
                if ( v184 == v60 )
                  goto LABEL_66;
                ++v132;
              }
            }
            v133 = a1[3];
            v134 = 1;
            v185 = v184;
            v135 = sub_2EBEE10(v133, v184);
            v174 = v46 + 1;
            v177 = v46 + 1;
            if ( v135 )
            {
              v173 = v46;
              v136 = v135;
              while ( 1 )
              {
                if ( *(_WORD *)(v136 + 68) != 68 && *(_WORD *)(v136 + 68)
                  || (v137 = a1[6], v137 != *(_QWORD *)(v136 + 24)) )
                {
LABEL_180:
                  v46 = v173;
                  goto LABEL_67;
                }
                v138 = sub_3598DB0(*a1, v136);
                v185 = v134 + v138 <= (int)v166 ? sub_3598190(v136, v137) : sub_3598140(v136, v137);
                v136 = sub_2EBEE10(a1[3], v185);
                v139 = sub_3598DB0(*a1, v136);
                if ( v139 != -1 )
                {
                  v140 = v181 - (v138 - v139);
                  if ( v140 >= v177 )
                  {
                    v144 = 32LL * (v140 - v177) + a6;
                    sub_359B060((__int64 **)&v188, (__int64 *)v144, &v185);
                    if ( v189 != *(_QWORD *)(v144 + 8) + 8LL * *(unsigned int *)(v144 + 24) )
                      break;
                  }
                }
                ++v177;
                ++v134;
                if ( !v136 )
                  goto LABEL_180;
              }
              v46 = v173;
              v185 = *(_DWORD *)(v189 + 4);
            }
          }
LABEL_67:
          v61 = sub_2EBEE10(a1[3], v185);
          if ( v61 && (!*(_WORD *)(v61 + 68) || *(_WORD *)(v61 + 68) == 68) && a5 == *(_QWORD *)(v61 + 24) )
          {
            v125 = *(_DWORD *)(v61 + 40) & 0xFFFFFF;
            if ( v125 == 1 )
            {
LABEL_169:
              v128 = 0;
            }
            else
            {
              v126 = *(_QWORD *)(v61 + 32);
              v127 = 1;
              while ( a5 == *(_QWORD *)(v126 + 40LL * (unsigned int)(v127 + 1) + 24) )
              {
                v127 = (unsigned int)(v127 + 2);
                if ( v125 == (_DWORD)v127 )
                  goto LABEL_169;
              }
              v128 = *(_DWORD *)(v126 + 40 * v127 + 8);
            }
            v185 = v128;
          }
          v64 = sub_2EBEE10(a1[3], v184);
          v172 = a6 + 32LL * (a9 - v46);
          if ( v64 )
          {
            v65 = *(_WORD *)(v64 + 68) == 68 || *(_WORD *)(v64 + 68) == 0;
            if ( a8 == a9 )
              goto LABEL_79;
          }
          else
          {
            v65 = 0;
            if ( a8 == a9 )
              goto LABEL_86;
          }
          if ( v46 )
          {
            if ( v164 == a8 )
            {
              v89 = a6 + 32LL * (v164 + 1 - v46);
              v90 = *(_DWORD *)(v89 + 24);
              v91 = *(_QWORD *)(v89 + 8);
              if ( v90 )
              {
                v92 = v90 - 1;
                v93 = v92 & (37 * v183);
                v62 = *(unsigned int *)(v91 + 8LL * v93);
                if ( v183 == (_DWORD)v62 )
                  goto LABEL_100;
                v119 = 1;
                while ( (_DWORD)v62 != -1 )
                {
                  v63 = (unsigned int)(v119 + 1);
                  v93 = v92 & (v119 + v93);
                  v62 = *(unsigned int *)(v91 + 8LL * v93);
                  if ( v183 == (_DWORD)v62 )
                    goto LABEL_100;
                  ++v119;
                }
              }
            }
LABEL_147:
            if ( v181 + 1 >= v167 )
              goto LABEL_105;
            v120 = a6 + 32LL * v165;
            v121 = *(_DWORD *)(v120 + 24);
            v62 = *(_QWORD *)(v120 + 8);
            if ( !v121 )
              goto LABEL_105;
            v68 = v184;
            v70 = 37 * v184;
            goto LABEL_150;
          }
          if ( v164 != a8 )
            goto LABEL_147;
          if ( !(v167 | v170) )
            goto LABEL_105;
          v66 = *(_DWORD *)(v154 + 24);
          v67 = *(_QWORD *)(v154 + 8);
          if ( !v66 )
            goto LABEL_105;
          v68 = v184;
          v69 = v66 - 1;
          v70 = 37 * v184;
          v62 = v69 & (37 * v184);
          v71 = *(_DWORD *)(v67 + 8 * v62);
          if ( v184 == v71 )
          {
LABEL_78:
            v182 = *sub_2FFAE70(v154, (int *)&v184);
            goto LABEL_79;
          }
          v145 = 1;
          while ( v71 != -1 )
          {
            v63 = (unsigned int)(v145 + 1);
            v62 = v69 & (unsigned int)(v145 + v62);
            v71 = *(_DWORD *)(v67 + 8LL * (unsigned int)v62);
            if ( v184 == v71 )
              goto LABEL_78;
            ++v145;
          }
          if ( v181 + 1 >= v167 )
            goto LABEL_105;
          v120 = a6 + 32LL * v165;
          v62 = *(_QWORD *)(v120 + 8);
          v121 = *(_DWORD *)(v120 + 24);
LABEL_150:
          v122 = v121 - 1;
          v123 = v122 & v70;
          v124 = *(_DWORD *)(v62 + 8LL * v123);
          if ( v68 == v124 )
          {
LABEL_151:
            v182 = *sub_2FFAE70(v120, (int *)&v184);
            goto LABEL_79;
          }
          v94 = 1;
          while ( v124 != -1 )
          {
            v63 = (unsigned int)(v94 + 1);
            v123 = v122 & (v94 + v123);
            v124 = *(_DWORD *)(v62 + 8LL * v123);
            if ( v68 == v124 )
              goto LABEL_151;
            ++v94;
          }
LABEL_105:
          v89 = a6 + 32LL * v169;
          v95 = *(_DWORD *)(v89 + 24);
          v96 = *(_QWORD *)(v89 + 8);
          if ( !v95 )
            goto LABEL_79;
          v97 = v95 - 1;
          v98 = v97 & (37 * v183);
          v62 = *(unsigned int *)(v96 + 8LL * v98);
          if ( v183 != (_DWORD)v62 )
          {
            v143 = 1;
            while ( (_DWORD)v62 != -1 )
            {
              v63 = (unsigned int)(v143 + 1);
              v98 = v97 & (v143 + v98);
              v62 = *(unsigned int *)(v96 + 8LL * v98);
              if ( v183 == (_DWORD)v62 )
                goto LABEL_107;
              ++v143;
            }
LABEL_79:
            if ( !v65 )
              goto LABEL_86;
            goto LABEL_80;
          }
LABEL_107:
          if ( v170 == v167 || v164 != a8 || !v65 )
          {
LABEL_100:
            v182 = *sub_2FFAE70(v89, (int *)&v183);
            goto LABEL_79;
          }
LABEL_80:
          if ( v170 <= (int)(v181 - v46) )
          {
            v99 = a1[12];
            v100 = (__int64)(a1 + 11);
            LODWORD(v187) = v184;
            if ( !v99 )
              goto LABEL_117;
            do
            {
              while ( 1 )
              {
                v101 = *(_QWORD *)(v99 + 16);
                v102 = *(_QWORD *)(v99 + 24);
                if ( *(_DWORD *)(v99 + 32) >= v184 )
                  break;
                v99 = *(_QWORD *)(v99 + 24);
                if ( !v102 )
                  goto LABEL_115;
              }
              v100 = v99;
              v99 = *(_QWORD *)(v99 + 16);
            }
            while ( v101 );
LABEL_115:
            if ( v156 == (__int64 *)v100 || v184 < *(_DWORD *)(v100 + 32) )
            {
LABEL_117:
              v188.m128i_i64[0] = (__int64)&v187;
              v100 = sub_359C130(a1 + 10, v100, (unsigned int **)&v188);
            }
            v176 = *(_DWORD *)(v100 + 36) - ((*(_BYTE *)(v100 + 40) == 0) + v170 - v167);
            if ( v176 > v46 && sub_359BBF0(v153, (int *)&v184) )
            {
              v103 = sub_3599670(a1, v64);
              v104 = j;
              if ( v103 )
                v104 = a6 + 32LL * (a9 - v46 - v176);
              if ( sub_359BBF0(v104, (int *)&v184) )
              {
                v105 = sub_2FFAE70(v104, (int *)&v184);
                v175 = *v105;
                v106 = *v105;
                sub_3599870(a1, a2, a7, a9, v46, v186, v183, *v105, 0);
                *sub_2FFAE70(j, (int *)&v183) = v106;
                v182 = v106;
                if ( sub_359BBF0(a6 + 32LL * (unsigned int)(a8 - 1 - v46), (int *)&v184) )
                  v182 = *sub_2FFAE70(a6 + 32LL * (unsigned int)(a8 - 1 - v46), (int *)&v184);
                if ( a10 && v171 - 1 == v46 )
                  sub_35988E0(v183, v175, a1[6], a1[3], a1[5], v107);
                goto LABEL_59;
              }
            }
          }
          if ( v158 > 0 && a8 == a9 )
          {
            v72 = a6 + 32LL * (a9 - v158 - v46);
            v73 = *(_DWORD *)(v72 + 24);
            v62 = *(_QWORD *)(v72 + 8);
            if ( v73 )
            {
              v74 = v73 - 1;
              v75 = (v73 - 1) & (37 * v184);
              v76 = *(_DWORD *)(v62 + 8LL * v75);
              if ( v76 == v184 )
              {
LABEL_85:
                v182 = *sub_2FFAE70(v72, (int *)&v184);
              }
              else
              {
                v63 = 1;
                while ( v76 != -1 )
                {
                  v75 = v74 & (v63 + v75);
                  v76 = *(_DWORD *)(v62 + 8LL * v75);
                  if ( v184 == v76 )
                    goto LABEL_85;
                  v63 = (unsigned int)(v63 + 1);
                }
              }
            }
          }
LABEL_86:
          v175 = sub_2EC06C0(
                   a1[3],
                   *(_QWORD *)(*(_QWORD *)(a1[3] + 56) + 16LL * (v183 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
                   byte_3F871B3,
                   0,
                   v62,
                   v63);
          v77 = *(_QWORD *)(a1[4] + 8);
          v187 = 0;
          v188 = 0u;
          v189 = 0;
          v78 = (__int64 *)sub_2E311E0(a2);
          v79 = sub_2F26260(a2, v78, v188.m128i_i64, v77, v175);
          v81 = v80;
          if ( v188.m128i_i64[0] )
            sub_B91220((__int64)&v188, v188.m128i_i64[0]);
          if ( v187 )
            sub_B91220((__int64)&v187, v187);
          v188.m128i_i64[0] = 0;
          v188.m128i_i32[2] = v185;
          v163 = v81;
          v189 = 0;
          v190 = 0;
          v191 = 0;
          sub_2E8EAD0(v81, (__int64)v79, &v188);
          v188.m128i_i8[0] = 4;
          v189 = 0;
          v188.m128i_i32[0] &= 0xFFF000FF;
          v190 = a3;
          sub_2E8EAD0(v81, (__int64)v79, &v188);
          v188.m128i_i64[0] = 0;
          v188.m128i_i32[2] = v182;
          v189 = 0;
          v190 = 0;
          v191 = 0;
          sub_2E8EAD0(v81, (__int64)v79, &v188);
          v188.m128i_i8[0] = 4;
          v189 = 0;
          v188.m128i_i32[0] &= 0xFFF000FF;
          v190 = a4;
          sub_2E8EAD0(v81, (__int64)v79, &v188);
          if ( v46 )
            goto LABEL_91;
          v108 = v186;
          v187 = v81;
          v109 = *(_DWORD *)(a7 + 24);
          if ( !v109 )
          {
            v188.m128i_i64[0] = 0;
            ++*(_QWORD *)a7;
            goto LABEL_207;
          }
          v110 = *(_QWORD *)(a7 + 8);
          v111 = 1;
          v112 = 0;
          v113 = (v109 - 1) & (((unsigned int)v81 >> 9) ^ ((unsigned int)v81 >> 4));
          v114 = (__int64 *)(v110 + 16LL * v113);
          v115 = *v114;
          if ( *v114 != v81 )
          {
            while ( v115 != -4096 )
            {
              if ( v115 == -8192 && !v112 )
                v112 = v114;
              v113 = (v109 - 1) & (v111 + v113);
              v114 = (__int64 *)(v110 + 16LL * v113);
              v115 = *v114;
              if ( v81 == *v114 )
                goto LABEL_130;
              ++v111;
            }
            if ( !v112 )
              v112 = v114;
            ++*(_QWORD *)a7;
            v117 = *(_DWORD *)(a7 + 16);
            v188.m128i_i64[0] = (__int64)v112;
            v118 = v117 + 1;
            if ( 4 * (v117 + 1) >= 3 * v109 )
            {
LABEL_207:
              v109 *= 2;
            }
            else if ( v109 - *(_DWORD *)(a7 + 20) - v118 > v109 >> 3 )
            {
LABEL_142:
              *(_DWORD *)(a7 + 16) = v118;
              if ( *v112 != -4096 )
                --*(_DWORD *)(a7 + 20);
              v112[1] = 0;
              *v112 = v163;
              v116 = v112 + 1;
              goto LABEL_131;
            }
            sub_2E48800(a7, v109);
            sub_3547B30(a7, &v187, &v188);
            v112 = (__int64 *)v188.m128i_i64[0];
            v163 = v187;
            v118 = *(_DWORD *)(a7 + 16) + 1;
            goto LABEL_142;
          }
LABEL_130:
          v116 = v114 + 1;
LABEL_131:
          *v116 = v108;
LABEL_91:
          v82 = 0;
          if ( a8 == a9 )
          {
            v83 = a6 + 32LL * v169;
            v84 = *(_DWORD *)(v83 + 24);
            v85 = *(_QWORD *)(v83 + 8);
            if ( v84 )
            {
              v86 = v84 - 1;
              v87 = v86 & (37 * v184);
              v88 = *(_DWORD *)(v85 + 8LL * v87);
              if ( v184 == v88 )
              {
LABEL_94:
                v82 = *sub_2FFAE70(v83, (int *)&v184);
              }
              else
              {
                v141 = 1;
                while ( v88 != -1 )
                {
                  v87 = v86 & (v141 + v87);
                  v88 = *(_DWORD *)(v85 + 8LL * v87);
                  if ( v184 == v88 )
                    goto LABEL_94;
                  ++v141;
                }
              }
            }
          }
          sub_3599870(a1, a2, a7, a9, v46, v186, v183, v175, v82);
          v48 = *(_QWORD *)(v172 + 8);
          v49 = *(_DWORD *)(v172 + 24);
          if ( v49 )
          {
            v50 = v49 - 1;
            v51 = (v49 - 1) & (37 * v183);
            v52 = *(_DWORD *)(v48 + 8LL * v51);
            if ( v183 == v52 )
            {
LABEL_52:
              v53 = sub_2FFAE70(j, (int *)&v183);
              sub_3599870(a1, a2, a7, a9, v46, v186, *v53, v175, 0);
            }
            else
            {
              v131 = 1;
              while ( v52 != -1 )
              {
                v47 = (unsigned int)(v131 + 1);
                v51 = v50 & (v131 + v51);
                v52 = *(_DWORD *)(v48 + 8LL * v51);
                if ( v183 == v52 )
                  goto LABEL_52;
                ++v131;
              }
            }
          }
          if ( a10 && v171 - 1 == v46 )
            sub_35988E0(v183, v175, a1[6], a1[3], a1[5], v47);
          v54 = v182;
          if ( a8 == a9 )
            v54 = v175;
          v182 = v54;
          *sub_2FFAE70(j, (int *)&v183) = v175;
LABEL_59:
          --v169;
          v46 = v174;
          --v165;
          --v166;
          if ( v174 == v171 )
            goto LABEL_159;
        }
      }
      v175 = 0;
LABEL_159:
      if ( v168 > v171 )
      {
        v129 = v171 + 1;
        do
        {
          sub_3599870(a1, a2, a7, a9, v129, v186, v183, v175, 0);
          v130 = v129++;
        }
        while ( v168 != v130 );
      }
      if ( v162 )
      {
        sub_359B060((__int64 **)&v188, (__int64 *)v153, (int *)&v184);
        if ( v189 != *(_QWORD *)(v153 + 8) + 8LL * *(unsigned int *)(v153 + 24) )
          sub_35988E0(v183, *(_DWORD *)(v189 + 4), a1[6], a1[3], a1[5], v142);
      }
      result = sub_2FD79B0(&v186);
      v13 = v186;
    }
    while ( v151 != v186 );
  }
  return result;
}
