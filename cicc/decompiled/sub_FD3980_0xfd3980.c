// Function: sub_FD3980
// Address: 0xfd3980
//
__int64 __fastcall sub_FD3980(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rsi
  __int64 v5; // rax
  int v6; // ebx
  int v7; // r14d
  char v8; // al
  __int64 v9; // r8
  char v10; // r9
  int v11; // eax
  __int64 v12; // r14
  __int64 v13; // rbx
  __int64 v14; // r14
  __int64 v15; // r13
  __int64 v16; // rcx
  unsigned int v17; // eax
  __int64 v18; // r9
  unsigned int v19; // edi
  __int64 *v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 result; // rax
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rax
  __int64 v28; // rbx
  __int64 v29; // rdx
  __int64 v30; // r14
  __int64 v31; // rcx
  unsigned int v32; // eax
  int v33; // r11d
  unsigned int v34; // edx
  _QWORD *v35; // rax
  __int64 v36; // r10
  _DWORD *v37; // rax
  _QWORD *v38; // r15
  int v39; // eax
  int v40; // r11d
  __int64 *v41; // rdx
  unsigned int v42; // edi
  __int64 *v43; // rcx
  _DWORD *v44; // rdx
  unsigned __int64 v45; // rcx
  __int64 v46; // rax
  unsigned __int64 v47; // rdx
  __int64 v48; // rbx
  __int64 v49; // rsi
  unsigned int v50; // r15d
  unsigned int v51; // edx
  __int64 *v52; // rax
  __int64 v53; // r9
  int v54; // r10d
  __int64 *v55; // rdx
  unsigned int v56; // edi
  __int64 *v57; // rax
  __int64 v58; // rcx
  int v59; // r11d
  unsigned int v60; // ecx
  __int64 v61; // rdx
  __int64 v62; // r8
  unsigned int v63; // r13d
  __int64 v64; // r12
  unsigned int v65; // eax
  int v66; // ecx
  __int64 v67; // rsi
  _DWORD *v68; // rax
  int v69; // edx
  _DWORD *v70; // rax
  int v71; // edx
  _DWORD *v72; // rax
  int v73; // edx
  int v74; // edx
  __int64 v75; // rdi
  __int64 v76; // rsi
  __int64 *v77; // r8
  unsigned int v78; // ebx
  int v79; // r9d
  __int64 v80; // rsi
  int v81; // r10d
  _QWORD *v82; // rax
  unsigned int v83; // edi
  _QWORD *v84; // rdx
  __int64 v85; // rcx
  unsigned int *v86; // rax
  unsigned __int64 v87; // rax
  __int64 v88; // r12
  int v89; // eax
  int v90; // ebx
  unsigned int v91; // r14d
  __int64 v92; // r8
  __int64 v93; // r9
  __int64 v94; // rax
  unsigned __int64 v95; // rdx
  __int64 *v96; // rdx
  int v97; // eax
  int v98; // ecx
  int v99; // eax
  int v100; // edi
  unsigned int v101; // eax
  __int64 v102; // r9
  int v103; // r11d
  __int64 *v104; // r10
  int v105; // eax
  int v106; // eax
  __int64 v107; // rdi
  __int64 *v108; // r9
  unsigned int v109; // r15d
  int v110; // r10d
  int v111; // edx
  __int64 v112; // rbx
  __int64 v113; // rdx
  __int64 v114; // rcx
  unsigned int v115; // eax
  int v116; // r10d
  __int64 v117; // rdx
  unsigned int v118; // ecx
  __int64 v119; // rax
  __int64 v120; // rdi
  unsigned int v121; // eax
  __int64 v122; // r14
  __int64 v123; // rdx
  __int64 v124; // rcx
  __int64 v125; // rdx
  __int64 v126; // rax
  __int64 v127; // r9
  unsigned int v128; // r8d
  __int64 *v129; // rdi
  __int64 v130; // r11
  int v131; // edi
  int v132; // ebx
  int v133; // edi
  int v134; // r10d
  __int64 *v135; // r9
  int v136; // edx
  unsigned int v137; // ecx
  __int64 v138; // rdi
  int v139; // eax
  int v140; // ebx
  unsigned int v141; // r13d
  int v142; // eax
  __int64 v143; // rcx
  int v144; // ecx
  unsigned int v145; // edx
  __int64 v146; // r11
  int v147; // edi
  int v148; // ecx
  unsigned int v149; // ecx
  __int64 v150; // r13
  _QWORD *v151; // rdx
  unsigned int v152; // ebx
  __int64 v153; // rdi
  int v154; // ecx
  __int64 v155; // r13
  unsigned int v156; // eax
  __int64 v157; // r15
  int v158; // edi
  int v159; // eax
  int v160; // r10d
  int v161; // r9d
  unsigned int v162; // r15d
  __int64 v163; // rax
  __int64 v164; // rdi
  unsigned int v165; // r11d
  __int64 v166; // [rsp+8h] [rbp-E8h]
  int v167; // [rsp+8h] [rbp-E8h]
  int v168; // [rsp+8h] [rbp-E8h]
  __int64 v169; // [rsp+18h] [rbp-D8h]
  __int64 v170; // [rsp+18h] [rbp-D8h]
  __int64 v171; // [rsp+18h] [rbp-D8h]
  int v172; // [rsp+18h] [rbp-D8h]
  __int64 v173; // [rsp+18h] [rbp-D8h]
  __int64 v174; // [rsp+18h] [rbp-D8h]
  __int64 v175; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v176; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v177; // [rsp+38h] [rbp-B8h]
  __int64 v178; // [rsp+40h] [rbp-B0h]
  unsigned int v179; // [rsp+48h] [rbp-A8h]
  __int64 v180; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v181; // [rsp+58h] [rbp-98h]
  __int64 v182; // [rsp+60h] [rbp-90h]
  unsigned int v183; // [rsp+68h] [rbp-88h]
  _BYTE *v184; // [rsp+70h] [rbp-80h] BYREF
  __int64 v185; // [rsp+78h] [rbp-78h]
  _BYTE v186[112]; // [rsp+80h] [rbp-70h] BYREF

  v3 = a1;
  v4 = *(_QWORD *)a1;
  v169 = a2;
  v5 = sub_DF9A70(a2);
  v6 = v5;
  v184 = (_BYTE *)v5;
  v7 = v5;
  v8 = sub_DF9980(v169);
  v9 = v169;
  v10 = v8;
  v11 = 7;
  if ( v10 )
  {
    v68 = sub_C94E20((__int64)qword_4F862F0);
    v9 = v169;
    v69 = v68 ? *v68 : LODWORD(qword_4F862F0[2]);
    v11 = 7;
    if ( v69 >= 0 )
    {
      v70 = sub_C94E20((__int64)qword_4F862F0);
      v9 = v169;
      v71 = v70 ? *v70 : LODWORD(qword_4F862F0[2]);
      v11 = 7;
      if ( v71 <= 10 )
      {
        v72 = sub_C94E20((__int64)qword_4F862F0);
        v9 = v169;
        v73 = v72 ? *v72 : LODWORD(qword_4F862F0[2]);
        v11 = 7;
        if ( (unsigned int)(v73 + 4) <= 0x12 )
        {
          v74 = v73 - 5;
          v7 = v6 + v74 * v6 / 10;
          v11 = 7 * v74 / 10 + 7;
        }
      }
    }
  }
  *(_DWORD *)(a1 + 40) = v7;
  v12 = *(_QWORD *)a1;
  *(_DWORD *)(a1 + 44) = v11;
  v13 = *(_QWORD *)(v12 + 80);
  v14 = v12 + 72;
  if ( v13 != v14 )
  {
    while ( 1 )
    {
      v23 = *(_QWORD *)(v3 + 16);
      if ( v13 )
      {
        v15 = v13 - 24;
        v16 = (unsigned int)(*(_DWORD *)(v13 + 20) + 1);
        v17 = *(_DWORD *)(v13 + 20) + 1;
      }
      else
      {
        v15 = 0;
        v16 = 0;
        v17 = 0;
      }
      if ( v17 >= *(_DWORD *)(v23 + 32) || !*(_QWORD *)(*(_QWORD *)(v23 + 24) + 8 * v16) )
        goto LABEL_11;
      v4 = *(unsigned int *)(v3 + 136);
      if ( !(_DWORD)v4 )
        break;
      v18 = *(_QWORD *)(v3 + 120);
      v19 = (v4 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v20 = (__int64 *)(v18 + 16LL * v19);
      v21 = *v20;
      if ( v15 != *v20 )
      {
        v172 = 1;
        v96 = 0;
        while ( v21 != -4096 )
        {
          if ( v21 == -8192 && !v96 )
            v96 = v20;
          v19 = (v4 - 1) & (v172 + v19);
          v20 = (__int64 *)(v18 + 16LL * v19);
          v21 = *v20;
          if ( *v20 == v15 )
            goto LABEL_9;
          ++v172;
        }
        if ( !v96 )
          v96 = v20;
        v97 = *(_DWORD *)(v3 + 128);
        ++*(_QWORD *)(v3 + 112);
        v98 = v97 + 1;
        if ( 4 * (v97 + 1) < (unsigned int)(3 * v4) )
        {
          if ( (int)v4 - *(_DWORD *)(v3 + 132) - v98 <= (unsigned int)v4 >> 3 )
          {
            v174 = v9;
            sub_CE3D80(v3 + 112, v4);
            v105 = *(_DWORD *)(v3 + 136);
            if ( !v105 )
            {
LABEL_370:
              ++*(_DWORD *)(v3 + 128);
              BUG();
            }
            v106 = v105 - 1;
            v107 = *(_QWORD *)(v3 + 120);
            v108 = 0;
            v109 = v106 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
            v9 = v174;
            v110 = 1;
            v98 = *(_DWORD *)(v3 + 128) + 1;
            v96 = (__int64 *)(v107 + 16LL * v109);
            v4 = *v96;
            if ( v15 != *v96 )
            {
              while ( v4 != -4096 )
              {
                if ( !v108 && v4 == -8192 )
                  v108 = v96;
                v109 = v106 & (v110 + v109);
                v96 = (__int64 *)(v107 + 16LL * v109);
                v4 = *v96;
                if ( *v96 == v15 )
                  goto LABEL_109;
                ++v110;
              }
              if ( v108 )
                v96 = v108;
            }
          }
          goto LABEL_109;
        }
LABEL_118:
        v173 = v9;
        sub_CE3D80(v3 + 112, 2 * v4);
        v99 = *(_DWORD *)(v3 + 136);
        if ( !v99 )
          goto LABEL_370;
        v100 = v99 - 1;
        v4 = *(_QWORD *)(v3 + 120);
        v9 = v173;
        v101 = (v99 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
        v98 = *(_DWORD *)(v3 + 128) + 1;
        v96 = (__int64 *)(v4 + 16LL * v101);
        v102 = *v96;
        if ( *v96 != v15 )
        {
          v103 = 1;
          v104 = 0;
          while ( v102 != -4096 )
          {
            if ( v102 == -8192 && !v104 )
              v104 = v96;
            v101 = v100 & (v103 + v101);
            v96 = (__int64 *)(v4 + 16LL * v101);
            v102 = *v96;
            if ( *v96 == v15 )
              goto LABEL_109;
            ++v103;
          }
          if ( v104 )
            v96 = v104;
        }
LABEL_109:
        *(_DWORD *)(v3 + 128) = v98;
        if ( *v96 != -4096 )
          --*(_DWORD *)(v3 + 132);
        *v96 = v15;
        v22 = 0;
        v96[1] = 0;
        goto LABEL_10;
      }
LABEL_9:
      v22 = v20[1];
LABEL_10:
      *(_QWORD *)(v22 + 16) = *(_QWORD *)(v3 + 40);
LABEL_11:
      v13 = *(_QWORD *)(v13 + 8);
      if ( v14 == v13 )
        goto LABEL_14;
    }
    ++*(_QWORD *)(v3 + 112);
    goto LABEL_118;
  }
LABEL_14:
  result = sub_DF9AE0(v9);
  if ( !(_BYTE)result )
    return result;
  result = sub_FCE0B0(v3, v4);
  if ( !(_BYTE)result )
    return result;
  v176 = 0;
  v184 = v186;
  v185 = 0x800000000LL;
  v27 = *(_QWORD *)v3;
  v177 = 0;
  v28 = *(_QWORD *)(v27 + 80);
  v178 = 0;
  v179 = 0;
  v180 = 0;
  v181 = 0;
  v182 = 0;
  v183 = 0;
  v170 = v27 + 72;
  if ( v28 == v27 + 72 )
  {
    v75 = 0;
    v76 = 0;
    if ( !*(_BYTE *)(v3 + 48) )
      goto LABEL_79;
    v47 = 1;
    v45 = 8;
    v46 = 0;
LABEL_38:
    v170 -= 24;
    v48 = v170;
LABEL_39:
    v175 = v170;
    if ( v47 > v45 )
    {
      sub_C8D5F0((__int64)&v184, v186, v47, 8u, v25, v26);
      v46 = (unsigned int)v185;
    }
    *(_QWORD *)&v184[8 * v46] = v48;
    v49 = v181;
    v50 = v185 + 1;
    LODWORD(v185) = v185 + 1;
    if ( v183 )
    {
      v51 = (v183 - 1) & (((unsigned int)v175 >> 9) ^ ((unsigned int)v175 >> 4));
      v52 = (__int64 *)(v181 + 16LL * v51);
      v53 = *v52;
      if ( v175 == *v52 )
      {
LABEL_43:
        if ( (__int64 *)(v181 + 16LL * v183) != v52 )
        {
LABEL_44:
          if ( v50 )
          {
            v171 = v3;
            while ( 1 )
            {
              v64 = *(_QWORD *)&v184[8 * v50 - 8];
              LODWORD(v185) = v50 - 1;
              if ( !v179 )
                break;
              v54 = 1;
              v55 = 0;
              v56 = (v179 - 1) & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
              v57 = (__int64 *)(v177 + 16LL * v56);
              v58 = *v57;
              if ( v64 == *v57 )
                goto LABEL_47;
              while ( 1 )
              {
                if ( v58 == -4096 )
                {
                  if ( !v55 )
                    v55 = v57;
                  ++v176;
                  v66 = v178 + 1;
                  if ( 4 * ((int)v178 + 1) < 3 * v179 )
                  {
                    if ( v179 - HIDWORD(v178) - v66 <= v179 >> 3 )
                    {
                      sub_A4A350((__int64)&v176, v179);
                      if ( !v179 )
                        goto LABEL_368;
                      v77 = 0;
                      v78 = (v179 - 1) & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
                      v79 = 1;
                      v66 = v178 + 1;
                      v55 = (__int64 *)(v177 + 16LL * v78);
                      v80 = *v55;
                      if ( v64 != *v55 )
                      {
                        while ( v80 != -4096 )
                        {
                          if ( !v77 && v80 == -8192 )
                            v77 = v55;
                          v78 = (v179 - 1) & (v79 + v78);
                          v55 = (__int64 *)(v177 + 16LL * v78);
                          v80 = *v55;
                          if ( v64 == *v55 )
                            goto LABEL_59;
                          ++v79;
                        }
                        if ( v77 )
                          v55 = v77;
                      }
                    }
LABEL_59:
                    LODWORD(v178) = v66;
                    if ( *v55 != -4096 )
                      --HIDWORD(v178);
                    *v55 = v64;
                    v59 = 0;
                    *((_DWORD *)v55 + 2) = 0;
                    goto LABEL_48;
                  }
LABEL_57:
                  sub_A4A350((__int64)&v176, 2 * v179);
                  if ( !v179 )
                    goto LABEL_368;
                  v65 = (v179 - 1) & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
                  v66 = v178 + 1;
                  v55 = (__int64 *)(v177 + 16LL * v65);
                  v67 = *v55;
                  if ( v64 != *v55 )
                  {
                    v134 = 1;
                    v135 = 0;
                    while ( v67 != -4096 )
                    {
                      if ( !v135 && v67 == -8192 )
                        v135 = v55;
                      v65 = (v179 - 1) & (v134 + v65);
                      v55 = (__int64 *)(v177 + 16LL * v65);
                      v67 = *v55;
                      if ( v64 == *v55 )
                        goto LABEL_59;
                      ++v134;
                    }
                    if ( v135 )
                      v55 = v135;
                  }
                  goto LABEL_59;
                }
                if ( v55 || v58 != -8192 )
                  v57 = v55;
                v56 = (v179 - 1) & (v54 + v56);
                v58 = *(_QWORD *)(v177 + 16LL * v56);
                if ( v64 == v58 )
                  break;
                ++v54;
                v55 = v57;
                v57 = (__int64 *)(v177 + 16LL * v56);
              }
              v57 = (__int64 *)(v177 + 16LL * v56);
LABEL_47:
              v59 = *((_DWORD *)v57 + 2);
LABEL_48:
              v49 = v181;
              if ( v183 )
              {
                v60 = (v183 - 1) & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
                v61 = v181 + 16LL * v60;
                v62 = *(_QWORD *)v61;
                if ( v64 == *(_QWORD *)v61 )
                {
LABEL_50:
                  if ( v61 != v181 + 16LL * v183 )
                  {
                    v63 = *(_DWORD *)(v61 + 8);
                    goto LABEL_52;
                  }
                }
                else
                {
                  v111 = 1;
                  while ( v62 != -4096 )
                  {
                    v161 = v111 + 1;
                    v60 = (v183 - 1) & (v111 + v60);
                    v61 = v181 + 16LL * v60;
                    v62 = *(_QWORD *)v61;
                    if ( v64 == *(_QWORD *)v61 )
                      goto LABEL_50;
                    v111 = v161;
                  }
                }
              }
              v112 = *(_QWORD *)(v64 + 16);
              if ( !v112 )
              {
LABEL_149:
                v63 = -1;
                goto LABEL_52;
              }
              while ( 1 )
              {
                v113 = *(_QWORD *)(v112 + 24);
                if ( (unsigned __int8)(*(_BYTE *)v113 - 30) <= 0xAu )
                  break;
                v112 = *(_QWORD *)(v112 + 8);
                if ( !v112 )
                  goto LABEL_149;
              }
              v63 = -1;
LABEL_146:
              v122 = *(_QWORD *)(v113 + 40);
              v123 = *(_QWORD *)(v171 + 16);
              if ( v122 )
              {
                v114 = (unsigned int)(*(_DWORD *)(v122 + 44) + 1);
                v115 = *(_DWORD *)(v122 + 44) + 1;
              }
              else
              {
                v114 = 0;
                v115 = 0;
              }
              if ( v115 >= *(_DWORD *)(v123 + 32) || !*(_QWORD *)(*(_QWORD *)(v123 + 24) + 8 * v114) )
                goto LABEL_144;
              v49 = v179;
              if ( !v179 )
              {
                ++v176;
                goto LABEL_284;
              }
              v116 = 1;
              v117 = 0;
              v118 = (v179 - 1) & (((unsigned int)v122 >> 9) ^ ((unsigned int)v122 >> 4));
              v119 = v177 + 16LL * v118;
              v120 = *(_QWORD *)v119;
              if ( v122 != *(_QWORD *)v119 )
              {
                while ( v120 != -4096 )
                {
                  if ( !v117 && v120 == -8192 )
                    v117 = v119;
                  v118 = (v179 - 1) & (v116 + v118);
                  v119 = v177 + 16LL * v118;
                  v120 = *(_QWORD *)v119;
                  if ( v122 == *(_QWORD *)v119 )
                    goto LABEL_141;
                  ++v116;
                }
                if ( !v117 )
                  v117 = v119;
                ++v176;
                v148 = v178 + 1;
                if ( 4 * ((int)v178 + 1) >= 3 * v179 )
                {
LABEL_284:
                  v49 = 2 * v179;
                  v167 = v59;
                  sub_A4A350((__int64)&v176, v49);
                  if ( !v179 )
                    goto LABEL_368;
                  v59 = v167;
                  v148 = v178 + 1;
                  v156 = (v179 - 1) & (((unsigned int)v122 >> 9) ^ ((unsigned int)v122 >> 4));
                  v117 = v177 + 16LL * v156;
                  v157 = *(_QWORD *)v117;
                  if ( v122 != *(_QWORD *)v117 )
                  {
                    v158 = 1;
                    v49 = 0;
                    while ( v157 != -4096 )
                    {
                      if ( v157 == -8192 && !v49 )
                        v49 = v117;
                      v156 = (v179 - 1) & (v158 + v156);
                      v117 = v177 + 16LL * v156;
                      v157 = *(_QWORD *)v117;
                      if ( v122 == *(_QWORD *)v117 )
                        goto LABEL_259;
                      ++v158;
                    }
                    if ( v49 )
                      v117 = v49;
                  }
                }
                else if ( v179 - HIDWORD(v178) - v148 <= v179 >> 3 )
                {
                  v168 = v59;
                  sub_A4A350((__int64)&v176, v179);
                  if ( !v179 )
                    goto LABEL_368;
                  v49 = 1;
                  v162 = (v179 - 1) & (((unsigned int)v122 >> 9) ^ ((unsigned int)v122 >> 4));
                  v59 = v168;
                  v148 = v178 + 1;
                  v163 = 0;
                  v117 = v177 + 16LL * v162;
                  v164 = *(_QWORD *)v117;
                  if ( v122 != *(_QWORD *)v117 )
                  {
                    while ( v164 != -4096 )
                    {
                      if ( !v163 && v164 == -8192 )
                        v163 = v117;
                      v162 = (v179 - 1) & (v49 + v162);
                      v117 = v177 + 16LL * v162;
                      v164 = *(_QWORD *)v117;
                      if ( v122 == *(_QWORD *)v117 )
                        goto LABEL_259;
                      v49 = (unsigned int)(v49 + 1);
                    }
                    if ( v163 )
                      v117 = v163;
                  }
                }
LABEL_259:
                LODWORD(v178) = v148;
                if ( *(_QWORD *)v117 != -4096 )
                  --HIDWORD(v178);
                *(_QWORD *)v117 = v122;
                v121 = 0;
                *(_DWORD *)(v117 + 8) = 0;
                goto LABEL_142;
              }
LABEL_141:
              v121 = *(_DWORD *)(v119 + 8);
LABEL_142:
              if ( v63 > v121 )
                v63 = v121;
LABEL_144:
              while ( 1 )
              {
                v112 = *(_QWORD *)(v112 + 8);
                if ( !v112 )
                  break;
                v113 = *(_QWORD *)(v112 + 24);
                if ( (unsigned __int8)(*(_BYTE *)v113 - 30) <= 0xAu )
                  goto LABEL_146;
              }
LABEL_52:
              if ( v63 == v59 )
              {
LABEL_53:
                v50 = v185;
                goto LABEL_54;
              }
              v49 = v179;
              if ( !v179 )
              {
                ++v176;
                goto LABEL_242;
              }
              v81 = 1;
              v82 = 0;
              v83 = (v179 - 1) & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
              v84 = (_QWORD *)(v177 + 16LL * v83);
              v85 = *v84;
              if ( v64 != *v84 )
              {
                while ( v85 != -4096 )
                {
                  if ( !v82 && v85 == -8192 )
                    v82 = v84;
                  v83 = (v179 - 1) & (v81 + v83);
                  v84 = (_QWORD *)(v177 + 16LL * v83);
                  v85 = *v84;
                  if ( v64 == *v84 )
                    goto LABEL_94;
                  ++v81;
                }
                if ( !v82 )
                  v82 = v84;
                ++v176;
                v144 = v178 + 1;
                if ( 4 * ((int)v178 + 1) >= 3 * v179 )
                {
LABEL_242:
                  v49 = 2 * v179;
                  sub_A4A350((__int64)&v176, v49);
                  if ( v179 )
                  {
                    v144 = v178 + 1;
                    v145 = (v179 - 1) & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
                    v82 = (_QWORD *)(v177 + 16LL * v145);
                    v146 = *v82;
                    if ( v64 != *v82 )
                    {
                      v147 = 1;
                      v49 = 0;
                      while ( v146 != -4096 )
                      {
                        if ( !v49 && v146 == -8192 )
                          v49 = (__int64)v82;
                        v145 = (v179 - 1) & (v147 + v145);
                        v82 = (_QWORD *)(v177 + 16LL * v145);
                        v146 = *v82;
                        if ( v64 == *v82 )
                          goto LABEL_233;
                        ++v147;
                      }
                      if ( v49 )
                        v82 = (_QWORD *)v49;
                    }
                    goto LABEL_233;
                  }
                }
                else
                {
                  if ( v179 - HIDWORD(v178) - v144 > v179 >> 3 )
                  {
LABEL_233:
                    LODWORD(v178) = v144;
                    if ( *v82 != -4096 )
                      --HIDWORD(v178);
                    *v82 = v64;
                    v86 = (unsigned int *)(v82 + 1);
                    *v86 = 0;
                    goto LABEL_95;
                  }
                  sub_A4A350((__int64)&v176, v179);
                  if ( v179 )
                  {
                    v151 = 0;
                    v152 = (v179 - 1) & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
                    v49 = 1;
                    v144 = v178 + 1;
                    v82 = (_QWORD *)(v177 + 16LL * v152);
                    v153 = *v82;
                    if ( v64 != *v82 )
                    {
                      while ( v153 != -4096 )
                      {
                        if ( v153 == -8192 && !v151 )
                          v151 = v82;
                        v152 = (v179 - 1) & (v49 + v152);
                        v82 = (_QWORD *)(v177 + 16LL * v152);
                        v153 = *v82;
                        if ( v64 == *v82 )
                          goto LABEL_233;
                        v49 = (unsigned int)(v49 + 1);
                      }
                      if ( v151 )
                        v82 = v151;
                    }
                    goto LABEL_233;
                  }
                }
LABEL_368:
                LODWORD(v178) = v178 + 1;
                BUG();
              }
LABEL_94:
              v86 = (unsigned int *)(v84 + 1);
LABEL_95:
              *v86 = v63;
              v87 = *(_QWORD *)(v64 + 48) & 0xFFFFFFFFFFFFFFF8LL;
              if ( v87 == v64 + 48 )
                goto LABEL_53;
              if ( !v87 )
                BUG();
              v88 = v87 - 24;
              if ( (unsigned int)*(unsigned __int8 *)(v87 - 24) - 30 > 0xA )
                goto LABEL_53;
              v89 = sub_B46E30(v88);
              v50 = v185;
              v90 = v89;
              if ( v89 )
              {
                v91 = 0;
                do
                {
                  v49 = v91;
                  v92 = sub_B46EC0(v88, v91);
                  v94 = v50;
                  v95 = v50 + 1LL;
                  if ( v95 > HIDWORD(v185) )
                  {
                    v49 = (__int64)v186;
                    v166 = v92;
                    sub_C8D5F0((__int64)&v184, v186, v95, 8u, v92, v93);
                    v94 = (unsigned int)v185;
                    v92 = v166;
                  }
                  ++v91;
                  *(_QWORD *)&v184[8 * v94] = v92;
                  v50 = v185 + 1;
                  LODWORD(v185) = v185 + 1;
                }
                while ( v90 != v91 );
              }
LABEL_54:
              if ( !v50 )
              {
                v3 = v171;
                goto LABEL_113;
              }
            }
            ++v176;
            goto LABEL_57;
          }
LABEL_113:
          if ( (_DWORD)v178 )
          {
            v49 = v177;
            v124 = v177 + 16LL * v179;
            if ( v177 != v124 )
            {
              while ( 1 )
              {
                v125 = *(_QWORD *)v49;
                v126 = v49;
                if ( *(_QWORD *)v49 != -8192 && v125 != -4096 )
                  break;
                v49 += 16;
                if ( v124 == v49 )
                  goto LABEL_114;
              }
              if ( v124 != v49 )
              {
                do
                {
                  v49 = *(unsigned int *)(v3 + 136);
                  v127 = *(_QWORD *)(v3 + 120);
                  if ( (_DWORD)v49 )
                  {
                    v128 = (v49 - 1) & (((unsigned int)v125 >> 9) ^ ((unsigned int)v125 >> 4));
                    v129 = (__int64 *)(v127 + 16LL * v128);
                    v130 = *v129;
                    if ( *v129 == v125 )
                    {
LABEL_159:
                      v49 = v127 + 16 * v49;
                      if ( v129 != (__int64 *)v49 )
                      {
                        v49 = *(unsigned int *)(v126 + 8);
                        *(_DWORD *)(v129[1] + 16) = v49;
                      }
                    }
                    else
                    {
                      v131 = 1;
                      while ( v130 != -4096 )
                      {
                        v132 = v131 + 1;
                        v128 = (v49 - 1) & (v131 + v128);
                        v129 = (__int64 *)(v127 + 16LL * v128);
                        v130 = *v129;
                        if ( v125 == *v129 )
                          goto LABEL_159;
                        v131 = v132;
                      }
                    }
                  }
                  v126 += 16;
                  if ( v126 == v124 )
                    break;
                  while ( 1 )
                  {
                    v125 = *(_QWORD *)v126;
                    if ( *(_QWORD *)v126 != -8192 && v125 != -4096 )
                      break;
                    v126 += 16;
                    if ( v124 == v126 )
                      goto LABEL_114;
                  }
                }
                while ( v126 != v124 );
              }
            }
          }
LABEL_114:
          if ( v184 != v186 )
            _libc_free(v184, v49);
          v75 = v181;
          v76 = 16LL * v183;
          goto LABEL_79;
        }
      }
      else
      {
        v159 = 1;
        while ( v53 != -4096 )
        {
          v160 = v159 + 1;
          v51 = (v183 - 1) & (v159 + v51);
          v52 = (__int64 *)(v181 + 16LL * v51);
          v53 = *v52;
          if ( v175 == *v52 )
            goto LABEL_43;
          v159 = v160;
        }
      }
    }
    v140 = *(_DWORD *)(v3 + 40);
    v49 = (__int64)&v175;
    *(_DWORD *)sub_FD3730((__int64)&v180, &v175) = v140;
    v50 = v185;
    goto LABEL_44;
  }
  do
  {
    while ( 1 )
    {
      v29 = *(_QWORD *)(v3 + 16);
      if ( v28 )
      {
        v30 = v28 - 24;
        v31 = (unsigned int)(*(_DWORD *)(v28 + 20) + 1);
        v32 = *(_DWORD *)(v28 + 20) + 1;
      }
      else
      {
        v30 = 0;
        v31 = 0;
        v32 = 0;
      }
      if ( v32 >= *(_DWORD *)(v29 + 32) || !*(_QWORD *)(*(_QWORD *)(v29 + 24) + 8 * v31) )
        goto LABEL_35;
      v4 = v179;
      if ( !v179 )
      {
        ++v176;
LABEL_206:
        v4 = 2 * v179;
        sub_A4A350((__int64)&v176, v4);
        if ( !v179 )
          goto LABEL_368;
        v26 = v179 - 1;
        v137 = v26 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
        v25 = v177 + 16LL * v137;
        v138 = *(_QWORD *)v25;
        v136 = v178 + 1;
        if ( v30 != *(_QWORD *)v25 )
        {
          v139 = 1;
          v4 = 0;
          while ( v138 != -4096 )
          {
            if ( !v4 && v138 == -8192 )
              v4 = v25;
            v137 = v26 & (v139 + v137);
            v25 = v177 + 16LL * v137;
            v138 = *(_QWORD *)v25;
            if ( *(_QWORD *)v25 == v30 )
              goto LABEL_201;
            ++v139;
          }
          if ( v4 )
            v25 = v4;
        }
        goto LABEL_201;
      }
      v33 = 1;
      v25 = 0;
      v34 = (v179 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
      v35 = (_QWORD *)(v177 + 16LL * v34);
      v36 = *v35;
      if ( *v35 == v30 )
      {
LABEL_24:
        v37 = v35 + 1;
        goto LABEL_25;
      }
      while ( v36 != -4096 )
      {
        if ( v36 == -8192 && !v25 )
          v25 = (__int64)v35;
        v26 = (unsigned int)(v33 + 1);
        v34 = (v179 - 1) & (v33 + v34);
        v35 = (_QWORD *)(v177 + 16LL * v34);
        v36 = *v35;
        if ( *v35 == v30 )
          goto LABEL_24;
        ++v33;
      }
      if ( !v25 )
        v25 = (__int64)v35;
      ++v176;
      v136 = v178 + 1;
      if ( 4 * ((int)v178 + 1) >= 3 * v179 )
        goto LABEL_206;
      if ( v179 - HIDWORD(v178) - v136 <= v179 >> 3 )
      {
        sub_A4A350((__int64)&v176, v179);
        if ( !v179 )
          goto LABEL_368;
        v4 = v177;
        v26 = 0;
        v141 = (v179 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
        v136 = v178 + 1;
        v142 = 1;
        v25 = v177 + 16LL * v141;
        v143 = *(_QWORD *)v25;
        if ( v30 != *(_QWORD *)v25 )
        {
          while ( v143 != -4096 )
          {
            if ( v143 == -8192 && !v26 )
              v26 = v25;
            v141 = (v179 - 1) & (v142 + v141);
            v25 = v177 + 16LL * v141;
            v143 = *(_QWORD *)v25;
            if ( *(_QWORD *)v25 == v30 )
              goto LABEL_201;
            ++v142;
          }
          if ( v26 )
            v25 = v26;
        }
      }
LABEL_201:
      LODWORD(v178) = v136;
      if ( *(_QWORD *)v25 != -4096 )
        --HIDWORD(v178);
      *(_QWORD *)v25 = v30;
      v37 = (_DWORD *)(v25 + 8);
      *(_DWORD *)(v25 + 8) = 0;
LABEL_25:
      *v37 = -1;
      v38 = (_QWORD *)(*(_QWORD *)(v30 + 48) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (_QWORD *)(v30 + 48) != v38 )
        break;
LABEL_35:
      v28 = *(_QWORD *)(v28 + 8);
      if ( v170 == v28 )
        goto LABEL_36;
    }
    do
    {
      v4 = (__int64)(v38 - 3);
      if ( !v38 )
        v4 = 0;
      v175 = sub_FCDCB0(v3, v4);
      if ( BYTE4(v175) )
      {
        v39 = v175;
        if ( (unsigned int)(v175 - 24) <= 0xE8 && (v175 & 7) == 0 )
        {
          v4 = v183;
          if ( v183 )
          {
            v40 = 1;
            v26 = v181;
            v41 = 0;
            v42 = (v183 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
            v43 = (__int64 *)(v181 + 16LL * v42);
            v25 = *v43;
            if ( v30 == *v43 )
            {
LABEL_33:
              v44 = v43 + 1;
LABEL_34:
              *v44 = v39;
              *(_BYTE *)(v3 + 48) = 1;
              goto LABEL_35;
            }
            while ( v25 != -4096 )
            {
              if ( v25 == -8192 && !v41 )
                v41 = v43;
              v42 = (v183 - 1) & (v40 + v42);
              v43 = (__int64 *)(v181 + 16LL * v42);
              v25 = *v43;
              if ( *v43 == v30 )
                goto LABEL_33;
              ++v40;
            }
            if ( !v41 )
              v41 = v43;
            ++v180;
            v133 = v182 + 1;
            if ( 4 * ((int)v182 + 1) < 3 * v183 )
            {
              v25 = v183 >> 3;
              if ( v183 - HIDWORD(v182) - v133 <= (unsigned int)v25 )
              {
                sub_A4A350((__int64)&v180, v183);
                if ( !v183 )
                {
LABEL_369:
                  LODWORD(v182) = v182 + 1;
                  BUG();
                }
                v26 = v183 - 1;
                v25 = 0;
                v154 = 1;
                LODWORD(v155) = v26 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
                v133 = v182 + 1;
                v39 = v175;
                v41 = (__int64 *)(v181 + 16LL * (unsigned int)v155);
                v4 = *v41;
                if ( v30 != *v41 )
                {
                  while ( v4 != -4096 )
                  {
                    if ( v4 == -8192 && !v25 )
                      v25 = (__int64)v41;
                    v155 = (unsigned int)v26 & ((_DWORD)v155 + v154);
                    v41 = (__int64 *)(v181 + 16 * v155);
                    v4 = *v41;
                    if ( *v41 == v30 )
                      goto LABEL_182;
                    ++v154;
                  }
                  if ( v25 )
                    v41 = (__int64 *)v25;
                }
              }
              goto LABEL_182;
            }
          }
          else
          {
            ++v180;
          }
          v4 = 2 * v183;
          sub_A4A350((__int64)&v180, v4);
          if ( !v183 )
            goto LABEL_369;
          v26 = v183 - 1;
          v133 = v182 + 1;
          v39 = v175;
          v149 = v26 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
          v41 = (__int64 *)(v181 + 16LL * v149);
          v150 = *v41;
          if ( v30 != *v41 )
          {
            v25 = 1;
            v4 = 0;
            while ( v150 != -4096 )
            {
              if ( v150 == -8192 && !v4 )
                v4 = (__int64)v41;
              v165 = v25 + 1;
              v25 = v149 + (unsigned int)v25;
              v149 = v26 & v25;
              v41 = (__int64 *)(v181 + 16LL * ((unsigned int)v26 & (unsigned int)v25));
              v150 = *v41;
              if ( *v41 == v30 )
                goto LABEL_182;
              v25 = v165;
            }
            if ( v4 )
              v41 = (__int64 *)v4;
          }
LABEL_182:
          LODWORD(v182) = v133;
          if ( *v41 != -4096 )
            --HIDWORD(v182);
          *v41 = v30;
          v44 = v41 + 1;
          *v44 = 0;
          goto LABEL_34;
        }
      }
      v38 = (_QWORD *)(*v38 & 0xFFFFFFFFFFFFFFF8LL);
    }
    while ( (_QWORD *)(v30 + 48) != v38 );
    v28 = *(_QWORD *)(v28 + 8);
  }
  while ( v170 != v28 );
LABEL_36:
  if ( *(_BYTE *)(v3 + 48) )
  {
    v45 = HIDWORD(v185);
    v170 = *(_QWORD *)(*(_QWORD *)v3 + 80LL);
    v46 = (unsigned int)v185;
    v47 = (unsigned int)v185 + 1LL;
    if ( v170 )
      goto LABEL_38;
    v48 = 0;
    goto LABEL_39;
  }
  if ( v184 != v186 )
    _libc_free(v184, v4);
  v75 = v181;
  v76 = 16LL * v183;
LABEL_79:
  sub_C7D6A0(v75, v76, 8);
  return sub_C7D6A0(v177, 16LL * v179, 8);
}
