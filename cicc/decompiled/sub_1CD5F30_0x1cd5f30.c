// Function: sub_1CD5F30
// Address: 0x1cd5f30
//
void __fastcall sub_1CD5F30(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 **a4,
        __int64 a5,
        __int64 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14)
{
  __int64 v14; // r15
  __int64 v16; // r15
  __int64 v17; // rsi
  __int64 *v18; // rbx
  __int64 v19; // r13
  __int64 *v20; // rax
  const char *v21; // r12
  __int64 *v22; // rax
  unsigned int v23; // edx
  __int64 *v24; // rax
  __int64 v25; // r13
  __int64 *v26; // rax
  __int64 *v27; // r10
  __int64 v28; // r9
  unsigned int v29; // r15d
  _QWORD *v30; // r14
  __int64 v31; // rax
  _QWORD *v32; // r12
  char v33; // di
  unsigned int v34; // esi
  __int64 v35; // rdx
  __int64 v36; // rax
  _QWORD *v37; // rcx
  _QWORD *v38; // rdx
  __int64 v39; // r13
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // r11
  __int64 v43; // rdx
  unsigned __int64 v44; // r11
  _QWORD *v45; // r13
  unsigned __int64 v46; // rax
  __int64 v47; // rax
  _QWORD *v48; // rdx
  __int64 v49; // rax
  unsigned int v50; // ebx
  int v51; // eax
  bool v52; // bl
  __int64 *v53; // rdi
  __int64 *v54; // r12
  __int64 v55; // rbx
  __int64 v56; // r14
  __int64 v57; // rdx
  __int64 v58; // rax
  double v59; // xmm4_8
  double v60; // xmm5_8
  __int64 v61; // r13
  __int64 **v62; // rax
  __int64 v63; // rax
  double v64; // xmm4_8
  double v65; // xmm5_8
  unsigned int v66; // eax
  __int64 v67; // r13
  __int64 v68; // rbx
  __int64 *v69; // rax
  __int64 **v70; // rax
  __int64 *v71; // rbx
  __int64 v72; // r8
  __int64 v73; // rax
  __int64 v74; // rdx
  __int64 v75; // r12
  char v76; // r9
  unsigned int v77; // edi
  __int64 v78; // rcx
  __int64 v79; // rdx
  __int64 v80; // rsi
  __int64 v81; // rsi
  int v82; // r9d
  int v83; // r11d
  __int64 v84; // rcx
  unsigned int v85; // r13d
  __int64 *v86; // rdx
  __int64 v87; // rdi
  __int64 v88; // r10
  __int64 v89; // r14
  __int64 v90; // rax
  __int64 v91; // r13
  __int64 v92; // rax
  double v93; // xmm4_8
  double v94; // xmm5_8
  double v95; // xmm4_8
  double v96; // xmm5_8
  int v97; // esi
  __int64 v98; // rcx
  unsigned int v99; // edx
  __int64 *v100; // rax
  __int64 v101; // r8
  __int64 *v102; // r12
  __int64 v103; // rsi
  __int64 *v104; // r14
  __int64 *v105; // rax
  __int64 v106; // rbx
  __int64 v107; // r14
  unsigned int v108; // edx
  __int64 *v109; // rax
  __int64 v110; // rax
  __int64 *v111; // rdi
  unsigned int v112; // r12d
  __int64 *v113; // r13
  __int64 v114; // rax
  double v115; // xmm4_8
  double v116; // xmm5_8
  __int64 *v117; // rax
  __int64 v118; // rax
  double v119; // xmm4_8
  double v120; // xmm5_8
  _QWORD *v121; // r14
  __int64 v122; // rbx
  __int64 v123; // rax
  __int64 v124; // rdx
  unsigned int v125; // ebx
  unsigned __int64 v126; // rdx
  __int64 v127; // rax
  __int64 *v128; // r10
  int v129; // r13d
  int v130; // eax
  __int64 v131; // rdi
  int v132; // esi
  unsigned __int64 v133; // rdx
  __int64 v134; // rax
  __int64 *v135; // rdi
  int v136; // r14d
  int v137; // edx
  int v138; // r14d
  __int64 *v139; // r11
  bool v140; // [rsp+7h] [rbp-109h]
  __int64 v141; // [rsp+10h] [rbp-100h]
  __int64 v142; // [rsp+18h] [rbp-F8h]
  __int64 v143; // [rsp+20h] [rbp-F0h]
  __int64 *v144; // [rsp+20h] [rbp-F0h]
  __int64 *v145; // [rsp+28h] [rbp-E8h]
  _QWORD *v146; // [rsp+28h] [rbp-E8h]
  __int64 v147; // [rsp+30h] [rbp-E0h]
  __int64 v148; // [rsp+30h] [rbp-E0h]
  __int64 *v150; // [rsp+38h] [rbp-D8h]
  __int64 v151; // [rsp+40h] [rbp-D0h]
  __int64 v152; // [rsp+40h] [rbp-D0h]
  bool v153; // [rsp+48h] [rbp-C8h]
  char v154; // [rsp+48h] [rbp-C8h]
  __int64 *v159; // [rsp+68h] [rbp-A8h]
  __int64 v160; // [rsp+68h] [rbp-A8h]
  __int64 *v161; // [rsp+70h] [rbp-A0h]
  __int64 **v162; // [rsp+78h] [rbp-98h]
  _QWORD *v163; // [rsp+78h] [rbp-98h]
  __int64 v164; // [rsp+80h] [rbp-90h]
  _QWORD *v165; // [rsp+80h] [rbp-90h]
  _QWORD *v166; // [rsp+80h] [rbp-90h]
  __int64 v167; // [rsp+80h] [rbp-90h]
  __int64 v168; // [rsp+80h] [rbp-90h]
  __int64 *v169; // [rsp+80h] [rbp-90h]
  __int64 *v170; // [rsp+88h] [rbp-88h]
  __int64 v171; // [rsp+88h] [rbp-88h]
  __int64 *v172; // [rsp+88h] [rbp-88h]
  char v173; // [rsp+88h] [rbp-88h]
  __int64 v174; // [rsp+88h] [rbp-88h]
  __int64 *v175; // [rsp+90h] [rbp-80h]
  unsigned int v176; // [rsp+90h] [rbp-80h]
  __int64 *v177; // [rsp+90h] [rbp-80h]
  __int64 *v178; // [rsp+98h] [rbp-78h]
  __int64 *v179; // [rsp+98h] [rbp-78h]
  __int64 v180; // [rsp+98h] [rbp-78h]
  __int64 v181; // [rsp+A8h] [rbp-68h] BYREF
  __int64 v182; // [rsp+B0h] [rbp-60h] BYREF
  __int64 *v183; // [rsp+B8h] [rbp-58h] BYREF
  const char *v184; // [rsp+C0h] [rbp-50h] BYREF
  __int64 v185; // [rsp+C8h] [rbp-48h]
  __int64 v186; // [rsp+D0h] [rbp-40h]
  unsigned int v187; // [rsp+D8h] [rbp-38h]

  v14 = a2;
  v161 = a4[1];
  v178 = *a4;
  if ( *a4 != v161 )
  {
    v16 = a5;
    do
    {
      v21 = (const char *)*v178;
      if ( *(_BYTE *)(*v178 + 16) == 13 )
      {
        v184 = (const char *)*v178;
        v22 = sub_1CD51A0(v16, (__int64 *)&v184);
        if ( *(_QWORD *)v22[1] != *(_QWORD *)(v22[1] + 8) )
        {
          v23 = *((_DWORD *)v21 + 8);
          v24 = (__int64 *)*((_QWORD *)v21 + 3);
          v17 = v23 <= 0x40
              ? (__int64)((_QWORD)v24 << (64 - (unsigned __int8)v23)) >> (64 - (unsigned __int8)v23)
              : *v24;
          v18 = (__int64 *)sub_159C470(*(_QWORD *)v21, -v17, 1u);
          v19 = *(_QWORD *)(v16 + 8) + 16LL * *(unsigned int *)(v16 + 24);
          sub_1CD1E70((__int64 **)&v184, (__int64 *)v16, (__int64)v18);
          if ( v19 != v186 )
          {
            v183 = v18;
            v20 = sub_1CD51A0(v16, (__int64 *)&v183);
            if ( *(_QWORD *)v20[1] != *(_QWORD *)(v20[1] + 8) )
            {
              v184 = v21;
              v25 = sub_1CD51A0(v16, (__int64 *)&v184)[1];
              v184 = (const char *)v18;
              v26 = sub_1CD51A0(v16, (__int64 *)&v184);
              v153 = 0;
              v27 = *(__int64 **)v25;
              v147 = v26[1];
              v175 = *(__int64 **)(v25 + 8);
              if ( *(__int64 **)v25 != v175 )
              {
                v151 = v16;
                v28 = a3;
                v29 = 0;
                v30 = (_QWORD *)v25;
                do
                {
                  v31 = 0x2FFFFFFFDLL;
                  v32 = (_QWORD *)*v27;
                  v33 = *(_BYTE *)(*v27 + 23) & 0x40;
                  v34 = *(_DWORD *)(*v27 + 20) & 0xFFFFFFF;
                  if ( v34 )
                  {
                    v35 = 24LL * *((unsigned int *)v32 + 14) + 8;
                    v36 = 0;
                    do
                    {
                      v37 = &v32[-3 * v34];
                      if ( v33 )
                        v37 = (_QWORD *)*(v32 - 1);
                      if ( v28 == *(_QWORD *)((char *)v37 + v35) )
                      {
                        v31 = 3 * v36;
                        goto LABEL_20;
                      }
                      ++v36;
                      v35 += 8;
                    }
                    while ( v34 != (_DWORD)v36 );
                    v31 = 0x2FFFFFFFDLL;
                  }
LABEL_20:
                  if ( v33 )
                    v38 = (_QWORD *)*(v32 - 1);
                  else
                    v38 = &v32[-3 * v34];
                  v39 = v38[v31];
                  v40 = *(_QWORD *)(v39 + 8);
                  if ( !v40 )
                    goto LABEL_25;
                  if ( *(_QWORD *)(v40 + 8) )
                    goto LABEL_25;
                  v164 = v28;
                  v170 = v27;
                  v41 = sub_1455EB0(*v27, a2);
                  v27 = v170;
                  v28 = v164;
                  v42 = v41;
                  if ( *(_BYTE *)(v41 + 16) != 37 )
                    goto LABEL_25;
                  v48 = (*(_BYTE *)(v41 + 23) & 0x40) != 0
                      ? *(_QWORD **)(v41 - 8)
                      : (_QWORD *)(v41 - 24LL * (*(_DWORD *)(v41 + 20) & 0xFFFFFFF));
                  v49 = *v48;
                  if ( *(_BYTE *)(*v48 + 16LL) != 13 )
                    goto LABEL_25;
                  v50 = *(_DWORD *)(v49 + 32);
                  if ( v50 <= 0x40 )
                  {
                    v52 = *(_QWORD *)(v49 + 24) == 0;
                  }
                  else
                  {
                    v143 = v164;
                    v145 = v170;
                    v165 = v48;
                    v171 = v42;
                    v51 = sub_16A57B0(v49 + 24);
                    v42 = v171;
                    v48 = v165;
                    v27 = v145;
                    v28 = v143;
                    v52 = v50 == v51;
                  }
                  if ( v52 && (v53 = *(__int64 **)(v147 + 8), v53 != *(__int64 **)v147) )
                  {
                    v146 = v32;
                    v54 = *(__int64 **)v147;
                    v140 = v52;
                    v55 = v48[3];
                    v166 = v30;
                    v141 = v42;
                    v144 = v27;
                    v142 = v28;
                    while ( 1 )
                    {
                      v56 = *v54;
                      if ( v55 == sub_1455EB0(*v54, a2) )
                        break;
                      if ( v53 == ++v54 )
                      {
                        v30 = v166;
                        v32 = v146;
                        v27 = v144;
                        v28 = v142;
                        goto LABEL_25;
                      }
                    }
                    v57 = v56;
                    v32 = v146;
                    v30 = v166;
                    v27 = v144;
                    v28 = v142;
                    if ( !v57 )
                      goto LABEL_25;
                    v167 = v57;
                    v58 = sub_1599EF0(*(__int64 ***)v39);
                    sub_164D160(v39, v58, a7, a8, a9, a10, v59, v60, a13, a14);
                    sub_15F20C0((_QWORD *)v39);
                    v61 = sub_157ED20(a1);
                    LOWORD(v186) = 259;
                    v184 = "oppositeIV";
                    v62 = (__int64 **)sub_13CF970(v141);
                    v63 = sub_15FB440(13, *v62, v167, (__int64)&v184, v61);
                    sub_164D160((__int64)v146, v63, a7, a8, a9, a10, v64, v65, a13, a14);
                    sub_15F20C0(v146);
                    v153 = v140;
                    v28 = v142;
                    v27 = v144;
                  }
                  else
                  {
LABEL_25:
                    v43 = v29++;
                    *(_QWORD *)(*v30 + 8 * v43) = v32;
                  }
                  ++v27;
                }
                while ( v175 != v27 );
                v44 = v29;
                v45 = v30;
                a3 = v28;
                v16 = v151;
                if ( v153 )
                {
                  v46 = (__int64)(v45[1] - *v45) >> 3;
                  if ( v44 > v46 )
                  {
                    sub_1CD22D0((__int64)v45, v44 - v46);
                  }
                  else if ( v44 < v46 )
                  {
                    v47 = *v45 + 8 * v44;
                    if ( v45[1] != v47 )
                      v45[1] = v47;
                  }
                }
              }
            }
          }
        }
      }
      ++v178;
    }
    while ( v161 != v178 );
    v14 = a2;
  }
  v66 = dword_4FBFB40;
  if ( (unsigned int)dword_4FBFB40 > 2 )
  {
    v159 = a4[1];
    if ( v159 == *a4 )
      return;
    v168 = a3;
    v179 = *a4;
    v67 = a5;
    do
    {
      v68 = *(_QWORD *)(v67 + 8) + 16LL * *(unsigned int *)(v67 + 24);
      v181 = *v179;
      sub_1CD1E70((__int64 **)&v184, (__int64 *)v67, v181);
      if ( v186 != v68 )
      {
        v69 = sub_1CD52D0(v67, &v181);
        if ( *(_QWORD *)(v69[1] + 8) - *(_QWORD *)v69[1] > 8u )
        {
          v70 = (__int64 **)sub_1CD52D0(v67, &v181)[1];
          v184 = 0;
          v185 = 0;
          v186 = 0;
          v187 = 0;
          v71 = *v70;
          v162 = v70;
          v172 = v70[1];
          if ( v172 != *v70 )
          {
            v154 = 0;
            v72 = 0;
            v73 = 0;
            v176 = 0;
            v152 = v67;
            while ( 1 )
            {
              v74 = 0x17FFFFFFE8LL;
              v75 = *v71;
              v76 = *(_BYTE *)(*v71 + 23) & 0x40;
              v77 = *(_DWORD *)(*v71 + 20) & 0xFFFFFFF;
              if ( v77 )
              {
                v78 = 24LL * *(unsigned int *)(v75 + 56) + 8;
                v79 = 0;
                do
                {
                  v80 = v75 - 24LL * v77;
                  if ( v76 )
                    v80 = *(_QWORD *)(v75 - 8);
                  if ( v14 == *(_QWORD *)(v80 + v78) )
                  {
                    v74 = 24 * v79;
                    goto LABEL_62;
                  }
                  ++v79;
                  v78 += 8;
                }
                while ( v77 != (_DWORD)v79 );
                v74 = 0x17FFFFFFE8LL;
              }
LABEL_62:
              if ( v76 )
              {
                v81 = *(_QWORD *)(*(_QWORD *)(v75 - 8) + v74);
                v182 = v81;
                if ( !(_DWORD)v72 )
                  goto LABEL_116;
              }
              else
              {
                v81 = *(_QWORD *)(v75 - 24LL * v77 + v74);
                v182 = v81;
                if ( !(_DWORD)v72 )
                {
LABEL_116:
                  ++v184;
                  goto LABEL_117;
                }
              }
              v82 = v72 - 1;
              v83 = 1;
              LODWORD(v84) = (v72 - 1) & (((unsigned int)v81 >> 9) ^ ((unsigned int)v81 >> 4));
              v85 = v84;
              v86 = (__int64 *)(v73 + 16LL * (unsigned int)v84);
              v87 = *v86;
              v88 = *v86;
              if ( v81 != *v86 )
              {
                while ( v88 != -8 )
                {
                  v138 = v83 + 1;
                  v85 = v82 & (v83 + v85);
                  v139 = (__int64 *)(v73 + 16LL * v85);
                  v88 = *v139;
                  if ( v81 == *v139 )
                  {
                    if ( v139 != (__int64 *)(v73 + 16LL * (unsigned int)v72) )
                      goto LABEL_66;
                    v128 = (__int64 *)(v73 + 16LL * (unsigned int)v84);
                    goto LABEL_101;
                  }
                  v83 = v138;
                }
                v84 = v82 & (((unsigned int)v81 >> 4) ^ ((unsigned int)v81 >> 9));
                v128 = (__int64 *)(v73 + 16 * v84);
                v87 = *v128;
                if ( v81 == *v128 )
                {
LABEL_155:
                  v86 = v128;
                  goto LABEL_110;
                }
LABEL_101:
                v129 = 1;
                v86 = 0;
                while ( v87 != -8 )
                {
                  if ( v87 == -16 && !v86 )
                    v86 = v128;
                  v84 = v82 & (unsigned int)(v84 + v129);
                  v128 = (__int64 *)(v73 + 16 * v84);
                  v87 = *v128;
                  if ( v81 == *v128 )
                    goto LABEL_155;
                  ++v129;
                }
                if ( !v86 )
                  v86 = v128;
                ++v184;
                v130 = v186 + 1;
                if ( 4 * ((int)v186 + 1) >= (unsigned int)(3 * v72) )
                {
LABEL_117:
                  v132 = 2 * v72;
                }
                else
                {
                  if ( (int)v72 - (v130 + HIDWORD(v186)) > (unsigned int)v72 >> 3 )
                    goto LABEL_107;
                  v132 = v72;
                }
                sub_1CD5DA0((__int64)&v184, v132);
                sub_1CD35C0((__int64)&v184, &v182, &v183);
                v86 = v183;
                v81 = v182;
                v130 = v186 + 1;
LABEL_107:
                LODWORD(v186) = v130;
                if ( *v86 != -8 )
                  --HIDWORD(v186);
                *v86 = v81;
                v86[1] = 0;
LABEL_110:
                v86[1] = v75;
                goto LABEL_111;
              }
              if ( v86 == (__int64 *)(16 * v72 + v73) )
                goto LABEL_110;
LABEL_66:
              v89 = sub_1455EB0(v75, v168);
              v90 = *(_QWORD *)(v89 + 8);
              if ( v90 )
              {
                v91 = *(_QWORD *)(v90 + 8);
                if ( !v91 )
                {
                  v92 = sub_1599EF0(*(__int64 ***)v89);
                  sub_164D160(v89, v92, a7, a8, a9, a10, v93, v94, a13, a14);
                  sub_15F20C0((_QWORD *)v89);
                  v97 = v187;
                  if ( v187 )
                  {
                    v98 = v182;
                    v99 = (v187 - 1) & (((unsigned int)v182 >> 9) ^ ((unsigned int)v182 >> 4));
                    v100 = (__int64 *)(v185 + 16LL * v99);
                    v101 = *v100;
                    if ( v182 == *v100 )
                    {
LABEL_70:
                      v91 = v100[1];
                      goto LABEL_71;
                    }
                    v135 = 0;
                    v136 = 1;
                    while ( v101 != -8 )
                    {
                      if ( !v135 && v101 == -16 )
                        v135 = v100;
                      v99 = (v187 - 1) & (v136 + v99);
                      v100 = (__int64 *)(v185 + 16LL * v99);
                      v101 = *v100;
                      if ( v182 == *v100 )
                        goto LABEL_70;
                      ++v136;
                    }
                    if ( !v135 )
                      v135 = v100;
                    ++v184;
                    v137 = v186 + 1;
                    if ( 4 * ((int)v186 + 1) < 3 * v187 )
                    {
                      if ( v187 - HIDWORD(v186) - v137 > v187 >> 3 )
                      {
LABEL_134:
                        LODWORD(v186) = v137;
                        if ( *v135 != -8 )
                          --HIDWORD(v186);
                        *v135 = v98;
                        v135[1] = 0;
LABEL_71:
                        ++v71;
                        sub_164D160(v75, v91, a7, a8, a9, a10, v95, v96, a13, a14);
                        sub_15F20C0((_QWORD *)v75);
                        v154 = 1;
                        if ( v172 == v71 )
                          goto LABEL_112;
                        goto LABEL_72;
                      }
LABEL_139:
                      sub_1CD5DA0((__int64)&v184, v97);
                      sub_1CD35C0((__int64)&v184, &v182, &v183);
                      v135 = v183;
                      v98 = v182;
                      v137 = v186 + 1;
                      goto LABEL_134;
                    }
                  }
                  else
                  {
                    ++v184;
                  }
                  v97 = 2 * v187;
                  goto LABEL_139;
                }
              }
LABEL_111:
              ++v71;
              (*v162)[v176++] = v75;
              if ( v172 == v71 )
              {
LABEL_112:
                v67 = v152;
                if ( !v154 )
                  goto LABEL_113;
                v133 = v162[1] - *v162;
                if ( v176 > v133 )
                {
                  sub_1CD22D0((__int64)v162, v176 - v133);
                }
                else if ( v176 < v133 )
                {
                  v134 = (__int64)&(*v162)[v176];
                  if ( v162[1] != (__int64 *)v134 )
                  {
                    v162[1] = (__int64 *)v134;
                    v131 = v185;
                    goto LABEL_114;
                  }
                }
LABEL_113:
                v131 = v185;
                goto LABEL_114;
              }
LABEL_72:
              v73 = v185;
              v72 = v187;
            }
          }
          v131 = 0;
LABEL_114:
          j___libc_free_0(v131);
        }
      }
      ++v179;
    }
    while ( v159 != v179 );
    a3 = v168;
    v66 = dword_4FBFB40;
  }
  if ( v66 > 1 )
  {
    v177 = a4[1];
    if ( *a4 != v177 )
    {
      v102 = *a4;
      v148 = a3;
      do
      {
        v106 = *v102;
        if ( *(_BYTE *)(*v102 + 16) == 13 )
        {
          v107 = *(_QWORD *)(a6 + 8) + 16LL * *(unsigned int *)(a6 + 24);
          sub_1CD1F20((__int64 **)&v184, (__int64 *)a6, *v102);
          if ( v186 != v107 )
          {
            v108 = *(_DWORD *)(v106 + 32);
            v109 = *(__int64 **)(v106 + 24);
            v103 = v108 <= 0x40
                 ? (__int64)((_QWORD)v109 << (64 - (unsigned __int8)v108)) >> (64 - (unsigned __int8)v108)
                 : *v109;
            v104 = (__int64 *)sub_159C470(*(_QWORD *)v106, -v103, 1u);
            v180 = *(_QWORD *)(a5 + 8) + 16LL * *(unsigned int *)(a5 + 24);
            sub_1CD1E70((__int64 **)&v184, (__int64 *)a5, (__int64)v104);
            if ( v186 != v180 )
            {
              v183 = v104;
              v105 = sub_1CD51A0(a5, (__int64 *)&v183);
              if ( *(_QWORD *)(v105[1] + 8) != *(_QWORD *)v105[1] )
              {
                v184 = (const char *)v104;
                v173 = 0;
                v110 = sub_1CD51A0(a5, (__int64 *)&v184)[1];
                v111 = *(__int64 **)(v110 + 8);
                v163 = (_QWORD *)v110;
                if ( v111 != *(__int64 **)v110 )
                {
                  v150 = v102;
                  v112 = 0;
                  v113 = *(__int64 **)v110;
                  v160 = v106;
                  do
                  {
                    while ( 1 )
                    {
                      v121 = (_QWORD *)*v113;
                      v122 = sub_1455EB0(*v113, v148);
                      v123 = *(_QWORD *)(v122 + 8);
                      if ( !v123 || *(_QWORD *)(v123 + 8) )
                        break;
                      ++v113;
                      v169 = (__int64 *)sub_1455EB0((__int64)v121, v14);
                      v114 = sub_1599EF0(*(__int64 ***)v122);
                      sub_164D160(v122, v114, a7, a8, a9, a10, v115, v116, a13, a14);
                      sub_15F20C0((_QWORD *)v122);
                      v174 = sub_157ED20(a1);
                      v184 = "oppositeIV";
                      LOWORD(v186) = 259;
                      v183 = (__int64 *)v160;
                      v117 = sub_1CD5400(a6, (__int64 *)&v183);
                      v118 = sub_15FB440(13, v169, v117[1], (__int64)&v184, v174);
                      sub_164D160((__int64)v121, v118, a7, a8, a9, a10, v119, v120, a13, a14);
                      sub_15F20C0(v121);
                      v173 = 1;
                      if ( v111 == v113 )
                        goto LABEL_93;
                    }
                    v124 = v112;
                    ++v113;
                    ++v112;
                    *(_QWORD *)(*v163 + 8 * v124) = v121;
                  }
                  while ( v111 != v113 );
LABEL_93:
                  v125 = v112;
                  v102 = v150;
                  if ( v173 )
                  {
                    v126 = (__int64)(v163[1] - *v163) >> 3;
                    if ( v125 > v126 )
                    {
                      sub_1CD22D0((__int64)v163, v125 - v126);
                    }
                    else if ( v125 < v126 )
                    {
                      v127 = *v163 + 8LL * v125;
                      if ( v163[1] != v127 )
                        v163[1] = v127;
                    }
                  }
                }
              }
            }
          }
        }
        ++v102;
      }
      while ( v177 != v102 );
    }
  }
}
