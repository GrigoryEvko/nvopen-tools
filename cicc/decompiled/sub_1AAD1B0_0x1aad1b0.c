// Function: sub_1AAD1B0
// Address: 0x1aad1b0
//
__int64 __fastcall sub_1AAD1B0(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  _QWORD *v11; // rax
  __int64 v12; // r14
  __int64 v13; // rdi
  unsigned __int64 v14; // rax
  _QWORD *v15; // r8
  unsigned __int64 v16; // r13
  int v17; // eax
  _QWORD *v18; // r11
  __int64 v19; // r12
  __int64 v20; // r15
  __int64 v21; // rbx
  char v22; // dl
  int v23; // esi
  _QWORD *v24; // rcx
  unsigned int v25; // eax
  __int64 v26; // rdi
  unsigned int v27; // esi
  _BYTE *v28; // rdi
  _QWORD *v29; // r14
  __int64 v30; // r13
  __int64 v31; // r15
  _QWORD *v32; // rbx
  __int64 v33; // rdx
  __int64 v34; // r12
  unsigned int v35; // eax
  int v36; // r8d
  int v37; // r9d
  __int64 v38; // r12
  unsigned __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // r15
  _QWORD *v43; // rax
  _QWORD *v44; // rdx
  char v45; // cl
  __int64 v46; // rax
  __int64 *v47; // r13
  __int64 v48; // r12
  unsigned __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // r12
  __int64 v52; // rax
  __int64 v53; // rdx
  bool v54; // zf
  __int64 v55; // rax
  __int64 v56; // r15
  __int64 v57; // r13
  unsigned int v58; // ecx
  unsigned int v59; // esi
  __int64 v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 v64; // r8
  __int64 v65; // r9
  __int64 v66; // r12
  __int64 v67; // r14
  char v68; // di
  __int64 v69; // rsi
  __int64 v70; // rdx
  __int64 v71; // rax
  __int64 v72; // rcx
  __int64 v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rcx
  int v76; // eax
  __int64 v77; // rax
  int v78; // edx
  __int64 v79; // rdx
  __int64 *v80; // rax
  __int64 v81; // rsi
  unsigned __int64 v82; // rdx
  __int64 v83; // rdx
  __int64 v84; // rdx
  __int64 v85; // r14
  __int64 v86; // rax
  __int64 v87; // rdx
  __int64 v88; // rcx
  __int64 v89; // r8
  __int64 v90; // r9
  double v91; // xmm4_8
  double v92; // xmm5_8
  __int64 v93; // rsi
  __int64 v94; // r14
  int v95; // eax
  __int64 v96; // rax
  int v97; // edx
  __int64 v98; // rdx
  __int64 *v99; // rax
  __int64 v100; // rcx
  unsigned __int64 v101; // rdx
  __int64 v102; // rcx
  __int64 v103; // rdx
  __int64 v104; // rax
  __int64 v105; // rcx
  __int64 v106; // rdx
  int v107; // eax
  __int64 v108; // rax
  int v109; // edx
  __int64 v110; // rdx
  __int64 *v111; // rax
  __int64 v112; // rcx
  unsigned __int64 v113; // rdx
  __int64 v114; // rcx
  __int64 v115; // rdx
  __int64 v116; // rcx
  unsigned int v117; // eax
  _QWORD *v118; // r9
  unsigned int v119; // ecx
  __int64 v120; // rax
  int v121; // r10d
  int v122; // ecx
  _QWORD *v123; // rsi
  unsigned int v124; // edx
  __int64 v125; // rdi
  _QWORD *v126; // rax
  __int64 v127; // rax
  int v128; // ecx
  _QWORD *v129; // rsi
  unsigned int v130; // edx
  __int64 v131; // rdi
  int v132; // r10d
  _QWORD *v134; // r12
  __int64 v135; // rax
  _QWORD *v136; // r13
  __int64 v137; // rdx
  __int64 v138; // rax
  _QWORD *v139; // rbx
  _QWORD *v140; // r12
  __int64 v141; // rsi
  __int64 v142; // rax
  unsigned __int64 v143; // rax
  int v144; // eax
  int v145; // r13d
  int v146; // r12d
  int v147; // eax
  int v148; // edx
  __int64 v149; // rax
  __int64 v153; // [rsp+18h] [rbp-308h]
  __int64 v154; // [rsp+20h] [rbp-300h]
  __int64 v155; // [rsp+28h] [rbp-2F8h]
  __int64 v156; // [rsp+38h] [rbp-2E8h]
  bool v157; // [rsp+47h] [rbp-2D9h]
  unsigned __int8 v158; // [rsp+48h] [rbp-2D8h]
  _BYTE *v159; // [rsp+50h] [rbp-2D0h]
  __int64 v160; // [rsp+58h] [rbp-2C8h]
  __int64 v161; // [rsp+58h] [rbp-2C8h]
  __int64 v162; // [rsp+58h] [rbp-2C8h]
  __int64 v163; // [rsp+58h] [rbp-2C8h]
  int v164; // [rsp+60h] [rbp-2C0h]
  __int64 v165; // [rsp+60h] [rbp-2C0h]
  _QWORD *v166; // [rsp+60h] [rbp-2C0h]
  _QWORD *v167; // [rsp+60h] [rbp-2C0h]
  _QWORD *v168; // [rsp+60h] [rbp-2C0h]
  __int64 v169; // [rsp+70h] [rbp-2B0h]
  __int64 *v170; // [rsp+70h] [rbp-2B0h]
  __int64 v171; // [rsp+70h] [rbp-2B0h]
  __int64 v172; // [rsp+78h] [rbp-2A8h]
  _BYTE *v173; // [rsp+78h] [rbp-2A8h]
  unsigned __int64 v174; // [rsp+88h] [rbp-298h] BYREF
  void *v175; // [rsp+90h] [rbp-290h]
  _QWORD v176[2]; // [rsp+98h] [rbp-288h] BYREF
  __int64 v177; // [rsp+A8h] [rbp-278h]
  __int64 v178; // [rsp+B0h] [rbp-270h]
  char *v179; // [rsp+C0h] [rbp-260h] BYREF
  __int64 v180; // [rsp+C8h] [rbp-258h] BYREF
  __int64 v181; // [rsp+D0h] [rbp-250h]
  __int64 v182; // [rsp+D8h] [rbp-248h]
  __int64 i; // [rsp+E0h] [rbp-240h]
  char *v184; // [rsp+F0h] [rbp-230h] BYREF
  _QWORD *v185; // [rsp+F8h] [rbp-228h]
  __int64 v186; // [rsp+100h] [rbp-220h]
  unsigned int v187; // [rsp+108h] [rbp-218h]
  _QWORD *v188; // [rsp+118h] [rbp-208h]
  unsigned int v189; // [rsp+128h] [rbp-1F8h]
  char v190; // [rsp+130h] [rbp-1F0h]
  char v191; // [rsp+139h] [rbp-1E7h]
  __int64 *v192; // [rsp+140h] [rbp-1E0h] BYREF
  __int64 v193; // [rsp+148h] [rbp-1D8h]
  _BYTE v194[128]; // [rsp+150h] [rbp-1D0h] BYREF
  __int64 v195; // [rsp+1D0h] [rbp-150h] BYREF
  __int64 v196; // [rsp+1D8h] [rbp-148h]
  _QWORD *v197; // [rsp+1E0h] [rbp-140h] BYREF
  unsigned int v198; // [rsp+1E8h] [rbp-138h]
  _BYTE *v199; // [rsp+260h] [rbp-C0h] BYREF
  __int64 v200; // [rsp+268h] [rbp-B8h]
  _BYTE v201[176]; // [rsp+270h] [rbp-B0h] BYREF

  v11 = &v197;
  v195 = 0;
  v196 = 1;
  do
    *v11++ = -8;
  while ( v11 != &v199 );
  v199 = v201;
  v200 = 0x1000000000LL;
  v12 = *(_QWORD *)(a1 + 80);
  v172 = a1 + 72;
  if ( v12 == a1 + 72 )
  {
    v158 = 0;
  }
  else
  {
    do
    {
      v13 = v12 - 24;
      if ( !v12 )
        v13 = 0;
      v14 = sub_157EBA0(v13);
      v16 = v14;
      if ( *(_BYTE *)(v14 + 16) == 28 )
      {
        v17 = *(_DWORD *)(v14 + 20) & 0xFFFFFFF;
        if ( v17 != 1 )
        {
          v18 = &v197;
          v19 = 24;
          v20 = 24LL * (unsigned int)(v17 - 2) + 48;
          do
          {
            if ( (*(_BYTE *)(v16 + 23) & 0x40) != 0 )
            {
              v21 = *(_QWORD *)(*(_QWORD *)(v16 - 8) + v19);
              v22 = v196 & 1;
              if ( (v196 & 1) != 0 )
                goto LABEL_10;
            }
            else
            {
              v21 = *(_QWORD *)(v16 - 24LL * (*(_DWORD *)(v16 + 20) & 0xFFFFFFF) + v19);
              v22 = v196 & 1;
              if ( (v196 & 1) != 0 )
              {
LABEL_10:
                v23 = 15;
                v24 = v18;
                goto LABEL_11;
              }
            }
            v27 = v198;
            v24 = v197;
            if ( !v198 )
            {
              v117 = v196;
              ++v195;
              v118 = 0;
              v119 = ((unsigned int)v196 >> 1) + 1;
              goto LABEL_124;
            }
            v23 = v198 - 1;
LABEL_11:
            v25 = v23 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
            v15 = &v24[v25];
            v26 = *v15;
            if ( v21 != *v15 )
            {
              v121 = 1;
              v118 = 0;
              while ( v26 != -8 )
              {
                if ( v118 || v26 != -16 )
                  v15 = v118;
                v25 = v23 & (v121 + v25);
                v26 = v24[v25];
                if ( v21 == v26 )
                  goto LABEL_12;
                ++v121;
                v118 = v15;
                v15 = &v24[v25];
              }
              v117 = v196;
              if ( !v118 )
                v118 = v15;
              ++v195;
              v119 = ((unsigned int)v196 >> 1) + 1;
              if ( v22 )
              {
                v27 = 16;
                if ( 4 * v119 >= 0x30 )
                  goto LABEL_138;
LABEL_125:
                if ( v27 - HIDWORD(v196) - v119 <= v27 >> 3 )
                {
                  v168 = v18;
                  sub_19B89E0((__int64)&v195, v27);
                  v18 = v168;
                  if ( (v196 & 1) != 0 )
                  {
                    v128 = 15;
                    v129 = v168;
                  }
                  else
                  {
                    v129 = v197;
                    if ( !v198 )
                    {
LABEL_224:
                      LODWORD(v196) = (2 * ((unsigned int)v196 >> 1) + 2) | v196 & 1;
                      BUG();
                    }
                    v128 = v198 - 1;
                  }
                  v130 = v128 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
                  v118 = &v129[v130];
                  v117 = v196;
                  v131 = *v118;
                  if ( v21 != *v118 )
                  {
                    LODWORD(v15) = 1;
                    v126 = 0;
                    while ( v131 != -8 )
                    {
                      if ( v131 == -16 && !v126 )
                        v126 = v118;
                      v132 = (_DWORD)v15 + 1;
                      LODWORD(v15) = v130 + (_DWORD)v15;
                      v130 = v128 & (unsigned int)v15;
                      v118 = &v129[v128 & (unsigned int)v15];
                      v131 = *v118;
                      if ( v21 == *v118 )
                        goto LABEL_145;
                      LODWORD(v15) = v132;
                    }
LABEL_143:
                    if ( v126 )
                      v118 = v126;
LABEL_145:
                    v117 = v196;
                  }
                }
              }
              else
              {
                v27 = v198;
LABEL_124:
                if ( 4 * v119 < 3 * v27 )
                  goto LABEL_125;
LABEL_138:
                v166 = v18;
                sub_19B89E0((__int64)&v195, 2 * v27);
                v18 = v166;
                if ( (v196 & 1) != 0 )
                {
                  v122 = 15;
                  v123 = v166;
                }
                else
                {
                  v123 = v197;
                  if ( !v198 )
                    goto LABEL_224;
                  v122 = v198 - 1;
                }
                v124 = v122 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
                v118 = &v123[v124];
                v117 = v196;
                v125 = *v118;
                if ( v21 != *v118 )
                {
                  LODWORD(v15) = 1;
                  v126 = 0;
                  while ( v125 != -8 )
                  {
                    if ( v125 != -16 || v126 )
                      v118 = v126;
                    v124 = v122 & ((_DWORD)v15 + v124);
                    v125 = v123[v124];
                    if ( v21 == v125 )
                    {
                      v117 = v196;
                      v118 = &v123[v124];
                      goto LABEL_126;
                    }
                    LODWORD(v15) = (_DWORD)v15 + 1;
                    v126 = v118;
                    v118 = &v123[v124];
                  }
                  goto LABEL_143;
                }
              }
LABEL_126:
              LODWORD(v196) = (2 * (v117 >> 1) + 2) | v117 & 1;
              if ( *v118 != -8 )
                --HIDWORD(v196);
              *v118 = v21;
              v120 = (unsigned int)v200;
              if ( (unsigned int)v200 >= HIDWORD(v200) )
              {
                v167 = v18;
                sub_16CD150((__int64)&v199, v201, 0, 8, (int)v15, (int)v118);
                v120 = (unsigned int)v200;
                v18 = v167;
              }
              *(_QWORD *)&v199[8 * v120] = v21;
              LODWORD(v200) = v200 + 1;
            }
LABEL_12:
            v19 += 24;
          }
          while ( v20 != v19 );
        }
      }
      v12 = *(_QWORD *)(v12 + 8);
    }
    while ( v172 != v12 );
    v28 = v199;
    if ( (_DWORD)v200 )
    {
      v173 = v199;
      v159 = &v199[8 * (unsigned int)v200];
      v158 = 0;
      v157 = a3 != 0 && a2 != 0;
      do
      {
        v29 = *(_QWORD **)v173;
        v30 = *(_QWORD *)(*(_QWORD *)v173 + 48LL);
        v192 = (__int64 *)v194;
        v193 = 0x1000000000LL;
        if ( !v30 )
          BUG();
        if ( *(_BYTE *)(v30 - 8) == 77 && (*(_DWORD *)(v30 - 4) & 0xFFFFFFF) != 0 )
        {
          v31 = 0;
          v32 = 0;
          v169 = 8LL * (*(_DWORD *)(v30 - 4) & 0xFFFFFFF);
          do
          {
            if ( (*(_BYTE *)(v30 - 1) & 0x40) != 0 )
              v33 = *(_QWORD *)(v30 - 32);
            else
              v33 = v30 - 24 - 24LL * (*(_DWORD *)(v30 - 4) & 0xFFFFFFF);
            v34 = *(_QWORD *)(v31 + v33 + 24LL * *(unsigned int *)(v30 + 32) + 8);
            v35 = *(unsigned __int8 *)(sub_157EBA0(v34) + 16) - 24;
            if ( v35 <= 3 )
            {
              if ( v35 <= 1 )
                goto LABEL_21;
              v127 = (unsigned int)v193;
              if ( (unsigned int)v193 >= HIDWORD(v193) )
              {
                sub_16CD150((__int64)&v192, v194, 0, 8, v36, v37);
                v127 = (unsigned int)v193;
              }
              v192[v127] = v34;
              LODWORD(v193) = v193 + 1;
            }
            else
            {
              if ( v35 != 4 || v32 )
                goto LABEL_21;
              v32 = (_QWORD *)v34;
            }
            v31 += 8;
          }
          while ( v169 != v31 );
          if ( !v32
            || !(_DWORD)v193
            || (v38 = sub_157ED20((__int64)v29),
                v39 = (unsigned int)*(unsigned __int8 *)(v38 + 16) - 34,
                (unsigned int)v39 <= 0x36)
            && (v40 = 0x40018000000001LL, _bittest64(&v40, v39))
            || sub_157F790((__int64)v29) )
          {
LABEL_21:
            if ( v192 != (__int64 *)v194 )
              _libc_free((unsigned __int64)v192);
          }
          else
          {
            v184 = ".split";
            LOWORD(v186) = 259;
            v41 = sub_157FBF0(v29, (__int64 *)(v38 + 24), (__int64)&v184);
            v42 = v41;
            if ( v157 )
            {
              v143 = sub_157EBA0(v41);
              v144 = sub_15F4D60(v143);
              if ( v144 )
              {
                v145 = 0;
                v146 = v144;
                do
                {
                  v147 = sub_1377370(a2, (__int64)v29, v145);
                  v148 = v145++;
                  sub_1379150(a2, v42, v148, v147);
                }
                while ( v146 != v145 );
              }
              v149 = sub_1368AA0(a3, (__int64)v29);
              sub_136C010(a3, v42, v149);
            }
            v184 = 0;
            v187 = 128;
            if ( v29 == v32 )
              v32 = (_QWORD *)v42;
            v43 = (_QWORD *)sub_22077B0(0x2000);
            v186 = 0;
            v185 = v43;
            v180 = 2;
            v181 = 0;
            v44 = &v43[8 * (unsigned __int64)v187];
            v182 = -8;
            for ( i = 0; v44 != v43; v43 += 8 )
            {
              if ( v43 )
              {
                v45 = v180;
                v43[2] = 0;
                v43[3] = -8;
                *v43 = &unk_49E6B50;
                v43[1] = v45 & 6;
                v43[4] = i;
              }
            }
            v179 = ".clone";
            v190 = 0;
            v191 = 1;
            LOWORD(v181) = 259;
            v46 = sub_1AB5760(v29, &v184, &v179, a1, 0, 0);
            v174 = 0;
            v156 = v46;
            v47 = v192;
            v170 = &v192[(unsigned int)v193];
            if ( v192 != v170 )
            {
              do
              {
                while ( 1 )
                {
                  v48 = *v47;
                  if ( v29 == (_QWORD *)*v47 )
                    v48 = v42;
                  v49 = sub_157EBA0(v48);
                  sub_1648780(v49, (__int64)v29, v156);
                  if ( v157 )
                    break;
                  if ( v170 == ++v47 )
                    goto LABEL_54;
                }
                ++v47;
                v164 = sub_13774B0(a2, v48, v156);
                v179 = (char *)sub_1368AA0(a3, v48);
                v50 = sub_16AF500((__int64 *)&v179, v164);
                sub_16AF570(&v174, v50);
              }
              while ( v170 != v47 );
            }
LABEL_54:
            if ( v157 )
            {
              sub_136C010(a3, v156, v174);
              v179 = (char *)sub_1368AA0(a3, (__int64)v29);
              v142 = sub_16AF5D0((__int64 *)&v179, v174);
              sub_136C010(a3, (__int64)v29, v142);
              sub_1377B70(a2, (__int64)v29);
            }
            v51 = v29[6];
            v171 = v51;
            v155 = sub_157ED20((__int64)v29) + 24;
            v165 = *(_QWORD *)(v156 + 48);
            v52 = sub_157EE30(v42);
            if ( v155 != v51 )
            {
              v53 = v52 - 24;
              v54 = v52 == 0;
              v55 = 0;
              v153 = (__int64)v29;
              if ( !v54 )
                v55 = v53;
              v154 = v55;
              do
              {
                v56 = 0;
                if ( v165 )
                  v56 = v165 - 24;
                v57 = v171 - 24;
                if ( !v171 )
                  v57 = 0;
                v58 = *(_DWORD *)(v56 + 20) & 0xFFFFFFF;
                if ( v58 )
                {
                  v59 = 0;
                  v60 = 24LL * *(unsigned int *)(v56 + 56) + 8;
                  while ( 1 )
                  {
                    v61 = v56 - 24LL * v58;
                    if ( (*(_BYTE *)(v56 + 23) & 0x40) != 0 )
                      v61 = *(_QWORD *)(v56 - 8);
                    if ( v32 == *(_QWORD **)(v61 + v60) )
                      break;
                    ++v59;
                    v60 += 8;
                    if ( v58 == v59 )
                      goto LABEL_171;
                  }
                }
                else
                {
LABEL_171:
                  v59 = -1;
                }
                sub_15F5350(v56, v59, 1);
                v165 = *(_QWORD *)(v165 + 8);
                v62 = *(_QWORD *)(v171 + 8);
                LOWORD(v181) = 259;
                v171 = v62;
                v179 = "ind";
                v160 = *(_QWORD *)v57;
                v63 = sub_1648B60(64);
                v66 = v63;
                if ( v63 )
                {
                  v67 = v63;
                  sub_15F1EA0(v63, v160, 53, 0, 0, v57);
                  *(_DWORD *)(v66 + 56) = 1;
                  sub_164B780(v66, (__int64 *)&v179);
                  sub_1648880(v66, *(_DWORD *)(v66 + 56), 1);
                }
                else
                {
                  v67 = 0;
                }
                v68 = *(_BYTE *)(v57 + 23) & 0x40;
                v69 = *(_DWORD *)(v57 + 20) & 0xFFFFFFF;
                if ( (*(_DWORD *)(v57 + 20) & 0xFFFFFFF) != 0 )
                {
                  v64 = v57 - 24LL * (unsigned int)v69;
                  v70 = 24LL * *(unsigned int *)(v57 + 56) + 8;
                  v71 = 0;
                  while ( 1 )
                  {
                    v72 = v57 - 24LL * (unsigned int)v69;
                    if ( v68 )
                      v72 = *(_QWORD *)(v57 - 8);
                    if ( v32 == *(_QWORD **)(v72 + v70) )
                      break;
                    ++v71;
                    v70 += 8;
                    if ( (_DWORD)v69 == (_DWORD)v71 )
                      goto LABEL_169;
                  }
                  v73 = 24 * v71;
                  if ( !v68 )
                  {
LABEL_170:
                    v69 = (unsigned int)v69;
                    v74 = v57 - 24LL * (unsigned int)v69;
                    goto LABEL_80;
                  }
                }
                else
                {
LABEL_169:
                  v73 = 0x17FFFFFFE8LL;
                  if ( !v68 )
                    goto LABEL_170;
                }
                v74 = *(_QWORD *)(v57 - 8);
LABEL_80:
                v75 = *(_QWORD *)(v74 + v73);
                v76 = *(_DWORD *)(v66 + 20) & 0xFFFFFFF;
                if ( v76 == *(_DWORD *)(v66 + 56) )
                {
                  v163 = v75;
                  sub_15F55D0(v66, v69, v74, v75, v64, v65);
                  v75 = v163;
                  v76 = *(_DWORD *)(v66 + 20) & 0xFFFFFFF;
                }
                v77 = (v76 + 1) & 0xFFFFFFF;
                v78 = v77 | *(_DWORD *)(v66 + 20) & 0xF0000000;
                *(_DWORD *)(v66 + 20) = v78;
                if ( (v78 & 0x40000000) != 0 )
                  v79 = *(_QWORD *)(v66 - 8);
                else
                  v79 = v67 - 24 * v77;
                v80 = (__int64 *)(v79 + 24LL * (unsigned int)(v77 - 1));
                if ( *v80 )
                {
                  v81 = v80[1];
                  v82 = v80[2] & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v82 = v81;
                  if ( v81 )
                    *(_QWORD *)(v81 + 16) = *(_QWORD *)(v81 + 16) & 3LL | v82;
                }
                *v80 = v75;
                if ( v75 )
                {
                  v83 = *(_QWORD *)(v75 + 8);
                  v80[1] = v83;
                  if ( v83 )
                    *(_QWORD *)(v83 + 16) = (unsigned __int64)(v80 + 1) | *(_QWORD *)(v83 + 16) & 3LL;
                  v80[2] = (v75 + 8) | v80[2] & 3;
                  *(_QWORD *)(v75 + 8) = v80;
                }
                v84 = *(_DWORD *)(v66 + 20) & 0xFFFFFFF;
                if ( (*(_BYTE *)(v66 + 23) & 0x40) != 0 )
                  v85 = *(_QWORD *)(v66 - 8);
                else
                  v85 = v67 - 24 * v84;
                *(_QWORD *)(v85 + 8LL * (unsigned int)(v84 - 1) + 24LL * *(unsigned int *)(v66 + 56) + 8) = v32;
                v179 = "merge";
                LOWORD(v181) = 259;
                v161 = *(_QWORD *)v57;
                v86 = sub_1648B60(64);
                v93 = v161;
                v94 = v86;
                if ( v86 )
                {
                  v162 = v86;
                  sub_15F1EA0(v86, v93, 53, 0, 0, v154);
                  *(_DWORD *)(v94 + 56) = 2;
                  sub_164B780(v94, (__int64 *)&v179);
                  v93 = *(unsigned int *)(v94 + 56);
                  sub_1648880(v94, v93, 1);
                }
                else
                {
                  v162 = 0;
                }
                v95 = *(_DWORD *)(v94 + 20) & 0xFFFFFFF;
                if ( v95 == *(_DWORD *)(v94 + 56) )
                {
                  sub_15F55D0(v94, v93, v87, v88, v89, v90);
                  v95 = *(_DWORD *)(v94 + 20) & 0xFFFFFFF;
                }
                v96 = (v95 + 1) & 0xFFFFFFF;
                v97 = v96 | *(_DWORD *)(v94 + 20) & 0xF0000000;
                *(_DWORD *)(v94 + 20) = v97;
                if ( (v97 & 0x40000000) != 0 )
                  v98 = *(_QWORD *)(v94 - 8);
                else
                  v98 = v162 - 24 * v96;
                v99 = (__int64 *)(v98 + 24LL * (unsigned int)(v96 - 1));
                if ( *v99 )
                {
                  v100 = v99[1];
                  v101 = v99[2] & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v101 = v100;
                  if ( v100 )
                    *(_QWORD *)(v100 + 16) = *(_QWORD *)(v100 + 16) & 3LL | v101;
                }
                *v99 = v66;
                v102 = *(_QWORD *)(v66 + 8);
                v99[1] = v102;
                if ( v102 )
                  *(_QWORD *)(v102 + 16) = (unsigned __int64)(v99 + 1) | *(_QWORD *)(v102 + 16) & 3LL;
                v99[2] = (v66 + 8) | v99[2] & 3;
                *(_QWORD *)(v66 + 8) = v99;
                v103 = *(_DWORD *)(v94 + 20) & 0xFFFFFFF;
                v104 = (unsigned int)(v103 - 1);
                if ( (*(_BYTE *)(v94 + 23) & 0x40) != 0 )
                  v105 = *(_QWORD *)(v94 - 8);
                else
                  v105 = v162 - 24 * v103;
                v106 = 3LL * *(unsigned int *)(v94 + 56);
                *(_QWORD *)(v105 + 8 * v104 + 24LL * *(unsigned int *)(v94 + 56) + 8) = v153;
                v107 = *(_DWORD *)(v94 + 20) & 0xFFFFFFF;
                if ( v107 == *(_DWORD *)(v94 + 56) )
                {
                  sub_15F55D0(v94, v153, v106, v105, v89, v90);
                  v107 = *(_DWORD *)(v94 + 20) & 0xFFFFFFF;
                }
                v108 = (v107 + 1) & 0xFFFFFFF;
                v109 = v108 | *(_DWORD *)(v94 + 20) & 0xF0000000;
                *(_DWORD *)(v94 + 20) = v109;
                if ( (v109 & 0x40000000) != 0 )
                  v110 = *(_QWORD *)(v94 - 8);
                else
                  v110 = v162 - 24 * v108;
                v111 = (__int64 *)(v110 + 24LL * (unsigned int)(v108 - 1));
                if ( *v111 )
                {
                  v112 = v111[1];
                  v113 = v111[2] & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v113 = v112;
                  if ( v112 )
                    *(_QWORD *)(v112 + 16) = *(_QWORD *)(v112 + 16) & 3LL | v113;
                }
                *v111 = v56;
                v114 = *(_QWORD *)(v56 + 8);
                v111[1] = v114;
                if ( v114 )
                  *(_QWORD *)(v114 + 16) = (unsigned __int64)(v111 + 1) | *(_QWORD *)(v114 + 16) & 3LL;
                v111[2] = (v56 + 8) | v111[2] & 3;
                *(_QWORD *)(v56 + 8) = v111;
                v115 = *(_DWORD *)(v94 + 20) & 0xFFFFFFF;
                if ( (*(_BYTE *)(v94 + 23) & 0x40) != 0 )
                  v116 = *(_QWORD *)(v94 - 8);
                else
                  v116 = v162 - 24 * v115;
                *(_QWORD *)(v116 + 8LL * (unsigned int)(v115 - 1) + 24LL * *(unsigned int *)(v94 + 56) + 8) = v156;
                sub_164D160(v57, v94, a4, a5, a6, a7, v91, v92, a10, a11);
                sub_15F20C0((_QWORD *)v57);
              }
              while ( v155 != v171 );
            }
            if ( v190 )
            {
              if ( v189 )
              {
                v139 = v188;
                v140 = &v188[2 * v189];
                do
                {
                  if ( *v139 != -8 && *v139 != -4 )
                  {
                    v141 = v139[1];
                    if ( v141 )
                      sub_161E7C0((__int64)(v139 + 1), v141);
                  }
                  v139 += 2;
                }
                while ( v140 != v139 );
              }
              j___libc_free_0(v188);
            }
            if ( v187 )
            {
              v134 = v185;
              v176[0] = 2;
              v176[1] = 0;
              v135 = -8;
              v136 = &v185[8 * (unsigned __int64)v187];
              v177 = -8;
              v175 = &unk_49E6B50;
              v178 = 0;
              v180 = 2;
              v181 = 0;
              v182 = -16;
              v179 = (char *)&unk_49E6B50;
              i = 0;
              while ( 1 )
              {
                v137 = v134[3];
                if ( v137 != v135 )
                {
                  v135 = v182;
                  if ( v137 != v182 )
                  {
                    v138 = v134[7];
                    if ( v138 != -8 && v138 != 0 && v138 != -16 )
                    {
                      sub_1649B30(v134 + 5);
                      v137 = v134[3];
                    }
                    v135 = v137;
                  }
                }
                *v134 = &unk_49EE2B0;
                if ( v135 != -8 && v135 != 0 && v135 != -16 )
                  sub_1649B30(v134 + 1);
                v134 += 8;
                if ( v136 == v134 )
                  break;
                v135 = v177;
              }
              v179 = (char *)&unk_49EE2B0;
              if ( v182 != 0 && v182 != -8 && v182 != -16 )
                sub_1649B30(&v180);
              v175 = &unk_49EE2B0;
              if ( v177 != -8 && v177 != 0 && v177 != -16 )
                sub_1649B30(v176);
            }
            j___libc_free_0(v185);
            if ( v192 != (__int64 *)v194 )
              _libc_free((unsigned __int64)v192);
            v158 = 1;
          }
        }
        v173 += 8;
      }
      while ( v159 != v173 );
      v28 = v199;
    }
    else
    {
      v158 = 0;
    }
    if ( v28 != v201 )
      _libc_free((unsigned __int64)v28);
  }
  if ( (v196 & 1) == 0 )
    j___libc_free_0(v197);
  return v158;
}
