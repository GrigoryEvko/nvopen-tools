// Function: sub_31D1050
// Address: 0x31d1050
//
__int64 __fastcall sub_31D1050(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 *v4; // r13
  __int64 *v5; // r14
  __int64 *v6; // r13
  __int64 v7; // rbx
  const char *v8; // rax
  unsigned __int64 v9; // rdx
  _QWORD *v10; // r12
  bool v11; // al
  size_t v12; // r9
  unsigned __int8 v13; // r13
  _QWORD *v14; // r13
  int v15; // esi
  __int64 v16; // r8
  __int64 v17; // rdx
  int v18; // esi
  __int64 v19; // rdi
  unsigned int v20; // ecx
  __int64 v21; // rax
  __int64 v22; // r11
  _QWORD *v23; // rbx
  __int64 *v24; // rdi
  __int64 v25; // rbx
  __int64 v26; // rdi
  __int64 v27; // rax
  __int64 v28; // r14
  __int64 v29; // rax
  const char *v30; // rsi
  __int64 v31; // r9
  const char *v32; // r14
  unsigned int *v33; // rax
  int v34; // ecx
  unsigned int *v35; // rdx
  __int64 *v36; // rdi
  char *v37; // rax
  char *v38; // r14
  unsigned int v39; // r10d
  __int64 *v40; // rax
  char *v41; // rbx
  char *v42; // r15
  unsigned __int64 v43; // r12
  __int64 v44; // r14
  __int64 v45; // r13
  _BYTE *v46; // rdx
  __int64 v47; // rdx
  char *v48; // rax
  __int64 *v49; // rax
  __int64 v50; // rcx
  __int64 v51; // rcx
  __int64 *v52; // rax
  int v53; // r14d
  _QWORD *v54; // rax
  __int64 v55; // rsi
  _QWORD *v56; // rbx
  unsigned int v57; // edx
  __int64 v58; // r12
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 v61; // rax
  _QWORD *v62; // r14
  _QWORD *v63; // rbx
  __int64 v64; // rsi
  __int64 v65; // rax
  _QWORD *v66; // rbx
  _QWORD *v67; // r12
  __int64 v68; // rax
  __int64 v69; // rdx
  int v70; // ecx
  __int64 v71; // rax
  __int64 v73; // rax
  _QWORD *v74; // rbx
  _QWORD *v75; // r12
  __int64 v76; // rsi
  __int64 v77; // rax
  _QWORD *v78; // r14
  _QWORD *v79; // rbx
  _QWORD *v80; // r13
  __int64 v81; // rdx
  _QWORD *v82; // rcx
  __int64 v83; // rax
  _QWORD *v84; // r13
  _QWORD *v85; // rbx
  __int64 v86; // rdx
  __int64 v87; // rax
  __int64 v88; // rdx
  _QWORD *v89; // rbx
  _QWORD *v90; // r13
  __int64 v91; // r12
  __int64 v92; // rax
  __int64 v93; // rax
  __int64 v94; // rax
  __int64 v95; // rsi
  unsigned int **v96; // rdx
  bool v97; // al
  bool v98; // zf
  __int64 *v99; // rdi
  __int64 v100; // rax
  const char *v101; // rdx
  char *v102; // r11
  size_t v103; // r9
  __int64 v104; // rax
  char v105; // dl
  __int64 *v106; // rdx
  __int64 *v107; // rax
  __int64 *v108; // rdi
  __int64 v109; // rax
  __int64 v110; // r14
  __int64 v111; // rax
  _QWORD *v112; // r12
  int v113; // edx
  __int64 v114; // rdx
  unsigned __int64 *v115; // rdi
  unsigned int **v116; // rdx
  _QWORD *v117; // r12
  __int64 v118; // rax
  __int64 v119; // rax
  __int64 v120; // rdx
  __int64 v121; // r8
  int v122; // edi
  _QWORD *v123; // rcx
  int v124; // esi
  _QWORD *v125; // rcx
  __int64 v126; // rdx
  __int64 v127; // rdi
  int v128; // esi
  __int64 v129; // rdx
  __int64 v130; // rdi
  _QWORD *v131; // r12
  _QWORD *v132; // rbx
  __int64 v133; // rax
  char *v134; // r13
  unsigned int v135; // eax
  _QWORD *v136; // r8
  unsigned __int64 v137; // rdx
  unsigned __int64 v138; // rax
  __int64 v139; // rax
  _QWORD *v140; // r14
  __int64 v141; // rax
  _QWORD *v142; // rdx
  __int64 v143; // rax
  _QWORD *v144; // rbx
  __int64 v145; // rcx
  unsigned __int64 v146; // r8
  _QWORD *v147; // rax
  _QWORD *v148; // r14
  _QWORD *v149; // rbx
  char v150; // al
  __int64 v151; // rax
  int v152; // r9d
  __int64 v153; // rax
  _QWORD *v154; // rbx
  __int64 v155; // rax
  __int64 *v156; // rax
  _QWORD *v157; // [rsp+8h] [rbp-178h]
  void **v158; // [rsp+10h] [rbp-170h]
  __int64 v159; // [rsp+18h] [rbp-168h]
  __int64 *v160; // [rsp+20h] [rbp-160h]
  _QWORD **v161; // [rsp+28h] [rbp-158h]
  size_t n; // [rsp+30h] [rbp-150h]
  void *src; // [rsp+38h] [rbp-148h]
  char *v164; // [rsp+40h] [rbp-140h]
  __int64 v165; // [rsp+48h] [rbp-138h]
  __int64 *i; // [rsp+50h] [rbp-130h]
  __int64 *v167; // [rsp+58h] [rbp-128h]
  __int64 *v168; // [rsp+60h] [rbp-120h] BYREF
  __int64 v169; // [rsp+68h] [rbp-118h] BYREF
  __int64 v170; // [rsp+70h] [rbp-110h] BYREF
  __int64 v171; // [rsp+78h] [rbp-108h]
  __int64 v172; // [rsp+80h] [rbp-100h]
  __int64 v173; // [rsp+90h] [rbp-F0h] BYREF
  __int64 v174; // [rsp+98h] [rbp-E8h] BYREF
  __int64 v175; // [rsp+A0h] [rbp-E0h]
  __int64 v176; // [rsp+A8h] [rbp-D8h]
  unsigned int **v177; // [rsp+B0h] [rbp-D0h]
  unsigned int *v178; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v179; // [rsp+C8h] [rbp-B8h] BYREF
  __int64 v180; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 v181; // [rsp+D8h] [rbp-A8h]
  __int64 *v182; // [rsp+E0h] [rbp-A0h]
  _QWORD *v183; // [rsp+E8h] [rbp-98h]
  __int64 v184; // [rsp+F0h] [rbp-90h]
  __int64 v185; // [rsp+F8h] [rbp-88h]
  __int16 v186; // [rsp+100h] [rbp-80h]
  __int64 v187; // [rsp+108h] [rbp-78h]
  void **v188; // [rsp+110h] [rbp-70h]
  _QWORD *v189; // [rsp+118h] [rbp-68h]
  __int64 v190; // [rsp+120h] [rbp-60h]
  int v191; // [rsp+128h] [rbp-58h]
  __int16 v192; // [rsp+12Ch] [rbp-54h]
  char v193; // [rsp+12Eh] [rbp-52h]
  __int64 v194; // [rsp+130h] [rbp-50h]
  __int64 v195; // [rsp+138h] [rbp-48h]
  void *v196; // [rsp+140h] [rbp-40h] BYREF
  _QWORD v197[7]; // [rsp+148h] [rbp-38h] BYREF

  v4 = (__int64 *)a2[2];
  v167 = a2;
  i = a2 + 1;
  v5 = v4;
  if ( v4 != a2 + 1 )
  {
    while ( 1 )
    {
      v6 = v5;
      v5 = (__int64 *)v5[1];
      if ( !(*(_DWORD *)(*(v6 - 6) + 8) >> 8) )
      {
        v7 = (__int64)(v6 - 7);
        if ( !(unsigned __int8)sub_CE8750((_BYTE *)v6 - 56)
          && !(unsigned __int8)sub_CE87C0((_BYTE *)v6 - 56)
          && !(unsigned __int8)sub_CE8830((_BYTE *)v6 - 56) )
        {
          v8 = sub_BD5D20((__int64)(v6 - 7));
          if ( v9 <= 4 || *(_DWORD *)v8 != 1836477548 || v8[4] != 46 )
            break;
        }
      }
LABEL_3:
      if ( i == v5 )
        goto LABEL_25;
    }
    v10 = (_QWORD *)*(v6 - 4);
    LOBYTE(v165) = v6[3] & 1;
    LOBYTE(v164) = *(_BYTE *)(v6 - 3) & 0xF;
    v11 = sub_B2FC80((__int64)(v6 - 7));
    v12 = 0;
    if ( !v11 )
      v12 = *(v6 - 11);
    n = v12;
    LOWORD(v182) = 257;
    v13 = *((_BYTE *)v6 - 23);
    v173 = 0x100000001LL;
    LODWORD(src) = (v13 >> 2) & 7;
    v14 = sub_BD2C40(88, unk_3F0FAE8);
    if ( v14 )
      sub_B30000(
        (__int64)v14,
        (__int64)v167,
        v10,
        v165,
        (unsigned __int8)v164 & 0xF,
        n,
        (__int64)&v178,
        v7,
        (__int16)src,
        v173,
        0);
    sub_B32030((__int64)v14, v7);
    sub_B9E560((__int64)v14, v7, 0);
    v181 = v7;
    v179 = 2;
    v180 = 0;
    if ( v7 != -8192 && v7 != -4096 )
      sub_BD73F0((__int64)&v179);
    v15 = *((_DWORD *)a1 + 6);
    v182 = a1;
    v16 = 0;
    v178 = (unsigned int *)&unk_4A259B8;
    if ( v15 )
    {
      v17 = v181;
      v18 = v15 - 1;
      v19 = a1[1];
      v20 = v18 & (((unsigned int)v181 >> 9) ^ ((unsigned int)v181 >> 4));
      v21 = v19 + 48LL * v20;
      v22 = *(_QWORD *)(v21 + 24);
      if ( v22 == v181 )
      {
LABEL_20:
        v23 = (_QWORD *)(v21 + 40);
LABEL_21:
        v178 = (unsigned int *)&unk_49DB368;
        if ( v17 != -4096 && v17 != 0 && v17 != -8192 )
          sub_BD60C0(&v179);
        *v23 = v14;
        goto LABEL_3;
      }
      v152 = 1;
      v16 = 0;
      while ( v22 != -4096 )
      {
        if ( !v16 && v22 == -8192 )
          v16 = v21;
        v20 = v18 & (v152 + v20);
        v21 = v19 + 48LL * v20;
        v22 = *(_QWORD *)(v21 + 24);
        if ( v181 == v22 )
          goto LABEL_20;
        ++v152;
      }
      if ( !v16 )
        v16 = v21;
    }
    v153 = sub_31CF770((__int64)a1, (__int64)&v178, v16);
    v17 = v181;
    v154 = (_QWORD *)v153;
    v155 = *(_QWORD *)(v153 + 24);
    if ( v155 != v181 )
    {
      if ( v155 != -4096 && v155 != 0 && v155 != -8192 )
      {
        sub_BD60C0(v154 + 1);
        v17 = v181;
      }
      v154[3] = v17;
      if ( v17 != -4096 && v17 != 0 && v17 != -8192 )
        sub_BD6050(v154 + 1, v179 & 0xFFFFFFFFFFFFFFF8LL);
      v17 = v181;
    }
    v156 = v182;
    v23 = v154 + 5;
    *v23 = 0;
    *(v23 - 1) = v156;
    goto LABEL_21;
  }
LABEL_25:
  if ( !*((_DWORD *)a1 + 4) )
    return 0;
  i = (__int64 *)&v178;
  v24 = (__int64 *)v167[4];
  v161 = (_QWORD **)(v167 + 3);
  n = (size_t)v24;
  if ( v167 + 3 != v24 )
  {
    do
    {
      v25 = n - 56;
      if ( !n )
        v25 = 0;
      if ( sub_B2FC80(v25) )
        goto LABEL_28;
      v26 = *(_QWORD *)(v25 + 80);
      if ( v26 )
        v26 -= 24;
      v27 = sub_AA5030(v26, 1);
      v28 = v27;
      if ( !v27 )
      {
        v2 = sub_BD5C60(0);
        v193 = 7;
        v187 = v2;
        v188 = &v196;
        v189 = v197;
        v178 = (unsigned int *)&v180;
        v179 = 0x200000000LL;
        v196 = &unk_49DA100;
        v190 = 0;
        v191 = 0;
        v192 = 512;
        v194 = 0;
        v195 = 0;
        v184 = 0;
        v185 = 0;
        v186 = 0;
        v197[0] = &unk_49DA0B0;
        BUG();
      }
      v165 = v27 - 24;
      v187 = sub_BD5C60(v27 - 24);
      v188 = &v196;
      v158 = &v196;
      v189 = v197;
      v157 = v197;
      v178 = (unsigned int *)&v180;
      v192 = 512;
      v196 = &unk_49DA100;
      v160 = &v180;
      v193 = 7;
      v179 = 0x200000000LL;
      v184 = 0;
      v185 = 0;
      v190 = 0;
      v191 = 0;
      v194 = 0;
      v195 = 0;
      v186 = 0;
      v197[0] = &unk_49DA0B0;
      v29 = *(_QWORD *)(v28 + 16);
      v185 = v28;
      v184 = v29;
      v30 = *(const char **)sub_B46C60(v165);
      v173 = (__int64)v30;
      if ( v30 && (v165 = (__int64)&v173, sub_B96E90((__int64)&v173, (__int64)v30, 1), (v32 = (const char *)v173) != 0) )
      {
        v33 = v178;
        v34 = v179;
        v35 = &v178[4 * (unsigned int)v179];
        if ( v178 != v35 )
        {
          v36 = (__int64 *)v165;
          while ( 1 )
          {
            v31 = *v33;
            if ( !(_DWORD)v31 )
              break;
            v33 += 4;
            if ( v35 == v33 )
              goto LABEL_87;
          }
          *((_QWORD *)v33 + 1) = v173;
          goto LABEL_42;
        }
LABEL_87:
        if ( (unsigned int)v179 >= (unsigned __int64)HIDWORD(v179) )
        {
          v146 = (unsigned int)v179 + 1LL;
          v147 = (_QWORD *)(v159 & 0xFFFFFFFF00000000LL);
          v159 &= 0xFFFFFFFF00000000LL;
          if ( HIDWORD(v179) < v146 )
          {
            v165 = (__int64)v147;
            sub_C8D5F0((__int64)i, v160, v146, 0x10u, v146, v31);
            v147 = (_QWORD *)v165;
            v35 = &v178[4 * (unsigned int)v179];
          }
          *(_QWORD *)v35 = v147;
          *((_QWORD *)v35 + 1) = v32;
          v32 = (const char *)v173;
          LODWORD(v179) = v179 + 1;
        }
        else
        {
          if ( v35 )
          {
            *v35 = 0;
            *((_QWORD *)v35 + 1) = v32;
            v34 = v179;
            v32 = (const char *)v173;
          }
          LODWORD(v179) = v34 + 1;
        }
      }
      else
      {
        sub_93FB40((__int64)i, 0);
        v32 = (const char *)v173;
      }
      if ( v32 )
      {
        v36 = &v173;
LABEL_42:
        sub_B91220((__int64)v36, (__int64)v32);
      }
      v37 = *(char **)(v25 + 80);
      src = (void *)(v25 + 72);
      v164 = v37;
      if ( (char *)(v25 + 72) != v37 )
      {
        while ( 1 )
        {
          if ( !v164 )
            BUG();
          v38 = (char *)*((_QWORD *)v164 + 4);
          v165 = (__int64)(v164 + 24);
          if ( v164 + 24 != v38 )
            break;
LABEL_64:
          v164 = (char *)*((_QWORD *)v164 + 1);
          if ( src == v164 )
            goto LABEL_65;
        }
        while ( 1 )
        {
          if ( !v38 )
            BUG();
          v39 = *((_DWORD *)v38 - 5) & 0x7FFFFFF;
          if ( v39 )
            break;
LABEL_63:
          v38 = (char *)*((_QWORD *)v38 + 1);
          if ( (char *)v165 == v38 )
            goto LABEL_64;
        }
        v40 = a1;
        v41 = v38 - 24;
        v42 = v38;
        v43 = 0;
        v44 = (__int64)v40;
        v45 = 32LL * v39;
        while ( 1 )
        {
          if ( (*(v42 - 17) & 0x40) != 0 )
          {
            v46 = *(_BYTE **)(*((_QWORD *)v42 - 4) + v43);
            if ( *v46 > 0x15u )
              goto LABEL_50;
          }
          else
          {
            v46 = *(_BYTE **)&v41[v43 + -32 * (*((_DWORD *)v42 - 5) & 0x7FFFFFF)];
            if ( *v46 > 0x15u )
              goto LABEL_50;
          }
          v47 = sub_31CFD60(v44, (_QWORD **)v167, (__int64)v46, i);
          if ( (*(v42 - 17) & 0x40) != 0 )
            v48 = (char *)*((_QWORD *)v42 - 4);
          else
            v48 = &v41[-32 * (*((_DWORD *)v42 - 5) & 0x7FFFFFF)];
          v49 = (__int64 *)&v48[v43];
          if ( *v49 )
          {
            v50 = v49[1];
            *(_QWORD *)v49[2] = v50;
            if ( v50 )
              *(_QWORD *)(v50 + 16) = v49[2];
          }
          *v49 = v47;
          if ( !v47 )
          {
LABEL_50:
            v43 += 32LL;
            if ( v45 == v43 )
              goto LABEL_62;
            continue;
          }
          v51 = *(_QWORD *)(v47 + 16);
          v49[1] = v51;
          if ( v51 )
            *(_QWORD *)(v51 + 16) = v49 + 1;
          v43 += 32LL;
          v49[2] = v47 + 16;
          *(_QWORD *)(v47 + 16) = v49;
          if ( v45 == v43 )
          {
LABEL_62:
            v52 = (__int64 *)v44;
            v38 = v42;
            a1 = v52;
            goto LABEL_63;
          }
        }
      }
LABEL_65:
      v53 = *((_DWORD *)a1 + 24);
      ++a1[10];
      if ( !v53 && !*((_DWORD *)a1 + 25) )
        goto LABEL_83;
      v54 = (_QWORD *)a1[11];
      v55 = *((unsigned int *)a1 + 26);
      v56 = &v54[6 * v55];
      v57 = 4 * v53;
      if ( (unsigned int)(4 * v53) < 0x40 )
        v57 = 64;
      if ( (unsigned int)v55 > v57 )
      {
        v169 = 2;
        v170 = 0;
        v171 = -4096;
        v168 = (__int64 *)&unk_4A34DD0;
        v172 = 0;
        v174 = 2;
        v175 = 0;
        v176 = -8192;
        v173 = (__int64)&unk_4A34DD0;
        v177 = 0;
        v165 = 48 * v55;
        v131 = &v54[6 * v55];
        v132 = v54;
        v164 = (char *)&unk_49DB358;
        do
        {
          v133 = v132[3];
          *v132 = &unk_49DB368;
          if ( v133 != -4096 && v133 != 0 && v133 != -8192 )
            sub_BD60C0(v132 + 1);
          v132 += 6;
        }
        while ( v131 != v132 );
        v134 = v164;
        v173 = (__int64)(v164 + 16);
        if ( v176 != 0 && v176 != -4096 && v176 != -8192 )
          sub_BD60C0(&v174);
        v168 = (__int64 *)(v164 + 16);
        if ( v171 != -4096 && v171 != 0 && v171 != -8192 )
          sub_BD60C0(&v169);
        if ( v53 )
        {
          v135 = v53 - 1;
          v53 = 64;
          if ( v135 )
          {
            _BitScanReverse(&v135, v135);
            v53 = 1 << (33 - (v135 ^ 0x1F));
            if ( v53 < 64 )
              v53 = 64;
          }
        }
        v136 = (_QWORD *)a1[11];
        if ( *((_DWORD *)a1 + 26) == v53 )
        {
          a1[12] = 0;
          v174 = 2;
          v175 = 0;
          v148 = &v136[6 * (unsigned int)v53];
          v176 = -4096;
          v173 = (__int64)&unk_4A34DD0;
          v177 = 0;
          if ( v148 == v136 )
            goto LABEL_83;
          v149 = v136;
          do
          {
            if ( v149 )
            {
              v150 = v174;
              v149[2] = 0;
              v149[1] = v150 & 6;
              v151 = v176;
              v98 = v176 == -4096;
              v149[3] = v176;
              if ( v151 != 0 && !v98 && v151 != -8192 )
                sub_BD6050(v149 + 1, v174 & 0xFFFFFFFFFFFFFFF8LL);
              *v149 = &unk_4A34DD0;
              v149[4] = v177;
            }
            v149 += 6;
          }
          while ( v148 != v149 );
          v143 = v176;
          v173 = (__int64)(v134 + 16);
          if ( v176 == -4096 || v176 == 0 )
            goto LABEL_83;
        }
        else
        {
          sub_C7D6A0(a1[11], v165, 8);
          if ( !v53 )
          {
            a1[11] = 0;
            a1[12] = 0;
            *((_DWORD *)a1 + 26) = 0;
            goto LABEL_83;
          }
          v137 = ((((((((4 * v53 / 3u + 1) | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1)) >> 2)
                    | (4 * v53 / 3u + 1)
                    | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1)) >> 4)
                  | (((4 * v53 / 3u + 1) | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1)) >> 2)
                  | (4 * v53 / 3u + 1)
                  | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1)) >> 8)
                | (((((4 * v53 / 3u + 1) | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1)) >> 2)
                  | (4 * v53 / 3u + 1)
                  | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1)) >> 4)
                | (((4 * v53 / 3u + 1) | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1)) >> 2)
                | (4 * v53 / 3u + 1)
                | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1)) >> 16;
          v138 = (v137
                | (((((((4 * v53 / 3u + 1) | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1)) >> 2)
                    | (4 * v53 / 3u + 1)
                    | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1)) >> 4)
                  | (((4 * v53 / 3u + 1) | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1)) >> 2)
                  | (4 * v53 / 3u + 1)
                  | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1)) >> 8)
                | (((((4 * v53 / 3u + 1) | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1)) >> 2)
                  | (4 * v53 / 3u + 1)
                  | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1)) >> 4)
                | (((4 * v53 / 3u + 1) | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1)) >> 2)
                | (4 * v53 / 3u + 1)
                | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1))
               + 1;
          *((_DWORD *)a1 + 26) = v138;
          v139 = sub_C7D670(48 * v138, 8);
          a1[12] = 0;
          v140 = (_QWORD *)v139;
          a1[11] = v139;
          v174 = 2;
          v175 = 0;
          v141 = *((unsigned int *)a1 + 26);
          v176 = -4096;
          v173 = (__int64)&unk_4A34DD0;
          v177 = 0;
          v142 = &v140[6 * v141];
          if ( v140 == v142 )
            goto LABEL_83;
          v143 = -4096;
          v144 = v142;
          do
          {
            if ( v140 )
            {
              v145 = v174;
              v140[2] = 0;
              v140[3] = v143;
              v140[1] = v145 & 6;
              if ( v143 != 0 && v143 != -4096 && v143 != -8192 )
              {
                sub_BD6050(v140 + 1, v145 & 0xFFFFFFFFFFFFFFF8LL);
                v143 = v176;
              }
              *v140 = &unk_4A34DD0;
              v140[4] = v177;
            }
            v140 += 6;
          }
          while ( v144 != v140 );
          v173 = (__int64)(v134 + 16);
          if ( v143 == 0 || v143 == -4096 )
            goto LABEL_83;
        }
        if ( v143 == -8192 )
          goto LABEL_83;
      }
      else
      {
        v58 = a1[11];
        v174 = 2;
        v175 = 0;
        v176 = -4096;
        v173 = (__int64)&unk_4A34DD0;
        v59 = -4096;
        v177 = 0;
        if ( v56 == v54 )
        {
          a1[12] = 0;
          goto LABEL_83;
        }
        do
        {
          v60 = *(_QWORD *)(v58 + 24);
          if ( v60 != v59 )
          {
            if ( v60 != -4096 && v60 != 0 && v60 != -8192 )
            {
              sub_BD60C0((_QWORD *)(v58 + 8));
              v59 = v176;
            }
            *(_QWORD *)(v58 + 24) = v59;
            if ( v59 != 0 && v59 != -4096 && v59 != -8192 )
              sub_BD6050((unsigned __int64 *)(v58 + 8), v174 & 0xFFFFFFFFFFFFFFF8LL);
            v59 = v176;
          }
          v58 += 48;
          *(_QWORD *)(v58 - 16) = v177;
        }
        while ( v56 != (_QWORD *)v58 );
        a1[12] = 0;
        v173 = (__int64)&unk_49DB368;
        if ( v59 == -4096 || v59 == 0 || v59 == -8192 )
          goto LABEL_83;
      }
      sub_BD60C0(&v174);
LABEL_83:
      if ( *((_BYTE *)a1 + 144) )
      {
        v61 = *((unsigned int *)a1 + 34);
        *((_BYTE *)a1 + 144) = 0;
        if ( (_DWORD)v61 )
        {
          v62 = (_QWORD *)a1[15];
          v63 = &v62[2 * (unsigned int)v61];
          do
          {
            if ( *v62 != -8192 && *v62 != -4096 )
            {
              v64 = v62[1];
              if ( v64 )
                sub_B91220((__int64)(v62 + 1), v64);
            }
            v62 += 2;
          }
          while ( v63 != v62 );
          v61 = *((unsigned int *)a1 + 34);
        }
        sub_C7D6A0(a1[15], 16 * v61, 8);
      }
      nullsub_61();
      v196 = &unk_49DA100;
      nullsub_63();
      if ( v178 != (unsigned int *)v160 )
        _libc_free((unsigned __int64)v178);
LABEL_28:
      n = *(_QWORD *)(n + 8);
    }
    while ( v161 != (_QWORD **)n );
  }
  v178 = 0;
  LODWORD(v181) = 128;
  v65 = sub_C7D670(0x2000, 8);
  v180 = 0;
  v66 = (_QWORD *)v65;
  v179 = v65;
  v174 = 2;
  v67 = (_QWORD *)(v65 + ((unsigned __int64)(unsigned int)v181 << 6));
  v175 = 0;
  v176 = -4096;
  v173 = (__int64)&unk_49DD7B0;
  v177 = 0;
  if ( (_QWORD *)v65 != v67 )
  {
    v68 = -4096;
    do
    {
      if ( v66 )
      {
        v69 = v174;
        v66[2] = 0;
        v66[3] = v68;
        v66[1] = v69 & 6;
        if ( v68 != 0 && v68 != -4096 && v68 != -8192 )
        {
          sub_BD6050(v66 + 1, v69 & 0xFFFFFFFFFFFFFFF8LL);
          v68 = v176;
        }
        *v66 = &unk_49DD7B0;
        v66[4] = v177;
      }
      v66 += 8;
    }
    while ( v67 != v66 );
    v173 = (__int64)&unk_49DB368;
    if ( v68 != 0 && v68 != -4096 && v68 != -8192 )
      sub_BD60C0(&v174);
  }
  v70 = *((_DWORD *)a1 + 4);
  LOBYTE(v186) = 0;
  if ( !v70 )
    goto LABEL_115;
  v77 = *((unsigned int *)a1 + 6);
  v78 = (_QWORD *)a1[1];
  v79 = v78;
  v80 = &v78[6 * v77];
  if ( v78 == v80 )
    goto LABEL_132;
  while ( 1 )
  {
    v81 = v79[3];
    if ( v81 != -8192 && v81 != -4096 )
      break;
    v79 += 6;
    if ( v80 == v79 )
      goto LABEL_132;
  }
  if ( v79 != v80 )
  {
    i = a1;
    v167 = (__int64 *)&unk_49DB358;
    v165 = (__int64)&v174;
    while ( 1 )
    {
      v109 = v79[3];
      v110 = v79[5];
      v174 = 2;
      v175 = 0;
      if ( v109 )
      {
        v176 = v109;
        if ( v109 != -4096 && v109 != -8192 )
          sub_BD73F0(v165);
      }
      else
      {
        v176 = 0;
      }
      v177 = &v178;
      v173 = (__int64)&unk_49DD7B0;
      if ( !(_DWORD)v181 )
        break;
      v111 = v176;
      v120 = ((_DWORD)v181 - 1) & (((unsigned int)v176 >> 9) ^ ((unsigned int)v176 >> 4));
      v112 = (_QWORD *)(v179 + (v120 << 6));
      v121 = v112[3];
      if ( v121 == v176 )
        goto LABEL_207;
      v122 = 1;
      v123 = 0;
      while ( v121 != -4096 )
      {
        if ( !v123 && v121 == -8192 )
          v123 = v112;
        LODWORD(v120) = (v181 - 1) & (v122 + v120);
        v112 = (_QWORD *)(v179 + ((unsigned __int64)(unsigned int)v120 << 6));
        v121 = v112[3];
        if ( v176 == v121 )
          goto LABEL_207;
        ++v122;
      }
      if ( v123 )
        v112 = v123;
      v178 = (unsigned int *)((char *)v178 + 1);
      v113 = v180 + 1;
      if ( 4 * ((int)v180 + 1) >= (unsigned int)(3 * v181) )
        goto LABEL_194;
      if ( (int)v181 - HIDWORD(v180) - v113 <= (unsigned int)v181 >> 3 )
      {
        sub_CF32C0((__int64)&v178, v181);
        if ( (_DWORD)v181 )
        {
          v111 = v176;
          v124 = 1;
          v125 = 0;
          LODWORD(v126) = (v181 - 1) & (((unsigned int)v176 >> 9) ^ ((unsigned int)v176 >> 4));
          v112 = (_QWORD *)(v179 + ((unsigned __int64)(unsigned int)v126 << 6));
          v127 = v112[3];
          if ( v127 != v176 )
          {
            while ( v127 != -4096 )
            {
              if ( !v125 && v127 == -8192 )
                v125 = v112;
              v126 = ((_DWORD)v181 - 1) & (unsigned int)(v126 + v124);
              v112 = (_QWORD *)(v179 + (v126 << 6));
              v127 = v112[3];
              if ( v176 == v127 )
                goto LABEL_196;
              ++v124;
            }
LABEL_235:
            if ( v125 )
              v112 = v125;
          }
LABEL_196:
          v113 = v180 + 1;
          goto LABEL_197;
        }
LABEL_195:
        v111 = v176;
        v112 = 0;
        goto LABEL_196;
      }
LABEL_197:
      LODWORD(v180) = v113;
      v114 = v112[3];
      if ( v114 == -4096 )
      {
        v115 = v112 + 1;
        if ( v111 != -4096 )
          goto LABEL_202;
      }
      else
      {
        --HIDWORD(v180);
        if ( v111 != v114 )
        {
          v115 = v112 + 1;
          if ( v114 != -8192 && v114 )
          {
            v164 = (char *)(v112 + 1);
            sub_BD60C0(v115);
            v111 = v176;
            v115 = v112 + 1;
          }
LABEL_202:
          v112[3] = v111;
          if ( v111 != -4096 && v111 != 0 && v111 != -8192 )
            sub_BD6050(v115, v174 & 0xFFFFFFFFFFFFFFF8LL);
          v111 = v176;
        }
      }
      v116 = v177;
      v112[5] = 6;
      v112[6] = 0;
      v112[4] = v116;
      v112[7] = 0;
LABEL_207:
      v117 = v112 + 5;
      v173 = (__int64)(v167 + 2);
      if ( v111 != -4096 && v111 != 0 && v111 != -8192 )
        sub_BD60C0((_QWORD *)v165);
      v118 = v117[2];
      if ( v118 != v110 )
      {
        if ( v118 != -4096 && v118 != 0 && v118 != -8192 )
          sub_BD60C0(v117);
        v117[2] = v110;
        if ( v110 != -4096 && v110 != 0 && v110 != -8192 )
          sub_BD73F0((__int64)v117);
      }
      v79 += 6;
      if ( v79 != v80 )
      {
        while ( 1 )
        {
          v119 = v79[3];
          if ( v119 != -8192 && v119 != -4096 )
            break;
          v79 += 6;
          if ( v80 == v79 )
            goto LABEL_221;
        }
        if ( v80 != v79 )
          continue;
      }
LABEL_221:
      a1 = i;
      if ( *((_DWORD *)i + 4) )
      {
        v78 = (_QWORD *)i[1];
        v77 = *((unsigned int *)i + 6);
        goto LABEL_132;
      }
LABEL_115:
      if ( (_BYTE)v186 )
        goto LABEL_118;
LABEL_116:
      v71 = (unsigned int)v181;
      if ( (_DWORD)v181 )
        goto LABEL_138;
      goto LABEL_117;
    }
    v178 = (unsigned int *)((char *)v178 + 1);
LABEL_194:
    sub_CF32C0((__int64)&v178, 2 * v181);
    if ( (_DWORD)v181 )
    {
      v111 = v176;
      v128 = 1;
      v125 = 0;
      LODWORD(v129) = (v181 - 1) & (((unsigned int)v176 >> 9) ^ ((unsigned int)v176 >> 4));
      v112 = (_QWORD *)(v179 + ((unsigned __int64)(unsigned int)v129 << 6));
      v130 = v112[3];
      if ( v130 != v176 )
      {
        while ( v130 != -4096 )
        {
          if ( v130 == -8192 && !v125 )
            v125 = v112;
          v129 = ((_DWORD)v181 - 1) & (unsigned int)(v129 + v128);
          v112 = (_QWORD *)(v179 + (v129 << 6));
          v130 = v112[3];
          if ( v176 == v130 )
            goto LABEL_196;
          ++v128;
        }
        goto LABEL_235;
      }
      goto LABEL_196;
    }
    goto LABEL_195;
  }
LABEL_132:
  v82 = &v78[6 * v77];
  if ( v78 == v82 )
    goto LABEL_115;
  do
  {
    v83 = v78[3];
    if ( v83 != -8192 && v83 != -4096 )
    {
      if ( v82 == v78 )
        goto LABEL_115;
      v89 = v82;
      v167 = &v170;
      v164 = (char *)&unk_4A259B8;
      while ( 1 )
      {
        v90 = v78 + 6;
        v91 = v78[3];
        for ( i = (__int64 *)v78[5]; v90 != v89; v90 += 6 )
        {
          v92 = v90[3];
          if ( v92 != -4096 && v92 != -8192 )
            break;
        }
        v174 = 2;
        v175 = 0;
        v176 = -8192;
        v173 = (__int64)v164;
        v177 = 0;
        v93 = v78[3];
        if ( v93 == -8192 )
        {
          v78[4] = 0;
        }
        else
        {
          if ( !v93 || v93 == -4096 )
          {
            v78[3] = -8192;
            v95 = v176;
            v96 = v177;
            v97 = v176 != 0;
            v98 = v176 == -4096;
          }
          else
          {
            v165 = (__int64)(v78 + 1);
            sub_BD60C0(v78 + 1);
            v94 = v176;
            v98 = v176 == 0;
            v78[3] = v176;
            if ( v94 == -4096 || v98 || v94 == -8192 )
            {
              v78[4] = v177;
              goto LABEL_170;
            }
            sub_BD6050((unsigned __int64 *)v165, v174 & 0xFFFFFFFFFFFFFFF8LL);
            v95 = v176;
            v96 = v177;
            v97 = v176 != -4096;
            v98 = v176 == 0;
          }
          v78[4] = v96;
          v173 = (__int64)&unk_49DB368;
          if ( v95 != -8192 && !v98 && v97 )
            sub_BD60C0(&v174);
        }
LABEL_170:
        --*((_DWORD *)a1 + 4);
        v99 = i;
        ++*((_DWORD *)a1 + 5);
        v100 = sub_ADAFB0((unsigned __int64)v99, *(_QWORD *)(v91 + 8));
        sub_BD84D0(v91, v100);
        v102 = (char *)sub_BD5D20(v91);
        v103 = (size_t)v101;
        v104 = (__int64)v101;
        v168 = v167;
        if ( &v102[(_QWORD)v101] && !v102 )
          sub_426248((__int64)"basic_string::_M_construct null not valid");
        v173 = (__int64)v101;
        if ( (unsigned __int64)v101 > 0xF )
        {
          n = (size_t)v101;
          src = v102;
          v165 = (__int64)&v168;
          v107 = (__int64 *)sub_22409D0((__int64)&v168, (unsigned __int64 *)&v173, 0);
          v102 = (char *)src;
          v103 = n;
          v168 = v107;
          v108 = v107;
          v170 = v173;
        }
        else
        {
          if ( v101 == (const char *)1 )
          {
            v105 = *v102;
            v165 = (__int64)&v168;
            LOBYTE(v170) = v105;
            v106 = v167;
            goto LABEL_175;
          }
          if ( !v101 )
          {
            v106 = v167;
            v165 = (__int64)&v168;
            goto LABEL_175;
          }
          v108 = v167;
          v165 = (__int64)&v168;
        }
        memcpy(v108, v102, v103);
        v104 = v173;
        v106 = v168;
LABEL_175:
        v169 = v104;
        *((_BYTE *)v106 + v104) = 0;
        sub_B30290(v91);
        LOWORD(v177) = 260;
        v173 = v165;
        sub_BD6B50((unsigned __int8 *)i, (const char **)&v173);
        if ( v168 != v167 )
          j_j___libc_free_0((unsigned __int64)v168);
        if ( v90 == v89 )
          goto LABEL_115;
        v78 = v90;
      }
    }
    v78 += 6;
  }
  while ( v82 != v78 );
  if ( !(_BYTE)v186 )
    goto LABEL_116;
LABEL_118:
  v73 = (unsigned int)v185;
  LOBYTE(v186) = 0;
  if ( (_DWORD)v185 )
  {
    v74 = v183;
    v75 = &v183[2 * (unsigned int)v185];
    do
    {
      if ( *v74 != -8192 && *v74 != -4096 )
      {
        v76 = v74[1];
        if ( v76 )
          sub_B91220((__int64)(v74 + 1), v76);
      }
      v74 += 2;
    }
    while ( v75 != v74 );
    v73 = (unsigned int)v185;
  }
  sub_C7D6A0((__int64)v183, 16 * v73, 8);
  v71 = (unsigned int)v181;
  if ( (_DWORD)v181 )
  {
LABEL_138:
    v84 = (_QWORD *)v179;
    v169 = 2;
    v170 = 0;
    v85 = (_QWORD *)(v179 + (v71 << 6));
    v171 = -4096;
    v168 = (__int64 *)&unk_49DD7B0;
    v173 = (__int64)&unk_49DD7B0;
    v86 = -4096;
    v172 = 0;
    v174 = 2;
    v175 = 0;
    v176 = -8192;
    v177 = 0;
    v167 = (__int64 *)&unk_49DB358;
    while ( 1 )
    {
      v87 = v84[3];
      if ( v86 != v87 && v87 != v176 )
      {
        v88 = v84[7];
        if ( v88 != -4096 && v88 != 0 && v88 != -8192 )
        {
          sub_BD60C0(v84 + 5);
          v87 = v84[3];
        }
      }
      *v84 = &unk_49DB368;
      if ( v87 != 0 && v87 != -4096 && v87 != -8192 )
        sub_BD60C0(v84 + 1);
      v84 += 8;
      if ( v85 == v84 )
        break;
      v86 = v171;
    }
    v173 = (__int64)(v167 + 2);
    if ( v176 != -4096 && v176 != 0 && v176 != -8192 )
      sub_BD60C0(&v174);
    v168 = v167 + 2;
    if ( v171 != 0 && v171 != -4096 && v171 != -8192 )
      sub_BD60C0(&v169);
    v71 = (unsigned int)v181;
  }
LABEL_117:
  sub_C7D6A0(v179, v71 << 6, 8);
  return 1;
}
