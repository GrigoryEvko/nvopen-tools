// Function: sub_2CFAA90
// Address: 0x2cfaa90
//
_BOOL8 __fastcall sub_2CFAA90(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  _QWORD *v3; // r14
  unsigned int v4; // eax
  __int64 v5; // rsi
  _QWORD *v6; // r12
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r13
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 **v14; // r14
  __int64 v15; // r15
  __int64 *v16; // rax
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  int v20; // edx
  __int64 v21; // r12
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rcx
  __int64 v30; // r12
  __int64 v31; // rax
  unsigned __int64 v32; // rax
  __int64 v33; // rdi
  __int16 v34; // dx
  __int64 v35; // rsi
  char v36; // al
  char v37; // dl
  __int64 v38; // rdi
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // r13
  __int64 v42; // rax
  unsigned __int64 v43; // r12
  __int64 v44; // rax
  __int64 **v45; // r15
  __int64 v46; // rax
  const char *v47; // rsi
  __int64 v48; // r8
  __int64 v49; // r9
  const char *v50; // r13
  __int64 v51; // rax
  int v52; // ecx
  char *v53; // rdx
  __int64 v54; // r15
  unsigned __int64 v55; // r15
  __int64 v56; // rax
  char v57; // al
  __int16 v58; // cx
  _QWORD *v59; // rax
  __int64 v60; // r9
  __int64 v61; // r13
  __int64 v62; // r15
  char *v63; // r12
  __int64 v64; // rdx
  unsigned int v65; // esi
  unsigned int v66; // r15d
  __int64 v67; // r12
  unsigned int v68; // r14d
  unsigned __int64 v69; // rax
  __int64 v70; // rax
  __int64 v71; // rax
  __int64 v72; // rdx
  __int64 v73; // rdx
  _QWORD *v74; // rax
  __int64 v75; // r14
  _DWORD *v76; // rax
  __int64 v77; // rax
  __int64 v78; // r13
  __int64 (__fastcall *v79)(__int64, __int64, unsigned __int8 *, _BYTE **, __int64, int); // rax
  _BYTE **v80; // rax
  __int64 *v81; // r10
  _BYTE **v82; // rcx
  __int64 v83; // r11
  unsigned int v84; // ecx
  __int64 v85; // rbx
  char *v86; // r13
  __int64 v87; // rdx
  unsigned int v88; // esi
  __int64 v89; // r14
  __int64 v90; // rax
  _QWORD *v91; // rdi
  _QWORD *v92; // r12
  unsigned int v93; // eax
  __int64 v94; // rsi
  _QWORD *v95; // r15
  __int64 v96; // rax
  __int64 v97; // rdx
  __int64 v99; // rax
  int v100; // edx
  char v101; // dl
  int v102; // eax
  __int64 v103; // rax
  unsigned __int64 v104; // rsi
  unsigned __int64 v105; // r15
  __int64 v106; // rax
  _QWORD *v107; // rbx
  _QWORD *v108; // r12
  __int64 v109; // rsi
  _QWORD *v110; // rbx
  _QWORD *v111; // r12
  __int64 v112; // rax
  __int64 v113; // r13
  __int64 v114; // rax
  __int64 v115; // rax
  _QWORD *v116; // rbx
  _QWORD *v117; // r13
  __int64 v118; // rsi
  __int64 v119; // rax
  unsigned int v120; // eax
  _QWORD *v121; // r12
  unsigned __int64 v122; // rdx
  unsigned __int64 v123; // rax
  _QWORD *v124; // rax
  __int64 v125; // rdx
  _QWORD *v126; // rdx
  char v127; // cl
  __int64 v128; // rax
  unsigned int v129; // eax
  _QWORD *v130; // r14
  unsigned __int64 v131; // rdx
  unsigned __int64 v132; // rax
  _QWORD *v133; // rax
  __int64 v134; // rdx
  _QWORD *v135; // rdx
  char v136; // cl
  _QWORD *v137; // r13
  char v138; // al
  __int64 v139; // rax
  bool v140; // zf
  _QWORD *v141; // r15
  char v142; // al
  __int64 v143; // rax
  __int64 v144; // [rsp+8h] [rbp-268h]
  unsigned int v145; // [rsp+10h] [rbp-260h]
  unsigned int v146; // [rsp+14h] [rbp-25Ch]
  __int64 v147; // [rsp+18h] [rbp-258h]
  __int64 v148; // [rsp+20h] [rbp-250h]
  __int64 v149; // [rsp+30h] [rbp-240h]
  __int64 v150; // [rsp+38h] [rbp-238h]
  __int64 v151; // [rsp+40h] [rbp-230h]
  __int64 v152; // [rsp+48h] [rbp-228h]
  __int64 v153; // [rsp+50h] [rbp-220h]
  _QWORD *v154; // [rsp+70h] [rbp-200h]
  __int64 v155; // [rsp+78h] [rbp-1F8h]
  _QWORD *v156; // [rsp+80h] [rbp-1F0h]
  __int64 v157; // [rsp+88h] [rbp-1E8h]
  __int64 v159; // [rsp+98h] [rbp-1D8h]
  __int64 j; // [rsp+A8h] [rbp-1C8h]
  __int64 v161; // [rsp+B0h] [rbp-1C0h]
  __int64 **v162; // [rsp+B8h] [rbp-1B8h]
  unsigned __int8 *v163; // [rsp+E0h] [rbp-190h]
  __int64 v164; // [rsp+E8h] [rbp-188h]
  _QWORD *v165; // [rsp+F0h] [rbp-180h]
  unsigned int v166; // [rsp+F8h] [rbp-178h]
  bool v167; // [rsp+FDh] [rbp-173h]
  __int16 v168; // [rsp+FEh] [rbp-172h]
  __int64 v169; // [rsp+100h] [rbp-170h]
  __int64 v170; // [rsp+108h] [rbp-168h]
  int v171; // [rsp+108h] [rbp-168h]
  __int64 v172; // [rsp+108h] [rbp-168h]
  __int64 v173; // [rsp+108h] [rbp-168h]
  __int64 v174; // [rsp+110h] [rbp-160h]
  _BYTE *v176; // [rsp+120h] [rbp-150h] BYREF
  __int64 v177; // [rsp+128h] [rbp-148h] BYREF
  _QWORD v178[2]; // [rsp+130h] [rbp-140h] BYREF
  unsigned __int64 v179; // [rsp+140h] [rbp-130h] BYREF
  __int64 v180; // [rsp+148h] [rbp-128h]
  const char *v181; // [rsp+150h] [rbp-120h] BYREF
  char v182; // [rsp+170h] [rbp-100h]
  char v183; // [rsp+171h] [rbp-FFh]
  const char *v184; // [rsp+180h] [rbp-F0h] BYREF
  __int64 v185; // [rsp+188h] [rbp-E8h] BYREF
  unsigned __int64 v186; // [rsp+190h] [rbp-E0h]
  __int64 v187; // [rsp+198h] [rbp-D8h]
  __int64 v188; // [rsp+1A0h] [rbp-D0h]
  char *v189; // [rsp+1B0h] [rbp-C0h] BYREF
  __int64 v190; // [rsp+1B8h] [rbp-B8h] BYREF
  __int64 v191; // [rsp+1C0h] [rbp-B0h] BYREF
  __int64 v192; // [rsp+1C8h] [rbp-A8h]
  __int64 i; // [rsp+1D0h] [rbp-A0h]
  __int64 v194; // [rsp+1E0h] [rbp-90h]
  __int64 v195; // [rsp+1E8h] [rbp-88h]
  __int64 v196; // [rsp+1F0h] [rbp-80h]
  __int64 v197; // [rsp+1F8h] [rbp-78h]
  void **v198; // [rsp+200h] [rbp-70h]
  void **v199; // [rsp+208h] [rbp-68h]
  __int64 v200; // [rsp+210h] [rbp-60h]
  int v201; // [rsp+218h] [rbp-58h]
  __int16 v202; // [rsp+21Ch] [rbp-54h]
  char v203; // [rsp+21Eh] [rbp-52h]
  __int64 v204; // [rsp+220h] [rbp-50h]
  __int64 v205; // [rsp+228h] [rbp-48h]
  void *v206; // [rsp+230h] [rbp-40h] BYREF
  void *v207; // [rsp+238h] [rbp-38h] BYREF

  v2 = *(unsigned int *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( __PAIR64__(*(_DWORD *)(a1 + 20), v2) )
  {
    v3 = *(_QWORD **)(a1 + 8);
    v4 = 4 * v2;
    v5 = 6LL * *(unsigned int *)(a1 + 24);
    if ( (unsigned int)(4 * v2) < 0x40 )
      v4 = 64;
    v6 = &v3[v5];
    if ( *(_DWORD *)(a1 + 24) > v4 )
    {
      v185 = 2;
      v186 = 0;
      v187 = -4096;
      v184 = (const char *)&unk_4A259B8;
      v188 = 0;
      v190 = 2;
      v191 = 0;
      v192 = -8192;
      v189 = (char *)&unk_4A259B8;
      i = 0;
      do
      {
        v128 = v3[3];
        *v3 = &unk_49DB368;
        if ( v128 != 0 && v128 != -4096 && v128 != -8192 )
          sub_BD60C0(v3 + 1);
        v3 += 6;
      }
      while ( v3 != v6 );
      v189 = (char *)&unk_49DB368;
      if ( v192 != -4096 && v192 != 0 && v192 != -8192 )
        sub_BD60C0(&v190);
      v184 = (const char *)&unk_49DB368;
      if ( v187 != 0 && v187 != -4096 && v187 != -8192 )
        sub_BD60C0(&v185);
      if ( (_DWORD)v2 )
      {
        v129 = v2 - 1;
        v2 = 64;
        if ( v129 )
        {
          _BitScanReverse(&v129, v129);
          v2 = (unsigned int)(1 << (33 - (v129 ^ 0x1F)));
          if ( (int)v2 < 64 )
            v2 = 64;
        }
      }
      v130 = *(_QWORD **)(a1 + 8);
      if ( *(_DWORD *)(a1 + 24) == (_DWORD)v2 )
      {
        *(_QWORD *)(a1 + 16) = 0;
        v190 = 2;
        v141 = &v130[6 * v2];
        v189 = (char *)&unk_4A259B8;
        v191 = 0;
        v192 = -4096;
        i = 0;
        if ( v141 != v130 )
        {
          do
          {
            if ( v130 )
            {
              v142 = v190;
              v130[2] = 0;
              v130[1] = v142 & 6;
              v143 = v192;
              v140 = v192 == -4096;
              v130[3] = v192;
              if ( v143 != 0 && !v140 && v143 != -8192 )
                sub_BD6050(v130 + 1, v190 & 0xFFFFFFFFFFFFFFF8LL);
              *v130 = &unk_4A259B8;
              v130[4] = i;
            }
            v130 += 6;
          }
          while ( v141 != v130 );
          v189 = (char *)&unk_49DB368;
          if ( v192 != 0 && v192 != -4096 && v192 != -8192 )
            goto LABEL_17;
        }
      }
      else
      {
        sub_C7D6A0((__int64)v130, v5 * 8, 8);
        if ( (_DWORD)v2 )
        {
          v131 = ((((((((4 * (int)v2 / 3u + 1) | ((unsigned __int64)(4 * (int)v2 / 3u + 1) >> 1)) >> 2)
                    | (4 * (int)v2 / 3u + 1)
                    | ((unsigned __int64)(4 * (int)v2 / 3u + 1) >> 1)) >> 4)
                  | (((4 * (int)v2 / 3u + 1) | ((unsigned __int64)(4 * (int)v2 / 3u + 1) >> 1)) >> 2)
                  | (4 * (int)v2 / 3u + 1)
                  | ((unsigned __int64)(4 * (int)v2 / 3u + 1) >> 1)) >> 8)
                | (((((4 * (int)v2 / 3u + 1) | ((unsigned __int64)(4 * (int)v2 / 3u + 1) >> 1)) >> 2)
                  | (4 * (int)v2 / 3u + 1)
                  | ((unsigned __int64)(4 * (int)v2 / 3u + 1) >> 1)) >> 4)
                | (((4 * (int)v2 / 3u + 1) | ((unsigned __int64)(4 * (int)v2 / 3u + 1) >> 1)) >> 2)
                | (4 * (int)v2 / 3u + 1)
                | ((unsigned __int64)(4 * (int)v2 / 3u + 1) >> 1)) >> 16;
          v132 = (v131
                | (((((((4 * (int)v2 / 3u + 1) | ((unsigned __int64)(4 * (int)v2 / 3u + 1) >> 1)) >> 2)
                    | (4 * (int)v2 / 3u + 1)
                    | ((unsigned __int64)(4 * (int)v2 / 3u + 1) >> 1)) >> 4)
                  | (((4 * (int)v2 / 3u + 1) | ((unsigned __int64)(4 * (int)v2 / 3u + 1) >> 1)) >> 2)
                  | (4 * (int)v2 / 3u + 1)
                  | ((unsigned __int64)(4 * (int)v2 / 3u + 1) >> 1)) >> 8)
                | (((((4 * (int)v2 / 3u + 1) | ((unsigned __int64)(4 * (int)v2 / 3u + 1) >> 1)) >> 2)
                  | (4 * (int)v2 / 3u + 1)
                  | ((unsigned __int64)(4 * (int)v2 / 3u + 1) >> 1)) >> 4)
                | (((4 * (int)v2 / 3u + 1) | ((unsigned __int64)(4 * (int)v2 / 3u + 1) >> 1)) >> 2)
                | (4 * (int)v2 / 3u + 1)
                | ((unsigned __int64)(4 * (int)v2 / 3u + 1) >> 1))
               + 1;
          *(_DWORD *)(a1 + 24) = v132;
          v133 = (_QWORD *)sub_C7D670(48 * v132, 8);
          v134 = *(unsigned int *)(a1 + 24);
          *(_QWORD *)(a1 + 16) = 0;
          *(_QWORD *)(a1 + 8) = v133;
          v190 = 2;
          v189 = (char *)&unk_4A259B8;
          v191 = 0;
          v135 = &v133[6 * v134];
          v192 = -4096;
          for ( i = 0; v135 != v133; v133 += 6 )
          {
            if ( v133 )
            {
              v136 = v190;
              v133[2] = 0;
              v133[3] = -4096;
              *v133 = &unk_4A259B8;
              v133[1] = v136 & 6;
              v133[4] = i;
            }
          }
        }
        else
        {
          *(_QWORD *)(a1 + 8) = 0;
          *(_QWORD *)(a1 + 16) = 0;
          *(_DWORD *)(a1 + 24) = 0;
        }
      }
    }
    else
    {
      v190 = 2;
      v191 = 0;
      v192 = -4096;
      v189 = (char *)&unk_4A259B8;
      v7 = -4096;
      i = 0;
      if ( v3 == v6 )
      {
        *(_QWORD *)(a1 + 16) = 0;
      }
      else
      {
        do
        {
          v8 = v3[3];
          if ( v8 != v7 )
          {
            if ( v8 != -4096 && v8 != 0 && v8 != -8192 )
            {
              sub_BD60C0(v3 + 1);
              v7 = v192;
            }
            v3[3] = v7;
            if ( v7 != 0 && v7 != -4096 && v7 != -8192 )
              sub_BD6050(v3 + 1, v190 & 0xFFFFFFFFFFFFFFF8LL);
            v7 = v192;
          }
          v3 += 6;
          *(v3 - 2) = i;
        }
        while ( v3 != v6 );
        *(_QWORD *)(a1 + 16) = 0;
        v189 = (char *)&unk_49DB368;
        if ( v7 != -8192 && v7 != 0 && v7 != -4096 )
LABEL_17:
          sub_BD60C0(&v190);
      }
    }
  }
  if ( *(_BYTE *)(a1 + 64) )
  {
    *(_BYTE *)(a1 + 64) = 0;
    v115 = *(unsigned int *)(a1 + 56);
    if ( (_DWORD)v115 )
    {
      v116 = *(_QWORD **)(a1 + 40);
      v117 = &v116[2 * (unsigned int)v115];
      do
      {
        if ( *v116 != -4096 && *v116 != -8192 )
        {
          v118 = v116[1];
          if ( v118 )
            sub_B91220((__int64)(v116 + 1), v118);
        }
        v116 += 2;
      }
      while ( v117 != v116 );
      v115 = *(unsigned int *)(a1 + 56);
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 40), 16 * v115, 8);
  }
  v167 = 0;
  for ( j = *(_QWORD *)(a2 + 32); a2 + 24 != j; j = *(_QWORD *)(j + 8) )
  {
    v9 = 0;
    if ( j )
      v9 = j - 56;
    v155 = v9;
    v10 = v9;
    if ( sub_B2FC80(v9) )
      continue;
    v174 = *(_QWORD *)(v10 + 80);
    v164 = v10 + 72;
    if ( v174 == v10 + 72 )
      continue;
    do
    {
      if ( !v174 )
        BUG();
      if ( *(_QWORD *)(v174 + 32) == v174 + 24 )
        goto LABEL_40;
      v11 = *(_QWORD *)(v174 + 32);
      while ( 2 )
      {
        v12 = v11;
        v11 = *(_QWORD *)(v11 + 8);
        if ( *(_BYTE *)(v12 - 24) != 85 )
          goto LABEL_29;
        v13 = *(_QWORD *)(v12 - 56);
        if ( !v13
          || *(_BYTE *)v13
          || *(_QWORD *)(v13 + 24) != *(_QWORD *)(v12 + 56)
          || (*(_BYTE *)(v13 + 33) & 0x20) == 0
          || *(_DWORD *)(v13 + 36) != 8922 )
        {
          goto LABEL_29;
        }
        v156 = *(_QWORD **)(v155 + 40);
        v154 = (_QWORD *)*v156;
        v14 = (__int64 **)sub_BCB2B0((_QWORD *)*v156);
        v178[0] = sub_BCE760(v14, 0);
        v15 = v178[0];
        v178[1] = v178[0];
        v16 = (__int64 *)sub_BCB2D0(v154);
        v17 = sub_BCF480(v16, v178, 2, 0);
        v18 = sub_BA8CA0((__int64)v156, (__int64)"vprintf", 7u, v17);
        v149 = v19;
        v20 = *(unsigned __int8 *)(v12 - 24);
        v150 = v18;
        v165 = (_QWORD *)(v12 - 24);
        if ( v20 == 40 )
        {
          v21 = 32LL * (unsigned int)sub_B491D0((__int64)v165);
        }
        else
        {
          v21 = 0;
          if ( v20 != 85 )
          {
            if ( v20 != 34 )
              BUG();
            v21 = 64;
          }
        }
        if ( *(char *)(v12 - 17) < 0 )
        {
          v22 = sub_BD2BC0((__int64)v165);
          v170 = v23 + v22;
          if ( *(char *)(v12 - 17) >= 0 )
          {
            if ( (unsigned int)(v170 >> 4) )
LABEL_265:
              BUG();
          }
          else if ( (unsigned int)((v170 - sub_BD2BC0((__int64)v165)) >> 4) )
          {
            if ( *(char *)(v12 - 17) >= 0 )
              goto LABEL_265;
            v171 = *(_DWORD *)(sub_BD2BC0((__int64)v165) + 8);
            if ( *(char *)(v12 - 17) >= 0 )
              BUG();
            v24 = sub_BD2BC0((__int64)v165);
            v26 = 32LL * (unsigned int)(*(_DWORD *)(v24 + v25 - 4) - v171);
            goto LABEL_49;
          }
        }
        v26 = 0;
LABEL_49:
        v27 = *(_DWORD *)(v12 - 20) & 0x7FFFFFF;
        v28 = -32 * v27;
        v29 = 32 * v27 - 32 - v21;
        v30 = *(_QWORD *)(v12 + v28 - 24);
        v172 = (v29 - v26) >> 5;
        if ( *(_BYTE *)v30 != 5 || *(_WORD *)(v30 + 2) != 34 || !(v167 = sub_2CF9460(v30)) )
          sub_C64ED0("The first argument for printf must be a string literal!", 1u);
        v31 = v151;
        LOWORD(v31) = 0;
        v151 = v31;
        v32 = sub_2CFA070(a1, v156, v30, v12, 0, 1);
        v180 = 0;
        v179 = v32;
        if ( (unsigned int)v172 <= 1 )
        {
          v180 = sub_AD6530(v15, (__int64)v156);
          goto LABEL_91;
        }
        v189 = "vprintfBuffer.local";
        LOWORD(i) = 259;
        v163 = (unsigned __int8 *)sub_BD2C40(80, 1u);
        if ( v163 )
          sub_B4CCA0((__int64)v163, (__int64 *)v14, 0, 0, 3u, (__int64)&v189, 0, 0);
        v33 = *(_QWORD *)(v155 + 80);
        if ( v33 )
          v33 -= 24;
        v35 = sub_AA5190(v33);
        if ( v35 )
        {
          v36 = v34;
          v37 = HIBYTE(v34);
        }
        else
        {
          v37 = 0;
          v36 = 0;
        }
        v38 = v148;
        v159 = (__int64)(v156 + 39);
        LOBYTE(v38) = v36;
        v39 = v38;
        BYTE1(v39) = v37;
        v148 = v39;
        sub_B44220(v163, v35, v39);
        v40 = (unsigned int)v172;
        v173 = 1;
        v147 = v11;
        v41 = 0;
        v157 = v40;
        do
        {
          v42 = 4 * (v173 - (*(_DWORD *)(v12 - 20) & 0x7FFFFFF));
          v43 = v165[v42];
          if ( *(_BYTE *)v43 == 5 && *(_WORD *)(v43 + 2) == 34 && sub_2CF9460(v165[v42]) )
          {
            sub_BD5D20(v43);
            v44 = v153;
            LOWORD(v44) = 0;
            v153 = v44;
            v43 = sub_2CFA070(a1, v156, v43, v12, 0, 0);
          }
          v45 = *(__int64 ***)(v43 + 8);
          v162 = (__int64 **)sub_BCE760(v45, 0);
          v161 = 1LL << sub_AE5020(v159, (__int64)v45);
          v169 = -v161 & (v161 + v41 - 1);
          v197 = sub_BD5C60((__int64)v165);
          v189 = (char *)&v191;
          v198 = &v206;
          v194 = 0;
          v199 = &v207;
          v195 = 0;
          v206 = &unk_49DA100;
          v190 = 0x200000000LL;
          v200 = 0;
          v201 = 0;
          v202 = 512;
          v203 = 7;
          v204 = 0;
          v205 = 0;
          LOWORD(v196) = 0;
          v207 = &unk_49DA0B0;
          v46 = *(_QWORD *)(v12 + 16);
          v195 = v12;
          v194 = v46;
          v47 = *(const char **)sub_B46C60((__int64)v165);
          v184 = v47;
          if ( v47 && (sub_B96E90((__int64)&v184, (__int64)v47, 1), (v50 = v184) != 0) )
          {
            v51 = (__int64)v189;
            v52 = v190;
            v53 = &v189[16 * (unsigned int)v190];
            if ( v189 != v53 )
            {
              while ( *(_DWORD *)v51 )
              {
                v51 += 16;
                if ( v53 == (char *)v51 )
                  goto LABEL_94;
              }
              *(_QWORD *)(v51 + 8) = v184;
LABEL_71:
              sub_B91220((__int64)&v184, (__int64)v50);
              goto LABEL_72;
            }
LABEL_94:
            if ( (unsigned int)v190 >= (unsigned __int64)HIDWORD(v190) )
            {
              v104 = (unsigned int)v190 + 1LL;
              v105 = v144 & 0xFFFFFFFF00000000LL;
              v144 &= 0xFFFFFFFF00000000LL;
              if ( HIDWORD(v190) < v104 )
              {
                sub_C8D5F0((__int64)&v189, &v191, v104, 0x10u, v48, v49);
                v53 = &v189[16 * (unsigned int)v190];
              }
              *(_QWORD *)v53 = v105;
              *((_QWORD *)v53 + 1) = v50;
              v50 = v184;
              LODWORD(v190) = v190 + 1;
            }
            else
            {
              if ( v53 )
              {
                *(_DWORD *)v53 = 0;
                *((_QWORD *)v53 + 1) = v50;
                v52 = v190;
                v50 = v184;
              }
              LODWORD(v190) = v52 + 1;
            }
          }
          else
          {
            sub_93FB40((__int64)&v189, 0);
            v50 = v184;
          }
          if ( v50 )
            goto LABEL_71;
LABEL_72:
          v54 = (__int64)v163;
          if ( !(_DWORD)v169 )
            goto LABEL_73;
          v183 = 1;
          v181 = "bufIndexed";
          v182 = 3;
          v76 = sub_AE2980(v159, 0);
          v77 = sub_BCCE00(v154, v76[1]);
          v176 = (_BYTE *)sub_ACD640(v77, (unsigned int)v169, 0);
          v78 = *((_QWORD *)v163 + 9);
          v79 = (__int64 (__fastcall *)(__int64, __int64, unsigned __int8 *, _BYTE **, __int64, int))*((_QWORD *)*v198 + 8);
          if ( v79 == sub_920540 )
          {
            if ( sub_BCEA30(*((_QWORD *)v163 + 9)) )
              goto LABEL_107;
            if ( *v163 > 0x15u )
              goto LABEL_107;
            v80 = sub_2CF94D0(&v176, (__int64)&v177);
            if ( v82 != v80 )
              goto LABEL_107;
            LOBYTE(v188) = 0;
            v54 = sub_AD9FD0(v78, v163, v81, 1, 0, (__int64)&v184, 0);
            if ( (_BYTE)v188 )
            {
              LOBYTE(v188) = 0;
              if ( (unsigned int)v187 > 0x40 && v186 )
                j_j___libc_free_0_0(v186);
              if ( (unsigned int)v185 > 0x40 && v184 )
                j_j___libc_free_0_0((unsigned __int64)v184);
            }
          }
          else
          {
            v54 = v79((__int64)v198, v78, v163, &v176, 1, 0);
          }
          if ( v54 )
            goto LABEL_73;
LABEL_107:
          LOWORD(v188) = 257;
          v54 = (__int64)sub_BD2C40(88, 2u);
          if ( !v54 )
            goto LABEL_110;
          v83 = *((_QWORD *)v163 + 1);
          v84 = v146 & 0xE0000000 | 2;
          v146 = v84;
          if ( (unsigned int)*(unsigned __int8 *)(v83 + 8) - 17 > 1 )
          {
            v99 = *((_QWORD *)v176 + 1);
            v100 = *(unsigned __int8 *)(v99 + 8);
            if ( v100 == 17 )
            {
              v101 = 0;
              goto LABEL_145;
            }
            if ( v100 == 18 )
            {
              v101 = 1;
LABEL_145:
              v102 = *(_DWORD *)(v99 + 32);
              BYTE4(v177) = v101;
              v145 = v84;
              LODWORD(v177) = v102;
              v103 = sub_BCE1B0((__int64 *)v83, v177);
              v84 = v145;
              v83 = v103;
            }
          }
          sub_B44260(v54, v83, 34, v84, 0, 0);
          *(_QWORD *)(v54 + 72) = v78;
          *(_QWORD *)(v54 + 80) = sub_B4DC50(v78, (__int64)&v176, 1);
          sub_B4D9A0(v54, (__int64)v163, (__int64 *)&v176, 1, (__int64)&v184);
LABEL_110:
          sub_B4DDE0(v54, 0);
          (*((void (__fastcall **)(void **, __int64, const char **, __int64, __int64))*v199 + 2))(
            v199,
            v54,
            &v181,
            v195,
            v196);
          if ( v189 != &v189[16 * (unsigned int)v190] )
          {
            v152 = v12;
            v85 = (__int64)v189;
            v86 = &v189[16 * (unsigned int)v190];
            do
            {
              v87 = *(_QWORD *)(v85 + 8);
              v88 = *(_DWORD *)v85;
              v85 += 16;
              sub_B99FD0(v54, v88, v87);
            }
            while ( v86 != (char *)v85 );
            v12 = v152;
          }
LABEL_73:
          v184 = "bcast";
          LOWORD(v188) = 259;
          v55 = sub_2CF9670((__int64 *)&v189, 0x31u, v54, v162, (__int64)&v184, 0, (int)v181, 0);
          v56 = sub_AA4E30(v194);
          v57 = sub_AE5020(v56, *(_QWORD *)(v43 + 8));
          HIBYTE(v58) = HIBYTE(v168);
          LOWORD(v188) = 257;
          LOBYTE(v58) = v57;
          v168 = v58;
          v59 = sub_BD2C40(80, unk_3F10A10);
          v61 = (__int64)v59;
          if ( v59 )
            sub_B4D3C0((__int64)v59, v43, v55, 0, v168, v60, 0, 0);
          (*((void (__fastcall **)(void **, __int64, const char **, __int64, __int64))*v199 + 2))(
            v199,
            v61,
            &v184,
            v195,
            v196);
          v62 = (__int64)v189;
          v63 = &v189[16 * (unsigned int)v190];
          if ( v189 != v63 )
          {
            do
            {
              v64 = *(_QWORD *)(v62 + 8);
              v65 = *(_DWORD *)v62;
              v62 += 16;
              sub_B99FD0(v61, v65, v64);
            }
            while ( v63 != (char *)v62 );
          }
          v41 = (unsigned int)(v161 + v169);
          nullsub_61();
          v206 = &unk_49DA100;
          nullsub_63();
          if ( v189 != (char *)&v191 )
            _libc_free((unsigned __int64)v189);
          ++v173;
        }
        while ( v157 != v173 );
        v66 = v161 + v169;
        v11 = v147;
        v67 = *((_QWORD *)v163 - 4);
        v68 = *(_DWORD *)(v67 + 32);
        if ( v68 > 0x40 )
        {
          if ( v68 - (unsigned int)sub_C444A0(v67 + 24) <= 0x40 )
          {
            v69 = **(_QWORD **)(v67 + 24);
            goto LABEL_82;
          }
        }
        else
        {
          v69 = *(_QWORD *)(v67 + 24);
LABEL_82:
          if ( v69 < v66 )
          {
            v70 = sub_BCB2D0(v154);
            v71 = sub_ACD640(v70, v66, 0);
            if ( *((_QWORD *)v163 - 4) )
            {
              v72 = *((_QWORD *)v163 - 3);
              **((_QWORD **)v163 - 2) = v72;
              if ( v72 )
                *(_QWORD *)(v72 + 16) = *((_QWORD *)v163 - 2);
            }
            *((_QWORD *)v163 - 4) = v71;
            if ( v71 )
            {
              v73 = *(_QWORD *)(v71 + 16);
              *((_QWORD *)v163 - 3) = v73;
              if ( v73 )
                *(_QWORD *)(v73 + 16) = v163 - 24;
              *((_QWORD *)v163 - 2) = v71 + 16;
              *(_QWORD *)(v71 + 16) = v163 - 32;
            }
          }
        }
        v180 = (__int64)v163;
LABEL_91:
        v189 = "vprintf";
        LOWORD(i) = 259;
        v74 = sub_BD2C40(88, 3u);
        v75 = (__int64)v74;
        if ( v74 )
        {
          v166 = v166 & 0xE0000000 | 3;
          sub_B44260((__int64)v74, **(_QWORD **)(v150 + 16), 56, v166, v12, 0);
          *(_QWORD *)(v75 + 72) = 0;
          sub_B4A290(v75, v150, v149, (__int64 *)&v179, 2, (__int64)&v189, 0, 0);
        }
        sub_BD84D0((__int64)v165, v75);
        sub_B43D60(v165);
LABEL_29:
        if ( v174 + 24 != v11 )
          continue;
        break;
      }
LABEL_40:
      v174 = *(_QWORD *)(v174 + 8);
    }
    while ( v164 != v174 );
  }
  v89 = *(unsigned int *)(a1 + 16);
  if ( !(_DWORD)v89 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_120;
  }
  v91 = *(_QWORD **)(a1 + 8);
  v110 = v91;
  v111 = &v91[6 * *(unsigned int *)(a1 + 24)];
  if ( v91 != v111 )
  {
    while ( 1 )
    {
      v112 = v110[3];
      if ( v112 != -8192 && v112 != -4096 )
        break;
      v110 += 6;
      if ( v111 == v110 )
        goto LABEL_172;
    }
    if ( v110 != v111 )
    {
      do
      {
        v113 = v110[3];
        sub_AD0030(v113);
        if ( !*(_QWORD *)(v113 + 16) && (*(_BYTE *)(v113 + 32) & 0xF) == 8 )
        {
          sub_BD6B90((unsigned __int8 *)v110[5], (unsigned __int8 *)v113);
          sub_B30290(v113);
        }
        v110 += 6;
        if ( v110 == v111 )
          break;
        while ( 1 )
        {
          v114 = v110[3];
          if ( v114 != -8192 && v114 != -4096 )
            break;
          v110 += 6;
          if ( v111 == v110 )
            goto LABEL_179;
        }
      }
      while ( v110 != v111 );
LABEL_179:
      v90 = a1;
      v89 = *(unsigned int *)(a1 + 16);
      ++*(_QWORD *)a1;
      if ( (_DWORD)v89 )
      {
LABEL_122:
        v91 = *(_QWORD **)(v90 + 8);
        goto LABEL_123;
      }
LABEL_120:
      v90 = a1;
      if ( !*(_DWORD *)(a1 + 20) )
        goto LABEL_139;
      v89 = 0;
      goto LABEL_122;
    }
  }
LABEL_172:
  ++*(_QWORD *)a1;
LABEL_123:
  v92 = v91;
  v93 = 4 * v89;
  v94 = 6LL * *(unsigned int *)(a1 + 24);
  if ( (unsigned int)(4 * v89) < 0x40 )
    v93 = 64;
  v95 = &v91[v94];
  if ( v93 < *(_DWORD *)(a1 + 24) )
  {
    v185 = 2;
    v186 = 0;
    v187 = -4096;
    v184 = (const char *)&unk_4A259B8;
    v188 = 0;
    v190 = 2;
    v191 = 0;
    v192 = -8192;
    v189 = (char *)&unk_4A259B8;
    i = 0;
    do
    {
      v119 = v92[3];
      *v92 = &unk_49DB368;
      if ( v119 != 0 && v119 != -4096 && v119 != -8192 )
        sub_BD60C0(v92 + 1);
      v92 += 6;
    }
    while ( v92 != v95 );
    v189 = (char *)&unk_49DB368;
    if ( v192 != 0 && v192 != -4096 && v192 != -8192 )
      sub_BD60C0(&v190);
    v184 = (const char *)&unk_49DB368;
    if ( v187 != 0 && v187 != -4096 && v187 != -8192 )
      sub_BD60C0(&v185);
    if ( (_DWORD)v89 )
    {
      v120 = v89 - 1;
      v89 = 64;
      if ( v120 )
      {
        _BitScanReverse(&v120, v120);
        v89 = (unsigned int)(1 << (33 - (v120 ^ 0x1F)));
        if ( (int)v89 < 64 )
          v89 = 64;
      }
    }
    v121 = *(_QWORD **)(a1 + 8);
    if ( *(_DWORD *)(a1 + 24) == (_DWORD)v89 )
    {
      *(_QWORD *)(a1 + 16) = 0;
      v190 = 2;
      v137 = &v121[6 * v89];
      v189 = (char *)&unk_4A259B8;
      v191 = 0;
      v192 = -4096;
      i = 0;
      if ( v137 != v121 )
      {
        do
        {
          if ( v121 )
          {
            v138 = v190;
            v121[2] = 0;
            v121[1] = v138 & 6;
            v139 = v192;
            v140 = v192 == 0;
            v121[3] = v192;
            if ( v139 != -4096 && !v140 && v139 != -8192 )
              sub_BD6050(v121 + 1, v190 & 0xFFFFFFFFFFFFFFF8LL);
            *v121 = &unk_4A259B8;
            v121[4] = i;
          }
          v121 += 6;
        }
        while ( v137 != v121 );
        v96 = v192;
        v189 = (char *)&unk_49DB368;
        if ( v192 != 0 && v192 != -4096 )
        {
LABEL_137:
          if ( v96 != -8192 )
            sub_BD60C0(&v190);
        }
      }
    }
    else
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 8), v94 * 8, 8);
      if ( (_DWORD)v89 )
      {
        v122 = ((((((((4 * (int)v89 / 3u + 1) | ((unsigned __int64)(4 * (int)v89 / 3u + 1) >> 1)) >> 2)
                  | (4 * (int)v89 / 3u + 1)
                  | ((unsigned __int64)(4 * (int)v89 / 3u + 1) >> 1)) >> 4)
                | (((4 * (int)v89 / 3u + 1) | ((unsigned __int64)(4 * (int)v89 / 3u + 1) >> 1)) >> 2)
                | (4 * (int)v89 / 3u + 1)
                | ((unsigned __int64)(4 * (int)v89 / 3u + 1) >> 1)) >> 8)
              | (((((4 * (int)v89 / 3u + 1) | ((unsigned __int64)(4 * (int)v89 / 3u + 1) >> 1)) >> 2)
                | (4 * (int)v89 / 3u + 1)
                | ((unsigned __int64)(4 * (int)v89 / 3u + 1) >> 1)) >> 4)
              | (((4 * (int)v89 / 3u + 1) | ((unsigned __int64)(4 * (int)v89 / 3u + 1) >> 1)) >> 2)
              | (4 * (int)v89 / 3u + 1)
              | ((unsigned __int64)(4 * (int)v89 / 3u + 1) >> 1)) >> 16;
        v123 = (v122
              | (((((((4 * (int)v89 / 3u + 1) | ((unsigned __int64)(4 * (int)v89 / 3u + 1) >> 1)) >> 2)
                  | (4 * (int)v89 / 3u + 1)
                  | ((unsigned __int64)(4 * (int)v89 / 3u + 1) >> 1)) >> 4)
                | (((4 * (int)v89 / 3u + 1) | ((unsigned __int64)(4 * (int)v89 / 3u + 1) >> 1)) >> 2)
                | (4 * (int)v89 / 3u + 1)
                | ((unsigned __int64)(4 * (int)v89 / 3u + 1) >> 1)) >> 8)
              | (((((4 * (int)v89 / 3u + 1) | ((unsigned __int64)(4 * (int)v89 / 3u + 1) >> 1)) >> 2)
                | (4 * (int)v89 / 3u + 1)
                | ((unsigned __int64)(4 * (int)v89 / 3u + 1) >> 1)) >> 4)
              | (((4 * (int)v89 / 3u + 1) | ((unsigned __int64)(4 * (int)v89 / 3u + 1) >> 1)) >> 2)
              | (4 * (int)v89 / 3u + 1)
              | ((unsigned __int64)(4 * (int)v89 / 3u + 1) >> 1))
             + 1;
        *(_DWORD *)(a1 + 24) = v123;
        v124 = (_QWORD *)sub_C7D670(48 * v123, 8);
        v125 = *(unsigned int *)(a1 + 24);
        *(_QWORD *)(a1 + 16) = 0;
        *(_QWORD *)(a1 + 8) = v124;
        v190 = 2;
        v189 = (char *)&unk_4A259B8;
        v191 = 0;
        v126 = &v124[6 * v125];
        v192 = -4096;
        for ( i = 0; v126 != v124; v124 += 6 )
        {
          if ( v124 )
          {
            v127 = v190;
            v124[2] = 0;
            v124[3] = -4096;
            *v124 = &unk_4A259B8;
            v124[1] = v127 & 6;
            v124[4] = i;
          }
        }
      }
      else
      {
        *(_QWORD *)(a1 + 8) = 0;
        *(_QWORD *)(a1 + 16) = 0;
        *(_DWORD *)(a1 + 24) = 0;
      }
    }
  }
  else
  {
    v190 = 2;
    v191 = 0;
    v192 = -4096;
    v189 = (char *)&unk_4A259B8;
    v96 = -4096;
    i = 0;
    if ( v91 == v95 )
    {
      *(_QWORD *)(a1 + 16) = 0;
      goto LABEL_139;
    }
    do
    {
      v97 = v92[3];
      if ( v97 != v96 )
      {
        if ( v97 != 0 && v97 != -4096 && v97 != -8192 )
        {
          sub_BD60C0(v92 + 1);
          v96 = v192;
        }
        v92[3] = v96;
        if ( v96 != 0 && v96 != -4096 && v96 != -8192 )
          sub_BD6050(v92 + 1, v190 & 0xFFFFFFFFFFFFFFF8LL);
        v96 = v192;
      }
      v92 += 6;
      *(v92 - 2) = i;
    }
    while ( v92 != v95 );
    *(_QWORD *)(a1 + 16) = 0;
    v189 = (char *)&unk_49DB368;
    if ( v96 != -4096 && v96 != 0 )
      goto LABEL_137;
  }
LABEL_139:
  if ( *(_BYTE *)(a1 + 64) )
  {
    *(_BYTE *)(a1 + 64) = 0;
    v106 = *(unsigned int *)(a1 + 56);
    if ( (_DWORD)v106 )
    {
      v107 = *(_QWORD **)(a1 + 40);
      v108 = &v107[2 * (unsigned int)v106];
      do
      {
        if ( *v107 != -8192 && *v107 != -4096 )
        {
          v109 = v107[1];
          if ( v109 )
            sub_B91220((__int64)(v107 + 1), v109);
        }
        v107 += 2;
      }
      while ( v108 != v107 );
      v106 = *(unsigned int *)(a1 + 56);
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 40), 16 * v106, 8);
  }
  return v167;
}
