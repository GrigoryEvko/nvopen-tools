// Function: sub_1CB1E60
// Address: 0x1cb1e60
//
_BOOL8 __fastcall sub_1CB1E60(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  int v11; // eax
  unsigned int v12; // ecx
  _QWORD *v13; // r13
  unsigned int v14; // edx
  _QWORD *v15; // r14
  __int64 j; // rdx
  __int64 v17; // rax
  __int64 v18; // rax
  bool v19; // zf
  __int64 v20; // rax
  __int64 v21; // rbx
  __int64 v22; // r15
  __int64 v23; // rbx
  __int64 v24; // rax
  __int64 *v25; // r14
  __int64 **v26; // r13
  __int64 *v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  int v30; // r12d
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rdx
  int v35; // edx
  unsigned int v36; // r12d
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 ***v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rcx
  _QWORD *v42; // rax
  __int64 v43; // rdi
  __int64 v44; // rsi
  __int64 v45; // rax
  unsigned int v46; // r12d
  __int64 v47; // rsi
  unsigned __int8 *v48; // rsi
  __int64 v49; // rdx
  __int64 ***v50; // r13
  __int64 *v51; // r15
  __int64 **v52; // r14
  __int64 v53; // rax
  __int64 v54; // rax
  unsigned __int8 *v55; // rsi
  __int64 v56; // r15
  _QWORD *v57; // rax
  _QWORD *v58; // r14
  unsigned __int64 *v59; // r13
  __int64 v60; // rax
  unsigned __int64 v61; // rcx
  int v62; // eax
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // r11
  __int64 v66; // rdi
  __int64 v67; // rax
  __int64 *v68; // r14
  __int64 v69; // rax
  __int64 v70; // rcx
  __int64 v71; // rsi
  unsigned __int8 *v72; // rsi
  _QWORD *v73; // rax
  _QWORD *v74; // rcx
  __int64 v75; // rax
  __int64 *v76; // rax
  __int64 *v77; // rax
  __int64 v78; // r11
  __int64 v79; // rcx
  __int64 v80; // r10
  __int64 v81; // rcx
  __int64 v82; // rax
  __int64 v83; // rsi
  __int64 v84; // rdx
  unsigned __int8 *v85; // rsi
  unsigned int v86; // r14d
  __int64 v87; // r12
  unsigned int v88; // ebx
  unsigned __int64 v89; // rax
  __int64 v90; // rax
  __int64 v91; // rax
  __int64 v92; // rcx
  unsigned __int64 v93; // rdx
  __int64 v94; // rdx
  __int64 v95; // r13
  _QWORD *v96; // rax
  double v97; // xmm4_8
  double v98; // xmm5_8
  __int64 v99; // r12
  _QWORD *v100; // rax
  __int64 v101; // rax
  _QWORD *v102; // rax
  int v103; // r14d
  __int64 v104; // rax
  _QWORD *v105; // rax
  _QWORD *v106; // rbx
  unsigned int v107; // eax
  _QWORD *v108; // r13
  __int64 m; // rdx
  __int64 v110; // rax
  __int64 v111; // rax
  unsigned int v113; // eax
  _QWORD *v114; // rbx
  _QWORD *v115; // r12
  __int64 v116; // rsi
  _QWORD *v117; // rbx
  _QWORD *v118; // r12
  __int64 v119; // rdx
  __int64 v120; // r13
  __int64 v121; // rax
  unsigned int v122; // eax
  _QWORD *v123; // r12
  _QWORD *v124; // r13
  __int64 v125; // rsi
  __int64 v126; // rax
  int v127; // edx
  __int64 v128; // rbx
  unsigned int v129; // eax
  _QWORD *v130; // r13
  unsigned __int64 v131; // rax
  unsigned __int64 v132; // rax
  __int64 v133; // rax
  _QWORD *v134; // rax
  __int64 v135; // rdx
  _QWORD *v136; // rdx
  char v137; // cl
  int v138; // r12d
  __int64 v139; // rdx
  int v140; // eax
  int v141; // edx
  __int64 v142; // r13
  unsigned int v143; // eax
  _QWORD *v144; // r14
  unsigned __int64 v145; // rdx
  unsigned __int64 v146; // rax
  _QWORD *v147; // rax
  __int64 v148; // rdx
  _QWORD *v149; // rdx
  char v150; // cl
  _QWORD *v151; // r12
  char v152; // al
  __int64 v153; // rax
  _QWORD *v154; // rbx
  char v155; // al
  __int64 v156; // rax
  __int64 v157; // [rsp+8h] [rbp-1C8h]
  __int64 v158; // [rsp+10h] [rbp-1C0h]
  __int64 v159; // [rsp+18h] [rbp-1B8h]
  int v160; // [rsp+20h] [rbp-1B0h]
  __int64 v161; // [rsp+28h] [rbp-1A8h]
  __int64 v162; // [rsp+30h] [rbp-1A0h]
  __int64 v163; // [rsp+38h] [rbp-198h]
  __int64 v164; // [rsp+40h] [rbp-190h]
  __int64 v165; // [rsp+48h] [rbp-188h]
  __int64 *v166; // [rsp+48h] [rbp-188h]
  __int64 v167; // [rsp+50h] [rbp-180h]
  __int64 v168; // [rsp+58h] [rbp-178h]
  _QWORD *v169; // [rsp+60h] [rbp-170h]
  _QWORD *v171; // [rsp+80h] [rbp-150h]
  __int64 v172; // [rsp+88h] [rbp-148h]
  __int64 v173; // [rsp+90h] [rbp-140h]
  bool v174; // [rsp+9Bh] [rbp-135h]
  unsigned int v175; // [rsp+9Ch] [rbp-134h]
  __int64 v176; // [rsp+A0h] [rbp-130h]
  __int64 v177; // [rsp+A8h] [rbp-128h]
  _QWORD *v178; // [rsp+B8h] [rbp-118h]
  __int64 v179; // [rsp+C0h] [rbp-110h]
  int v180; // [rsp+C0h] [rbp-110h]
  __int64 v181; // [rsp+C0h] [rbp-110h]
  __int64 k; // [rsp+C0h] [rbp-110h]
  int v183; // [rsp+C8h] [rbp-108h]
  __int64 v184; // [rsp+D0h] [rbp-100h] BYREF
  unsigned __int8 *v185; // [rsp+D8h] [rbp-F8h] BYREF
  _QWORD v186[2]; // [rsp+E0h] [rbp-F0h] BYREF
  __int64 ***v187; // [rsp+F0h] [rbp-E0h] BYREF
  __int64 v188; // [rsp+F8h] [rbp-D8h]
  __int64 v189[2]; // [rsp+100h] [rbp-D0h] BYREF
  char v190; // [rsp+110h] [rbp-C0h]
  char v191; // [rsp+111h] [rbp-BFh]
  unsigned __int8 *v192; // [rsp+120h] [rbp-B0h] BYREF
  __int64 v193; // [rsp+128h] [rbp-A8h] BYREF
  __int64 v194; // [rsp+130h] [rbp-A0h]
  __int64 v195; // [rsp+138h] [rbp-98h]
  __int64 v196; // [rsp+140h] [rbp-90h]
  char *v197; // [rsp+150h] [rbp-80h] BYREF
  __int64 v198; // [rsp+158h] [rbp-78h] BYREF
  __int64 *v199; // [rsp+160h] [rbp-70h]
  __int64 v200; // [rsp+168h] [rbp-68h]
  __int64 i; // [rsp+170h] [rbp-60h]
  int v202; // [rsp+178h] [rbp-58h]
  __int64 v203; // [rsp+180h] [rbp-50h]
  __int64 v204; // [rsp+188h] [rbp-48h]

  v11 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( v11 || *(_DWORD *)(a1 + 20) )
  {
    v193 = 2;
    v194 = 0;
    v12 = *(_DWORD *)(a1 + 24);
    v13 = *(_QWORD **)(a1 + 8);
    v195 = -8;
    v14 = 4 * v11;
    v15 = &v13[6 * v12];
    if ( (unsigned int)(4 * v11) < 0x40 )
      v14 = 64;
    if ( v12 > v14 )
    {
      v196 = 0;
      v198 = 2;
      v199 = 0;
      v192 = (unsigned __int8 *)&unk_49F8530;
      v200 = -16;
      v197 = (char *)&unk_49F8530;
      i = 0;
      v138 = v11;
      do
      {
        v139 = v13[3];
        *v13 = &unk_49EE2B0;
        if ( v139 != -8 && v139 != 0 && v139 != -16 )
          sub_1649B30(v13 + 1);
        v13 += 6;
      }
      while ( v13 != v15 );
      v140 = v138;
      v197 = (char *)&unk_49EE2B0;
      if ( v200 != -8 && v200 != 0 && v200 != -16 )
      {
        sub_1649B30(&v198);
        v140 = v138;
      }
      v192 = (unsigned __int8 *)&unk_49EE2B0;
      if ( v195 != 0 && v195 != -8 && v195 != -16 )
      {
        v183 = v140;
        sub_1649B30(&v193);
        v140 = v183;
      }
      v141 = *(_DWORD *)(a1 + 24);
      if ( v140 )
      {
        v142 = 64;
        v143 = v140 - 1;
        if ( v143 )
        {
          _BitScanReverse(&v143, v143);
          v142 = (unsigned int)(1 << (33 - (v143 ^ 0x1F)));
          if ( (int)v142 < 64 )
            v142 = 64;
        }
        v144 = *(_QWORD **)(a1 + 8);
        if ( (_DWORD)v142 == v141 )
        {
          v198 = 2;
          v199 = 0;
          *(_QWORD *)(a1 + 16) = 0;
          v151 = &v144[6 * v142];
          v200 = -8;
          v197 = (char *)&unk_49F8530;
          i = 0;
          do
          {
            if ( v144 )
            {
              v152 = v198;
              v144[2] = 0;
              v144[1] = v152 & 6;
              v153 = v200;
              v19 = v200 == -8;
              v144[3] = v200;
              if ( v153 != 0 && !v19 && v153 != -16 )
                sub_1649AC0(v144 + 1, v198 & 0xFFFFFFFFFFFFFFF8LL);
              *v144 = &unk_49F8530;
              v144[4] = i;
            }
            v144 += 6;
          }
          while ( v151 != v144 );
          v197 = (char *)&unk_49EE2B0;
          if ( v200 != -16 && v200 != -8 && v200 )
            sub_1649B30(&v198);
        }
        else
        {
          j___libc_free_0(*(_QWORD *)(a1 + 8));
          v145 = ((((((((4 * (int)v142 / 3u + 1) | ((unsigned __int64)(4 * (int)v142 / 3u + 1) >> 1)) >> 2)
                    | (4 * (int)v142 / 3u + 1)
                    | ((unsigned __int64)(4 * (int)v142 / 3u + 1) >> 1)) >> 4)
                  | (((4 * (int)v142 / 3u + 1) | ((unsigned __int64)(4 * (int)v142 / 3u + 1) >> 1)) >> 2)
                  | (4 * (int)v142 / 3u + 1)
                  | ((unsigned __int64)(4 * (int)v142 / 3u + 1) >> 1)) >> 8)
                | (((((4 * (int)v142 / 3u + 1) | ((unsigned __int64)(4 * (int)v142 / 3u + 1) >> 1)) >> 2)
                  | (4 * (int)v142 / 3u + 1)
                  | ((unsigned __int64)(4 * (int)v142 / 3u + 1) >> 1)) >> 4)
                | (((4 * (int)v142 / 3u + 1) | ((unsigned __int64)(4 * (int)v142 / 3u + 1) >> 1)) >> 2)
                | (4 * (int)v142 / 3u + 1)
                | ((unsigned __int64)(4 * (int)v142 / 3u + 1) >> 1)) >> 16;
          v146 = (v145
                | (((((((4 * (int)v142 / 3u + 1) | ((unsigned __int64)(4 * (int)v142 / 3u + 1) >> 1)) >> 2)
                    | (4 * (int)v142 / 3u + 1)
                    | ((unsigned __int64)(4 * (int)v142 / 3u + 1) >> 1)) >> 4)
                  | (((4 * (int)v142 / 3u + 1) | ((unsigned __int64)(4 * (int)v142 / 3u + 1) >> 1)) >> 2)
                  | (4 * (int)v142 / 3u + 1)
                  | ((unsigned __int64)(4 * (int)v142 / 3u + 1) >> 1)) >> 8)
                | (((((4 * (int)v142 / 3u + 1) | ((unsigned __int64)(4 * (int)v142 / 3u + 1) >> 1)) >> 2)
                  | (4 * (int)v142 / 3u + 1)
                  | ((unsigned __int64)(4 * (int)v142 / 3u + 1) >> 1)) >> 4)
                | (((4 * (int)v142 / 3u + 1) | ((unsigned __int64)(4 * (int)v142 / 3u + 1) >> 1)) >> 2)
                | (4 * (int)v142 / 3u + 1)
                | ((unsigned __int64)(4 * (int)v142 / 3u + 1) >> 1))
               + 1;
          *(_DWORD *)(a1 + 24) = v146;
          v147 = (_QWORD *)sub_22077B0(48 * v146);
          v148 = *(unsigned int *)(a1 + 24);
          *(_QWORD *)(a1 + 16) = 0;
          *(_QWORD *)(a1 + 8) = v147;
          v198 = 2;
          v199 = 0;
          v149 = &v147[6 * v148];
          v200 = -8;
          v197 = (char *)&unk_49F8530;
          for ( i = 0; v149 != v147; v147 += 6 )
          {
            if ( v147 )
            {
              v150 = v198;
              v147[2] = 0;
              v147[3] = -8;
              *v147 = &unk_49F8530;
              v147[1] = v150 & 6;
              v147[4] = i;
            }
          }
        }
      }
      else if ( v141 )
      {
        j___libc_free_0(*(_QWORD *)(a1 + 8));
        *(_QWORD *)(a1 + 8) = 0;
        *(_QWORD *)(a1 + 16) = 0;
        *(_DWORD *)(a1 + 24) = 0;
      }
      else
      {
        *(_QWORD *)(a1 + 16) = 0;
      }
    }
    else
    {
      v198 = 2;
      v196 = 0;
      v199 = 0;
      v192 = (unsigned __int8 *)&unk_49F8530;
      v200 = -16;
      v197 = (char *)&unk_49F8530;
      i = 0;
      if ( v13 == v15 )
      {
        *(_QWORD *)(a1 + 16) = 0;
      }
      else
      {
        for ( j = -8; ; j = v195 )
        {
          v17 = v13[3];
          if ( v17 != j )
          {
            if ( v17 != -8 && v17 != 0 && v17 != -16 )
            {
              sub_1649B30(v13 + 1);
              j = v195;
            }
            v13[3] = j;
            if ( j != 0 && j != -8 && j != -16 )
              sub_1649AC0(v13 + 1, v193 & 0xFFFFFFFFFFFFFFF8LL);
            v13[4] = v196;
          }
          v13 += 6;
          if ( v13 == v15 )
            break;
        }
        v18 = v200;
        v19 = v200 == -8;
        *(_QWORD *)(a1 + 16) = 0;
        v197 = (char *)&unk_49EE2B0;
        if ( v18 != 0 && !v19 && v18 != -16 )
          sub_1649B30(&v198);
      }
      v192 = (unsigned __int8 *)&unk_49EE2B0;
      if ( v195 != 0 && v195 != -8 && v195 != -16 )
        sub_1649B30(&v193);
    }
  }
  if ( *(_BYTE *)(a1 + 64) )
  {
    v122 = *(_DWORD *)(a1 + 56);
    if ( v122 )
    {
      v123 = *(_QWORD **)(a1 + 40);
      v124 = &v123[2 * v122];
      do
      {
        if ( *v123 != -8 && *v123 != -4 )
        {
          v125 = v123[1];
          if ( v125 )
            sub_161E7C0((__int64)(v123 + 1), v125);
        }
        v123 += 2;
      }
      while ( v124 != v123 );
    }
    j___libc_free_0(*(_QWORD *)(a1 + 40));
    *(_BYTE *)(a1 + 64) = 0;
  }
  v174 = 0;
  v157 = a2 + 24;
  v164 = *(_QWORD *)(a2 + 32);
  if ( v164 != a2 + 24 )
  {
    while ( 1 )
    {
      v20 = 0;
      if ( v164 )
        v20 = v164 - 56;
      v167 = v20;
      v21 = v20;
      if ( !sub_15E4F60(v20) )
      {
        v176 = *(_QWORD *)(v21 + 80);
        v168 = v21 + 72;
        if ( v176 != v21 + 72 )
          break;
      }
LABEL_27:
      v164 = *(_QWORD *)(v164 + 8);
      if ( v157 == v164 )
        goto LABEL_138;
    }
    while ( 1 )
    {
      if ( !v176 )
        BUG();
      v22 = *(_QWORD *)(v176 + 24);
      if ( v22 != v176 + 16 )
        break;
LABEL_39:
      v176 = *(_QWORD *)(v176 + 8);
      if ( v168 == v176 )
        goto LABEL_27;
    }
    while ( 1 )
    {
      v23 = v22;
      v22 = *(_QWORD *)(v22 + 8);
      if ( *(_BYTE *)(v23 - 8) == 78 )
      {
        v24 = *(_QWORD *)(v23 - 48);
        if ( !*(_BYTE *)(v24 + 16) && (*(_BYTE *)(v24 + 33) & 0x20) != 0 && *(_DWORD *)(v24 + 36) == 4045 )
          break;
      }
LABEL_38:
      if ( v176 + 16 == v22 )
        goto LABEL_39;
    }
    v173 = sub_1632FA0(*(_QWORD *)(v167 + 40));
    if ( !v173 )
      sub_16BD130("DataLayout must be available for lowering printf!", 1u);
    v171 = *(_QWORD **)(v167 + 40);
    v169 = (_QWORD *)*v171;
    v25 = (__int64 *)sub_1643330((_QWORD *)*v171);
    v186[0] = sub_1646BA0(v25, 0);
    v26 = (__int64 **)v186[0];
    v186[1] = v186[0];
    v27 = (__int64 *)sub_1643350(v169);
    v28 = sub_1644EA0(v27, v186, 2, 0);
    v163 = sub_1632190((__int64)v171, (__int64)"vprintf", 7, v28);
    v178 = (_QWORD *)(v23 - 24);
    v29 = *(_DWORD *)(v23 - 4) & 0xFFFFFFF;
    v30 = *(_DWORD *)(v23 - 4) & 0xFFFFFFF;
    if ( *(char *)(v23 - 1) < 0 )
    {
      v31 = sub_1648A40(v23 - 24);
      v179 = v32 + v31;
      if ( *(char *)(v23 - 1) >= 0 )
      {
        if ( (unsigned int)(v179 >> 4) )
LABEL_273:
          BUG();
      }
      else if ( (unsigned int)((v179 - sub_1648A40((__int64)v178)) >> 4) )
      {
        if ( *(char *)(v23 - 1) >= 0 )
          goto LABEL_273;
        v180 = *(_DWORD *)(sub_1648A40((__int64)v178) + 8);
        if ( *(char *)(v23 - 1) >= 0 )
          BUG();
        v33 = sub_1648A40((__int64)v178);
        v35 = *(_DWORD *)(v33 + v34 - 4) - v180;
        v29 = *(_DWORD *)(v23 - 4) & 0xFFFFFFF;
LABEL_48:
        v36 = v30 - 1 - v35;
        v37 = -3 * v29;
        v38 = v178[v37];
        if ( *(_BYTE *)(v38 + 16) != 5 || *(_WORD *)(v38 + 18) != 32 || (v181 = v178[v37], !(v174 = sub_1CB0BE0(v181))) )
          sub_16BD130("The first argument for printf must be a string literal!", 1u);
        v39 = sub_1CB14E0(a1, v171, v181, v23, 1);
        v188 = 0;
        v187 = v39;
        if ( v36 <= 1 )
        {
          v188 = sub_15A06D0(v26, (__int64)v171, v40, v41);
          goto LABEL_125;
        }
        v197 = "vprintfBuffer.local";
        LOWORD(v199) = 259;
        v42 = sub_1648A60(64, 1u);
        v177 = (__int64)v42;
        if ( v42 )
          sub_15F8A50((__int64)v42, v25, 0, 0, 8u, (__int64)&v197, 0);
        v43 = *(_QWORD *)(v167 + 80);
        if ( v43 )
          v43 -= 24;
        v44 = sub_157EE30(v43);
        if ( v44 )
          v44 -= 24;
        sub_15F2120(v177, v44);
        v45 = v36;
        v158 = v22;
        v46 = 0;
        v172 = v45;
        for ( k = 1; k != v172; ++k )
        {
          v49 = k - (*(_DWORD *)(v23 - 4) & 0xFFFFFFF);
          v50 = (__int64 ***)v178[3 * v49];
          if ( *((_BYTE *)v50 + 16) == 5 && *((_WORD *)v50 + 9) == 32 && sub_1CB0BE0(v178[3 * v49]) )
          {
            sub_1649960((__int64)v50);
            v50 = sub_1CB14E0(a1, v171, (__int64)v50, v23, 0);
          }
          v51 = (__int64 *)*v50;
          v52 = (__int64 **)sub_1646BA0((__int64 *)*v50, 0);
          v175 = sub_15A9FE0(v173, (__int64)v51);
          if ( v46 % v175 )
            v46 = v175 + v46 - v46 % v175;
          v53 = sub_16498A0((__int64)v178);
          v199 = 0;
          v197 = 0;
          v200 = v53;
          i = 0;
          v202 = 0;
          v203 = 0;
          v204 = 0;
          v54 = *(_QWORD *)(v23 + 16);
          v199 = (__int64 *)v23;
          v198 = v54;
          v55 = *(unsigned __int8 **)(v23 + 24);
          v192 = v55;
          if ( v55 )
          {
            sub_1623A60((__int64)&v192, (__int64)v55, 2);
            if ( v197 )
              sub_161E7C0((__int64)&v197, (__int64)v197);
            v197 = (char *)v192;
            if ( v192 )
              sub_1623210((__int64)&v192, v192, (__int64)&v197);
          }
          if ( v46 )
          {
            v191 = 1;
            v189[0] = (__int64)"bufIndexed";
            v190 = 3;
            v62 = sub_15A9520(v173, 0);
            v63 = sub_1644900(v169, 8 * v62);
            v64 = sub_159C470(v63, v46, 0);
            v184 = v64;
            v65 = *(_QWORD *)(v177 + 56);
            if ( *(_BYTE *)(v177 + 16) > 0x10u || *(_BYTE *)(v64 + 16) > 0x10u )
            {
              LOWORD(v194) = 257;
              if ( !v65 )
              {
                v101 = *(_QWORD *)v177;
                if ( *(_BYTE *)(*(_QWORD *)v177 + 8LL) == 16 )
                  v101 = **(_QWORD **)(v101 + 16);
                v65 = *(_QWORD *)(v101 + 24);
              }
              v165 = v65;
              v73 = sub_1648A60(72, 2u);
              v56 = (__int64)v73;
              if ( v73 )
              {
                v162 = (__int64)v73;
                v74 = v73 - 6;
                v75 = *(_QWORD *)v177;
                if ( *(_BYTE *)(*(_QWORD *)v177 + 8LL) == 16 )
                  v75 = **(_QWORD **)(v75 + 16);
                v159 = (__int64)v74;
                v160 = *(_DWORD *)(v75 + 8) >> 8;
                v76 = (__int64 *)sub_15F9F50(v165, (__int64)&v184, 1);
                v77 = (__int64 *)sub_1646BA0(v76, v160);
                v78 = v165;
                v79 = v159;
                v80 = (__int64)v77;
                if ( *(_BYTE *)(*(_QWORD *)v177 + 8LL) == 16 )
                {
                  v102 = sub_16463B0(v77, *(_QWORD *)(*(_QWORD *)v177 + 32LL));
                  v78 = v165;
                  v79 = v159;
                  v80 = (__int64)v102;
                }
                else if ( *(_BYTE *)(*(_QWORD *)v184 + 8LL) == 16 )
                {
                  v100 = sub_16463B0(v77, *(_QWORD *)(*(_QWORD *)v184 + 32LL));
                  v79 = v159;
                  v78 = v165;
                  v80 = (__int64)v100;
                }
                v161 = v78;
                sub_15F1EA0(v56, v80, 32, v79, 2, 0);
                *(_QWORD *)(v56 + 56) = v161;
                *(_QWORD *)(v56 + 64) = sub_15F9F50(v161, (__int64)&v184, 1);
                sub_15F9CE0(v56, v177, &v184, 1, (__int64)&v192);
              }
              else
              {
                v162 = 0;
              }
              if ( v198 )
              {
                v166 = v199;
                sub_157E9D0(v198 + 40, v56);
                v81 = *v166;
                v82 = *(_QWORD *)(v56 + 24) & 7LL;
                *(_QWORD *)(v56 + 32) = v166;
                v81 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)(v56 + 24) = v81 | v82;
                *(_QWORD *)(v81 + 8) = v56 + 24;
                *v166 = *v166 & 7 | (v56 + 24);
              }
              sub_164B780(v162, v189);
              if ( v197 )
              {
                v185 = (unsigned __int8 *)v197;
                sub_1623A60((__int64)&v185, (__int64)v197, 2);
                v83 = *(_QWORD *)(v56 + 48);
                v84 = v56 + 48;
                if ( v83 )
                {
                  sub_161E7C0(v56 + 48, v83);
                  v84 = v56 + 48;
                }
                v85 = v185;
                *(_QWORD *)(v56 + 48) = v185;
                if ( v85 )
                  sub_1623210((__int64)&v185, v85, v84);
              }
            }
            else
            {
              v66 = *(_QWORD *)(v177 + 56);
              BYTE4(v192) = 0;
              v185 = (unsigned __int8 *)v64;
              v56 = sub_15A2E80(v66, v177, (__int64 **)&v185, 1u, 0, (__int64)&v192, 0);
            }
          }
          else
          {
            v56 = v177;
          }
          v191 = 1;
          v189[0] = (__int64)"bcast";
          v190 = 3;
          if ( v52 != *(__int64 ***)v56 )
          {
            if ( *(_BYTE *)(v56 + 16) > 0x10u )
            {
              LOWORD(v194) = 257;
              v67 = sub_15FDBD0(47, v56, (__int64)v52, (__int64)&v192, 0);
              v56 = v67;
              if ( v198 )
              {
                v68 = v199;
                sub_157E9D0(v198 + 40, v67);
                v69 = *(_QWORD *)(v56 + 24);
                v70 = *v68;
                *(_QWORD *)(v56 + 32) = v68;
                v70 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)(v56 + 24) = v70 | v69 & 7;
                *(_QWORD *)(v70 + 8) = v56 + 24;
                *v68 = *v68 & 7 | (v56 + 24);
              }
              sub_164B780(v56, v189);
              if ( v197 )
              {
                v185 = (unsigned __int8 *)v197;
                sub_1623A60((__int64)&v185, (__int64)v197, 2);
                v71 = *(_QWORD *)(v56 + 48);
                if ( v71 )
                  sub_161E7C0(v56 + 48, v71);
                v72 = v185;
                *(_QWORD *)(v56 + 48) = v185;
                if ( v72 )
                  sub_1623210((__int64)&v185, v72, v56 + 48);
              }
            }
            else
            {
              v56 = sub_15A46C0(47, (__int64 ***)v56, v52, 0);
            }
          }
          LOWORD(v194) = 257;
          v57 = sub_1648A60(64, 2u);
          v58 = v57;
          if ( v57 )
            sub_15F9650((__int64)v57, (__int64)v50, v56, 0, 0);
          if ( v198 )
          {
            v59 = (unsigned __int64 *)v199;
            sub_157E9D0(v198 + 40, (__int64)v58);
            v60 = v58[3];
            v61 = *v59;
            v58[4] = v59;
            v61 &= 0xFFFFFFFFFFFFFFF8LL;
            v58[3] = v61 | v60 & 7;
            *(_QWORD *)(v61 + 8) = v58 + 3;
            *v59 = *v59 & 7 | (unsigned __int64)(v58 + 3);
          }
          sub_164B780((__int64)v58, (__int64 *)&v192);
          if ( v197 )
          {
            v189[0] = (__int64)v197;
            sub_1623A60((__int64)v189, (__int64)v197, 2);
            v47 = v58[6];
            if ( v47 )
              sub_161E7C0((__int64)(v58 + 6), v47);
            v48 = (unsigned __int8 *)v189[0];
            v58[6] = v189[0];
            if ( v48 )
              sub_1623210((__int64)v189, v48, (__int64)(v58 + 6));
            v46 += v175;
            if ( v197 )
              sub_161E7C0((__int64)&v197, (__int64)v197);
          }
          else
          {
            v46 += v175;
          }
        }
        v86 = v46;
        v22 = v158;
        v87 = *(_QWORD *)(v177 - 24);
        v88 = *(_DWORD *)(v87 + 32);
        if ( v88 > 0x40 )
        {
          if ( v88 - (unsigned int)sub_16A57B0(v87 + 24) > 0x40 )
          {
LABEL_124:
            v188 = v177;
LABEL_125:
            LOWORD(v199) = 259;
            v197 = "vprintf";
            v95 = *(_QWORD *)(*(_QWORD *)v163 + 24LL);
            v96 = sub_1648AB0(72, 3u, 0);
            v99 = (__int64)v96;
            if ( v96 )
            {
              sub_15F1EA0((__int64)v96, **(_QWORD **)(v95 + 16), 54, (__int64)(v96 - 9), 3, (__int64)v178);
              *(_QWORD *)(v99 + 56) = 0;
              sub_15F5B40(v99, v95, v163, (__int64 *)&v187, 2, (__int64)&v197, 0, 0);
            }
            sub_164D160((__int64)v178, v99, a3, a4, a5, a6, v97, v98, a9, a10);
            sub_15F20C0(v178);
            goto LABEL_38;
          }
          v89 = **(_QWORD **)(v87 + 24);
        }
        else
        {
          v89 = *(_QWORD *)(v87 + 24);
        }
        if ( v86 > v89 )
        {
          v90 = sub_1643350(v169);
          v91 = sub_159C470(v90, v86, 0);
          if ( *(_QWORD *)(v177 - 24) )
          {
            v92 = *(_QWORD *)(v177 - 16);
            v93 = *(_QWORD *)(v177 - 8) & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v93 = v92;
            if ( v92 )
              *(_QWORD *)(v92 + 16) = *(_QWORD *)(v92 + 16) & 3LL | v93;
          }
          *(_QWORD *)(v177 - 24) = v91;
          if ( v91 )
          {
            v94 = *(_QWORD *)(v91 + 8);
            *(_QWORD *)(v177 - 16) = v94;
            if ( v94 )
              *(_QWORD *)(v94 + 16) = (v177 - 16) | *(_QWORD *)(v94 + 16) & 3LL;
            *(_QWORD *)(v177 - 8) = (v91 + 8) | *(_QWORD *)(v177 - 8) & 3LL;
            *(_QWORD *)(v91 + 8) = v177 - 24;
          }
        }
        goto LABEL_124;
      }
      v35 = 0;
      v29 = *(_DWORD *)(v23 - 4) & 0xFFFFFFF;
      goto LABEL_48;
    }
    v35 = 0;
    goto LABEL_48;
  }
LABEL_138:
  v103 = *(_DWORD *)(a1 + 16);
  if ( !v103 )
  {
    ++*(_QWORD *)a1;
LABEL_140:
    v104 = a1;
    if ( !*(_DWORD *)(a1 + 20) )
      goto LABEL_164;
    v103 = 0;
    goto LABEL_142;
  }
  v105 = *(_QWORD **)(a1 + 8);
  v117 = v105;
  v118 = &v105[6 * *(unsigned int *)(a1 + 24)];
  if ( v105 == v118 )
    goto LABEL_178;
  while ( 1 )
  {
    v119 = v117[3];
    if ( v119 != -16 && v119 != -8 )
      break;
    v117 += 6;
    if ( v118 == v117 )
      goto LABEL_178;
  }
  if ( v117 == v118 )
  {
LABEL_178:
    ++*(_QWORD *)a1;
    goto LABEL_143;
  }
  do
  {
    v120 = v117[3];
    sub_159D9E0(v120);
    if ( !*(_QWORD *)(v120 + 8) && (*(_BYTE *)(v120 + 32) & 0xF) == 8 )
    {
      sub_164B7C0(v117[5], v120);
      sub_15E55B0(v120);
    }
    v117 += 6;
    if ( v117 == v118 )
      break;
    while ( 1 )
    {
      v121 = v117[3];
      if ( v121 != -8 && v121 != -16 )
        break;
      v117 += 6;
      if ( v118 == v117 )
        goto LABEL_185;
    }
  }
  while ( v117 != v118 );
LABEL_185:
  v104 = a1;
  v103 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v103 )
    goto LABEL_140;
LABEL_142:
  v105 = *(_QWORD **)(v104 + 8);
LABEL_143:
  v106 = v105;
  v193 = 2;
  v194 = 0;
  v195 = -8;
  v107 = 4 * v103;
  v108 = &v106[6 * *(unsigned int *)(a1 + 24)];
  if ( (unsigned int)(4 * v103) < 0x40 )
    v107 = 64;
  if ( v107 < *(_DWORD *)(a1 + 24) )
  {
    v196 = 0;
    v198 = 2;
    v199 = 0;
    v192 = (unsigned __int8 *)&unk_49F8530;
    v200 = -16;
    v197 = (char *)&unk_49F8530;
    i = 0;
    do
    {
      v126 = v106[3];
      *v106 = &unk_49EE2B0;
      if ( v126 != 0 && v126 != -8 && v126 != -16 )
        sub_1649B30(v106 + 1);
      v106 += 6;
    }
    while ( v106 != v108 );
    v197 = (char *)&unk_49EE2B0;
    if ( v200 != -8 && v200 != 0 && v200 != -16 )
      sub_1649B30(&v198);
    v192 = (unsigned __int8 *)&unk_49EE2B0;
    if ( v195 != 0 && v195 != -8 && v195 != -16 )
      sub_1649B30(&v193);
    v127 = *(_DWORD *)(a1 + 24);
    if ( v103 )
    {
      v128 = 64;
      if ( v103 != 1 )
      {
        _BitScanReverse(&v129, v103 - 1);
        v128 = (unsigned int)(1 << (33 - (v129 ^ 0x1F)));
        if ( (int)v128 < 64 )
          v128 = 64;
      }
      v130 = *(_QWORD **)(a1 + 8);
      if ( (_DWORD)v128 == v127 )
      {
        v198 = 2;
        v199 = 0;
        *(_QWORD *)(a1 + 16) = 0;
        v154 = &v130[6 * v128];
        v200 = -8;
        v197 = (char *)&unk_49F8530;
        i = 0;
        do
        {
          if ( v130 )
          {
            v155 = v198;
            v130[2] = 0;
            v130[1] = v155 & 6;
            v156 = v200;
            v19 = v200 == -8;
            v130[3] = v200;
            if ( v156 != 0 && !v19 && v156 != -16 )
              sub_1649AC0(v130 + 1, v198 & 0xFFFFFFFFFFFFFFF8LL);
            *v130 = &unk_49F8530;
            v130[4] = i;
          }
          v130 += 6;
        }
        while ( v154 != v130 );
        v197 = (char *)&unk_49EE2B0;
        if ( v200 != 0 && v200 != -8 && v200 != -16 )
          sub_1649B30(&v198);
      }
      else
      {
        j___libc_free_0(*(_QWORD *)(a1 + 8));
        v131 = (4 * (int)v128 / 3u + 1) | ((unsigned __int64)(4 * (int)v128 / 3u + 1) >> 1);
        v132 = (((v131 >> 2) | v131) >> 4) | (v131 >> 2) | v131;
        v133 = ((((v132 >> 8) | v132) >> 16) | (v132 >> 8) | v132) + 1;
        *(_DWORD *)(a1 + 24) = v133;
        v134 = (_QWORD *)sub_22077B0(48 * v133);
        v135 = *(unsigned int *)(a1 + 24);
        *(_QWORD *)(a1 + 16) = 0;
        *(_QWORD *)(a1 + 8) = v134;
        v198 = 2;
        v199 = 0;
        v136 = &v134[6 * v135];
        v200 = -8;
        v197 = (char *)&unk_49F8530;
        for ( i = 0; v136 != v134; v134 += 6 )
        {
          if ( v134 )
          {
            v137 = v198;
            v134[2] = 0;
            v134[3] = -8;
            *v134 = &unk_49F8530;
            v134[1] = v137 & 6;
            v134[4] = i;
          }
        }
      }
    }
    else if ( v127 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 8));
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
    }
    else
    {
      *(_QWORD *)(a1 + 16) = 0;
    }
  }
  else
  {
    v198 = 2;
    v196 = 0;
    v199 = 0;
    v192 = (unsigned __int8 *)&unk_49F8530;
    v200 = -16;
    v197 = (char *)&unk_49F8530;
    i = 0;
    if ( v106 == v108 )
    {
      *(_QWORD *)(a1 + 16) = 0;
    }
    else
    {
      for ( m = -8; ; m = v195 )
      {
        v110 = v106[3];
        if ( v110 != m )
        {
          if ( v110 != -8 && v110 != 0 && v110 != -16 )
          {
            sub_1649B30(v106 + 1);
            m = v195;
          }
          v106[3] = m;
          if ( m != 0 && m != -8 && m != -16 )
            sub_1649AC0(v106 + 1, v193 & 0xFFFFFFFFFFFFFFF8LL);
          v106[4] = v196;
        }
        v106 += 6;
        if ( v106 == v108 )
          break;
      }
      v111 = v200;
      v19 = v200 == -8;
      *(_QWORD *)(a1 + 16) = 0;
      v197 = (char *)&unk_49EE2B0;
      if ( v111 != -16 && !v19 && v111 )
        sub_1649B30(&v198);
    }
    v192 = (unsigned __int8 *)&unk_49EE2B0;
    if ( v195 != 0 && v195 != -8 && v195 != -16 )
      sub_1649B30(&v193);
  }
LABEL_164:
  if ( *(_BYTE *)(a1 + 64) )
  {
    v113 = *(_DWORD *)(a1 + 56);
    if ( v113 )
    {
      v114 = *(_QWORD **)(a1 + 40);
      v115 = &v114[2 * v113];
      do
      {
        if ( *v114 != -4 && *v114 != -8 )
        {
          v116 = v114[1];
          if ( v116 )
            sub_161E7C0((__int64)(v114 + 1), v116);
        }
        v114 += 2;
      }
      while ( v115 != v114 );
    }
    j___libc_free_0(*(_QWORD *)(a1 + 40));
    *(_BYTE *)(a1 + 64) = 0;
  }
  return v174;
}
