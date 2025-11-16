// Function: sub_2755460
// Address: 0x2755460
//
__int64 __fastcall sub_2755460(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4, __int64 a5, bool a6)
{
  __int64 v7; // r12
  unsigned __int16 v9; // ax
  char v10; // r13
  __int64 v11; // rcx
  __int64 v12; // rsi
  __int64 v13; // rax
  unsigned int v14; // r8d
  unsigned __int64 v15; // rdx
  __int64 v16; // rsi
  __int64 v17; // rbx
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rcx
  __int64 *v22; // rax
  __int64 *v23; // rax
  __int64 v24; // r13
  __int64 *v25; // rax
  __int64 v26; // r15
  __int64 v27; // r14
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rax
  __int64 v31; // r13
  __int64 v32; // rdx
  __int64 v33; // r14
  __int64 v34; // rax
  __int64 v35; // rax
  __m128i *v36; // rsi
  __int64 v37; // rbx
  signed __int64 v38; // r12
  unsigned __int8 *v39; // rax
  unsigned __int8 *v40; // rbx
  __int64 v41; // r12
  __int64 v42; // r13
  __int64 v43; // rax
  _BYTE *v44; // r14
  __int64 *v45; // rbx
  __int64 *v46; // r13
  unsigned __int8 *v47; // rsi
  __int64 v48; // r14
  __int64 v50; // rax
  __int64 v51; // rax
  unsigned int v52; // r14d
  __int128 v53; // kr00_16
  __int64 v54; // rax
  unsigned int v55; // esi
  _BYTE *v56; // rax
  __int64 v57; // rdx
  __int64 *v58; // rax
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rax
  __int64 *v62; // rdi
  __int64 v63; // rdx
  __int64 v64; // rax
  __int64 v65; // rcx
  __int64 v66; // rcx
  unsigned int v67; // r14d
  __int128 v68; // kr10_16
  __int64 v69; // rax
  unsigned int v70; // esi
  __int64 v71; // rax
  _BYTE *v72; // rax
  __int64 v73; // rdx
  __int64 *v74; // rax
  __int64 v75; // rax
  __int64 v76; // rdx
  __int64 v77; // rsi
  _QWORD *v78; // rax
  __int64 v79; // r14
  _QWORD *v80; // rbx
  __int64 v81; // r10
  __int64 v82; // r8
  __int64 *v83; // r13
  __int64 v84; // rsi
  __int64 v85; // rax
  __int64 v86; // rdx
  __int64 v87; // rdx
  __int64 v88; // rax
  __int64 *v89; // rbx
  int v90; // eax
  bool v91; // r14
  void *v92; // rax
  unsigned __int64 v93; // rdx
  __int64 v94; // r8
  __int64 v95; // r9
  size_t v96; // r13
  __int64 v97; // r15
  __int64 *v98; // r13
  __int64 v99; // r8
  _BYTE *v100; // r10
  __int64 v101; // rdi
  _BYTE *v102; // rsi
  _BYTE *v103; // rcx
  _BYTE *v104; // r9
  _BYTE *v105; // rdx
  _BYTE *v106; // rax
  __int64 *v107; // rax
  bool v108; // dl
  __int64 v109; // rsi
  unsigned __int8 *v110; // rsi
  __int64 v111; // rdx
  int v112; // eax
  __int8 v113; // al
  _BYTE *v114; // r11
  _BYTE *v115; // rax
  _BOOL4 v116; // r8d
  __int64 v117; // rax
  __int64 v118; // r9
  unsigned __int64 v119; // r15
  char v120; // r8
  _QWORD *v121; // r14
  void *v122; // rax
  bool v123; // zf
  __int64 *v124; // rax
  unsigned __int64 v125; // rax
  _QWORD *v126; // rdi
  __int64 v127; // rbx
  unsigned int v128; // eax
  __int64 v129; // rax
  unsigned __int16 v130; // ax
  _BYTE *v131; // rdi
  __int64 *v132; // rdi
  __int64 v133; // rdx
  __int64 v134; // rax
  __int64 v135; // rcx
  __int64 v136; // rcx
  __int64 v137; // rsi
  __int64 v138; // rax
  __int64 v139; // rax
  _QWORD *v140; // rcx
  _BYTE *v141; // rsi
  size_t v142; // rdx
  __int64 v143; // rdi
  _BYTE *v144; // rax
  _BYTE *v145; // rcx
  _BYTE *v146; // rdx
  __int32 v147; // edx
  __int64 v148; // rax
  __int64 v149; // rax
  __int64 v151; // [rsp+10h] [rbp-180h]
  __int64 v152; // [rsp+20h] [rbp-170h]
  int v153; // [rsp+28h] [rbp-168h]
  __int64 v155; // [rsp+38h] [rbp-158h]
  unsigned __int64 v156; // [rsp+40h] [rbp-150h]
  unsigned __int64 v158; // [rsp+50h] [rbp-140h]
  __int64 *v161; // [rsp+70h] [rbp-120h]
  unsigned __int8 *v162; // [rsp+70h] [rbp-120h]
  __int64 v163; // [rsp+70h] [rbp-120h]
  __int64 v164; // [rsp+78h] [rbp-118h]
  __int64 v165; // [rsp+78h] [rbp-118h]
  __int64 v166; // [rsp+80h] [rbp-110h]
  int src; // [rsp+88h] [rbp-108h]
  _BOOL4 srca; // [rsp+88h] [rbp-108h]
  void *srcb; // [rsp+88h] [rbp-108h]
  char srcc; // [rsp+88h] [rbp-108h]
  __int64 v171; // [rsp+98h] [rbp-F8h] BYREF
  __int128 v172; // [rsp+A0h] [rbp-F0h] BYREF
  __int64 v173; // [rsp+B0h] [rbp-E0h]
  _BYTE *v174; // [rsp+C0h] [rbp-D0h] BYREF
  __int64 v175; // [rsp+C8h] [rbp-C8h]
  char v176; // [rsp+D0h] [rbp-C0h]
  __m128i v177; // [rsp+E0h] [rbp-B0h] BYREF
  __int64 v178; // [rsp+F0h] [rbp-A0h] BYREF
  _BYTE v179[40]; // [rsp+F8h] [rbp-98h] BYREF
  unsigned __int8 *v180; // [rsp+120h] [rbp-70h] BYREF
  __int64 v181; // [rsp+128h] [rbp-68h]
  char v182[8]; // [rsp+130h] [rbp-60h] BYREF
  __int64 v183; // [rsp+138h] [rbp-58h] BYREF
  _QWORD *v184; // [rsp+140h] [rbp-50h]
  __int64 *v185; // [rsp+148h] [rbp-48h]
  __int64 *v186; // [rsp+150h] [rbp-40h]
  __int64 v187; // [rsp+158h] [rbp-38h]

  v7 = a1;
  v161 = (__int64 *)(a1 + 72);
  v9 = sub_A74840((_QWORD *)(a1 + 72), 0);
  v10 = v9;
  if ( HIBYTE(v9) )
  {
    v11 = 1LL << v9;
    v12 = -(1LL << v9);
  }
  else
  {
    v12 = -1;
    v11 = 1;
    v10 = 0;
  }
  v13 = *a2;
  if ( a6 )
  {
    v14 = 0;
    v166 = *a3;
    v15 = v12 & (a4 - v13 + v11 - 1);
    if ( *a3 <= v15 )
      return v14;
    v158 = *a3 - v15;
  }
  else
  {
    v50 = v13 - a4;
    v158 = a5 - v50;
    v51 = (v12 & (a5 - v50 + v11 - 1)) + v50 - a5;
    if ( v51 )
    {
      v14 = 0;
      if ( v11 - v51 >= v158 )
        return v14;
      v158 = v158 - v11 + v51;
    }
    v166 = *a3;
  }
  v16 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  v156 = v166 - v158;
  if ( *(_BYTE *)a1 != 85 )
    goto LABEL_7;
  v138 = *(_QWORD *)(a1 - 32);
  if ( !v138
    || *(_BYTE *)v138
    || *(_QWORD *)(v138 + 24) != *(_QWORD *)(a1 + 80)
    || (*(_BYTE *)(v138 + 33) & 0x20) == 0
    || (unsigned int)(*(_DWORD *)(v138 + 36) - 239) > 5
    || ((1LL << (*(_BYTE *)(v138 + 36) + 17)) & 0x29) == 0 )
  {
    goto LABEL_7;
  }
  v139 = *(_QWORD *)(a1 + 32 * (3 - v16));
  v140 = *(_QWORD **)(v139 + 24);
  if ( *(_DWORD *)(v139 + 32) > 0x40u )
    v140 = (_QWORD *)*v140;
  v14 = 0;
  if ( !(v156 % (unsigned int)v140) )
  {
LABEL_7:
    v17 = *(_QWORD *)(a1 + 32 * (2 - v16));
    v18 = sub_AD64C0(*(_QWORD *)(v17 + 8), v156, 0);
    v19 = a1 + 32 * (2LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
    if ( *(_QWORD *)v19 )
    {
      v20 = *(_QWORD *)(v19 + 8);
      **(_QWORD **)(v19 + 16) = v20;
      if ( v20 )
        *(_QWORD *)(v20 + 16) = *(_QWORD *)(v19 + 16);
    }
    *(_QWORD *)v19 = v18;
    if ( v18 )
    {
      v21 = *(_QWORD *)(v18 + 16);
      *(_QWORD *)(v19 + 8) = v21;
      if ( v21 )
        *(_QWORD *)(v21 + 16) = v19 + 8;
      *(_QWORD *)(v19 + 16) = v18 + 16;
      *(_QWORD *)(v18 + 16) = v19;
    }
    v22 = (__int64 *)sub_BD5C60(a1);
    *(_QWORD *)(a1 + 72) = sub_A7B980(v161, v22, 1, 86);
    v23 = (__int64 *)sub_BD5C60(a1);
    v24 = sub_A77A40(v23, v10);
    LODWORD(v180) = 0;
    v25 = (__int64 *)sub_BD5C60(a1);
    *(_QWORD *)(a1 + 72) = sub_A7B660(v161, v25, &v180, 1, v24);
    v26 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
    if ( a6 )
    {
      v27 = 8 * *a2;
      v166 = 8 * *a3 - 8 * v156;
      src = sub_B43CC0(a1);
      LODWORD(v164) = v27 + 8 * v156;
      goto LABEL_16;
    }
    v171 = sub_AD64C0(*(_QWORD *)(v17 + 8), v158, 0);
    LOWORD(v184) = 257;
    v78 = (_QWORD *)sub_BD5C60(a1);
    v79 = sub_BCB2B0(v78);
    v80 = sub_BD2C40(88, 2u);
    if ( !v80 )
    {
LABEL_95:
      v83 = v80 + 6;
      sub_B4DDE0((__int64)v80, 3);
      v84 = *(_QWORD *)(a1 + 48);
      v180 = (unsigned __int8 *)v84;
      if ( v84 )
      {
        sub_B96E90((__int64)&v180, v84, 1);
        if ( v83 == (__int64 *)&v180 )
        {
          if ( v180 )
            sub_B91220((__int64)&v180, (__int64)v180);
          goto LABEL_99;
        }
        v109 = v80[6];
        if ( !v109 )
        {
LABEL_130:
          v110 = v180;
          v80[6] = v180;
          if ( v110 )
            sub_B976B0((__int64)&v180, v110, (__int64)(v80 + 6));
          goto LABEL_99;
        }
      }
      else if ( v83 == (__int64 *)&v180 || (v109 = v80[6]) == 0 )
      {
LABEL_99:
        v85 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
        if ( *(_QWORD *)v85 )
        {
          v86 = *(_QWORD *)(v85 + 8);
          **(_QWORD **)(v85 + 16) = v86;
          if ( v86 )
            *(_QWORD *)(v86 + 16) = *(_QWORD *)(v85 + 16);
        }
        *(_QWORD *)v85 = v80;
        if ( v80 )
        {
          v87 = v80[2];
          *(_QWORD *)(v85 + 8) = v87;
          if ( v87 )
            *(_QWORD *)(v87 + 16) = v85 + 8;
          *(_QWORD *)(v85 + 16) = v80 + 2;
          v80[2] = v85;
        }
        v180 = *(unsigned __int8 **)(a1 + 72);
        v88 = sub_A744E0(&v180, 0);
        v180 = 0;
        *(_QWORD *)&v172 = v88;
        v181 = 0;
        LODWORD(v183) = 0;
        v184 = 0;
        v185 = &v183;
        v186 = &v183;
        v187 = 0;
        v89 = (__int64 *)sub_A73280((__int64 *)&v172);
        v165 = sub_A73290((__int64 *)&v172);
        if ( v89 == (__int64 *)v165 )
        {
LABEL_154:
          v124 = (__int64 *)sub_BD5C60(v7);
          v125 = sub_A7A440(v161, v124, 1, (__int64)&v180);
          v126 = v184;
          *(_QWORD *)(v7 + 72) = v125;
          sub_2754130(v126);
          v127 = *a3;
          v164 = 8 * *a2;
          src = sub_B43CC0(v7);
          LODWORD(v166) = 8 * (v158 + v127 - v166);
LABEL_16:
          v153 = sub_BD5C60(v7);
          if ( (*(_BYTE *)(v7 + 7) & 0x20) != 0 )
          {
            v30 = sub_B91C10(v7, 38);
            v31 = v30;
            if ( v30 )
            {
              v31 = sub_AE94B0(v30);
              v33 = v32;
            }
            else
            {
              v33 = 0;
            }
            if ( (*(_BYTE *)(v7 + 7) & 0x20) != 0 )
            {
              v34 = sub_B91C10(v7, 38);
              if ( v34 )
              {
                v35 = *(_QWORD *)(v34 + 8);
                v36 = (__m128i *)(v35 & 0xFFFFFFFFFFFFFFF8LL);
                if ( (v35 & 4) == 0 )
                  v36 = 0;
                sub_B967C0(&v177, v36);
                goto LABEL_24;
              }
            }
          }
          else
          {
            v31 = 0;
            v33 = 0;
          }
          v177.m128i_i64[0] = (__int64)&v178;
          v177.m128i_i64[1] = 0x600000000LL;
LABEL_24:
          v180 = (unsigned __int8 *)v182;
          v181 = 0x600000000LL;
          if ( v31 == v33 )
          {
            LODWORD(v181) = 0;
            v41 = 0;
          }
          else
          {
            v37 = v31;
            v38 = 0;
            do
            {
              v37 = *(_QWORD *)(v37 + 8);
              ++v38;
            }
            while ( v37 != v33 );
            v39 = (unsigned __int8 *)v182;
            if ( v38 > 6 )
            {
              sub_C8D5F0((__int64)&v180, v182, v38, 8u, v28, v29);
              v39 = &v180[8 * (unsigned int)v181];
            }
            do
            {
              v39 += 8;
              *((_QWORD *)v39 - 1) = *(_QWORD *)(v31 + 24);
              v31 = *(_QWORD *)(v31 + 8);
            }
            while ( v37 != v31 );
            v40 = v180;
            LODWORD(v181) = v181 + v38;
            v41 = 0;
            v162 = &v180[8 * (unsigned int)v181];
            if ( v162 != v180 )
            {
              do
              {
                while ( 1 )
                {
                  v44 = *(_BYTE **)v40;
                  v173 = 0;
                  v172 = 0;
                  if ( (unsigned __int8)sub_AEA6D0(src, v26, v164, v166, (__int64)v44, (__int64)&v172) )
                  {
                    if ( (_BYTE)v173 )
                      break;
                  }
                  sub_B59B20((__int64)v44);
                  if ( !v41 )
                    v41 = sub_AF40E0(v153, 1u);
                  v40 += 8;
                  sub_B59600((__int64)v44, v41);
                  if ( v40 == v162 )
                    goto LABEL_43;
                }
                if ( (_QWORD)v172 )
                {
                  v42 = sub_B47F80(v44);
                  v43 = v155;
                  LOWORD(v43) = 0;
                  v155 = v43;
                  sub_B43E90(v42, (__int64)(v44 + 24));
                  if ( !v41 )
                    v41 = sub_AF40E0(v153, 1u);
                  sub_B59600(v42, v41);
                  if ( (_BYTE)v173 )
                  {
                    v52 = DWORD2(v172);
                    v53 = v172;
                    v54 = *(_QWORD *)(*(_QWORD *)(v42 + 32 * (2LL - (*(_DWORD *)(v42 + 4) & 0x7FFFFFF))) + 24LL);
                    sub_AF47B0((__int64)&v174, *(unsigned __int64 **)(v54 + 16), *(unsigned __int64 **)(v54 + 24));
                    v55 = v52;
                    if ( v176 )
                      v55 = v52 - v175;
                    v56 = (_BYTE *)sub_B0E470(
                                     *(_QWORD *)(*(_QWORD *)(v42 + 32 * (2LL - (*(_DWORD *)(v42 + 4) & 0x7FFFFFF)))
                                               + 24LL),
                                     v55,
                                     v53);
                    v175 = v57;
                    v174 = v56;
                    if ( (_BYTE)v57 )
                    {
                      v132 = (__int64 *)(*((_QWORD *)v174 + 1) & 0xFFFFFFFFFFFFFFF8LL);
                      if ( (*((_QWORD *)v174 + 1) & 4) != 0 )
                        v132 = (__int64 *)*v132;
                      v133 = sub_B9F6F0(v132, v174);
                      v134 = v42 + 32 * (2LL - (*(_DWORD *)(v42 + 4) & 0x7FFFFFF));
                      if ( *(_QWORD *)v134 )
                      {
                        v135 = *(_QWORD *)(v134 + 8);
                        **(_QWORD **)(v134 + 16) = v135;
                        if ( v135 )
                          *(_QWORD *)(v135 + 16) = *(_QWORD *)(v134 + 16);
                      }
                      *(_QWORD *)v134 = v133;
                      if ( v133 )
                      {
                        v136 = *(_QWORD *)(v133 + 16);
                        *(_QWORD *)(v134 + 8) = v136;
                        if ( v136 )
                          *(_QWORD *)(v136 + 16) = v134 + 8;
                        *(_QWORD *)(v134 + 16) = v133 + 16;
                        *(_QWORD *)(v133 + 16) = v134;
                      }
                    }
                    else
                    {
                      v58 = (__int64 *)sub_BD5C60(v42);
                      v59 = sub_B0D000(v58, 0, 0, 0, 1);
                      v174 = (_BYTE *)sub_B0E470(v59, DWORD2(v53), v53);
                      v175 = v60;
                      v61 = *((_QWORD *)v174 + 1);
                      v62 = (__int64 *)(v61 & 0xFFFFFFFFFFFFFFF8LL);
                      if ( (v61 & 4) != 0 )
                        v62 = (__int64 *)*v62;
                      v63 = sub_B9F6F0(v62, v174);
                      v64 = v42 + 32 * (2LL - (*(_DWORD *)(v42 + 4) & 0x7FFFFFF));
                      if ( *(_QWORD *)v64 )
                      {
                        v65 = *(_QWORD *)(v64 + 8);
                        **(_QWORD **)(v64 + 16) = v65;
                        if ( v65 )
                          *(_QWORD *)(v65 + 16) = *(_QWORD *)(v64 + 16);
                      }
                      *(_QWORD *)v64 = v63;
                      if ( v63 )
                      {
                        v66 = *(_QWORD *)(v63 + 16);
                        *(_QWORD *)(v64 + 8) = v66;
                        if ( v66 )
                          *(_QWORD *)(v66 + 16) = v64 + 8;
                        *(_QWORD *)(v64 + 16) = v63 + 16;
                        *(_QWORD *)(v63 + 16) = v64;
                      }
                      sub_F507F0(v42);
                    }
                  }
                  sub_B59B20(v42);
                }
                v40 += 8;
              }
              while ( v40 != v162 );
            }
          }
LABEL_43:
          v45 = (__int64 *)v177.m128i_i64[0];
          v163 = v177.m128i_i64[0] + 8LL * v177.m128i_u32[2];
          if ( v177.m128i_i64[0] != v163 )
          {
            do
            {
              while ( 1 )
              {
                v48 = *v45;
                v173 = 0;
                v172 = 0;
                if ( (unsigned __int8)sub_AEA880(src, v26, v164, v166, v48, (__int64)&v172) )
                {
                  if ( (_BYTE)v173 )
                    break;
                }
                sub_B14010(v48, v26);
                if ( !v41 )
                  v41 = sub_AF40E0(v153, 1u);
                ++v45;
                sub_B13D10(v48, v41);
                if ( (__int64 *)v163 == v45 )
                  goto LABEL_56;
              }
              if ( (_QWORD)v172 )
              {
                v46 = (__int64 *)sub_B13070(v48);
                sub_B143F0(v46, v48);
                if ( !v41 )
                  v41 = sub_AF40E0(v153, 1u);
                v47 = (unsigned __int8 *)v41;
                sub_B13D10((__int64)v46, v41);
                if ( (_BYTE)v173 )
                {
                  v67 = DWORD2(v172);
                  v68 = v172;
                  v152 = (__int64)(v46 + 10);
                  v69 = sub_B11F60((__int64)(v46 + 10));
                  sub_AF47B0((__int64)&v174, *(unsigned __int64 **)(v69 + 16), *(unsigned __int64 **)(v69 + 24));
                  v70 = v67;
                  if ( v176 )
                    v70 = v67 - v175;
                  v71 = sub_B11F60(v152);
                  v72 = (_BYTE *)sub_B0E470(v71, v70, v68);
                  v175 = v73;
                  v174 = v72;
                  if ( (_BYTE)v73 )
                  {
                    sub_B11F20(&v171, (__int64)v174);
                    v137 = v46[10];
                    if ( v137 )
                      sub_B91220(v152, v137);
                    v47 = (unsigned __int8 *)v171;
                    v46[10] = v171;
                    if ( v47 )
                      sub_B976B0((__int64)&v171, v47, v152);
                  }
                  else
                  {
                    v74 = (__int64 *)sub_B141C0((__int64)v46);
                    v75 = sub_B0D000(v74, 0, 0, 0, 1);
                    v174 = (_BYTE *)sub_B0E470(v75, DWORD2(v68), v68);
                    v175 = v76;
                    sub_B11F20(&v174, (__int64)v174);
                    v77 = v46[10];
                    if ( v77 )
                      sub_B91220(v152, v77);
                    v47 = v174;
                    v46[10] = (__int64)v174;
                    if ( v47 )
                      sub_B976B0((__int64)&v174, v47, v152);
                    sub_B13710((__int64)v46);
                  }
                }
                sub_B14010((__int64)v46, (__int64)v47);
              }
              ++v45;
            }
            while ( (__int64 *)v163 != v45 );
          }
LABEL_56:
          if ( v180 != (unsigned __int8 *)v182 )
            _libc_free((unsigned __int64)v180);
          if ( (__int64 *)v177.m128i_i64[0] != &v178 )
            _libc_free(v177.m128i_u64[0]);
          if ( !a6 )
            *a2 += v158;
          v14 = 1;
          *a3 = v156;
          return v14;
        }
        v151 = v26;
LABEL_108:
        if ( !sub_A71840((__int64)v89) )
        {
          v90 = sub_A71AE0(v89);
          switch ( v90 )
          {
            case '+':
              goto LABEL_152;
            case 'V':
              v130 = sub_A71F30(v89);
              if ( !HIBYTE(v130) || (~(-1LL << v130) & v158) == 0 )
                goto LABEL_152;
              break;
            case '(':
              goto LABEL_152;
          }
        }
        v174 = (_BYTE *)*v89;
        v91 = sub_A71840((__int64)&v174);
        if ( !v91 )
        {
          v128 = sub_A71AE0((__int64 *)&v174);
          (&v180)[(unsigned __int64)v128 >> 6] = (unsigned __int8 *)((1LL << v128)
                                                                   | (unsigned __int64)(&v180)[(unsigned __int64)v128 >> 6]);
          goto LABEL_152;
        }
        v92 = (void *)sub_A71FD0((__int64 *)&v174);
        v177.m128i_i64[1] = 0;
        v96 = v93;
        v97 = v93;
        v177.m128i_i64[0] = (__int64)v179;
        v178 = 32;
        if ( v93 > 0x20 )
        {
          srcb = v92;
          sub_C8D290((__int64)&v177, v179, v93, 1u, v94, v95);
          v92 = srcb;
          v131 = (_BYTE *)(v177.m128i_i64[0] + v177.m128i_i64[1]);
        }
        else
        {
          if ( !v93 )
          {
LABEL_115:
            v98 = v184;
            v177.m128i_i64[1] = v97;
            if ( !v184 )
            {
              v98 = &v183;
              goto LABEL_158;
            }
            v99 = v177.m128i_i64[0];
            v100 = (_BYTE *)(v177.m128i_i64[0] + v97);
            while ( 1 )
            {
              v101 = v98[5];
              v102 = (_BYTE *)v98[4];
              v103 = (_BYTE *)(v177.m128i_i64[0] + v101);
              v104 = &v102[v101];
              v105 = v102;
              if ( v97 <= v101 )
                v103 = (_BYTE *)(v177.m128i_i64[0] + v97);
              if ( (_BYTE *)v177.m128i_i64[0] == v103 )
              {
LABEL_136:
                if ( v104 != v105 )
                {
LABEL_125:
                  v107 = (__int64 *)v98[2];
                  v108 = v91;
                  if ( !v107 )
                    goto LABEL_138;
                  goto LABEL_126;
                }
              }
              else
              {
                v106 = (_BYTE *)v177.m128i_i64[0];
                while ( 1 )
                {
                  if ( *v106 < *v105 )
                  {
                    v102 = (_BYTE *)v98[4];
                    goto LABEL_125;
                  }
                  if ( *v106 > *v105 )
                    break;
                  ++v106;
                  ++v105;
                  if ( v103 == v106 )
                  {
                    v102 = (_BYTE *)v98[4];
                    goto LABEL_136;
                  }
                }
                v102 = (_BYTE *)v98[4];
              }
              v107 = (__int64 *)v98[3];
              v108 = a6;
              if ( !v107 )
              {
LABEL_138:
                v114 = (_BYTE *)v177.m128i_i64[0];
                if ( !v108 )
                  goto LABEL_139;
LABEL_158:
                if ( v185 == v98 )
                  goto LABEL_147;
                v129 = sub_220EF80((__int64)v98);
                v99 = v177.m128i_i64[0];
                v102 = *(_BYTE **)(v129 + 32);
                v101 = *(_QWORD *)(v129 + 40);
                v114 = (_BYTE *)v177.m128i_i64[0];
                v100 = (_BYTE *)(v177.m128i_i64[0] + v97);
                v104 = &v102[v101];
LABEL_139:
                if ( v97 < v101 )
                  v104 = &v102[v97];
                v115 = (_BYTE *)v99;
                if ( v104 == v102 )
                {
LABEL_165:
                  if ( v100 != v115 )
                    goto LABEL_146;
                  goto LABEL_150;
                }
                while ( *v102 >= *v115 )
                {
                  if ( *v102 > *v115 )
                    goto LABEL_150;
                  ++v102;
                  ++v115;
                  if ( v104 == v102 )
                    goto LABEL_165;
                }
LABEL_146:
                v114 = (_BYTE *)v99;
                if ( v98 )
                {
LABEL_147:
                  v116 = 1;
                  if ( v98 != &v183 )
                  {
                    v143 = v98[5];
                    v144 = (_BYTE *)v177.m128i_i64[0];
                    v145 = (_BYTE *)(v177.m128i_i64[0] + v143);
                    if ( v143 >= v97 )
                      v145 = (_BYTE *)(v177.m128i_i64[0] + v97);
                    v146 = (_BYTE *)v98[4];
                    if ( (_BYTE *)v177.m128i_i64[0] == v145 )
                    {
LABEL_213:
                      v116 = v146 != (_BYTE *)(v143 + v98[4]);
                    }
                    else
                    {
                      while ( 1 )
                      {
                        if ( *v144 < *v146 )
                        {
                          v116 = 1;
                          goto LABEL_148;
                        }
                        if ( *v144 > *v146 )
                          break;
                        ++v144;
                        ++v146;
                        if ( v145 == v144 )
                          goto LABEL_213;
                      }
                      v116 = 0;
                    }
                  }
LABEL_148:
                  srca = v116;
                  v117 = sub_22077B0(0x58u);
                  v119 = v177.m128i_u64[1];
                  v120 = srca;
                  v121 = (_QWORD *)v117;
                  v122 = (void *)(v117 + 56);
                  v123 = v177.m128i_i64[1] == 0;
                  v121[4] = v122;
                  v121[5] = 0;
                  v121[6] = 32;
                  if ( !v123 )
                  {
                    if ( (_BYTE *)v177.m128i_i64[0] == v179 )
                    {
                      v141 = v179;
                      v142 = v119;
                      if ( v119 <= 0x20
                        || (sub_C8D290((__int64)(v121 + 4), v122, v119, 1u, srca, v118),
                            v142 = v177.m128i_u64[1],
                            v122 = (void *)v121[4],
                            v141 = (_BYTE *)v177.m128i_i64[0],
                            v120 = srca,
                            v177.m128i_i64[1]) )
                      {
                        srcc = v120;
                        memcpy(v122, v141, v142);
                        v120 = srcc;
                      }
                      v121[5] = v119;
                      v177.m128i_i64[1] = 0;
                    }
                    else
                    {
                      v149 = v178;
                      v121[4] = v177.m128i_i64[0];
                      v121[5] = v119;
                      v121[6] = v149;
                      v178 = 0;
                      v177.m128i_i64[0] = (__int64)v179;
                      v177.m128i_i64[1] = 0;
                    }
                  }
                  sub_220F040(v120, (__int64)v121, v98, &v183);
                  ++v187;
                  v114 = (_BYTE *)v177.m128i_i64[0];
                }
LABEL_150:
                if ( v114 != v179 )
                  _libc_free((unsigned __int64)v114);
LABEL_152:
                if ( (__int64 *)v165 == ++v89 )
                {
                  v26 = v151;
                  v7 = a1;
                  goto LABEL_154;
                }
                goto LABEL_108;
              }
LABEL_126:
              v98 = v107;
            }
          }
          v131 = v179;
        }
        memcpy(v131, v92, v96);
        v97 = v96 + v177.m128i_i64[1];
        goto LABEL_115;
      }
      sub_B91220((__int64)(v80 + 6), v109);
      goto LABEL_130;
    }
    v81 = *(_QWORD *)(v26 + 8);
    v82 = a1 + 24;
    if ( (unsigned int)*(unsigned __int8 *)(v81 + 8) - 17 <= 1 )
    {
LABEL_94:
      sub_B44260((__int64)v80, v81, 34, 2u, v82, 0);
      v80[9] = v79;
      v80[10] = sub_B4DC50(v79, (__int64)&v171, 1);
      sub_B4D9A0((__int64)v80, v26, &v171, 1, (__int64)&v180);
      goto LABEL_95;
    }
    v111 = *(_QWORD *)(v171 + 8);
    v112 = *(unsigned __int8 *)(v111 + 8);
    if ( v112 == 17 )
    {
      v113 = 0;
    }
    else
    {
      if ( v112 != 18 )
        goto LABEL_94;
      v113 = 1;
    }
    v147 = *(_DWORD *)(v111 + 32);
    v177.m128i_i8[4] = v113;
    v177.m128i_i32[0] = v147;
    v148 = sub_BCE1B0((__int64 *)v81, v177.m128i_i64[0]);
    v82 = a1 + 24;
    v81 = v148;
    goto LABEL_94;
  }
  return v14;
}
