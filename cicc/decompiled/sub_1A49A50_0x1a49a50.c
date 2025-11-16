// Function: sub_1A49A50
// Address: 0x1a49a50
//
unsigned __int64 __fastcall sub_1A49A50(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v11; // r15
  _QWORD *v12; // rax
  unsigned __int8 *v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 **v16; // r12
  __int64 v17; // rdx
  int v18; // eax
  __int64 v19; // rcx
  __int64 v20; // rsi
  int v21; // edi
  unsigned int v22; // edx
  __int64 *v23; // rax
  __int64 v24; // r8
  __int64 v25; // rax
  __int64 v26; // rbx
  __int64 *v27; // rbx
  __int64 v28; // rax
  double v29; // xmm4_8
  double v30; // xmm5_8
  int i; // ebx
  unsigned __int64 v32; // r13
  unsigned __int64 v33; // r12
  __int64 v34; // r14
  unsigned int v35; // r15d
  __int64 v36; // rsi
  __int64 v37; // r15
  unsigned int v38; // eax
  __int64 v39; // rcx
  unsigned __int64 v40; // r8
  __int64 v41; // rax
  __int64 v42; // rsi
  unsigned int v43; // r15d
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rdi
  char v48; // al
  bool v49; // zf
  int v50; // eax
  __int64 v51; // rax
  __int64 v52; // r12
  __int64 v53; // rax
  int v54; // edx
  __int64 v55; // rax
  _QWORD *v56; // r13
  __int64 v57; // rbx
  _QWORD *v58; // r14
  unsigned __int8 v59; // al
  int v60; // edx
  __int64 *v61; // rdx
  __int64 v62; // rdx
  __int64 *v63; // rax
  __int64 v64; // rcx
  __int64 v65; // rdi
  unsigned __int64 v66; // rsi
  __int64 v67; // rsi
  _QWORD *v68; // rax
  __int64 v69; // rsi
  unsigned __int64 v70; // rdx
  __int64 v71; // rdx
  __int64 v72; // rax
  __int64 v73; // r14
  _QWORD *v74; // rax
  unsigned int v75; // ebx
  unsigned __int8 *v76; // rax
  __int64 **v77; // rdx
  unsigned __int64 result; // rax
  unsigned __int64 v79; // rax
  unsigned int v80; // esi
  int v81; // eax
  unsigned int v82; // eax
  __int64 v83; // rdx
  __int64 v84; // rsi
  unsigned __int64 v85; // r9
  _QWORD *v86; // rax
  __int64 v87; // rax
  __int64 v88; // rdx
  __int64 *v89; // rsi
  int v90; // edi
  __int64 v91; // rsi
  __int64 v92; // rax
  __int64 v93; // rsi
  __int64 v94; // rdx
  unsigned __int8 *v95; // rsi
  int v96; // eax
  __int64 v97; // r12
  __int64 v98; // rax
  int v99; // ebx
  __int64 v100; // r12
  _QWORD *v101; // r14
  _QWORD *v102; // rax
  __int64 v103; // rax
  _QWORD *v104; // rax
  __int64 v105; // r15
  _QWORD *v106; // rsi
  __int64 v107; // rax
  __int64 v108; // rdx
  __int64 v109; // rsi
  __int64 *v110; // rbx
  __int64 v111; // rax
  __int64 v112; // rcx
  __int64 v113; // rsi
  unsigned __int8 *v114; // rsi
  __int64 *v115; // rbx
  __int64 v116; // rax
  __int64 v117; // rcx
  __int64 v118; // rsi
  unsigned __int8 *v119; // rsi
  __int64 v120; // rax
  unsigned __int64 v121; // rax
  unsigned int v122; // esi
  int v123; // eax
  _QWORD *v124; // rax
  int v125; // r9d
  _QWORD *v126; // rdx
  __int64 v127; // rdi
  unsigned __int64 v128; // rsi
  __int64 v129; // [rsp+0h] [rbp-160h]
  __int64 v130; // [rsp+8h] [rbp-158h]
  __int64 v131; // [rsp+8h] [rbp-158h]
  __int64 v132; // [rsp+8h] [rbp-158h]
  __int64 v133; // [rsp+10h] [rbp-150h]
  unsigned __int64 v134; // [rsp+10h] [rbp-150h]
  __int64 v135; // [rsp+10h] [rbp-150h]
  __int64 v136; // [rsp+10h] [rbp-150h]
  __int64 v137; // [rsp+18h] [rbp-148h]
  __int64 v138; // [rsp+18h] [rbp-148h]
  unsigned __int64 v139; // [rsp+18h] [rbp-148h]
  unsigned __int64 v140; // [rsp+18h] [rbp-148h]
  unsigned __int64 v141; // [rsp+20h] [rbp-140h]
  __int64 v142; // [rsp+20h] [rbp-140h]
  unsigned __int64 v143; // [rsp+20h] [rbp-140h]
  __int64 v144; // [rsp+20h] [rbp-140h]
  __int64 v145; // [rsp+20h] [rbp-140h]
  __int64 v146; // [rsp+20h] [rbp-140h]
  __int64 v147; // [rsp+28h] [rbp-138h]
  __int64 v149; // [rsp+40h] [rbp-120h]
  __int64 v150; // [rsp+48h] [rbp-118h]
  __int64 v152; // [rsp+58h] [rbp-108h]
  __int64 v153; // [rsp+58h] [rbp-108h]
  __int64 v154; // [rsp+58h] [rbp-108h]
  __int64 *v155; // [rsp+58h] [rbp-108h]
  bool v156; // [rsp+63h] [rbp-FDh]
  int v157; // [rsp+64h] [rbp-FCh]
  __int64 v158; // [rsp+68h] [rbp-F8h]
  __int64 v159; // [rsp+68h] [rbp-F8h]
  __int64 v160; // [rsp+70h] [rbp-F0h]
  __int64 *v161; // [rsp+78h] [rbp-E8h]
  __int64 **v162; // [rsp+78h] [rbp-E8h]
  unsigned __int8 *v163; // [rsp+88h] [rbp-D8h] BYREF
  __int64 v164; // [rsp+90h] [rbp-D0h] BYREF
  unsigned int v165; // [rsp+98h] [rbp-C8h]
  __int64 v166[2]; // [rsp+A0h] [rbp-C0h] BYREF
  __int16 v167; // [rsp+B0h] [rbp-B0h]
  unsigned __int8 *v168; // [rsp+C0h] [rbp-A0h] BYREF
  unsigned int v169; // [rsp+C8h] [rbp-98h]
  __int16 v170; // [rsp+D0h] [rbp-90h]
  unsigned __int8 *v171; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v172; // [rsp+E8h] [rbp-78h]
  __int64 *v173; // [rsp+F0h] [rbp-70h]
  _QWORD *v174; // [rsp+F8h] [rbp-68h]
  __int64 v175; // [rsp+100h] [rbp-60h]
  int v176; // [rsp+108h] [rbp-58h]
  __int64 v177; // [rsp+110h] [rbp-50h]
  __int64 v178; // [rsp+118h] [rbp-48h]

  v11 = a2;
  v12 = (_QWORD *)sub_16498A0(a2);
  v13 = *(unsigned __int8 **)(a2 + 48);
  v171 = 0;
  v174 = v12;
  v14 = *(_QWORD *)(v11 + 40);
  v175 = 0;
  v172 = v14;
  v176 = 0;
  v177 = 0;
  v178 = 0;
  v173 = (__int64 *)(v11 + 24);
  v168 = v13;
  if ( v13 )
  {
    sub_1623A60((__int64)&v168, (__int64)v13, 2);
    if ( v171 )
      sub_161E7C0((__int64)&v171, (__int64)v171);
    v171 = v168;
    if ( v168 )
      sub_1623210((__int64)&v168, v168, (__int64)&v171);
  }
  v150 = sub_15A9650(a1[20], *(_QWORD *)v11);
  v15 = *(_QWORD *)v11;
  if ( *(_BYTE *)(*(_QWORD *)v11 + 8LL) == 16 )
    v15 = **(_QWORD **)(v15 + 16);
  v16 = (__int64 **)sub_16471D0(v174, *(_DWORD *)(v15 + 8) >> 8);
  v160 = *(_QWORD *)(v11 - 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF));
  v17 = a1[23];
  v18 = *(_DWORD *)(v17 + 24);
  if ( v18 )
  {
    v19 = *(_QWORD *)(v11 + 40);
    v20 = *(_QWORD *)(v17 + 8);
    v21 = v18 - 1;
    v22 = (v18 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
    v23 = (__int64 *)(v20 + 16LL * v22);
    v24 = *v23;
    if ( v19 == *v23 )
    {
LABEL_10:
      v25 = v23[1];
      v147 = v25;
      if ( v25 )
      {
        v156 = sub_13FC1A0(v25, v160);
        if ( v156 && *(_QWORD *)(v160 + 8) )
        {
          v162 = v16;
          v99 = 0;
          v100 = *(_QWORD *)(v160 + 8);
          v159 = v11;
          while ( 1 )
          {
            v104 = sub_1648700(v100);
            if ( *((_BYTE *)v104 + 16) > 0x17u )
              break;
LABEL_155:
            v100 = *(_QWORD *)(v100 + 8);
            if ( !v100 )
            {
              v16 = v162;
              v11 = v159;
              goto LABEL_12;
            }
          }
          v105 = v104[5];
          v106 = *(_QWORD **)(v147 + 72);
          v102 = *(_QWORD **)(v147 + 64);
          if ( v106 == v102 )
          {
            v101 = &v102[*(unsigned int *)(v147 + 84)];
            if ( v102 == v101 )
            {
              v126 = *(_QWORD **)(v147 + 64);
            }
            else
            {
              do
              {
                if ( v105 == *v102 )
                  break;
                ++v102;
              }
              while ( v101 != v102 );
              v126 = v101;
            }
          }
          else
          {
            v101 = &v106[*(unsigned int *)(v147 + 80)];
            v102 = sub_16CC9F0(v147 + 56, v105);
            if ( v105 == *v102 )
            {
              v108 = *(_QWORD *)(v147 + 72);
              if ( v108 == *(_QWORD *)(v147 + 64) )
                v109 = *(unsigned int *)(v147 + 84);
              else
                v109 = *(unsigned int *)(v147 + 80);
              v126 = (_QWORD *)(v108 + 8 * v109);
            }
            else
            {
              v103 = *(_QWORD *)(v147 + 72);
              if ( v103 != *(_QWORD *)(v147 + 64) )
              {
                v102 = (_QWORD *)(v103 + 8LL * *(unsigned int *)(v147 + 80));
                goto LABEL_152;
              }
              v102 = (_QWORD *)(v103 + 8LL * *(unsigned int *)(v147 + 84));
              v126 = v102;
            }
          }
          while ( v126 != v102 && *v102 >= 0xFFFFFFFFFFFFFFFELL )
            ++v102;
LABEL_152:
          if ( v102 != v101 )
          {
            if ( v99 == 1 )
            {
              v16 = v162;
              v11 = v159;
              goto LABEL_175;
            }
            v99 = 1;
          }
          goto LABEL_155;
        }
      }
      else
      {
LABEL_175:
        v156 = 0;
      }
      goto LABEL_12;
    }
    v96 = 1;
    while ( v24 != -8 )
    {
      v125 = v96 + 1;
      v22 = v21 & (v96 + v22);
      v23 = (__int64 *)(v20 + 16LL * v22);
      v24 = *v23;
      if ( v19 == *v23 )
        goto LABEL_10;
      v96 = v125;
    }
  }
  v147 = 0;
  v156 = 0;
LABEL_12:
  if ( v16 != *(__int64 ***)v160 )
  {
    v167 = 257;
    if ( v16 != *(__int64 ***)v160 )
    {
      if ( *(_BYTE *)(v160 + 16) > 0x10u )
      {
        v170 = 257;
        v160 = sub_15FDBD0(47, v160, (__int64)v16, (__int64)&v168, 0);
        if ( v172 )
        {
          v110 = v173;
          sub_157E9D0(v172 + 40, v160);
          v111 = *(_QWORD *)(v160 + 24);
          v112 = *v110;
          *(_QWORD *)(v160 + 32) = v110;
          v112 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v160 + 24) = v112 | v111 & 7;
          *(_QWORD *)(v112 + 8) = v160 + 24;
          *v110 = *v110 & 7 | (v160 + 24);
        }
        sub_164B780(v160, v166);
        if ( v171 )
        {
          v164 = (__int64)v171;
          sub_1623A60((__int64)&v164, (__int64)v171, 2);
          v113 = *(_QWORD *)(v160 + 48);
          if ( v113 )
            sub_161E7C0(v160 + 48, v113);
          v114 = (unsigned __int8 *)v164;
          *(_QWORD *)(v160 + 48) = v164;
          if ( v114 )
            sub_1623210((__int64)&v164, v114, v160 + 48);
        }
      }
      else
      {
        v160 = sub_15A46C0(47, (__int64 ***)v160, v16, 0);
      }
    }
  }
  if ( (*(_BYTE *)(v11 + 23) & 0x40) != 0 )
    v26 = *(_QWORD *)(v11 - 8);
  else
    v26 = v11 - 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF);
  v27 = (__int64 *)(v26 + 24);
  v28 = sub_16348C0(v11) | 4;
  v157 = *(_DWORD *)(v11 + 20) & 0xFFFFFFF;
  if ( v157 == 1 )
  {
    if ( a3 )
    {
      v158 = 0;
      goto LABEL_146;
    }
    goto LABEL_108;
  }
  v149 = v11;
  v158 = 0;
  v161 = v27;
  for ( i = 2; ; ++i )
  {
    v32 = v28 & 0xFFFFFFFFFFFFFFF8LL;
    v33 = v28 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v28 & 4) == 0 )
      goto LABEL_46;
    v34 = *(_QWORD *)(v149 + 24 * ((unsigned int)(i - 1) - (unsigned __int64)(*(_DWORD *)(v149 + 20) & 0xFFFFFFF)));
    if ( *(_BYTE *)(v34 + 16) == 13 )
    {
      v35 = *(_DWORD *)(v34 + 32);
      if ( v35 <= 0x40 )
      {
        if ( !*(_QWORD *)(v34 + 24) )
          goto LABEL_41;
      }
      else if ( v35 == (unsigned int)sub_16A57B0(v34 + 24) )
      {
        goto LABEL_41;
      }
    }
    v36 = v32;
    v37 = a1[20];
    if ( !v32 )
      v36 = sub_1643D30(0, *v161);
    v38 = sub_15A9FE0(v37, v36);
    v39 = 1;
    v40 = v38;
    while ( 2 )
    {
      switch ( *(_BYTE *)(v36 + 8) )
      {
        case 0:
        case 8:
        case 0xA:
        case 0xC:
        case 0x10:
          v87 = *(_QWORD *)(v36 + 32);
          v36 = *(_QWORD *)(v36 + 24);
          v39 *= v87;
          continue;
        case 1:
          v41 = 16;
          goto LABEL_29;
        case 2:
          v41 = 32;
          goto LABEL_29;
        case 3:
        case 9:
          v41 = 64;
          goto LABEL_29;
        case 4:
          v41 = 80;
          goto LABEL_29;
        case 5:
        case 6:
          v41 = 128;
          goto LABEL_29;
        case 7:
          v141 = v40;
          v80 = 0;
          v152 = v39;
          goto LABEL_122;
        case 0xB:
          v41 = *(_DWORD *)(v36 + 8) >> 8;
          goto LABEL_29;
        case 0xD:
          v143 = v40;
          v154 = v39;
          v86 = (_QWORD *)sub_15A9930(v37, v36);
          v39 = v154;
          v40 = v143;
          v41 = 8LL * *v86;
          goto LABEL_29;
        case 0xE:
          v133 = v40;
          v137 = v39;
          v142 = *(_QWORD *)(v36 + 24);
          v153 = *(_QWORD *)(v36 + 32);
          v82 = sub_15A9FE0(v37, v142);
          v40 = v133;
          v83 = 1;
          v84 = v142;
          v39 = v137;
          v85 = v82;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v84 + 8) )
            {
              case 0:
              case 8:
              case 0xA:
              case 0xC:
              case 0x10:
                v120 = *(_QWORD *)(v84 + 32);
                v84 = *(_QWORD *)(v84 + 24);
                v83 *= v120;
                continue;
              case 1:
                v107 = 16;
                goto LABEL_169;
              case 2:
                v107 = 32;
                goto LABEL_169;
              case 3:
              case 9:
                v107 = 64;
                goto LABEL_169;
              case 4:
                v107 = 80;
                goto LABEL_169;
              case 5:
              case 6:
                v107 = 128;
                goto LABEL_169;
              case 7:
                v131 = v133;
                v122 = 0;
                v135 = v137;
                v139 = v85;
                v145 = v83;
                goto LABEL_195;
              case 0xB:
                v107 = *(_DWORD *)(v84 + 8) >> 8;
                goto LABEL_169;
              case 0xD:
                v132 = v133;
                v136 = v137;
                v140 = v85;
                v146 = v83;
                v124 = (_QWORD *)sub_15A9930(v37, v84);
                v83 = v146;
                v85 = v140;
                v39 = v136;
                v40 = v132;
                v107 = 8LL * *v124;
                goto LABEL_169;
              case 0xE:
                v129 = v133;
                v130 = v137;
                v134 = v85;
                v138 = v83;
                v144 = *(_QWORD *)(v84 + 32);
                v121 = sub_12BE0A0(v37, *(_QWORD *)(v84 + 24));
                v83 = v138;
                v85 = v134;
                v39 = v130;
                v40 = v129;
                v107 = 8 * v144 * v121;
                goto LABEL_169;
              case 0xF:
                v131 = v133;
                v135 = v137;
                v139 = v85;
                v122 = *(_DWORD *)(v84 + 8) >> 8;
                v145 = v83;
LABEL_195:
                v123 = sub_15A9520(v37, v122);
                v83 = v145;
                v85 = v139;
                v39 = v135;
                v40 = v131;
                v107 = (unsigned int)(8 * v123);
LABEL_169:
                v41 = 8 * v153 * v85 * ((v85 + ((unsigned __int64)(v107 * v83 + 7) >> 3) - 1) / v85);
                break;
            }
            goto LABEL_29;
          }
        case 0xF:
          v141 = v40;
          v152 = v39;
          v80 = *(_DWORD *)(v36 + 8) >> 8;
LABEL_122:
          v81 = sub_15A9520(v37, v80);
          v39 = v152;
          v40 = v141;
          v41 = (unsigned int)(8 * v81);
LABEL_29:
          v42 = v40 * ((v40 + ((unsigned __int64)(v41 * v39 + 7) >> 3) - 1) / v40);
          v43 = *(_DWORD *)(v150 + 8) >> 8;
          v165 = v43;
          if ( v43 <= 0x40 )
          {
            v164 = v42 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v43);
LABEL_31:
            if ( v164 == 1 )
              goto LABEL_36;
            if ( !v164 || (v164 & (v164 - 1)) != 0 )
              goto LABEL_33;
            _BitScanReverse64(&v79, v164);
            v167 = 257;
            v50 = v43 + (v79 ^ 0x3F) - 64;
            goto LABEL_54;
          }
          sub_16A4EF0((__int64)&v164, v42, 0);
          v43 = v165;
          if ( v165 <= 0x40 )
            goto LABEL_31;
          if ( v43 - (unsigned int)sub_16A57B0((__int64)&v164) <= 0x40 && *(_QWORD *)v164 == 1 )
            goto LABEL_36;
          if ( (unsigned int)sub_16A5940((__int64)&v164) != 1 )
          {
LABEL_33:
            v167 = 257;
            v44 = sub_15A1070(v150, (__int64)&v164);
            if ( *(_BYTE *)(v34 + 16) <= 0x10u && *(_BYTE *)(v44 + 16) <= 0x10u )
            {
              v34 = sub_15A2C20((__int64 *)v34, v44, 0, 0, *(double *)a4.m128_u64, a5, a6);
              goto LABEL_36;
            }
            v88 = v44;
            v170 = 257;
            v89 = (__int64 *)v34;
            v90 = 15;
            goto LABEL_133;
          }
          v167 = 257;
          v50 = sub_16A57B0((__int64)&v164);
LABEL_54:
          v51 = sub_15A0680(v150, v43 - 1 - v50, 0);
          if ( *(_BYTE *)(v34 + 16) <= 0x10u && *(_BYTE *)(v51 + 16) <= 0x10u )
          {
            v34 = sub_15A2D50((__int64 *)v34, v51, 0, 0, *(double *)a4.m128_u64, a5, a6);
            goto LABEL_36;
          }
          v88 = v51;
          v170 = 257;
          v89 = (__int64 *)v34;
          v90 = 23;
LABEL_133:
          v34 = sub_15FB440(v90, v89, v88, (__int64)&v168, 0);
          if ( v172 )
          {
            v155 = v173;
            sub_157E9D0(v172 + 40, v34);
            v91 = *v155;
            v92 = *(_QWORD *)(v34 + 24) & 7LL;
            *(_QWORD *)(v34 + 32) = v155;
            v91 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v34 + 24) = v91 | v92;
            *(_QWORD *)(v91 + 8) = v34 + 24;
            *v155 = *v155 & 7 | (v34 + 24);
          }
          sub_164B780(v34, v166);
          if ( v171 )
          {
            v163 = v171;
            sub_1623A60((__int64)&v163, (__int64)v171, 2);
            v93 = *(_QWORD *)(v34 + 48);
            v94 = v34 + 48;
            if ( v93 )
            {
              sub_161E7C0(v34 + 48, v93);
              v94 = v34 + 48;
            }
            v95 = v163;
            *(_QWORD *)(v34 + 48) = v163;
            if ( v95 )
              sub_1623210((__int64)&v163, v95, v94);
          }
LABEL_36:
          v168 = "uglygep";
          v170 = 259;
          v45 = sub_1643330(v174);
          v46 = sub_12815B0((__int64 *)&v171, v45, (_BYTE *)v160, v34, (__int64)&v168);
          v47 = v158;
          v160 = v46;
          if ( !v158 )
            v47 = v46;
          v158 = v47;
          if ( v165 > 0x40 && v164 )
            j_j___libc_free_0_0(v164);
          break;
      }
      break;
    }
LABEL_41:
    if ( v32 )
    {
      v48 = *(_BYTE *)(v33 + 8);
      if ( ((v48 - 14) & 0xFD) == 0 )
        goto LABEL_43;
      goto LABEL_47;
    }
LABEL_46:
    v33 = sub_1643D30(v32, *v161);
    v48 = *(_BYTE *)(v33 + 8);
    if ( ((v48 - 14) & 0xFD) == 0 )
    {
LABEL_43:
      v28 = *(_QWORD *)(v33 + 24) | 4LL;
      goto LABEL_44;
    }
LABEL_47:
    v49 = v48 == 13;
    v28 = 0;
    if ( v49 )
      v28 = v33;
LABEL_44:
    v161 += 3;
    if ( v157 == i )
      break;
  }
  v11 = v149;
  if ( !a3 )
  {
    v156 = 0;
    goto LABEL_61;
  }
LABEL_146:
  v97 = sub_15A0680(v150, a3, 0);
  v170 = 259;
  v168 = "uglygep";
  v98 = sub_1643330(v174);
  v160 = sub_12815B0((__int64 *)&v171, v98, (_BYTE *)v160, v97, (__int64)&v168);
LABEL_61:
  if ( !v158 || *(_BYTE *)(v158 + 16) != 56 )
    goto LABEL_108;
  if ( v160 )
  {
    v52 = 0;
    if ( *(_BYTE *)(v160 + 16) == 56 )
      v52 = v160;
  }
  else
  {
    v52 = 0;
  }
  if ( !v156 )
    goto LABEL_108;
  v53 = *(_QWORD *)(v158 + 8);
  if ( !v53 )
    goto LABEL_108;
  if ( *(_QWORD *)(v53 + 8) )
    goto LABEL_108;
  if ( !v52 )
    goto LABEL_108;
  if ( *(_QWORD *)(v158 + 40) != *(_QWORD *)(v52 + 40) )
    goto LABEL_108;
  if ( v52 == v158 )
    goto LABEL_108;
  v54 = *(_DWORD *)(v158 + 20) & 0xFFFFFFF;
  v55 = *(_DWORD *)(v52 + 20) & 0xFFFFFFF;
  if ( (_DWORD)v55 != v54 )
    goto LABEL_108;
  if ( v54 != 2 )
    goto LABEL_108;
  v56 = *(_QWORD **)(v158 - 48);
  v57 = *(_QWORD *)(v158 - 24);
  v58 = *(_QWORD **)(v52 - 24 * v55);
  if ( sub_13FC1A0(v147, v57) || *v56 != *v58 )
    goto LABEL_108;
  v59 = *(_BYTE *)(v57 + 16);
  if ( v59 <= 0x17u )
    goto LABEL_84;
  v60 = v59;
  if ( (unsigned int)v59 - 47 > 2 )
  {
LABEL_214:
    if ( (unsigned int)(v60 - 35) <= 0x11 )
      goto LABEL_81;
LABEL_84:
    v62 = *(_QWORD *)(v52 + 24 * (1LL - (*(_DWORD *)(v52 + 20) & 0xFFFFFFF)));
    v63 = (__int64 *)(v158 + 24 * (1LL - (*(_DWORD *)(v158 + 20) & 0xFFFFFFF)));
    v64 = *v63;
    if ( v62 )
    {
      if ( v64 )
      {
        v65 = v63[1];
        v66 = v63[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v66 = v65;
        if ( v65 )
          *(_QWORD *)(v65 + 16) = *(_QWORD *)(v65 + 16) & 3LL | v66;
      }
      *v63 = v62;
      v67 = *(_QWORD *)(v62 + 8);
      v63[1] = v67;
      if ( v67 )
        *(_QWORD *)(v67 + 16) = (unsigned __int64)(v63 + 1) | *(_QWORD *)(v67 + 16) & 3LL;
      v63[2] = (v62 + 8) | v63[2] & 3;
      *(_QWORD *)(v62 + 8) = v63;
    }
    else
    {
      if ( !v64 )
        goto LABEL_98;
      v127 = v63[1];
      v128 = v63[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v128 = v127;
      if ( v127 )
        *(_QWORD *)(v127 + 16) = v128 | *(_QWORD *)(v127 + 16) & 3LL;
      *v63 = 0;
    }
    v68 = (_QWORD *)(v52 + 24 * (1LL - (*(_DWORD *)(v52 + 20) & 0xFFFFFFF)));
    if ( *v68 )
    {
      v69 = v68[1];
      v70 = v68[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v70 = v69;
      if ( v69 )
        *(_QWORD *)(v69 + 16) = *(_QWORD *)(v69 + 16) & 3LL | v70;
    }
    *v68 = v64;
    if ( v64 )
    {
      v71 = *(_QWORD *)(v64 + 8);
      v68[1] = v71;
      if ( v71 )
        *(_QWORD *)(v71 + 16) = (unsigned __int64)(v68 + 1) | *(_QWORD *)(v71 + 16) & 3LL;
      v68[2] = (v64 + 8) | v68[2] & 3LL;
      *(_QWORD *)(v64 + 8) = v68;
    }
LABEL_98:
    v72 = sub_15F2050(v158);
    v73 = sub_1632FA0(v72);
    v169 = 8 * sub_15A95A0(v73, *(_DWORD *)(*(_QWORD *)v158 + 8LL) >> 8);
    if ( v169 > 0x40 )
      sub_16A4EF0((__int64)&v168, 0, 0);
    else
      v168 = 0;
    v74 = (_QWORD *)sub_164A410(v158, v73, (__int64)&v168);
    BYTE2(v163) = 0;
    LOWORD(v163) = 0;
    if ( !(unsigned __int8)sub_140E950(v74, v166, v73, a1[24], (int)v163) )
      goto LABEL_204;
    v75 = v169;
    if ( v169 > 0x40 )
    {
      if ( v75 - (unsigned int)sub_16A57B0((__int64)&v168) > 0x40 )
        goto LABEL_204;
      v76 = *(unsigned __int8 **)v168;
    }
    else
    {
      v76 = v168;
    }
    if ( v166[0] >= (unsigned __int64)v76 )
    {
      sub_15FA2E0(v158, 1);
LABEL_105:
      if ( v169 > 0x40 && v168 )
        j_j___libc_free_0_0(v168);
      goto LABEL_108;
    }
LABEL_204:
    sub_15FA2E0(v158, 0);
    sub_15FA2E0(v52, 0);
    goto LABEL_105;
  }
  if ( (*(_BYTE *)(v57 + 23) & 0x40) != 0 )
    v61 = *(__int64 **)(v57 - 8);
  else
    v61 = (__int64 *)(v57 - 24LL * (*(_DWORD *)(v57 + 20) & 0xFFFFFFF));
  if ( *(_BYTE *)(v61[3] + 16) == 13 )
  {
    v57 = *v61;
    v59 = *(_BYTE *)(*v61 + 16);
    if ( v59 <= 0x17u )
      goto LABEL_84;
    v60 = v59;
    goto LABEL_214;
  }
LABEL_81:
  if ( ((v59 - 35) & 0xFD) != 0
    || *(_BYTE *)(*(_QWORD *)(v57 - 48) + 16LL) != 13 && *(_BYTE *)(*(_QWORD *)(v57 - 24) + 16LL) != 13 )
  {
    goto LABEL_84;
  }
LABEL_108:
  v77 = *(__int64 ***)v11;
  if ( *(_QWORD *)v11 != *(_QWORD *)v160 )
  {
    v167 = 257;
    if ( v77 != *(__int64 ***)v160 )
    {
      if ( *(_BYTE *)(v160 + 16) > 0x10u )
      {
        v170 = 257;
        v160 = sub_15FDBD0(47, v160, (__int64)v77, (__int64)&v168, 0);
        if ( v172 )
        {
          v115 = v173;
          sub_157E9D0(v172 + 40, v160);
          v116 = *(_QWORD *)(v160 + 24);
          v117 = *v115;
          *(_QWORD *)(v160 + 32) = v115;
          v117 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v160 + 24) = v117 | v116 & 7;
          *(_QWORD *)(v117 + 8) = v160 + 24;
          *v115 = *v115 & 7 | (v160 + 24);
        }
        sub_164B780(v160, v166);
        if ( v171 )
        {
          v164 = (__int64)v171;
          sub_1623A60((__int64)&v164, (__int64)v171, 2);
          v118 = *(_QWORD *)(v160 + 48);
          if ( v118 )
            sub_161E7C0(v160 + 48, v118);
          v119 = (unsigned __int8 *)v164;
          *(_QWORD *)(v160 + 48) = v164;
          if ( v119 )
            sub_1623210((__int64)&v164, v119, v160 + 48);
        }
      }
      else
      {
        v160 = sub_15A46C0(47, (__int64 ***)v160, v77, 0);
      }
    }
  }
  sub_164D160(v11, v160, a4, a5, a6, a7, v29, v30, a10, a11);
  result = sub_15F20C0((_QWORD *)v11);
  if ( v171 )
    return sub_161E7C0((__int64)&v171, (__int64)v171);
  return result;
}
