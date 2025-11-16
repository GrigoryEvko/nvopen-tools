// Function: sub_1A04A80
// Address: 0x1a04a80
//
__int64 __fastcall sub_1A04A80(
        __int64 *a1,
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
  __int64 v10; // rdi
  unsigned int v11; // eax
  int v12; // ebx
  unsigned int v13; // r15d
  unsigned int v14; // ebx
  int v15; // edx
  unsigned __int64 v16; // r12
  _BYTE *v17; // rax
  unsigned int v18; // eax
  int v19; // edx
  _BYTE *v20; // rax
  unsigned int v21; // r13d
  __int64 *v22; // rbx
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  char *v26; // rsi
  _BYTE *v27; // rax
  __int64 v28; // rdi
  __int64 v29; // r15
  __int64 v30; // rsi
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  int v35; // r9d
  __int64 v36; // r14
  __int64 v37; // r10
  int v38; // eax
  bool v39; // al
  __int64 v40; // r11
  int v41; // eax
  int v42; // edx
  __int64 v43; // rax
  unsigned int v44; // ecx
  unsigned int i; // eax
  __int64 v46; // rax
  unsigned int v47; // ecx
  __int64 v48; // rdi
  __int64 v49; // rax
  unsigned int v50; // ecx
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rdi
  int v54; // r10d
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // rcx
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rdi
  __int64 v61; // r11
  int v62; // r8d
  int v63; // ecx
  unsigned __int64 v64; // rsi
  unsigned int v65; // edx
  unsigned __int64 v66; // rdi
  __int64 v67; // r11
  int v68; // r8d
  int v69; // ecx
  unsigned __int64 v70; // rsi
  unsigned int v71; // edx
  unsigned __int64 v72; // rdi
  __int64 v73; // rax
  unsigned __int64 v74; // rdx
  __int64 v75; // r15
  unsigned int v76; // edx
  __int64 v77; // rax
  int v78; // r10d
  unsigned __int64 v79; // r13
  unsigned int v80; // edx
  __int64 v81; // r14
  __int64 v82; // rdi
  __int64 v83; // rbx
  __int64 v84; // r12
  __int64 v85; // rdi
  _BYTE *v86; // rbx
  unsigned __int64 v87; // r12
  __int64 v88; // rdi
  bool v90; // cc
  __int64 v91; // rax
  unsigned int v92; // r13d
  __int64 v93; // rcx
  int v94; // eax
  __int64 v95; // rbx
  unsigned int v96; // edx
  __int64 v97; // rax
  __int64 v98; // rax
  double v99; // xmm4_8
  double v100; // xmm5_8
  int v101; // edx
  _BYTE *v102; // rax
  unsigned int v103; // edx
  __int64 v104; // rdi
  int v105; // r11d
  int v106; // r10d
  unsigned int v107; // r13d
  __int64 v108; // rsi
  unsigned int v109; // edx
  __int64 v110; // rsi
  int v111; // r11d
  int v112; // edx
  unsigned int v113; // r13d
  __int64 v114; // rsi
  __int64 v115; // rax
  unsigned __int64 v116; // [rsp+0h] [rbp-260h]
  __int64 v117; // [rsp+8h] [rbp-258h]
  unsigned int v118; // [rsp+10h] [rbp-250h]
  unsigned int v119; // [rsp+1Ch] [rbp-244h]
  unsigned int v120; // [rsp+40h] [rbp-220h]
  __int64 v121; // [rsp+40h] [rbp-220h]
  __int64 v122; // [rsp+48h] [rbp-218h]
  __int64 v123; // [rsp+48h] [rbp-218h]
  __int64 v124; // [rsp+48h] [rbp-218h]
  __int64 v125; // [rsp+48h] [rbp-218h]
  __int64 v126; // [rsp+48h] [rbp-218h]
  __int64 v127; // [rsp+48h] [rbp-218h]
  __int64 v128; // [rsp+48h] [rbp-218h]
  unsigned __int8 v130; // [rsp+5Bh] [rbp-205h]
  int v131; // [rsp+5Ch] [rbp-204h]
  __int64 v132; // [rsp+60h] [rbp-200h]
  __int64 v133; // [rsp+60h] [rbp-200h]
  __int64 v134; // [rsp+60h] [rbp-200h]
  unsigned int v135; // [rsp+60h] [rbp-200h]
  __int64 v136; // [rsp+60h] [rbp-200h]
  __int64 v137; // [rsp+60h] [rbp-200h]
  unsigned int v138; // [rsp+60h] [rbp-200h]
  __int64 v139; // [rsp+60h] [rbp-200h]
  __int64 v140; // [rsp+60h] [rbp-200h]
  __int64 v141; // [rsp+60h] [rbp-200h]
  int v142; // [rsp+68h] [rbp-1F8h]
  unsigned int v143; // [rsp+70h] [rbp-1F0h]
  __int64 v144; // [rsp+70h] [rbp-1F0h]
  __int64 *v145; // [rsp+78h] [rbp-1E8h]
  char *v146; // [rsp+80h] [rbp-1E0h] BYREF
  unsigned int v147; // [rsp+88h] [rbp-1D8h]
  unsigned __int64 v148; // [rsp+90h] [rbp-1D0h] BYREF
  unsigned int v149; // [rsp+98h] [rbp-1C8h]
  unsigned __int64 v150; // [rsp+A0h] [rbp-1C0h] BYREF
  int v151; // [rsp+A8h] [rbp-1B8h]
  __int64 *v152; // [rsp+B0h] [rbp-1B0h] BYREF
  char *v153; // [rsp+B8h] [rbp-1A8h] BYREF
  unsigned int v154; // [rsp+C0h] [rbp-1A0h]
  unsigned __int64 v155; // [rsp+D0h] [rbp-190h] BYREF
  char *v156; // [rsp+D8h] [rbp-188h] BYREF
  unsigned int v157; // [rsp+E0h] [rbp-180h]
  __int64 v158; // [rsp+F0h] [rbp-170h] BYREF
  __int64 v159; // [rsp+F8h] [rbp-168h]
  __int64 v160; // [rsp+100h] [rbp-160h]
  unsigned int v161; // [rsp+108h] [rbp-158h]
  _BYTE *v162; // [rsp+110h] [rbp-150h] BYREF
  __int64 v163; // [rsp+118h] [rbp-148h]
  _BYTE v164[64]; // [rsp+120h] [rbp-140h] BYREF
  _BYTE *v165; // [rsp+160h] [rbp-100h] BYREF
  __int64 v166; // [rsp+168h] [rbp-F8h]
  _BYTE v167[240]; // [rsp+170h] [rbp-F0h] BYREF

  v145 = a1;
  v10 = *a1;
  if ( *(_BYTE *)(v10 + 8) == 16 )
    v10 = **(_QWORD **)(v10 + 16);
  v11 = sub_1643030(v10);
  v119 = v11;
  v12 = *((unsigned __int8 *)v145 + 16);
  v165 = v167;
  v131 = v12;
  v13 = v12 - 24;
  v166 = 0x800000000LL;
  LODWORD(v163) = v11;
  if ( v11 > 0x40 )
  {
    sub_16A4EF0((__int64)&v162, 1, 0);
    v14 = v163;
    v16 = (unsigned __int64)v162;
    LODWORD(v163) = 0;
    v15 = v166;
    if ( HIDWORD(v166) <= (unsigned int)v166 )
    {
      sub_1A01B90((__int64)&v165, 0);
      v15 = v166;
    }
  }
  else
  {
    v14 = v11;
    v15 = 0;
    LODWORD(v163) = 0;
    v16 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v11) & 1;
    v162 = (_BYTE *)v16;
  }
  v17 = &v165[24 * v15];
  if ( v17 )
  {
    *((_QWORD *)v17 + 1) = v16;
    *((_DWORD *)v17 + 4) = v14;
    *(_QWORD *)v17 = v145;
    LODWORD(v166) = v166 + 1;
  }
  else
  {
    LODWORD(v166) = v15 + 1;
    if ( v14 > 0x40 && v16 )
      j_j___libc_free_0_0(v16);
  }
  if ( (unsigned int)v163 > 0x40 && v162 )
    j_j___libc_free_0_0(v162);
  v158 = 0;
  v162 = v164;
  v163 = 0x800000000LL;
  v18 = v166;
  v159 = 0;
  v160 = 0;
  v161 = 0;
  if ( !(_DWORD)v166 )
  {
    v130 = 0;
    if ( *(_DWORD *)(a2 + 8) )
      goto LABEL_145;
    goto LABEL_189;
  }
  v130 = 0;
  v143 = v13;
  while ( 2 )
  {
    v23 = v18;
    v24 = v18 - 1;
    v25 = (__int64)&v165[24 * v23 - 24];
    v152 = *(__int64 **)v25;
    v154 = *(_DWORD *)(v25 + 16);
    v26 = *(char **)(v25 + 8);
    LODWORD(v166) = v24;
    v27 = &v165[24 * v24];
    v153 = v26;
    *(_DWORD *)(v25 + 16) = 0;
    if ( *((_DWORD *)v27 + 4) > 0x40u )
    {
      v28 = *((_QWORD *)v27 + 1);
      if ( v28 )
        j_j___libc_free_0_0(v28);
    }
    v145 = v152;
    v22 = v152 - 6;
    do
    {
      v29 = *v22;
      v147 = v154;
      if ( v154 > 0x40 )
        sub_16A4FD0((__int64)&v146, (const void **)&v153);
      else
        v146 = v153;
      v30 = v143;
      v31 = sub_19FF050(v29, v143);
      v36 = v31;
      if ( v31 )
      {
        v155 = v31;
        v157 = v147;
        if ( v147 > 0x40 )
        {
          sub_16A4FD0((__int64)&v156, (const void **)&v146);
          v19 = v166;
          if ( (unsigned int)v166 < HIDWORD(v166) )
            goto LABEL_14;
        }
        else
        {
          v19 = v166;
          v156 = v146;
          if ( (unsigned int)v166 < HIDWORD(v166) )
            goto LABEL_14;
        }
        sub_1A01B90((__int64)&v165, 0);
        v19 = v166;
LABEL_14:
        v20 = &v165[24 * v19];
        if ( v20 )
        {
          *(_QWORD *)v20 = v155;
          *((_DWORD *)v20 + 4) = v157;
          *((_QWORD *)v20 + 1) = v156;
          LODWORD(v166) = v166 + 1;
        }
        else
        {
          LODWORD(v166) = v19 + 1;
          if ( v157 > 0x40 && v156 )
            j_j___libc_free_0_0(v156);
        }
        goto LABEL_16;
      }
      if ( v161 )
      {
        v30 = v159;
        v32 = (v161 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
        v37 = v159 + 24 * v32;
        v33 = *(_QWORD *)v37;
        if ( v29 == *(_QWORD *)v37 )
        {
LABEL_34:
          if ( v37 != v159 + 24LL * v161 )
          {
            v21 = v147;
            if ( v147 <= 0x40 )
            {
              v39 = v146 == 0;
            }
            else
            {
              v132 = v37;
              v38 = sub_16A57B0((__int64)&v146);
              v37 = v132;
              v39 = v21 == v38;
            }
            v40 = v37 + 8;
            if ( v39 )
              goto LABEL_52;
            v34 = *(unsigned int *)(v37 + 16);
            if ( (unsigned int)v34 <= 0x40 )
            {
              if ( *(_QWORD *)(v37 + 8) )
                goto LABEL_40;
              if ( v21 <= 0x40 )
              {
                v30 = (__int64)v146;
                *(_QWORD *)(v37 + 8) = v146;
                v73 = v147;
                *(_DWORD *)(v37 + 16) = v147;
                v74 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v73;
                if ( (unsigned int)v73 > 0x40 )
                {
                  v115 = (unsigned int)((unsigned __int64)(v73 + 63) >> 6) - 1;
                  *(_QWORD *)(v30 + 8 * v115) &= v74;
                }
                else
                {
                  *(_QWORD *)(v37 + 8) = v30 & v74;
                }
                v21 = v147;
                goto LABEL_52;
              }
            }
            else
            {
              v120 = *(_DWORD *)(v37 + 16);
              v122 = v37;
              v133 = v37 + 8;
              v41 = sub_16A57B0(v37 + 8);
              v34 = v120;
              v40 = v133;
              v37 = v122;
              if ( v120 != v41 )
              {
LABEL_40:
                if ( (unsigned int)(v131 - 50) <= 1 )
                  goto LABEL_52;
                if ( v143 == 28 )
                {
                  if ( (unsigned int)v34 > 0x40 )
                  {
                    v30 = 0;
                    v128 = v40;
                    v141 = v37;
                    **(_QWORD **)(v37 + 8) = 0;
                    memset(
                      (void *)(*(_QWORD *)(v37 + 8) + 8LL),
                      0,
                      8 * (unsigned int)(((unsigned __int64)*(unsigned int *)(v37 + 16) + 63) >> 6) - 8);
                    v21 = v147;
                    v37 = v141;
                    v40 = v128;
                  }
                  else
                  {
                    *(_QWORD *)(v37 + 8) = 0;
                    v21 = v147;
                  }
                  goto LABEL_52;
                }
                if ( (unsigned int)(v131 - 35) > 1 )
                {
                  if ( (unsigned int)v34 <= 3 )
                  {
                    v30 = *(_QWORD *)(v37 + 8);
                    v42 = 1 << (v34 - 1);
                    if ( (_DWORD)v34 == 3 )
                      v42 = 2;
                    LODWORD(v43) = (_DWORD)v146;
                    v44 = v34 + v42;
                    if ( v21 > 0x40 )
                      v43 = *(_QWORD *)v146;
                    for ( i = v30 + v43; v44 <= i; i -= v42 )
                      ;
                    *(_QWORD *)(v37 + 8) = (unsigned int)(0xFFFFFFFFFFFFFFFFLL >> -(char)v34) & i;
LABEL_51:
                    v21 = v147;
                    goto LABEL_52;
                  }
                  v92 = v34 - 2;
                  v149 = v34;
                  v93 = 1LL << ((unsigned __int8)v34 - 2);
                  if ( (unsigned int)v34 <= 0x40 )
                  {
                    v148 = 0;
                    v94 = v34;
                  }
                  else
                  {
                    v118 = v34;
                    v121 = 1LL << ((unsigned __int8)v34 - 2);
                    v124 = v37;
                    v137 = v40;
                    sub_16A4EF0((__int64)&v148, 0, 0);
                    v94 = v149;
                    v40 = v137;
                    v37 = v124;
                    v93 = v121;
                    v34 = v118;
                    if ( v149 > 0x40 )
                    {
                      *(_QWORD *)(v148 + 8LL * (v92 >> 6)) |= v121;
                      LODWORD(v156) = v149;
                      if ( v149 > 0x40 )
                      {
                        sub_16A4FD0((__int64)&v155, (const void **)&v148);
                        v34 = v118;
                        v37 = v124;
                        v40 = v137;
                        goto LABEL_178;
                      }
LABEL_198:
                      v155 = v148;
LABEL_178:
                      v117 = v37;
                      v125 = v40;
                      sub_16A7490((__int64)&v155, v34);
                      v138 = (unsigned int)v156;
                      v151 = (int)v156;
                      v116 = v155;
                      v150 = v155;
                      sub_16A7200(v125, (__int64 *)&v146);
                      while ( 1 )
                      {
                        v30 = (__int64)&v150;
                        if ( (int)sub_16A9900(v125, &v150) < 0 )
                          break;
                        sub_16A7590(v125, (__int64 *)&v148);
                      }
                      v40 = v125;
                      v36 = 0;
                      v37 = v117;
                      if ( v138 > 0x40 && v116 )
                      {
                        j_j___libc_free_0_0(v116);
                        v37 = v117;
                        v40 = v125;
                      }
                      if ( v149 > 0x40 && v148 )
                      {
                        v126 = v37;
                        v139 = v40;
                        j_j___libc_free_0_0(v148);
                        v37 = v126;
                        v40 = v139;
                      }
                      goto LABEL_51;
                    }
                  }
                  v148 |= v93;
                  LODWORD(v156) = v94;
                  goto LABEL_198;
                }
                v30 = (__int64)&v146;
                v127 = v37;
                v140 = v40;
                sub_16A7200(v40, (__int64 *)&v146);
                v21 = v147;
                v40 = v140;
                v37 = v127;
LABEL_52:
                v46 = *(_QWORD *)(v29 + 8);
                if ( !v46 || *(_QWORD *)(v46 + 8) )
                  goto LABEL_17;
                if ( v21 <= 0x40 && (v47 = *(_DWORD *)(v37 + 16), v47 <= 0x40) )
                {
                  v32 = *(_QWORD *)(v37 + 8);
                  v147 = *(_DWORD *)(v37 + 16);
                  v33 = -v47;
                  v146 = (char *)(v32 & (0xFFFFFFFFFFFFFFFFLL >> v33));
                }
                else
                {
                  v30 = v40;
                  v134 = v37;
                  sub_16A51C0((__int64)&v146, v40);
                  v37 = v134;
                  if ( *(_DWORD *)(v134 + 16) > 0x40u )
                  {
                    v48 = *(_QWORD *)(v134 + 8);
                    if ( v48 )
                    {
                      j_j___libc_free_0_0(v48);
                      v37 = v134;
                    }
                  }
                }
                *(_QWORD *)v37 = -16;
                LODWORD(v160) = v160 - 1;
                ++HIDWORD(v160);
LABEL_60:
                if ( (unsigned __int8)(*(_BYTE *)(v29 + 16) - 35) > 0x11u )
                  goto LABEL_63;
                if ( v143 == 15 )
                {
                  if ( !sub_15FB6B0(v29, v30, v32, v33) )
                    goto LABEL_63;
                  goto LABEL_204;
                }
                if ( v143 == 16 && sub_15FB6D0(v29, 0, v32, v33) )
                {
LABEL_204:
                  v155 = sub_19FFCB0((__int64 ***)v29, a3, a4, a5, a6, v99, v100, a9, a10);
                  v157 = v147;
                  if ( v147 > 0x40 )
                    sub_16A4FD0((__int64)&v156, (const void **)&v146);
                  else
                    v156 = v146;
                  v101 = v166;
                  if ( (unsigned int)v166 >= HIDWORD(v166) )
                  {
                    sub_1A01B90((__int64)&v165, 0);
                    v101 = v166;
                  }
                  v102 = &v165[24 * v101];
                  if ( v102 )
                  {
                    *(_QWORD *)v102 = v155;
                    *((_DWORD *)v102 + 4) = v157;
                    *((_QWORD *)v102 + 1) = v156;
                    LODWORD(v166) = v166 + 1;
                  }
                  else
                  {
                    LODWORD(v166) = v101 + 1;
                    if ( v157 > 0x40 && v156 )
                      j_j___libc_free_0_0(v156);
                  }
                  v130 = 1;
                  v21 = v147;
                  goto LABEL_17;
                }
LABEL_63:
                v49 = (unsigned int)v163;
                if ( (unsigned int)v163 >= HIDWORD(v163) )
                {
                  sub_16CD150((__int64)&v162, v164, 0, 8, v34, v35);
                  v49 = (unsigned int)v163;
                }
                *(_QWORD *)&v162[8 * v49] = v29;
                LODWORD(v163) = v163 + 1;
                if ( v161 )
                {
                  v50 = (v161 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
                  v51 = v159 + 24LL * v50;
                  v52 = *(_QWORD *)v51;
                  if ( v29 == *(_QWORD *)v51 )
                  {
LABEL_67:
                    v53 = v51 + 8;
                    if ( *(_DWORD *)(v51 + 16) > 0x40u )
                      goto LABEL_68;
                    goto LABEL_96;
                  }
                  v61 = 0;
                  v62 = 1;
                  while ( v52 != -8 )
                  {
                    if ( !v61 && v52 == -16 )
                      v61 = v51;
                    v50 = (v161 - 1) & (v62 + v50);
                    v51 = v159 + 24LL * v50;
                    v52 = *(_QWORD *)v51;
                    if ( v29 == *(_QWORD *)v51 )
                      goto LABEL_67;
                    ++v62;
                  }
                  if ( v61 )
                    v51 = v61;
                  ++v158;
                  v63 = v160 + 1;
                  if ( 4 * ((int)v160 + 1) < 3 * v161 )
                  {
                    if ( v161 - HIDWORD(v160) - v63 > v161 >> 3 )
                      goto LABEL_93;
                    sub_1A04880((__int64)&v158, v161);
                    if ( !v161 )
                    {
LABEL_278:
                      LODWORD(v160) = v160 + 1;
                      BUG();
                    }
                    v106 = 1;
                    v107 = (v161 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
                    v63 = v160 + 1;
                    v51 = v159 + 24LL * v107;
                    v108 = *(_QWORD *)v51;
                    if ( v29 == *(_QWORD *)v51 )
                      goto LABEL_93;
                    while ( v108 != -8 )
                    {
                      if ( !v36 && v108 == -16 )
                        v36 = v51;
                      v107 = (v161 - 1) & (v106 + v107);
                      v51 = v159 + 24LL * v107;
                      v108 = *(_QWORD *)v51;
                      if ( v29 == *(_QWORD *)v51 )
                        goto LABEL_93;
                      ++v106;
                    }
                    goto LABEL_218;
                  }
                }
                else
                {
                  ++v158;
                }
                sub_1A04880((__int64)&v158, 2 * v161);
                if ( !v161 )
                  goto LABEL_278;
                v103 = (v161 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
                v63 = v160 + 1;
                v51 = v159 + 24LL * v103;
                v104 = *(_QWORD *)v51;
                if ( v29 == *(_QWORD *)v51 )
                  goto LABEL_93;
                v105 = 1;
                while ( v104 != -8 )
                {
                  if ( v104 == -16 && !v36 )
                    v36 = v51;
                  v103 = (v161 - 1) & (v105 + v103);
                  v51 = v159 + 24LL * v103;
                  v104 = *(_QWORD *)v51;
                  if ( v29 == *(_QWORD *)v51 )
                    goto LABEL_93;
                  ++v105;
                }
LABEL_218:
                if ( v36 )
                  v51 = v36;
LABEL_93:
                LODWORD(v160) = v63;
                if ( *(_QWORD *)v51 != -8 )
                  --HIDWORD(v160);
                *(_QWORD *)v51 = v29;
                v53 = v51 + 8;
                *(_DWORD *)(v51 + 16) = 1;
                *(_QWORD *)(v51 + 8) = 0;
LABEL_96:
                if ( v147 <= 0x40 )
                {
                  v64 = (unsigned __int64)v146;
                  *(_QWORD *)(v51 + 8) = v146;
                  v65 = v147;
                  *(_DWORD *)(v51 + 16) = v147;
                  v66 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v65;
                  if ( v65 > 0x40 )
                  {
                    v91 = (unsigned int)(((unsigned __int64)v65 + 63) >> 6) - 1;
                    *(_QWORD *)(v64 + 8 * v91) &= v66;
                  }
                  else
                  {
                    *(_QWORD *)(v51 + 8) = v66 & v64;
                  }
LABEL_69:
                  if ( v147 <= 0x40 )
                    goto LABEL_20;
                  goto LABEL_18;
                }
LABEL_68:
                sub_16A51C0(v53, (__int64)&v146);
                goto LABEL_69;
              }
            }
            v30 = (__int64)&v146;
            v123 = v37;
            v136 = v40;
            sub_16A51C0(v40, (__int64)&v146);
            v21 = v147;
            v40 = v136;
            v37 = v123;
            goto LABEL_52;
          }
        }
        else
        {
          v54 = 1;
          while ( v33 != -8 )
          {
            LODWORD(v34) = v54 + 1;
            v32 = (v161 - 1) & (v54 + (_DWORD)v32);
            v37 = v159 + 24LL * (unsigned int)v32;
            v33 = *(_QWORD *)v37;
            if ( v29 == *(_QWORD *)v37 )
              goto LABEL_34;
            v54 = v34;
          }
        }
      }
      v55 = *(_QWORD *)(v29 + 8);
      if ( v55 && !*(_QWORD *)(v55 + 8) )
        goto LABEL_60;
      v56 = (unsigned int)v163;
      if ( (unsigned int)v163 >= HIDWORD(v163) )
      {
        sub_16CD150((__int64)&v162, v164, 0, 8, v34, v35);
        v56 = (unsigned int)v163;
      }
      *(_QWORD *)&v162[8 * v56] = v29;
      LODWORD(v163) = v163 + 1;
      if ( v161 )
      {
        LODWORD(v57) = (v161 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
        v58 = v159 + 24LL * (unsigned int)v57;
        v59 = *(_QWORD *)v58;
        if ( v29 == *(_QWORD *)v58 )
        {
LABEL_80:
          v60 = v58 + 8;
          if ( *(_DWORD *)(v58 + 16) > 0x40u )
            goto LABEL_81;
          goto LABEL_109;
        }
        v67 = 0;
        v68 = 1;
        while ( v59 != -8 )
        {
          if ( v59 == -16 && !v67 )
            v67 = v58;
          v57 = (v161 - 1) & ((_DWORD)v57 + v68);
          v58 = v159 + 24 * v57;
          v59 = *(_QWORD *)v58;
          if ( v29 == *(_QWORD *)v58 )
            goto LABEL_80;
          ++v68;
        }
        if ( v67 )
          v58 = v67;
        ++v158;
        v69 = v160 + 1;
        if ( 4 * ((int)v160 + 1) < 3 * v161 )
        {
          if ( v161 - HIDWORD(v160) - v69 > v161 >> 3 )
            goto LABEL_106;
          sub_1A04880((__int64)&v158, v161);
          if ( !v161 )
          {
LABEL_279:
            LODWORD(v160) = v160 + 1;
            BUG();
          }
          v112 = 1;
          v113 = (v161 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
          v69 = v160 + 1;
          v58 = v159 + 24LL * v113;
          v114 = *(_QWORD *)v58;
          if ( v29 == *(_QWORD *)v58 )
            goto LABEL_106;
          while ( v114 != -8 )
          {
            if ( v114 == -16 && !v36 )
              v36 = v58;
            v113 = (v161 - 1) & (v112 + v113);
            v58 = v159 + 24LL * v113;
            v114 = *(_QWORD *)v58;
            if ( v29 == *(_QWORD *)v58 )
              goto LABEL_106;
            ++v112;
          }
          goto LABEL_248;
        }
      }
      else
      {
        ++v158;
      }
      sub_1A04880((__int64)&v158, 2 * v161);
      if ( !v161 )
        goto LABEL_279;
      v109 = (v161 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
      v69 = v160 + 1;
      v58 = v159 + 24LL * v109;
      v110 = *(_QWORD *)v58;
      if ( v29 == *(_QWORD *)v58 )
        goto LABEL_106;
      v111 = 1;
      while ( v110 != -8 )
      {
        if ( !v36 && v110 == -16 )
          v36 = v58;
        v109 = (v161 - 1) & (v111 + v109);
        v58 = v159 + 24LL * v109;
        v110 = *(_QWORD *)v58;
        if ( v29 == *(_QWORD *)v58 )
          goto LABEL_106;
        ++v111;
      }
LABEL_248:
      if ( v36 )
        v58 = v36;
LABEL_106:
      LODWORD(v160) = v69;
      if ( *(_QWORD *)v58 != -8 )
        --HIDWORD(v160);
      *(_QWORD *)v58 = v29;
      v60 = v58 + 8;
      *(_DWORD *)(v58 + 16) = 1;
      *(_QWORD *)(v58 + 8) = 0;
LABEL_109:
      if ( v147 > 0x40 )
      {
LABEL_81:
        sub_16A51C0(v60, (__int64)&v146);
        v21 = v147;
        goto LABEL_17;
      }
      v70 = (unsigned __int64)v146;
      *(_QWORD *)(v58 + 8) = v146;
      v71 = v147;
      *(_DWORD *)(v58 + 16) = v147;
      v72 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v71;
      if ( v71 > 0x40 )
      {
        v98 = (unsigned int)(((unsigned __int64)v71 + 63) >> 6) - 1;
        *(_QWORD *)(v70 + 8 * v98) &= v72;
LABEL_16:
        v21 = v147;
        goto LABEL_17;
      }
      *(_QWORD *)(v58 + 8) = v72 & v70;
      v21 = v147;
LABEL_17:
      if ( v21 <= 0x40 )
        goto LABEL_20;
LABEL_18:
      if ( v146 )
        j_j___libc_free_0_0(v146);
LABEL_20:
      v22 += 3;
    }
    while ( v145 != v22 );
    if ( v154 > 0x40 && v153 )
      j_j___libc_free_0_0(v153);
    v18 = v166;
    if ( (_DWORD)v166 )
      continue;
    break;
  }
  v13 = v143;
  if ( !(_DWORD)v163 )
    goto LABEL_142;
  v135 = v143;
  v144 = 8LL * (unsigned int)v163;
  v75 = 0;
  while ( 2 )
  {
    while ( 2 )
    {
      if ( !v161 )
        goto LABEL_128;
      v78 = 1;
      v79 = *(_QWORD *)&v162[v75];
      v80 = (v161 - 1) & (((unsigned int)v79 >> 9) ^ ((unsigned int)v79 >> 4));
      v81 = v159 + 24LL * v80;
      v82 = *(_QWORD *)v81;
      if ( v79 == *(_QWORD *)v81 )
      {
LABEL_131:
        if ( v81 == v159 + 24LL * v161 )
          goto LABEL_128;
        LODWORD(v153) = *(_DWORD *)(v81 + 16);
        if ( (unsigned int)v153 <= 0x40 )
        {
          v152 = *(__int64 **)(v81 + 8);
          if ( !v152 )
            goto LABEL_128;
        }
        else
        {
          sub_16A4FD0((__int64)&v152, (const void **)(v81 + 8));
          if ( (unsigned int)v153 > 0x40 )
          {
            v142 = (int)v153;
            if ( v142 != (unsigned int)sub_16A57B0((__int64)&v152) )
              goto LABEL_135;
LABEL_126:
            if ( v152 )
              j_j___libc_free_0_0(v152);
LABEL_128:
            v75 += 8;
            if ( v144 == v75 )
              goto LABEL_141;
            continue;
          }
          if ( !v152 )
            goto LABEL_128;
LABEL_135:
          if ( *(_DWORD *)(v81 + 16) > 0x40u )
          {
            **(_QWORD **)(v81 + 8) = 0;
            memset(
              (void *)(*(_QWORD *)(v81 + 8) + 8LL),
              0,
              8 * (unsigned int)(((unsigned __int64)*(unsigned int *)(v81 + 16) + 63) >> 6) - 8);
LABEL_121:
            v155 = v79;
            v157 = (unsigned int)v153;
            if ( (unsigned int)v153 > 0x40 )
            {
              sub_16A4FD0((__int64)&v156, (const void **)&v152);
              v76 = *(_DWORD *)(a2 + 8);
              if ( v76 >= *(_DWORD *)(a2 + 12) )
                goto LABEL_163;
            }
            else
            {
              v76 = *(_DWORD *)(a2 + 8);
              v156 = (char *)v152;
              if ( v76 < *(_DWORD *)(a2 + 12) )
                goto LABEL_123;
LABEL_163:
              sub_1A01D10(a2, 0);
              v76 = *(_DWORD *)(a2 + 8);
            }
LABEL_123:
            v77 = *(_QWORD *)a2 + 24LL * v76;
            if ( v77 )
            {
              *(_QWORD *)v77 = v155;
              *(_DWORD *)(v77 + 16) = v157;
              *(_QWORD *)(v77 + 8) = v156;
              ++*(_DWORD *)(a2 + 8);
            }
            else
            {
              v90 = v157 <= 0x40;
              *(_DWORD *)(a2 + 8) = v76 + 1;
              if ( !v90 && v156 )
                j_j___libc_free_0_0(v156);
            }
            if ( (unsigned int)v153 > 0x40 )
              goto LABEL_126;
            goto LABEL_128;
          }
        }
        *(_QWORD *)(v81 + 8) = 0;
        goto LABEL_121;
      }
      break;
    }
    while ( v82 != -8 )
    {
      v80 = (v161 - 1) & (v78 + v80);
      v81 = v159 + 24LL * v80;
      v82 = *(_QWORD *)v81;
      if ( v79 == *(_QWORD *)v81 )
        goto LABEL_131;
      ++v78;
    }
    v75 += 8;
    if ( v144 != v75 )
      continue;
    break;
  }
LABEL_141:
  v13 = v135;
LABEL_142:
  if ( !*(_DWORD *)(a2 + 8) )
  {
LABEL_189:
    v95 = sub_15A14F0(v13, (__int64 **)*v145, 0);
    LODWORD(v156) = v119;
    if ( v119 <= 0x40 )
      v155 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v119) & 1;
    else
      sub_16A4EF0((__int64)&v155, 1, 0);
    v96 = *(_DWORD *)(a2 + 8);
    if ( v96 >= *(_DWORD *)(a2 + 12) )
    {
      sub_1A01D10(a2, 0);
      v96 = *(_DWORD *)(a2 + 8);
    }
    v97 = *(_QWORD *)a2 + 24LL * v96;
    if ( v97 )
    {
      *(_QWORD *)v97 = v95;
      *(_DWORD *)(v97 + 16) = (_DWORD)v156;
      *(_QWORD *)(v97 + 8) = v155;
      ++*(_DWORD *)(a2 + 8);
    }
    else
    {
      v90 = (unsigned int)v156 <= 0x40;
      *(_DWORD *)(a2 + 8) = v96 + 1;
      if ( !v90 && v155 )
        j_j___libc_free_0_0(v155);
    }
  }
  if ( v162 != v164 )
    _libc_free((unsigned __int64)v162);
LABEL_145:
  if ( v161 )
  {
    v83 = v159;
    v84 = v159 + 24LL * v161;
    do
    {
      if ( *(_QWORD *)v83 != -16 && *(_QWORD *)v83 != -8 && *(_DWORD *)(v83 + 16) > 0x40u )
      {
        v85 = *(_QWORD *)(v83 + 8);
        if ( v85 )
          j_j___libc_free_0_0(v85);
      }
      v83 += 24;
    }
    while ( v84 != v83 );
  }
  j___libc_free_0(v159);
  v86 = v165;
  v87 = (unsigned __int64)&v165[24 * (unsigned int)v166];
  if ( v165 != (_BYTE *)v87 )
  {
    do
    {
      v87 -= 24LL;
      if ( *(_DWORD *)(v87 + 16) > 0x40u )
      {
        v88 = *(_QWORD *)(v87 + 8);
        if ( v88 )
          j_j___libc_free_0_0(v88);
      }
    }
    while ( v86 != (_BYTE *)v87 );
    v87 = (unsigned __int64)v165;
  }
  if ( (_BYTE *)v87 != v167 )
    _libc_free(v87);
  return v130;
}
