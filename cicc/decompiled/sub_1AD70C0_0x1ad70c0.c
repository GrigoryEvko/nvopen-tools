// Function: sub_1AD70C0
// Address: 0x1ad70c0
//
void __fastcall sub_1AD70C0(
        __int64 a1,
        __int64 a2,
        _BYTE *a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  const char *v11; // rax
  __int64 v12; // rdx
  __int64 v14; // r14
  __int64 v15; // r12
  __int64 v16; // rdi
  __int64 v17; // rbx
  __int64 v18; // r9
  __int64 v19; // rax
  char v20; // di
  unsigned int v21; // esi
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rcx
  __int64 v25; // r9
  __int64 v26; // r8
  __int64 v27; // rax
  __int64 v28; // r13
  __int64 v29; // r12
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rdi
  unsigned __int64 v33; // rdi
  __int64 v34; // rsi
  __int64 *v35; // rax
  __int64 *v36; // rdi
  __int64 *v37; // rcx
  __int64 *v38; // rax
  __int64 v39; // rcx
  __int64 v40; // rdx
  __int64 v41; // r14
  __int64 *v42; // rbx
  __int64 v43; // rsi
  __int64 v44; // r12
  __int64 v45; // r13
  __int64 v46; // rdx
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // r14
  __int64 v50; // r15
  __int64 v51; // rbx
  __int64 v52; // r15
  __int64 v53; // r13
  __int64 v54; // rdx
  __int64 *v55; // rax
  unsigned __int64 v56; // rdx
  __int64 v57; // rdx
  __int64 v58; // rdx
  __int64 v59; // rax
  __int64 v60; // r12
  __int64 v61; // r12
  __int64 v62; // rcx
  int v63; // eax
  __int64 v64; // rax
  int v65; // edx
  __int64 v66; // rbx
  __int64 v67; // r12
  int v68; // eax
  __int64 v69; // rax
  int v70; // edx
  __int64 v71; // rdx
  _QWORD *v72; // rax
  __int64 v73; // rcx
  unsigned __int64 v74; // rdx
  __int64 v75; // rdx
  __int64 v76; // rdx
  __int64 v77; // rcx
  __int64 v78; // rbx
  __int64 v79; // rsi
  __int64 v80; // r15
  __int64 v81; // rdx
  __int64 v82; // rcx
  _QWORD *v83; // rdi
  __int64 v84; // r9
  __int64 v85; // rbx
  __int64 v86; // r14
  __int64 v87; // r8
  __int64 v88; // r12
  __int64 v89; // r13
  __int64 v90; // r13
  int v91; // eax
  __int64 v92; // rax
  int v93; // edx
  __int64 v94; // rdx
  __int64 *v95; // rax
  unsigned __int64 v96; // rdx
  __int64 v97; // rdx
  __int64 v98; // rdx
  __int64 v99; // rax
  _QWORD *v100; // r12
  __int64 *v101; // r13
  __int64 v102; // rdx
  __int64 v103; // rax
  __int64 v104; // rdi
  __int64 v105; // rax
  int v106; // r14d
  __int64 v107; // rbx
  __int64 v108; // rcx
  _QWORD *v109; // rdx
  __int64 v110; // rdi
  unsigned __int64 v111; // rcx
  __int64 v112; // rcx
  __int64 v113; // rcx
  __int64 v114; // r12
  __int64 v115; // rdx
  __int64 v116; // rax
  double v117; // xmm4_8
  double v118; // xmm5_8
  __int64 v119; // r15
  __int64 v120; // r12
  __int64 v121; // rcx
  __int64 v122; // r8
  __int64 v123; // r9
  __int64 v124; // rsi
  __int64 v125; // rdx
  __int64 v126; // rdx
  int v127; // ecx
  __int64 v128; // r13
  __int64 v129; // rax
  double v130; // xmm4_8
  double v131; // xmm5_8
  __int64 v132; // r12
  __int64 v133; // rsi
  __int64 v134; // rdx
  __int64 v135; // rcx
  __int64 v136; // r8
  __int64 v137; // r9
  __int64 v138; // rbx
  __int64 v139; // r13
  _QWORD *v140; // r12
  int v141; // eax
  __int64 v142; // rax
  int v143; // edx
  __int64 v144; // rdx
  _QWORD *v145; // rax
  __int64 v146; // rcx
  unsigned __int64 v147; // rdx
  __int64 v148; // rdx
  __int64 v149; // rdx
  __int64 v150; // rcx
  _QWORD *v151; // r15
  __int64 v152; // r12
  __int64 v153; // r13
  __int64 v154; // r12
  char *v155; // rax
  __int64 v156; // rsi
  __int64 *v157; // rax
  __int64 v159; // [rsp+8h] [rbp-1D8h]
  __int64 v161; // [rsp+20h] [rbp-1C0h]
  int v162; // [rsp+34h] [rbp-1ACh]
  __int64 v163; // [rsp+40h] [rbp-1A0h]
  __int64 v164; // [rsp+48h] [rbp-198h]
  __int64 v165; // [rsp+48h] [rbp-198h]
  __int64 v166; // [rsp+48h] [rbp-198h]
  __int64 v167; // [rsp+50h] [rbp-190h]
  __int64 *v168; // [rsp+58h] [rbp-188h]
  unsigned __int64 v169; // [rsp+58h] [rbp-188h]
  __int64 v170; // [rsp+58h] [rbp-188h]
  __int64 v171; // [rsp+58h] [rbp-188h]
  const char *v172; // [rsp+60h] [rbp-180h] BYREF
  __int64 v173; // [rsp+68h] [rbp-178h]
  const char **v174; // [rsp+70h] [rbp-170h] BYREF
  char *v175; // [rsp+78h] [rbp-168h]
  __int16 v176; // [rsp+80h] [rbp-160h]
  _QWORD *v177; // [rsp+90h] [rbp-150h]
  __int64 v178; // [rsp+98h] [rbp-148h]
  _QWORD *v179; // [rsp+A0h] [rbp-140h]
  __int64 v180; // [rsp+A8h] [rbp-138h]
  _BYTE *v181; // [rsp+B0h] [rbp-130h] BYREF
  __int64 v182; // [rsp+B8h] [rbp-128h]
  _BYTE v183[64]; // [rsp+C0h] [rbp-120h] BYREF
  __int64 v184; // [rsp+100h] [rbp-E0h] BYREF
  __int64 *v185; // [rsp+108h] [rbp-D8h]
  __int64 *v186; // [rsp+110h] [rbp-D0h]
  __int64 v187; // [rsp+118h] [rbp-C8h]
  int v188; // [rsp+120h] [rbp-C0h]
  _BYTE v189[184]; // [rsp+128h] [rbp-B8h] BYREF

  v14 = *(_QWORD *)(a2 + 56);
  v15 = *(_QWORD *)(a1 + 40);
  v16 = *(_QWORD *)(a1 - 24);
  v181 = v183;
  v17 = *(_QWORD *)(v16 + 48);
  v159 = v16;
  v177 = (_QWORD *)v16;
  v178 = 0;
  v179 = 0;
  v180 = 0;
  v182 = 0x800000000LL;
  while ( 1 )
  {
    if ( !v17 )
      BUG();
    if ( *(_BYTE *)(v17 - 8) != 77 )
      break;
    v18 = v17 - 24;
    v19 = 0x17FFFFFFE8LL;
    v20 = *(_BYTE *)(v17 - 1) & 0x40;
    v21 = *(_DWORD *)(v17 - 4) & 0xFFFFFFF;
    if ( v21 )
    {
      v22 = 24LL * *(unsigned int *)(v17 + 32) + 8;
      v23 = 0;
      do
      {
        v24 = v18 - 24LL * v21;
        if ( v20 )
          v24 = *(_QWORD *)(v17 - 32);
        if ( v15 == *(_QWORD *)(v24 + v22) )
        {
          v19 = 24 * v23;
          goto LABEL_11;
        }
        ++v23;
        v22 += 8;
      }
      while ( v21 != (_DWORD)v23 );
      v19 = 0x17FFFFFFE8LL;
    }
LABEL_11:
    if ( v20 )
      v25 = *(_QWORD *)(v17 - 32);
    else
      v25 = v18 - 24LL * v21;
    v26 = *(_QWORD *)(v25 + v19);
    v27 = (unsigned int)v182;
    if ( (unsigned int)v182 >= HIDWORD(v182) )
    {
      v171 = v26;
      sub_16CD150((__int64)&v181, v183, 0, 8, v26, v25);
      v27 = (unsigned int)v182;
      v26 = v171;
    }
    *(_QWORD *)&v181[8 * v27] = v26;
    LODWORD(v182) = v182 + 1;
    v17 = *(_QWORD *)(v17 + 8);
  }
  v28 = v14 + 72;
  v184 = 0;
  v185 = (__int64 *)v189;
  v186 = (__int64 *)v189;
  v179 = (_QWORD *)(v17 - 24);
  v187 = 16;
  v188 = 0;
  v167 = a2 + 24;
  v161 = v14 + 72;
  if ( a2 + 24 != v14 + 72 )
  {
    v29 = a2 + 24;
    while ( 1 )
    {
      v32 = v29 - 24;
      if ( !v29 )
        v32 = 0;
      v33 = sub_157EBA0(v32);
      if ( *(_BYTE *)(v33 + 16) != 29 )
        goto LABEL_21;
      v34 = sub_15F6E60(v33);
      v35 = v185;
      if ( v186 != v185 )
        goto LABEL_20;
      v36 = &v185[HIDWORD(v187)];
      v30 = HIDWORD(v187);
      if ( v185 == v36 )
        goto LABEL_167;
      v37 = 0;
      do
      {
        if ( v34 == *v35 )
          goto LABEL_21;
        if ( *v35 == -2 )
          v37 = v35;
        ++v35;
      }
      while ( v36 != v35 );
      if ( !v37 )
      {
LABEL_167:
        if ( HIDWORD(v187) < (unsigned int)v187 )
        {
          v30 = (unsigned int)++HIDWORD(v187);
          *v36 = v34;
          ++v184;
          goto LABEL_21;
        }
LABEL_20:
        sub_16CCBA0((__int64)&v184, v34);
LABEL_21:
        v29 = *(_QWORD *)(v29 + 8);
        if ( v28 == v29 )
          goto LABEL_34;
      }
      else
      {
        *v37 = v34;
        --v188;
        v29 = *(_QWORD *)(v29 + 8);
        ++v184;
        if ( v28 == v29 )
        {
LABEL_34:
          v38 = v186;
          if ( v186 == v185 )
          {
            v39 = HIDWORD(v187);
            v168 = &v186[HIDWORD(v187)];
          }
          else
          {
            v39 = (unsigned int)v187;
            v168 = &v186[(unsigned int)v187];
          }
          v40 = (__int64)v168;
          if ( v186 != v168 )
          {
            while ( 1 )
            {
              v41 = *v38;
              v42 = v38;
              if ( (unsigned __int64)*v38 < 0xFFFFFFFFFFFFFFFELL )
                break;
              if ( v168 == ++v38 )
                goto LABEL_39;
            }
            if ( v168 != v38 )
            {
              v151 = v179;
              do
              {
                v152 = *((_DWORD *)v151 + 5) & 0xFFFFFFF;
                sub_15F5A20(v41, *((_DWORD *)v151 + 5) & 0xFFFFFFF, v40, v39, v30, v31);
                if ( (_DWORD)v152 )
                {
                  v153 = 0;
                  v154 = 24 * v152;
                  do
                  {
                    if ( (*((_BYTE *)v151 + 23) & 0x40) != 0 )
                    {
                      v155 = (char *)*(v151 - 1);
                    }
                    else
                    {
                      v39 = 24LL * (*((_DWORD *)v151 + 5) & 0xFFFFFFF);
                      v155 = (char *)v151 - v39;
                    }
                    v156 = *(_QWORD *)&v155[v153];
                    v153 += 24;
                    sub_15F5A60(v41, v156, v40, v39, v30, v31);
                  }
                  while ( v153 != v154 );
                }
                if ( (*((_BYTE *)v151 + 18) & 1) != 0 )
                  *(_WORD *)(v41 + 18) |= 1u;
                v157 = v42 + 1;
                if ( v42 + 1 == v168 )
                  break;
                v41 = *v157;
                for ( ++v42; (unsigned __int64)*v157 >= 0xFFFFFFFFFFFFFFFELL; v42 = v157 )
                {
                  if ( v168 == ++v157 )
                    goto LABEL_39;
                  v41 = *v157;
                }
              }
              while ( v42 != v168 );
            }
          }
          while ( 1 )
          {
LABEL_39:
            if ( !*a3 )
              goto LABEL_81;
            v43 = (__int64)v177;
            if ( !v167 )
              break;
            v44 = v167 - 24;
            v45 = v167 - 24;
            v49 = sub_1AD6F20(v167 - 24, (__int64)v177, 0);
            if ( v49 )
            {
              v50 = (unsigned int)v182;
              v51 = v177[6];
              if ( (_DWORD)v182 )
              {
LABEL_43:
                v52 = 8 * v50;
                v53 = 0;
                do
                {
                  if ( !v51 )
                    BUG();
                  v61 = v51 - 24;
                  v62 = *(_QWORD *)&v181[v53];
                  v63 = *(_DWORD *)(v51 - 4) & 0xFFFFFFF;
                  if ( v63 == *(_DWORD *)(v51 + 32) )
                  {
                    v170 = *(_QWORD *)&v181[v53];
                    sub_15F55D0(v51 - 24, v43, v46, v62, v47, v48);
                    v62 = v170;
                    v63 = *(_DWORD *)(v51 - 4) & 0xFFFFFFF;
                  }
                  v64 = (v63 + 1) & 0xFFFFFFF;
                  v43 = (unsigned int)(v64 - 1);
                  v65 = v64 | *(_DWORD *)(v51 - 4) & 0xF0000000;
                  *(_DWORD *)(v51 - 4) = v65;
                  if ( (v65 & 0x40000000) != 0 )
                    v54 = *(_QWORD *)(v51 - 32);
                  else
                    v54 = v61 - 24 * v64;
                  v55 = (__int64 *)(v54 + 24LL * (unsigned int)v43);
                  if ( *v55 )
                  {
                    v43 = v55[1];
                    v56 = v55[2] & 0xFFFFFFFFFFFFFFFCLL;
                    *(_QWORD *)v56 = v43;
                    if ( v43 )
                      *(_QWORD *)(v43 + 16) = *(_QWORD *)(v43 + 16) & 3LL | v56;
                  }
                  *v55 = v62;
                  if ( v62 )
                  {
                    v57 = *(_QWORD *)(v62 + 8);
                    v55[1] = v57;
                    if ( v57 )
                    {
                      v47 = (__int64)(v55 + 1);
                      v43 = (unsigned __int64)(v55 + 1) | *(_QWORD *)(v57 + 16) & 3LL;
                      *(_QWORD *)(v57 + 16) = v43;
                    }
                    v55[2] = (v62 + 8) | v55[2] & 3;
                    *(_QWORD *)(v62 + 8) = v55;
                  }
                  v58 = *(_DWORD *)(v51 - 4) & 0xFFFFFFF;
                  v59 = (unsigned int)(v58 - 1);
                  if ( (*(_BYTE *)(v51 - 1) & 0x40) != 0 )
                    v60 = *(_QWORD *)(v51 - 32);
                  else
                    v60 = v61 - 24 * v58;
                  v53 += 8;
                  v46 = 3LL * *(unsigned int *)(v51 + 32);
                  *(_QWORD *)(v60 + 8 * v59 + 24LL * *(unsigned int *)(v51 + 32) + 8) = v49;
                  v51 = *(_QWORD *)(v51 + 8);
                }
                while ( v52 != v53 );
LABEL_81:
                v44 = v167 - 24;
                if ( !v167 )
                  goto LABEL_146;
              }
              v45 = v44;
            }
LABEL_83:
            v169 = sub_157EBA0(v45);
            if ( *(_BYTE *)(v169 + 16) == 30 )
            {
              v78 = v178;
              if ( !v178 )
              {
                v100 = v177;
                v101 = (__int64 *)v179[4];
                v172 = sub_1649960((__int64)v177);
                v174 = &v172;
                v173 = v102;
                v175 = ".body";
                v176 = 773;
                v103 = sub_157FBF0(v100, v101, (__int64)&v174);
                v104 = *(_QWORD *)(v103 + 48);
                v178 = v103;
                v105 = v104 - 24;
                if ( !v104 )
                  v105 = 0;
                v106 = 0;
                v163 = v105;
                v107 = v177[6];
                v162 = v182;
                if ( (_DWORD)v182 )
                {
                  do
                  {
                    if ( !v107 )
                    {
                      v11 = sub_1649960(0);
                      v176 = 773;
                      v172 = v11;
                      v173 = v12;
                      v174 = &v172;
                      v175 = ".lpad-body";
                      BUG();
                    }
                    v172 = sub_1649960(v107 - 24);
                    v176 = 773;
                    v173 = v115;
                    v174 = &v172;
                    v175 = ".lpad-body";
                    v165 = *(_QWORD *)(v107 - 24);
                    v116 = sub_1648B60(64);
                    v119 = v116;
                    if ( v116 )
                    {
                      v120 = v116;
                      sub_15F1EA0(v116, v165, 53, 0, 0, v163);
                      *(_DWORD *)(v119 + 56) = 2;
                      sub_164B780(v119, (__int64 *)&v174);
                      sub_1648880(v119, *(_DWORD *)(v119 + 56), 1);
                    }
                    else
                    {
                      v120 = 0;
                    }
                    sub_164D160(v107 - 24, v119, a4, a5, a6, a7, v117, v118, a10, a11);
                    v124 = (__int64)v177;
                    v125 = *(_DWORD *)(v119 + 20) & 0xFFFFFFF;
                    if ( (_DWORD)v125 == *(_DWORD *)(v119 + 56) )
                    {
                      v166 = (__int64)v177;
                      sub_15F55D0(v119, (__int64)v177, v125, v121, v122, v123);
                      v124 = v166;
                      LODWORD(v125) = *(_DWORD *)(v119 + 20) & 0xFFFFFFF;
                    }
                    v126 = ((_DWORD)v125 + 1) & 0xFFFFFFF;
                    v127 = v126 | *(_DWORD *)(v119 + 20) & 0xF0000000;
                    *(_DWORD *)(v119 + 20) = v127;
                    if ( (v127 & 0x40000000) != 0 )
                      v108 = *(_QWORD *)(v119 - 8);
                    else
                      v108 = v120 - 24 * v126;
                    v109 = (_QWORD *)(v108 + 24LL * (unsigned int)(v126 - 1));
                    if ( *v109 )
                    {
                      v110 = v109[1];
                      v111 = v109[2] & 0xFFFFFFFFFFFFFFFCLL;
                      *(_QWORD *)v111 = v110;
                      if ( v110 )
                        *(_QWORD *)(v110 + 16) = *(_QWORD *)(v110 + 16) & 3LL | v111;
                    }
                    *v109 = v107 - 24;
                    v112 = *(_QWORD *)(v107 - 16);
                    v109[1] = v112;
                    if ( v112 )
                      *(_QWORD *)(v112 + 16) = (unsigned __int64)(v109 + 1) | *(_QWORD *)(v112 + 16) & 3LL;
                    v109[2] = (v107 - 16) | v109[2] & 3LL;
                    *(_QWORD *)(v107 - 16) = v109;
                    v113 = *(_DWORD *)(v119 + 20) & 0xFFFFFFF;
                    if ( (*(_BYTE *)(v119 + 23) & 0x40) != 0 )
                      v114 = *(_QWORD *)(v119 - 8);
                    else
                      v114 = v120 - 24 * v113;
                    ++v106;
                    *(_QWORD *)(v114 + 8LL * (unsigned int)(v113 - 1) + 24LL * *(unsigned int *)(v119 + 56) + 8) = v124;
                    v107 = *(_QWORD *)(v107 + 8);
                  }
                  while ( v162 != v106 );
                }
                v174 = (const char **)"eh.lpad-body";
                v176 = 259;
                v128 = *v179;
                v129 = sub_1648B60(64);
                v132 = v129;
                if ( v129 )
                {
                  sub_15F1EA0(v129, v128, 53, 0, 0, v163);
                  *(_DWORD *)(v132 + 56) = 2;
                  sub_164B780(v132, (__int64 *)&v174);
                  sub_1648880(v132, *(_DWORD *)(v132 + 56), 1);
                }
                v133 = v132;
                v180 = v132;
                sub_164D160((__int64)v179, v132, a4, a5, a6, a7, v130, v131, a10, a11);
                v138 = v180;
                v139 = (__int64)v177;
                v140 = v179;
                v141 = *(_DWORD *)(v180 + 20) & 0xFFFFFFF;
                if ( v141 == *(_DWORD *)(v180 + 56) )
                {
                  sub_15F55D0(v180, v133, v134, v135, v136, v137);
                  v141 = *(_DWORD *)(v138 + 20) & 0xFFFFFFF;
                }
                v142 = (v141 + 1) & 0xFFFFFFF;
                v143 = v142 | *(_DWORD *)(v138 + 20) & 0xF0000000;
                *(_DWORD *)(v138 + 20) = v143;
                if ( (v143 & 0x40000000) != 0 )
                  v144 = *(_QWORD *)(v138 - 8);
                else
                  v144 = v138 - 24 * v142;
                v145 = (_QWORD *)(v144 + 24LL * (unsigned int)(v142 - 1));
                if ( *v145 )
                {
                  v146 = v145[1];
                  v147 = v145[2] & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v147 = v146;
                  if ( v146 )
                    *(_QWORD *)(v146 + 16) = *(_QWORD *)(v146 + 16) & 3LL | v147;
                }
                *v145 = v140;
                if ( v140 )
                {
                  v148 = v140[1];
                  v145[1] = v148;
                  if ( v148 )
                    *(_QWORD *)(v148 + 16) = (unsigned __int64)(v145 + 1) | *(_QWORD *)(v148 + 16) & 3LL;
                  v145[2] = (unsigned __int64)(v140 + 1) | v145[2] & 3LL;
                  v140[1] = v145;
                }
                v149 = *(_DWORD *)(v138 + 20) & 0xFFFFFFF;
                if ( (*(_BYTE *)(v138 + 23) & 0x40) != 0 )
                  v150 = *(_QWORD *)(v138 - 8);
                else
                  v150 = v138 - 24 * v149;
                *(_QWORD *)(v150 + 8LL * (unsigned int)(v149 - 1) + 24LL * *(unsigned int *)(v138 + 56) + 8) = v139;
                v78 = v178;
              }
              v79 = 1;
              v80 = *(_QWORD *)(v169 + 40);
              v83 = sub_1648A60(56, 1u);
              if ( v83 )
              {
                v79 = v78;
                sub_15F8590((__int64)v83, v78, v80);
              }
              v85 = *(_QWORD *)(v78 + 48);
              v86 = 0;
              v87 = 8LL * (unsigned int)v182;
              v88 = v87;
              if ( (_DWORD)v182 )
              {
                do
                {
                  if ( !v85 )
                    BUG();
                  v90 = v85 - 24;
                  v82 = *(_QWORD *)&v181[v86];
                  v91 = *(_DWORD *)(v85 - 4) & 0xFFFFFFF;
                  if ( v91 == *(_DWORD *)(v85 + 32) )
                  {
                    v164 = *(_QWORD *)&v181[v86];
                    sub_15F55D0(v85 - 24, v79, v81, v82, v87, v84);
                    v82 = v164;
                    v91 = *(_DWORD *)(v85 - 4) & 0xFFFFFFF;
                  }
                  v92 = (v91 + 1) & 0xFFFFFFF;
                  v79 = (unsigned int)(v92 - 1);
                  v93 = v92 | *(_DWORD *)(v85 - 4) & 0xF0000000;
                  *(_DWORD *)(v85 - 4) = v93;
                  if ( (v93 & 0x40000000) != 0 )
                    v94 = *(_QWORD *)(v85 - 32);
                  else
                    v94 = v90 - 24 * v92;
                  v95 = (__int64 *)(v94 + 24LL * (unsigned int)v79);
                  if ( *v95 )
                  {
                    v79 = v95[1];
                    v96 = v95[2] & 0xFFFFFFFFFFFFFFFCLL;
                    *(_QWORD *)v96 = v79;
                    if ( v79 )
                      *(_QWORD *)(v79 + 16) = *(_QWORD *)(v79 + 16) & 3LL | v96;
                  }
                  *v95 = v82;
                  if ( v82 )
                  {
                    v97 = *(_QWORD *)(v82 + 8);
                    v95[1] = v97;
                    if ( v97 )
                    {
                      v87 = (__int64)(v95 + 1);
                      v79 = (unsigned __int64)(v95 + 1) | *(_QWORD *)(v97 + 16) & 3LL;
                      *(_QWORD *)(v97 + 16) = v79;
                    }
                    v95[2] = (v82 + 8) | v95[2] & 3;
                    *(_QWORD *)(v82 + 8) = v95;
                  }
                  v98 = *(_DWORD *)(v85 - 4) & 0xFFFFFFF;
                  v99 = (unsigned int)(v98 - 1);
                  if ( (*(_BYTE *)(v85 - 1) & 0x40) != 0 )
                    v89 = *(_QWORD *)(v85 - 32);
                  else
                    v89 = v90 - 24 * v98;
                  v86 += 8;
                  v81 = 3LL * *(unsigned int *)(v85 + 32);
                  *(_QWORD *)(v89 + 8 * v99 + 24LL * *(unsigned int *)(v85 + 32) + 8) = v80;
                  v85 = *(_QWORD *)(v85 + 8);
                }
                while ( v88 != v86 );
              }
              v66 = v180;
              v67 = *(_QWORD *)(v169 - 24);
              v68 = *(_DWORD *)(v180 + 20) & 0xFFFFFFF;
              if ( v68 == *(_DWORD *)(v180 + 56) )
              {
                sub_15F55D0(v180, v79, v81, v82, v87, v84);
                v68 = *(_DWORD *)(v66 + 20) & 0xFFFFFFF;
              }
              v69 = (v68 + 1) & 0xFFFFFFF;
              v70 = v69 | *(_DWORD *)(v66 + 20) & 0xF0000000;
              *(_DWORD *)(v66 + 20) = v70;
              if ( (v70 & 0x40000000) != 0 )
                v71 = *(_QWORD *)(v66 - 8);
              else
                v71 = v66 - 24 * v69;
              v72 = (_QWORD *)(v71 + 24LL * (unsigned int)(v69 - 1));
              if ( *v72 )
              {
                v73 = v72[1];
                v74 = v72[2] & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v74 = v73;
                if ( v73 )
                  *(_QWORD *)(v73 + 16) = *(_QWORD *)(v73 + 16) & 3LL | v74;
              }
              *v72 = v67;
              if ( v67 )
              {
                v75 = *(_QWORD *)(v67 + 8);
                v72[1] = v75;
                if ( v75 )
                  *(_QWORD *)(v75 + 16) = (unsigned __int64)(v72 + 1) | *(_QWORD *)(v75 + 16) & 3LL;
                v72[2] = (v67 + 8) | v72[2] & 3LL;
                *(_QWORD *)(v67 + 8) = v72;
              }
              v76 = *(_DWORD *)(v66 + 20) & 0xFFFFFFF;
              if ( (*(_BYTE *)(v66 + 23) & 0x40) != 0 )
                v77 = *(_QWORD *)(v66 - 8);
              else
                v77 = v66 - 24 * v76;
              *(_QWORD *)(v77 + 8LL * (unsigned int)(v76 - 1) + 24LL * *(unsigned int *)(v66 + 56) + 8) = v80;
              sub_15F20C0((_QWORD *)v169);
            }
            v167 = *(_QWORD *)(v167 + 8);
            if ( v161 == v167 )
              goto LABEL_75;
          }
          v49 = sub_1AD6F20(0, (__int64)v177, 0);
          if ( !v49 || (v50 = (unsigned int)v182, v51 = v177[6], !(_DWORD)v182) )
          {
LABEL_146:
            v45 = 0;
            goto LABEL_83;
          }
          goto LABEL_43;
        }
      }
    }
  }
LABEL_75:
  sub_157F2D0(v159, *(_QWORD *)(a1 + 40), 0);
  if ( v186 != v185 )
    _libc_free((unsigned __int64)v186);
  if ( v181 != v183 )
    _libc_free((unsigned __int64)v181);
}
