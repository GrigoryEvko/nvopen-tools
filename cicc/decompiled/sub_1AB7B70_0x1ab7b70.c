// Function: sub_1AB7B70
// Address: 0x1ab7b70
//
void __fastcall sub_1AB7B70(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int8 a5,
        __int64 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        __int64 a15,
        __int64 a16)
{
  __int64 v16; // r15
  __int64 v18; // rdx
  __int64 v19; // rsi
  __int64 v20; // rbx
  unsigned __int64 *v21; // r13
  unsigned __int64 v22; // r12
  __int64 v23; // r12
  __int64 v24; // r14
  __int64 v25; // rcx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r12
  int v29; // r8d
  int v30; // r9d
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // r12
  __int64 v34; // rdx
  __int64 v35; // r15
  __int64 v36; // rbx
  __int64 v37; // r14
  int v38; // ebx
  __int64 v39; // rax
  unsigned __int64 *v40; // rax
  __int64 v41; // r13
  __int64 v42; // rbx
  __int64 v43; // rdx
  __int64 v44; // rcx
  _QWORD *v45; // r8
  _QWORD *v46; // r9
  __int64 v47; // rax
  int v48; // r14d
  __int64 v49; // rax
  __int64 v50; // r8
  __int64 v51; // r12
  unsigned __int8 v52; // al
  unsigned __int64 v53; // rdx
  double v54; // xmm4_8
  double v55; // xmm5_8
  __int64 k; // r15
  __int64 v57; // rdx
  __int64 v58; // rcx
  _QWORD *v59; // r8
  _QWORD *v60; // r9
  __int64 v61; // r12
  __int64 v62; // rbx
  __int64 v63; // r14
  double v64; // xmm4_8
  double v65; // xmm5_8
  __int64 *v66; // r13
  unsigned __int64 v67; // rdx
  __int64 v68; // rcx
  double v69; // xmm4_8
  double v70; // xmm5_8
  unsigned __int64 v71; // rax
  __int64 m; // r13
  __int64 v73; // rdi
  int v74; // r8d
  int v75; // r9d
  unsigned __int64 v76; // r14
  __int64 v77; // rax
  __int64 v78; // r14
  __int64 v79; // rdi
  __int64 v80; // rax
  __int64 v81; // rsi
  unsigned int v82; // r14d
  _QWORD *v83; // rax
  __int64 v84; // r10
  __int64 v85; // rbx
  __int64 v86; // rdx
  __int64 v87; // rax
  __int64 v88; // rsi
  __int64 v89; // rdi
  unsigned int v90; // ecx
  _QWORD *v91; // rdx
  __int64 v92; // r8
  __int64 v93; // r15
  __int64 v94; // rcx
  __int64 v95; // r12
  __int64 v96; // rdx
  __int64 v97; // rdx
  __int64 *v98; // r12
  __int64 v99; // rcx
  unsigned __int64 v100; // rdx
  __int64 v101; // rdx
  __int64 v102; // rdx
  int v103; // edx
  int v104; // r9d
  __int64 v105; // rbx
  _QWORD *v106; // rdx
  __int64 v107; // r12
  __int64 *v108; // rax
  unsigned __int64 v109; // rdi
  __int64 *v110; // r8
  __int64 v111; // rcx
  __int64 v112; // rdx
  __int64 v113; // r14
  __int64 v114; // r15
  __int64 v115; // rbx
  __int64 **v116; // rdi
  _QWORD *v117; // r14
  __int64 v118; // r13
  double v119; // xmm4_8
  double v120; // xmm5_8
  __int64 v121; // rsi
  _QWORD *v122; // rdi
  __int64 v123; // rax
  unsigned int v124; // eax
  unsigned __int64 *v125; // r12
  __int64 v126; // r13
  __int64 v127; // r15
  __int64 v128; // r14
  __int64 v129; // rdx
  __int64 *v130; // rsi
  unsigned __int64 v131; // r10
  __int64 *v132; // rax
  __int64 v133; // rcx
  __int64 v134; // rdx
  __int64 i; // r13
  __int64 v136; // r15
  __int64 v137; // r12
  __int64 v138; // r14
  int v139; // ebx
  __int64 j; // r13
  unsigned int v141; // ecx
  unsigned int v142; // esi
  __int64 v143; // rax
  __int64 v144; // rdx
  unsigned __int64 v145; // rax
  __int64 *v146; // rcx
  __int64 *v147; // rax
  __int64 v148; // rax
  _QWORD *v149; // rdi
  __int64 v150; // rax
  __int64 v151; // rax
  __int64 v153; // [rsp+10h] [rbp-260h]
  int v154; // [rsp+1Ch] [rbp-254h]
  __int64 v156; // [rsp+30h] [rbp-240h]
  __int64 v157; // [rsp+40h] [rbp-230h]
  int v158; // [rsp+48h] [rbp-228h]
  int v159; // [rsp+4Ch] [rbp-224h]
  unsigned __int8 v160; // [rsp+50h] [rbp-220h]
  __int64 v161; // [rsp+58h] [rbp-218h]
  int v162; // [rsp+58h] [rbp-218h]
  __int64 v164; // [rsp+60h] [rbp-210h]
  __int64 v165; // [rsp+60h] [rbp-210h]
  unsigned __int64 *v166; // [rsp+60h] [rbp-210h]
  __int64 v167; // [rsp+68h] [rbp-208h]
  __int64 v168; // [rsp+68h] [rbp-208h]
  __int64 v169; // [rsp+68h] [rbp-208h]
  __int64 v170; // [rsp+68h] [rbp-208h]
  __int64 v171; // [rsp+68h] [rbp-208h]
  __int64 v172; // [rsp+70h] [rbp-200h]
  __int64 v173; // [rsp+70h] [rbp-200h]
  __int64 v174; // [rsp+70h] [rbp-200h]
  _QWORD *v175; // [rsp+70h] [rbp-200h]
  __int64 v176; // [rsp+70h] [rbp-200h]
  __int64 v177; // [rsp+70h] [rbp-200h]
  __int64 v178; // [rsp+78h] [rbp-1F8h]
  __int64 v179; // [rsp+78h] [rbp-1F8h]
  bool v180; // [rsp+78h] [rbp-1F8h]
  __int64 v181; // [rsp+78h] [rbp-1F8h]
  __int64 v182; // [rsp+78h] [rbp-1F8h]
  __int64 v183; // [rsp+78h] [rbp-1F8h]
  unsigned __int64 *v184; // [rsp+78h] [rbp-1F8h]
  unsigned __int64 v185; // [rsp+88h] [rbp-1E8h] BYREF
  __int64 v186; // [rsp+90h] [rbp-1E0h] BYREF
  __int64 v187; // [rsp+98h] [rbp-1D8h]
  __int64 v188; // [rsp+A0h] [rbp-1D0h]
  __m128i v189; // [rsp+B0h] [rbp-1C0h] BYREF
  __int64 v190; // [rsp+C0h] [rbp-1B0h]
  __int64 v191; // [rsp+C8h] [rbp-1A8h]
  __int64 v192; // [rsp+D0h] [rbp-1A0h]
  _QWORD v193[3]; // [rsp+E0h] [rbp-190h] BYREF
  unsigned __int8 v194; // [rsp+F8h] [rbp-178h]
  __int64 v195; // [rsp+100h] [rbp-170h]
  __int64 v196; // [rsp+108h] [rbp-168h]
  _BYTE *v197; // [rsp+110h] [rbp-160h] BYREF
  __int64 v198; // [rsp+118h] [rbp-158h]
  _BYTE v199[128]; // [rsp+120h] [rbp-150h] BYREF
  unsigned __int64 v200; // [rsp+1A0h] [rbp-D0h] BYREF
  __int64 v201; // [rsp+1A8h] [rbp-C8h] BYREF
  __int64 *v202; // [rsp+1B0h] [rbp-C0h] BYREF
  __int64 *v203; // [rsp+1B8h] [rbp-B8h]
  __int64 *v204; // [rsp+1C0h] [rbp-B0h]
  __int64 v205; // [rsp+1C8h] [rbp-A8h]
  _BYTE *v206; // [rsp+1F0h] [rbp-80h] BYREF
  __int64 v207; // [rsp+1F8h] [rbp-78h]
  _BYTE v208[112]; // [rsp+200h] [rbp-70h] BYREF

  v16 = a4;
  v195 = a15;
  v193[0] = a1;
  v193[1] = a2;
  v193[2] = a4;
  v194 = a5;
  v196 = a16;
  if ( a3 )
  {
    v153 = *(_QWORD *)(a3 + 40);
  }
  else
  {
    v151 = *(_QWORD *)(a2 + 80);
    if ( !v151 )
      BUG();
    a3 = *(_QWORD *)(v151 + 24);
    v153 = v151 - 24;
    if ( a3 )
      a3 -= 24;
  }
  v18 = a3 + 24;
  v186 = 0;
  v19 = v153;
  v187 = 0;
  v188 = 0;
  while ( 1 )
  {
    sub_1AB45A0((__int64)v193, v19, v18, (__int64)&v186, a7, a8, a9, a10, a11, a12, a13, a14);
    if ( v186 == v187 )
      break;
    v19 = *(_QWORD *)(v187 - 8);
    v187 -= 8;
    v18 = *(_QWORD *)(v19 + 48);
  }
  v20 = *(_QWORD *)(a2 + 80);
  v21 = &v200;
  v197 = v199;
  v198 = 0x1000000000LL;
  v178 = a1 + 72;
  v167 = a2 + 72;
  if ( v20 != a2 + 72 )
  {
    do
    {
      v23 = v20 - 24;
      if ( !v20 )
        v23 = 0;
      v189.m128i_i64[0] = v23;
      sub_1A51850(&v200, v16, v189.m128i_i64);
      v24 = (__int64)v202;
      if ( v202 )
      {
        if ( v202 != (__int64 *)-16LL && v202 != (__int64 *)-8LL )
          sub_1649B30(&v200);
        sub_15E01D0(v178, v24);
        v25 = *(_QWORD *)(a1 + 72);
        *(_QWORD *)(v24 + 32) = v178;
        v25 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v24 + 24) = v25 | *(_QWORD *)(v24 + 24) & 7LL;
        *(_QWORD *)(v25 + 8) = v24 + 24;
        *(_QWORD *)(a1 + 72) = *(_QWORD *)(a1 + 72) & 7LL | (v24 + 24);
        v26 = sub_157F280(v23);
        v172 = v27;
        v28 = v26;
        while ( v172 != v28 )
        {
          if ( *(_BYTE *)(sub_1AB4240(v16, v28)[2] + 16LL) != 77 )
            break;
          v31 = (unsigned int)v198;
          if ( (unsigned int)v198 >= HIDWORD(v198) )
          {
            sub_16CD150((__int64)&v197, v199, 0, 8, v29, v30);
            v31 = (unsigned int)v198;
          }
          *(_QWORD *)&v197[8 * v31] = v28;
          LODWORD(v198) = v198 + 1;
          if ( !v28 )
            BUG();
          v32 = *(_QWORD *)(v28 + 32);
          if ( !v32 )
            BUG();
          v28 = 0;
          if ( *(_BYTE *)(v32 - 8) == 77 )
            v28 = v32 - 24;
        }
        v22 = sub_157EBA0(v24);
        sub_1B75040(&v200, v16, a5 ^ 1u, 0, 0);
        sub_1B79630(&v200, v22);
        sub_1B75110(&v200);
      }
      v20 = *(_QWORD *)(v20 + 8);
    }
    while ( v20 != v167 );
    v154 = v198;
    if ( (_DWORD)v198 )
    {
      v33 = v16;
      v159 = 0;
      v173 = 0;
      v160 = a5 ^ 1;
      while ( 1 )
      {
        v80 = *(_QWORD *)&v197[8 * v173];
        v157 = *(_QWORD *)(v80 + 40);
        v158 = *(_DWORD *)(v80 + 20) & 0xFFFFFFF;
        v156 = sub_1AB4240(v33, v157)[2];
        if ( v159 != (_DWORD)v198 )
        {
          v34 = v173;
          do
          {
            v81 = *(_QWORD *)&v197[8 * v34];
            if ( *(_QWORD *)(v81 + 40) != v157 )
              break;
            v82 = 0;
            v83 = sub_1AB4240(v33, v81);
            v84 = v33;
            v85 = v83[2];
            v162 = v158;
            if ( v158 )
            {
              while ( 1 )
              {
                if ( (*(_BYTE *)(v85 + 23) & 0x40) != 0 )
                  v86 = *(_QWORD *)(v85 - 8);
                else
                  v86 = v85 - 24LL * (*(_DWORD *)(v85 + 20) & 0xFFFFFFF);
                v87 = *(unsigned int *)(v84 + 24);
                v182 = 8LL * v82;
                if ( (_DWORD)v87 )
                {
                  v88 = *(_QWORD *)(v84 + 8);
                  v89 = *(_QWORD *)(8LL * v82 + v86 + 24LL * *(unsigned int *)(v85 + 56) + 8);
                  v90 = (v87 - 1) & (((unsigned int)v89 >> 9) ^ ((unsigned int)v89 >> 4));
                  v91 = (_QWORD *)(v88 + ((unsigned __int64)v90 << 6));
                  v92 = v91[3];
                  if ( v89 == v92 )
                  {
LABEL_117:
                    if ( v91 != (_QWORD *)(v88 + (v87 << 6)) )
                    {
                      v200 = 6;
                      v201 = 0;
                      v202 = (__int64 *)v91[7];
                      v93 = (__int64)v202;
                      if ( v202 != 0 && v202 + 1 != 0 && v202 != (__int64 *)-16LL )
                      {
                        v176 = v84;
                        sub_1649AC0(v21, v91[5] & 0xFFFFFFFFFFFFFFF8LL);
                        v93 = (__int64)v202;
                        v84 = v176;
                      }
                      if ( v93 )
                      {
                        if ( v93 != -16 && v93 != -8 )
                        {
                          v177 = v84;
                          sub_1649B30(v21);
                          v84 = v177;
                        }
                        if ( (*(_BYTE *)(v85 + 23) & 0x40) != 0 )
                          v94 = *(_QWORD *)(v85 - 8);
                        else
                          v94 = v85 - 24LL * (*(_DWORD *)(v85 + 20) & 0xFFFFFFF);
                        v165 = v84;
                        v95 = *(_QWORD *)(v94 + 24LL * v82);
                        sub_1B75040(v21, v84, v160, 0, 0);
                        v168 = sub_1B79660(v21, v95, v96);
                        sub_1B75110(v21);
                        v84 = v165;
                        if ( (*(_BYTE *)(v85 + 23) & 0x40) != 0 )
                          v97 = *(_QWORD *)(v85 - 8);
                        else
                          v97 = v85 - 24LL * (*(_DWORD *)(v85 + 20) & 0xFFFFFFF);
                        v98 = (__int64 *)(v97 + 24LL * v82);
                        if ( *v98 )
                        {
                          v99 = v98[1];
                          v100 = v98[2] & 0xFFFFFFFFFFFFFFFCLL;
                          *(_QWORD *)v100 = v99;
                          if ( v99 )
                            *(_QWORD *)(v99 + 16) = *(_QWORD *)(v99 + 16) & 3LL | v100;
                        }
                        *v98 = v168;
                        if ( v168 )
                        {
                          v101 = *(_QWORD *)(v168 + 8);
                          v98[1] = v101;
                          if ( v101 )
                            *(_QWORD *)(v101 + 16) = (unsigned __int64)(v98 + 1) | *(_QWORD *)(v101 + 16) & 3LL;
                          v98[2] = (v168 + 8) | v98[2] & 3;
                          *(_QWORD *)(v168 + 8) = v98;
                        }
                        if ( (*(_BYTE *)(v85 + 23) & 0x40) != 0 )
                          v102 = *(_QWORD *)(v85 - 8);
                        else
                          v102 = v85 - 24LL * (*(_DWORD *)(v85 + 20) & 0xFFFFFFF);
                        ++v82;
                        *(_QWORD *)(v182 + v102 + 24LL * *(unsigned int *)(v85 + 56) + 8) = v93;
                        goto LABEL_113;
                      }
                    }
                  }
                  else
                  {
                    v103 = 1;
                    while ( v92 != -8 )
                    {
                      v104 = v103 + 1;
                      v90 = (v87 - 1) & (v103 + v90);
                      v91 = (_QWORD *)(v88 + ((unsigned __int64)v90 << 6));
                      v92 = v91[3];
                      if ( v89 == v92 )
                        goto LABEL_117;
                      v103 = v104;
                    }
                  }
                }
                v183 = v84;
                sub_15F5350(v85, v82, 0);
                --v162;
                v84 = v183;
LABEL_113:
                if ( v162 == v82 )
                {
                  v33 = v84;
                  break;
                }
              }
            }
            v34 = (unsigned int)++v159;
          }
          while ( (_DWORD)v198 != v159 );
          v173 = v34;
        }
        v35 = 0;
        v36 = *(_QWORD *)(v156 + 48);
        v37 = *(_QWORD *)(v156 + 8);
        v179 = v36;
        if ( v36 )
          v35 = v36 - 24;
        if ( v37 )
          break;
LABEL_147:
        if ( (*(_DWORD *)(v35 + 20) & 0xFFFFFFF) != 0 )
          goto LABEL_148;
LABEL_38:
        if ( !v179 )
          BUG();
        if ( (*(_DWORD *)(v179 - 4) & 0xFFFFFFF) == 0 )
        {
          v113 = v179;
          if ( *(_BYTE *)(v179 - 8) == 77 )
          {
            v114 = *(_QWORD *)(v179 + 8);
            v184 = v21;
            v115 = *(_QWORD *)(v157 + 48);
            while ( 1 )
            {
              v116 = *(__int64 ***)(v113 - 24);
              v117 = (_QWORD *)(v113 - 24);
              v118 = sub_1599EF0(v116);
              sub_164D160((__int64)v117, v118, a7, a8, a9, a10, v119, v120, a13, a14);
              v121 = v115 - 24;
              if ( !v115 )
                v121 = 0;
              v122 = sub_1AB4240(v33, v121);
              v123 = v122[2];
              if ( v118 != v123 )
              {
                if ( v123 != 0 && v123 != -8 && v123 != -16 )
                  sub_1649B30(v122);
                v122[2] = v118;
                if ( v118 != 0 && v118 != -8 && v118 != -16 )
                  sub_164C220((__int64)v122);
              }
              sub_15F20C0(v117);
              if ( *(_BYTE *)(v114 - 8) != 77 )
                break;
              v113 = v114;
              v115 = *(_QWORD *)(v115 + 8);
              v114 = *(_QWORD *)(v114 + 8);
            }
            v21 = v184;
          }
        }
        if ( v154 == v159 )
        {
          v16 = v33;
          goto LABEL_42;
        }
      }
      while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v37) + 16) - 25) > 9u )
      {
        v37 = *(_QWORD *)(v37 + 8);
        if ( !v37 )
          goto LABEL_147;
      }
      v38 = 0;
      while ( 1 )
      {
        v37 = *(_QWORD *)(v37 + 8);
        if ( !v37 )
          break;
        while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v37) + 16) - 25) <= 9u )
        {
          v37 = *(_QWORD *)(v37 + 8);
          ++v38;
          if ( !v37 )
            goto LABEL_37;
        }
      }
LABEL_37:
      if ( (*(_DWORD *)(v35 + 20) & 0xFFFFFFF) == v38 + 1 )
        goto LABEL_38;
LABEL_148:
      LODWORD(v201) = 0;
      v203 = &v201;
      v204 = &v201;
      v202 = 0;
      v205 = 0;
      v105 = *(_QWORD *)(v156 + 8);
      if ( v105 )
      {
        while ( 1 )
        {
          v106 = sub_1648700(v105);
          if ( (unsigned __int8)(*((_BYTE *)v106 + 16) - 25) <= 9u )
            break;
          v105 = *(_QWORD *)(v105 + 8);
          if ( !v105 )
            goto LABEL_177;
        }
        v169 = v33;
        v107 = v105;
        v108 = 0;
LABEL_151:
        v109 = v106[5];
        v110 = &v201;
        v185 = v109;
        if ( !v108 )
          goto LABEL_158;
        do
        {
          while ( 1 )
          {
            v111 = v108[2];
            v112 = v108[3];
            if ( v108[4] >= v109 )
              break;
            v108 = (__int64 *)v108[3];
            if ( !v112 )
              goto LABEL_156;
          }
          v110 = v108;
          v108 = (__int64 *)v108[2];
        }
        while ( v111 );
LABEL_156:
        if ( v110 == &v201 || v110[4] > v109 )
        {
LABEL_158:
          v189.m128i_i64[0] = (__int64)&v185;
          v110 = (__int64 *)sub_1AB3F50(v21, v110, (unsigned __int64 **)&v189);
        }
        --*((_DWORD *)v110 + 10);
        while ( 1 )
        {
          v107 = *(_QWORD *)(v107 + 8);
          if ( !v107 )
            break;
          v106 = sub_1648700(v107);
          if ( (unsigned __int8)(*((_BYTE *)v106 + 16) - 25) <= 9u )
          {
            v108 = v202;
            goto LABEL_151;
          }
        }
        v33 = v169;
      }
LABEL_177:
      v124 = *(_DWORD *)(v35 + 20) & 0xFFFFFFF;
      if ( v124 )
      {
        v170 = v33;
        v125 = v21;
        v126 = v35;
        v127 = 0;
        v128 = 8LL * v124;
        do
        {
          if ( (*(_BYTE *)(v126 + 23) & 0x40) != 0 )
            v129 = *(_QWORD *)(v126 - 8);
          else
            v129 = v126 - 24LL * (*(_DWORD *)(v126 + 20) & 0xFFFFFFF);
          v130 = &v201;
          v131 = *(_QWORD *)(v127 + v129 + 24LL * *(unsigned int *)(v126 + 56) + 8);
          v132 = v202;
          v185 = v131;
          if ( !v202 )
            goto LABEL_188;
          do
          {
            while ( 1 )
            {
              v133 = v132[2];
              v134 = v132[3];
              if ( v132[4] >= v131 )
                break;
              v132 = (__int64 *)v132[3];
              if ( !v134 )
                goto LABEL_186;
            }
            v130 = v132;
            v132 = (__int64 *)v132[2];
          }
          while ( v133 );
LABEL_186:
          if ( v130 == &v201 || v130[4] > v131 )
          {
LABEL_188:
            v189.m128i_i64[0] = (__int64)&v185;
            v130 = (__int64 *)sub_1AB3F50(v125, v130, (unsigned __int64 **)&v189);
          }
          v127 += 8;
          ++*((_DWORD *)v130 + 10);
        }
        while ( v127 != v128 );
        v21 = v125;
        v33 = v170;
      }
      v171 = v33;
      v166 = v21;
      for ( i = *(_QWORD *)(v156 + 48); ; i = *(_QWORD *)(i + 8) )
      {
        if ( !i )
          BUG();
        if ( *(_BYTE *)(i - 8) != 77 )
          break;
        v136 = (__int64)v203;
        v137 = i - 24;
        v138 = i;
        if ( v203 != &v201 )
        {
          do
          {
            v139 = *(_DWORD *)(v136 + 40);
            for ( j = *(_QWORD *)(v136 + 32); v139; --v139 )
            {
              while ( 1 )
              {
                v141 = *(_DWORD *)(v138 - 4) & 0xFFFFFFF;
                if ( v141 )
                  break;
LABEL_206:
                sub_15F5350(v137, 0xFFFFFFFF, 0);
                if ( !--v139 )
                  goto LABEL_203;
              }
              v142 = 0;
              v143 = 24LL * *(unsigned int *)(v138 + 32) + 8;
              while ( 1 )
              {
                v144 = v137 - 24LL * v141;
                if ( (*(_BYTE *)(v138 - 1) & 0x40) != 0 )
                  v144 = *(_QWORD *)(v138 - 32);
                if ( j == *(_QWORD *)(v144 + v143) )
                  break;
                ++v142;
                v143 += 8;
                if ( v141 == v142 )
                  goto LABEL_206;
              }
              sub_15F5350(v137, v142, 0);
            }
LABEL_203:
            v136 = sub_220EEE0(v136);
          }
          while ( (__int64 *)v136 != &v201 );
          i = v138;
        }
      }
      v33 = v171;
      v21 = v166;
      sub_1AB3C40((__int64)v202);
      v179 = *(_QWORD *)(v156 + 48);
      goto LABEL_38;
    }
  }
LABEL_42:
  v39 = sub_1632FA0(*(_QWORD *)(a1 + 40));
  v200 = 0;
  v201 = 1;
  v164 = v39;
  v40 = (unsigned __int64 *)&v202;
  do
    *v40++ = -8;
  while ( v40 != (unsigned __int64 *)&v206 );
  v41 = 0;
  v206 = v208;
  v207 = 0x800000000LL;
  v42 = 8LL * (unsigned int)v198;
  if ( (_DWORD)v198 )
  {
    do
    {
      while ( *(_BYTE *)(sub_1AB4240(v16, *(_QWORD *)&v197[v41])[2] + 16LL) != 77 )
      {
        v41 += 8;
        if ( v42 == v41 )
          goto LABEL_49;
      }
      v47 = *(_QWORD *)&v197[v41];
      v41 += 8;
      v189.m128i_i64[0] = v47;
      sub_1AB7890((__int64)&v200, &v189, v43, v44, v45, v46);
    }
    while ( v42 != v41 );
LABEL_49:
    if ( !(_DWORD)v207 )
      goto LABEL_65;
    v174 = v16;
    v48 = 0;
    v49 = 0;
    while ( 1 )
    {
      v185 = *(_QWORD *)&v206[8 * v49];
      sub_1A51850((unsigned __int64 *)&v189, v174, (__int64 *)&v185);
      v51 = v190;
      if ( v190 )
        break;
LABEL_63:
      v49 = (unsigned int)(v48 + 1);
      v48 = v49;
      if ( (_DWORD)v49 == (_DWORD)v207 )
      {
        v16 = v174;
        goto LABEL_65;
      }
    }
    v180 = v190 != -16 && v190 != -8;
    if ( *(_BYTE *)(v190 + 16) <= 0x17u )
    {
      if ( v180 )
        sub_1649B30(&v189);
      goto LABEL_63;
    }
    if ( v180 )
      sub_1649B30(&v189);
    v52 = *(_BYTE *)(v51 + 16);
    if ( v52 > 0x17u )
    {
      v53 = v51 | 4;
      if ( v52 != 78 )
      {
        if ( v52 != 29 )
          goto LABEL_58;
        v53 = v51 & 0xFFFFFFFFFFFFFFFBLL;
      }
      v145 = v53 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v53 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v146 = (__int64 *)(v145 - 24);
        v147 = (__int64 *)(v145 - 72);
        if ( (v53 & 4) != 0 )
          v147 = v146;
        v148 = *v147;
        if ( !*(_BYTE *)(v148 + 16) && (*(_BYTE *)(v148 + 33) & 0x20) == 0 )
          goto LABEL_63;
      }
    }
LABEL_58:
    v189 = (__m128i)(unsigned __int64)v164;
    v190 = 0;
    v191 = 0;
    v192 = 0;
    v161 = sub_13E3350(v51, &v189, 0, 1, v50);
    if ( v161 )
    {
      for ( k = *(_QWORD *)(v185 + 8); k; k = *(_QWORD *)(k + 8) )
      {
        v189.m128i_i64[0] = (__int64)sub_1648700(k);
        sub_1AB7890((__int64)&v200, &v189, v57, v58, v59, v60);
      }
      sub_164D160(v51, v161, a7, a8, a9, a10, v54, v55, a13, a14);
      if ( (unsigned __int8)sub_1AE9990(v51, 0) )
      {
        sub_15F20C0((_QWORD *)v51);
      }
      else
      {
        v149 = sub_1AB4240(v174, v185);
        v150 = v149[2];
        if ( v51 != v150 )
        {
          if ( v150 != 0 && v150 != -8 && v150 != -16 )
            sub_1649B30(v149);
          v149[2] = v51;
          if ( v180 )
            sub_164C220((__int64)v149);
        }
      }
    }
    goto LABEL_63;
  }
LABEL_65:
  v61 = a1 + 72;
  v181 = sub_1AB4240(v16, v153)[2] + 24LL;
  v62 = v181;
  if ( v181 != a1 + 72 )
  {
    while ( 1 )
    {
      sub_1AEE9C0(v62 - 24, 0, 0, 0);
      if ( v181 != v62 )
      {
        v78 = *(_QWORD *)(v62 - 16);
        if ( !v78 )
          break;
        while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v78) + 16) - 25) > 9u )
        {
          v78 = *(_QWORD *)(v78 + 8);
          if ( !v78 )
            goto LABEL_102;
        }
        if ( v62 - 24 == sub_157F0B0(v62 - 24) )
          break;
      }
      v71 = sub_157EBA0(v62 - 24);
      if ( *(_BYTE *)(v71 + 16) == 26 && (*(_DWORD *)(v71 + 20) & 0xFFFFFFF) != 3 )
      {
        v63 = *(_QWORD *)(v71 - 24);
        v175 = (_QWORD *)v71;
        if ( sub_157F0B0(v63) )
        {
          sub_15F20C0(v175);
          sub_164D160(v63, v62 - 24, a7, a8, a9, a10, v64, v65, a13, a14);
          if ( v63 + 40 != (*(_QWORD *)(v63 + 40) & 0xFFFFFFFFFFFFFFF8LL) )
          {
            v66 = *(__int64 **)(v63 + 48);
            if ( v62 + 16 != v63 + 40 )
            {
              sub_157EA80(v62 + 16, v63 + 40, *(_QWORD *)(v63 + 48), v63 + 40);
              if ( (__int64 *)(v63 + 40) != v66 )
              {
                v67 = *(_QWORD *)(v63 + 40) & 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)((*v66 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v63 + 40;
                *(_QWORD *)(v63 + 40) = *(_QWORD *)(v63 + 40) & 7LL | *v66 & 0xFFFFFFFFFFFFFFF8LL;
                v68 = *(_QWORD *)(v62 + 16);
                *(_QWORD *)(v67 + 8) = v62 + 16;
                v68 &= 0xFFFFFFFFFFFFFFF8LL;
                *v66 = v68 | *v66 & 7;
                *(_QWORD *)(v68 + 8) = v66;
                *(_QWORD *)(v62 + 16) = v67 | *(_QWORD *)(v62 + 16) & 7LL;
              }
            }
          }
          sub_157F980(v63);
LABEL_74:
          if ( v62 == v61 )
            goto LABEL_79;
          goto LABEL_75;
        }
      }
      v62 = *(_QWORD *)(v62 + 8);
      if ( v62 == v61 )
        goto LABEL_79;
LABEL_75:
      if ( !v62 )
      {
        sub_1AEE9C0(0, 0, 0, 0);
        BUG();
      }
    }
LABEL_102:
    v79 = v62 - 24;
    v62 = *(_QWORD *)(v62 + 8);
    sub_1AA7270(v79, 0, a7, a8, a9, a10, v69, v70, a13, a14);
    goto LABEL_74;
  }
LABEL_79:
  for ( m = sub_1AB4240(v16, v153)[2] + 24LL; m != v61; m = *(_QWORD *)(m + 8) )
  {
    while ( 1 )
    {
      v73 = m - 24;
      if ( !m )
        v73 = 0;
      v76 = sub_157EBA0(v73);
      if ( *(_BYTE *)(v76 + 16) == 25 )
        break;
      m = *(_QWORD *)(m + 8);
      if ( m == v61 )
        goto LABEL_88;
    }
    v77 = *(unsigned int *)(a6 + 8);
    if ( (unsigned int)v77 >= *(_DWORD *)(a6 + 12) )
    {
      sub_16CD150(a6, (const void *)(a6 + 16), 0, 8, v74, v75);
      v77 = *(unsigned int *)(a6 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a6 + 8 * v77) = v76;
    ++*(_DWORD *)(a6 + 8);
  }
LABEL_88:
  if ( v206 != v208 )
    _libc_free((unsigned __int64)v206);
  if ( (v201 & 1) == 0 )
    j___libc_free_0(v202);
  if ( v197 != v199 )
    _libc_free((unsigned __int64)v197);
  if ( v186 )
    j_j___libc_free_0(v186, v188 - v186);
}
