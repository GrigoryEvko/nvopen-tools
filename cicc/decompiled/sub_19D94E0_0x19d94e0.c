// Function: sub_19D94E0
// Address: 0x19d94e0
//
__int64 __fastcall sub_19D94E0(
        __int64 a1,
        __int64 *a2,
        unsigned __int64 a3,
        __int64 a4,
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
  __int64 *v16; // rbx
  __int64 v17; // r13
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rax
  __int64 *v23; // r15
  __int64 v24; // rdx
  __int64 *v25; // rax
  __int64 *v26; // rdx
  __int64 v27; // rcx
  int v28; // r9d
  __int64 v29; // r8
  __int64 v30; // rdx
  __int64 v31; // r10
  __int64 v32; // r14
  unsigned int v33; // ecx
  _QWORD *v34; // rsi
  _QWORD *v35; // r13
  unsigned __int8 *v36; // r14
  __int64 v37; // r13
  unsigned __int8 *v38; // rbx
  __int64 v39; // r14
  __int64 v40; // rsi
  _QWORD *v41; // rdi
  __int64 *v42; // rax
  int v43; // edx
  __int64 v44; // r14
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // r15
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // r15
  __int64 v52; // rax
  __int64 v53; // rax
  _QWORD *v54; // r14
  _QWORD *v55; // rax
  _QWORD *v56; // r15
  unsigned __int64 *v57; // r14
  __int64 v58; // rax
  unsigned __int64 v59; // rcx
  unsigned __int8 *v60; // rsi
  __int64 v61; // rsi
  __int64 v62; // rdx
  __int64 v63; // rcx
  __int64 *v64; // r8
  __int64 v65; // r9
  double v66; // xmm4_8
  double v67; // xmm5_8
  int v68; // eax
  __int64 v69; // rax
  int v70; // edx
  __int64 v71; // rdx
  _QWORD *v72; // rax
  __int64 v73; // rcx
  unsigned __int64 v74; // rdx
  __int64 v75; // rdx
  __int64 v76; // rdx
  __int64 v77; // rax
  __int64 v78; // rcx
  __int64 v79; // rdx
  __int64 *v80; // rbx
  __int64 v81; // r12
  __int64 v82; // r14
  __int64 result; // rax
  __int64 v84; // rdi
  int v85; // eax
  __int64 v86; // rax
  __int64 v87; // rcx
  __int64 v88; // r15
  _QWORD *v89; // rax
  _QWORD *v90; // r14
  unsigned __int64 *v91; // r15
  __int64 v92; // rax
  unsigned __int64 v93; // rcx
  __int64 v94; // rdx
  __int64 v95; // rcx
  __int64 *v96; // r8
  __int64 v97; // r9
  unsigned __int8 *v98; // rsi
  __int64 v99; // rsi
  __int64 v100; // rbx
  int v101; // eax
  __int64 v102; // rax
  int v103; // edx
  __int64 v104; // rdx
  __int64 *v105; // rax
  __int64 v106; // rcx
  unsigned __int64 v107; // rdx
  __int64 v108; // rdx
  __int64 v109; // rdx
  __int64 v110; // rcx
  _QWORD *v111; // rax
  __int64 v112; // r9
  _QWORD **v113; // rax
  __int64 *v114; // rax
  __int64 v115; // rax
  __int64 v116; // r9
  unsigned __int64 v117; // rsi
  __int64 v118; // rax
  __int64 v119; // rsi
  __int64 v120; // rdx
  unsigned __int8 *v121; // rsi
  __int64 v122; // rsi
  __int64 v123; // rdx
  unsigned __int64 v124; // rax
  __int64 v125; // rdx
  __int64 v126; // rdx
  __int64 v127; // rcx
  __int64 v128; // rbx
  __int64 v129; // r8
  __int64 v130; // r9
  int v131; // eax
  __int64 v132; // rax
  int v133; // edx
  __int64 v134; // rdx
  __int64 *v135; // rax
  __int64 v136; // rcx
  unsigned __int64 v137; // rdx
  __int64 v138; // rdx
  __int64 v139; // rdx
  __int64 v140; // rcx
  _QWORD *v141; // rax
  _QWORD *v142; // r15
  unsigned __int64 v143; // rsi
  __int64 v144; // rax
  __int64 v145; // rsi
  __int64 v146; // rbx
  int v147; // eax
  __int64 v148; // rax
  int v149; // edx
  __int64 v150; // rdx
  __int64 *v151; // rax
  __int64 v152; // rcx
  unsigned __int64 v153; // rdx
  __int64 v154; // rdx
  __int64 v155; // r15
  _QWORD *v156; // rax
  _QWORD *v157; // r14
  unsigned __int64 *v158; // r15
  __int64 v159; // rax
  unsigned __int64 v160; // rcx
  __int64 v161; // rdx
  __int64 v162; // rcx
  __int64 *v163; // r8
  __int64 v164; // r9
  unsigned __int8 *v165; // rsi
  __int64 v166; // rsi
  __int64 v167; // rbx
  int v168; // eax
  __int64 v169; // rax
  int v170; // edx
  __int64 v171; // rdx
  __int64 *v172; // rax
  __int64 v173; // rcx
  unsigned __int64 v174; // rdx
  __int64 v175; // rcx
  __int64 v176; // [rsp+0h] [rbp-120h]
  __int64 v177; // [rsp+8h] [rbp-118h]
  __int64 *v178; // [rsp+8h] [rbp-118h]
  __int64 *v179; // [rsp+10h] [rbp-110h]
  __int64 v180; // [rsp+10h] [rbp-110h]
  __int64 v181; // [rsp+10h] [rbp-110h]
  __int64 v182; // [rsp+10h] [rbp-110h]
  _QWORD *v183; // [rsp+18h] [rbp-108h]
  unsigned __int64 *v184; // [rsp+18h] [rbp-108h]
  __int64 v185; // [rsp+18h] [rbp-108h]
  __int64 v187; // [rsp+28h] [rbp-F8h]
  __int64 v188; // [rsp+28h] [rbp-F8h]
  __int64 v189; // [rsp+28h] [rbp-F8h]
  __int64 *v191; // [rsp+40h] [rbp-E0h]
  __int64 v192; // [rsp+40h] [rbp-E0h]
  __int64 v194; // [rsp+48h] [rbp-D8h]
  unsigned __int64 *v195; // [rsp+48h] [rbp-D8h]
  __int64 v196; // [rsp+58h] [rbp-C8h] BYREF
  __int64 v197[2]; // [rsp+60h] [rbp-C0h] BYREF
  __int16 v198; // [rsp+70h] [rbp-B0h]
  __int64 v199; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v200; // [rsp+88h] [rbp-98h]
  __int16 v201; // [rsp+90h] [rbp-90h]
  int v202; // [rsp+98h] [rbp-88h]
  unsigned __int8 *v203; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v204; // [rsp+A8h] [rbp-78h]
  unsigned __int64 *v205; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v206; // [rsp+B8h] [rbp-68h]
  __int64 v207; // [rsp+C0h] [rbp-60h]
  __int64 v208; // [rsp+C8h] [rbp-58h]
  __int64 v209; // [rsp+D0h] [rbp-50h]
  __int64 v210; // [rsp+D8h] [rbp-48h]

  v16 = a2;
  v17 = *a2;
  v191 = (__int64 *)sub_157E9C0(*a2);
  if ( a3 <= 1 )
  {
    v84 = a2[2];
    v85 = *(_DWORD *)(v84 + 20) & 0xFFFFFFF;
    if ( a4 )
    {
      if ( v85 != 3 )
      {
        sub_15F20C0((_QWORD *)v84);
        v86 = sub_157E9C0(v17);
        v87 = a2[1];
        v206 = v86;
        v88 = *(_QWORD *)(a5 + 40);
        v203 = 0;
        v207 = 0;
        LODWORD(v208) = 0;
        v209 = 0;
        v210 = 0;
        v204 = v17;
        v205 = (unsigned __int64 *)(v17 + 40);
        v192 = v87;
        v201 = 257;
        v89 = sub_1648A60(56, 3u);
        v90 = v89;
        if ( v89 )
          sub_15F83E0((__int64)v89, a4, v88, v192, 0);
        if ( v204 )
        {
          v91 = v205;
          sub_157E9D0(v204 + 40, (__int64)v90);
          v92 = v90[3];
          v93 = *v205;
          v90[4] = v205;
          v93 &= 0xFFFFFFFFFFFFFFF8LL;
          v90[3] = v93 | v92 & 7;
          *(_QWORD *)(v93 + 8) = v90 + 3;
          *v91 = *v91 & 7 | (unsigned __int64)(v90 + 3);
        }
        sub_164B780((__int64)v90, &v199);
        v98 = v203;
        if ( v203 )
        {
          v197[0] = (__int64)v203;
          sub_1623A60((__int64)v197, (__int64)v203, 2);
          v99 = v90[6];
          v96 = v197;
          if ( v99 )
          {
            sub_161E7C0((__int64)(v90 + 6), v99);
            v96 = v197;
          }
          v98 = (unsigned __int8 *)v197[0];
          v90[6] = v197[0];
          if ( v98 )
            sub_1623210((__int64)v197, v98, (__int64)(v90 + 6));
        }
        v100 = v16[1];
        v101 = *(_DWORD *)(a5 + 20) & 0xFFFFFFF;
        if ( v101 == *(_DWORD *)(a5 + 56) )
        {
          sub_15F55D0(a5, (__int64)v98, v94, v95, (__int64)v96, v97);
          v101 = *(_DWORD *)(a5 + 20) & 0xFFFFFFF;
        }
        v102 = (v101 + 1) & 0xFFFFFFF;
        v103 = v102 | *(_DWORD *)(a5 + 20) & 0xF0000000;
        *(_DWORD *)(a5 + 20) = v103;
        if ( (v103 & 0x40000000) != 0 )
          v104 = *(_QWORD *)(a5 - 8);
        else
          v104 = a5 - 24 * v102;
        v105 = (__int64 *)(v104 + 24LL * (unsigned int)(v102 - 1));
        if ( *v105 )
        {
          v106 = v105[1];
          v107 = v105[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v107 = v106;
          if ( v106 )
            *(_QWORD *)(v106 + 16) = *(_QWORD *)(v106 + 16) & 3LL | v107;
        }
        *v105 = v100;
        if ( v100 )
        {
          v108 = *(_QWORD *)(v100 + 8);
          v105[1] = v108;
          if ( v108 )
            *(_QWORD *)(v108 + 16) = (unsigned __int64)(v105 + 1) | *(_QWORD *)(v108 + 16) & 3LL;
          v105[2] = (v100 + 8) | v105[2] & 3;
          *(_QWORD *)(v100 + 8) = v105;
        }
LABEL_89:
        v109 = *(_DWORD *)(a5 + 20) & 0xFFFFFFF;
        if ( (*(_BYTE *)(a5 + 23) & 0x40) != 0 )
          v110 = *(_QWORD *)(a5 - 8);
        else
          v110 = a5 - 24 * v109;
        result = 8LL * (unsigned int)(v109 - 1) + 24LL * *(unsigned int *)(a5 + 56);
        *(_QWORD *)(v110 + result + 8) = v17;
        if ( v203 )
          return sub_161E7C0((__int64)&v203, (__int64)v203);
        return result;
      }
      v122 = v84 - 24;
      if ( *(_QWORD *)(v84 - 24) )
      {
        v123 = *(_QWORD *)(v84 - 16);
        v124 = *(_QWORD *)(v84 - 8) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v124 = v123;
        if ( v123 )
          *(_QWORD *)(v123 + 16) = *(_QWORD *)(v123 + 16) & 3LL | v124;
      }
      *(_QWORD *)(v84 - 24) = a4;
      v125 = *(_QWORD *)(a4 + 8);
      *(_QWORD *)(v84 - 16) = v125;
      if ( v125 )
        *(_QWORD *)(v125 + 16) = (v84 - 16) | *(_QWORD *)(v125 + 16) & 3LL;
      *(_QWORD *)(v84 - 8) = *(_QWORD *)(v84 - 8) & 3LL | (a4 + 8);
      *(_QWORD *)(a4 + 8) = v122;
      v128 = sub_159C540(v191);
      v131 = *(_DWORD *)(a5 + 20) & 0xFFFFFFF;
      if ( v131 == *(_DWORD *)(a5 + 56) )
      {
        sub_15F55D0(a5, v122, v126, v127, v129, v130);
        v131 = *(_DWORD *)(a5 + 20) & 0xFFFFFFF;
      }
      v132 = (v131 + 1) & 0xFFFFFFF;
      v133 = v132 | *(_DWORD *)(a5 + 20) & 0xF0000000;
      *(_DWORD *)(a5 + 20) = v133;
      if ( (v133 & 0x40000000) != 0 )
        v134 = *(_QWORD *)(a5 - 8);
      else
        v134 = a5 - 24 * v132;
      v135 = (__int64 *)(v134 + 24LL * (unsigned int)(v132 - 1));
      if ( *v135 )
      {
        v136 = v135[1];
        v137 = v135[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v137 = v136;
        if ( v136 )
          *(_QWORD *)(v136 + 16) = *(_QWORD *)(v136 + 16) & 3LL | v137;
      }
      *v135 = v128;
      if ( v128 )
      {
        v138 = *(_QWORD *)(v128 + 8);
        v135[1] = v138;
        if ( v138 )
          *(_QWORD *)(v138 + 16) = (unsigned __int64)(v135 + 1) | *(_QWORD *)(v138 + 16) & 3LL;
        v135[2] = (v128 + 8) | v135[2] & 3;
        *(_QWORD *)(v128 + 8) = v135;
      }
    }
    else
    {
      if ( v85 == 3 )
      {
        sub_15F20C0((_QWORD *)v84);
        v203 = 0;
        v206 = sub_157E9C0(v17);
        v155 = *(_QWORD *)(a5 + 40);
        v205 = (unsigned __int64 *)(v17 + 40);
        v207 = 0;
        LODWORD(v208) = 0;
        v209 = 0;
        v210 = 0;
        v204 = v17;
        v201 = 257;
        v156 = sub_1648A60(56, 1u);
        v157 = v156;
        if ( v156 )
          sub_15F8320((__int64)v156, v155, 0);
        if ( v204 )
        {
          v158 = v205;
          sub_157E9D0(v204 + 40, (__int64)v157);
          v159 = v157[3];
          v160 = *v205;
          v157[4] = v205;
          v160 &= 0xFFFFFFFFFFFFFFF8LL;
          v157[3] = v160 | v159 & 7;
          *(_QWORD *)(v160 + 8) = v157 + 3;
          *v158 = *v158 & 7 | (unsigned __int64)(v157 + 3);
        }
        sub_164B780((__int64)v157, &v199);
        v165 = v203;
        if ( v203 )
        {
          v197[0] = (__int64)v203;
          sub_1623A60((__int64)v197, (__int64)v203, 2);
          v166 = v157[6];
          v163 = v197;
          if ( v166 )
          {
            sub_161E7C0((__int64)(v157 + 6), v166);
            v163 = v197;
          }
          v165 = (unsigned __int8 *)v197[0];
          v157[6] = v197[0];
          if ( v165 )
            sub_1623210((__int64)v197, v165, (__int64)(v157 + 6));
        }
        v167 = v16[1];
        v168 = *(_DWORD *)(a5 + 20) & 0xFFFFFFF;
        if ( v168 == *(_DWORD *)(a5 + 56) )
        {
          sub_15F55D0(a5, (__int64)v165, v161, v162, (__int64)v163, v164);
          v168 = *(_DWORD *)(a5 + 20) & 0xFFFFFFF;
        }
        v169 = (v168 + 1) & 0xFFFFFFF;
        v170 = v169 | *(_DWORD *)(a5 + 20) & 0xF0000000;
        *(_DWORD *)(a5 + 20) = v170;
        if ( (v170 & 0x40000000) != 0 )
          v171 = *(_QWORD *)(a5 - 8);
        else
          v171 = a5 - 24 * v169;
        v172 = (__int64 *)(v171 + 24LL * (unsigned int)(v169 - 1));
        if ( *v172 )
        {
          v173 = v172[1];
          v174 = v172[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v174 = v173;
          if ( v173 )
            *(_QWORD *)(v173 + 16) = *(_QWORD *)(v173 + 16) & 3LL | v174;
        }
        *v172 = v167;
        if ( v167 )
        {
          v175 = *(_QWORD *)(v167 + 8);
          v172[1] = v175;
          if ( v175 )
            *(_QWORD *)(v175 + 16) = (unsigned __int64)(v172 + 1) | *(_QWORD *)(v175 + 16) & 3LL;
          v172[2] = v172[2] & 3 | (v167 + 8);
          *(_QWORD *)(v167 + 8) = v172;
        }
        goto LABEL_89;
      }
      v146 = a2[1];
      v147 = *(_DWORD *)(a5 + 20) & 0xFFFFFFF;
      if ( v147 == *(_DWORD *)(a5 + 56) )
      {
        sub_15F55D0(a5, (__int64)a2, v18, v19, v20, v21);
        v147 = *(_DWORD *)(a5 + 20) & 0xFFFFFFF;
      }
      v148 = (v147 + 1) & 0xFFFFFFF;
      v149 = v148 | *(_DWORD *)(a5 + 20) & 0xF0000000;
      *(_DWORD *)(a5 + 20) = v149;
      if ( (v149 & 0x40000000) != 0 )
        v150 = *(_QWORD *)(a5 - 8);
      else
        v150 = a5 - 24 * v148;
      v151 = (__int64 *)(v150 + 24LL * (unsigned int)(v148 - 1));
      if ( *v151 )
      {
        v152 = v151[1];
        v153 = v151[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v153 = v152;
        if ( v152 )
          *(_QWORD *)(v152 + 16) = *(_QWORD *)(v152 + 16) & 3LL | v153;
      }
      *v151 = v146;
      if ( v146 )
      {
        v154 = *(_QWORD *)(v146 + 8);
        v151[1] = v154;
        if ( v154 )
          *(_QWORD *)(v154 + 16) = (unsigned __int64)(v151 + 1) | *(_QWORD *)(v154 + 16) & 3LL;
        v151[2] = (v146 + 8) | v151[2] & 3;
        *(_QWORD *)(v146 + 8) = v151;
      }
    }
    v139 = *(_DWORD *)(a5 + 20) & 0xFFFFFFF;
    if ( (*(_BYTE *)(a5 + 23) & 0x40) != 0 )
      v140 = *(_QWORD *)(a5 - 8);
    else
      v140 = a5 - 24 * v139;
    result = 8LL * (unsigned int)(v139 - 1) + 24LL * *(unsigned int *)(a5 + 56);
    *(_QWORD *)(v140 + result + 8) = v17;
    return result;
  }
  v22 = 104 * a3;
  v23 = &a2[13 * a3];
  v24 = 0x4EC4EC4EC4EC4EC5LL * (v22 >> 3);
  if ( v24 >> 2 > 0 )
  {
    v25 = a2;
    v26 = &a2[52 * (v24 >> 2)];
    while ( !*((_BYTE *)v25 + 24) )
    {
      if ( *((_BYTE *)v25 + 128) )
      {
        v25 += 13;
        if ( v23 == v25 )
          goto LABEL_33;
        goto LABEL_10;
      }
      if ( *((_BYTE *)v25 + 232) )
      {
        v25 += 26;
        if ( v23 == v25 )
          goto LABEL_33;
        goto LABEL_10;
      }
      if ( *((_BYTE *)v25 + 336) )
      {
        v25 += 39;
        if ( v23 == v25 )
          goto LABEL_33;
        goto LABEL_10;
      }
      v25 += 52;
      if ( v26 == v25 )
      {
        v24 = 0x4EC4EC4EC4EC4EC5LL * (v23 - v25);
        goto LABEL_30;
      }
    }
    goto LABEL_9;
  }
  v25 = a2;
LABEL_30:
  if ( v24 == 2 )
  {
LABEL_162:
    if ( *((_BYTE *)v25 + 24) )
      goto LABEL_9;
    v25 += 13;
    goto LABEL_164;
  }
  if ( v24 == 3 )
  {
    if ( *((_BYTE *)v25 + 24) )
      goto LABEL_9;
    v25 += 13;
    goto LABEL_162;
  }
  if ( v24 != 1 )
    goto LABEL_33;
LABEL_164:
  if ( !*((_BYTE *)v25 + 24) )
    goto LABEL_33;
LABEL_9:
  if ( v23 == v25 )
    goto LABEL_33;
LABEL_10:
  v179 = v25;
  v27 = *(_QWORD *)(a1 + 32);
  v203 = (unsigned __int8 *)v25[4];
  v177 = v27;
  v204 = v25[8];
  v205 = (unsigned __int64 *)v25[5];
  v206 = v25[9];
  v207 = v25[1];
  v208 = v25[2];
  sub_19D91E0((__int64)&v199, (__int64 *)&v203, 6);
  v203 = (unsigned __int8 *)&v205;
  v204 = 0x400000000LL;
  v29 = *v179 + 40;
  if ( *(_QWORD *)(*v179 + 48) == v29 )
    goto LABEL_28;
  v30 = 0;
  v31 = v17;
  v32 = *(_QWORD *)(*v179 + 48);
  do
  {
    while ( 1 )
    {
      v35 = (_QWORD *)(v32 - 24);
      if ( !v32 )
        v35 = 0;
      if ( !v202 )
        goto LABEL_17;
      v33 = (v202 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
      v34 = *(_QWORD **)(v200 + 8LL * v33);
      if ( v35 != v34 )
        break;
LABEL_13:
      v32 = *(_QWORD *)(v32 + 8);
      if ( v29 == v32 )
        goto LABEL_20;
    }
    v28 = 1;
    while ( v34 != (_QWORD *)-8LL )
    {
      v33 = (v202 - 1) & (v28 + v33);
      v34 = *(_QWORD **)(v200 + 8LL * v33);
      if ( v35 == v34 )
        goto LABEL_13;
      ++v28;
    }
LABEL_17:
    if ( HIDWORD(v204) <= (unsigned int)v30 )
    {
      v176 = v29;
      v182 = v31;
      sub_16CD150((__int64)&v203, &v205, 0, 8, v29, v28);
      v30 = (unsigned int)v204;
      v29 = v176;
      v31 = v182;
    }
    *(_QWORD *)&v203[8 * v30] = v35;
    v30 = (unsigned int)(v204 + 1);
    LODWORD(v204) = v204 + 1;
    v32 = *(_QWORD *)(v32 + 8);
  }
  while ( v29 != v32 );
LABEL_20:
  v17 = v31;
  v36 = &v203[8 * v30];
  if ( v203 != v36 )
  {
    v180 = v31;
    v37 = v177;
    v178 = v16;
    v38 = &v203[8 * v30];
    v39 = (__int64)v203;
    do
    {
      v40 = *(_QWORD *)(v37 + 48);
      v41 = (_QWORD *)*((_QWORD *)v38 - 1);
      if ( v40 )
        v40 -= 24;
      v38 -= 8;
      sub_15F22F0(v41, v40);
    }
    while ( (unsigned __int8 *)v39 != v38 );
    v17 = v180;
    v16 = v178;
    v36 = v203;
  }
  if ( v36 != (unsigned __int8 *)&v205 )
    _libc_free((unsigned __int64)v36);
LABEL_28:
  j___libc_free_0(v200);
LABEL_33:
  if ( v23 == v16 )
  {
    v44 = 0;
  }
  else
  {
    v42 = v16;
    v43 = 0;
    do
    {
      v43 += *((_DWORD *)v42 + 24);
      v42 += 13;
    }
    while ( v23 != v42 );
    v44 = v43 / 8;
  }
  sub_15F20C0((_QWORD *)v16[2]);
  sub_15F20C0((_QWORD *)v16[1]);
  sub_15F20C0((_QWORD *)v16[5]);
  sub_15F20C0((_QWORD *)v16[9]);
  v45 = sub_157E9C0(v17);
  v204 = v17;
  v206 = v45;
  v203 = 0;
  v207 = 0;
  LODWORD(v208) = 0;
  v209 = 0;
  v210 = 0;
  v205 = (unsigned __int64 *)(v17 + 40);
  v46 = sub_15F2050(a5);
  v47 = sub_1632FA0(v46);
  v48 = sub_15A9620(v47, (__int64)v191, 0);
  v49 = sub_159C470(v48, v44, 0);
  v50 = sub_1AB1ED0(v16[4], v16[8], v49, &v203, v47, a6);
  v198 = 257;
  v51 = v50;
  v52 = sub_1643350(v191);
  v53 = sub_159C470(v52, 0, 0);
  if ( *(_BYTE *)(v51 + 16) > 0x10u || *(_BYTE *)(v53 + 16) > 0x10u )
  {
    v188 = v53;
    v201 = 257;
    v111 = sub_1648A60(56, 2u);
    v112 = v188;
    v54 = v111;
    if ( v111 )
    {
      v189 = (__int64)v111;
      v113 = *(_QWORD ***)v51;
      if ( *(_BYTE *)(*(_QWORD *)v51 + 8LL) == 16 )
      {
        v181 = v112;
        v183 = v113[4];
        v114 = (__int64 *)sub_1643320(*v113);
        v115 = (__int64)sub_16463B0(v114, (unsigned int)v183);
        v116 = v181;
      }
      else
      {
        v185 = v112;
        v115 = sub_1643320(*v113);
        v116 = v185;
      }
      sub_15FEC10((__int64)v54, v115, 51, 32, v51, v116, (__int64)&v199, 0);
    }
    else
    {
      v189 = 0;
    }
    if ( v204 )
    {
      v184 = v205;
      sub_157E9D0(v204 + 40, (__int64)v54);
      v117 = *v184;
      v118 = v54[3] & 7LL;
      v54[4] = v184;
      v117 &= 0xFFFFFFFFFFFFFFF8LL;
      v54[3] = v117 | v118;
      *(_QWORD *)(v117 + 8) = v54 + 3;
      *v184 = *v184 & 7 | (unsigned __int64)(v54 + 3);
    }
    sub_164B780(v189, v197);
    if ( v203 )
    {
      v196 = (__int64)v203;
      sub_1623A60((__int64)&v196, (__int64)v203, 2);
      v119 = v54[6];
      v120 = (__int64)(v54 + 6);
      if ( v119 )
      {
        sub_161E7C0((__int64)(v54 + 6), v119);
        v120 = (__int64)(v54 + 6);
      }
      v121 = (unsigned __int8 *)v196;
      v54[6] = v196;
      if ( v121 )
        sub_1623210((__int64)&v196, v121, v120);
    }
  }
  else
  {
    v54 = (_QWORD *)sub_15A37B0(0x20u, (_QWORD *)v51, (_QWORD *)v53, 0);
  }
  if ( a4 )
  {
    v187 = *(_QWORD *)(a5 + 40);
    v201 = 257;
    v55 = sub_1648A60(56, 3u);
    v56 = v55;
    if ( v55 )
      sub_15F83E0((__int64)v55, a4, v187, (__int64)v54, 0);
    if ( v204 )
    {
      v57 = v205;
      sub_157E9D0(v204 + 40, (__int64)v56);
      v58 = v56[3];
      v59 = *v57;
      v56[4] = v57;
      v59 &= 0xFFFFFFFFFFFFFFF8LL;
      v56[3] = v59 | v58 & 7;
      *(_QWORD *)(v59 + 8) = v56 + 3;
      *v57 = *v57 & 7 | (unsigned __int64)(v56 + 3);
    }
    sub_164B780((__int64)v56, &v199);
    v60 = v203;
    if ( v203 )
    {
      v197[0] = (__int64)v203;
      sub_1623A60((__int64)v197, (__int64)v203, 2);
      v61 = v56[6];
      if ( v61 )
        sub_161E7C0((__int64)(v56 + 6), v61);
      v60 = (unsigned __int8 *)v197[0];
      v56[6] = v197[0];
      if ( v60 )
        sub_1623210((__int64)v197, v60, (__int64)(v56 + 6));
    }
    v54 = (_QWORD *)sub_159C540(v191);
    v68 = *(_DWORD *)(a5 + 20) & 0xFFFFFFF;
    if ( v68 != *(_DWORD *)(a5 + 56) )
      goto LABEL_51;
LABEL_136:
    sub_15F55D0(a5, (__int64)v60, v62, v63, (__int64)v64, v65);
    v68 = *(_DWORD *)(a5 + 20) & 0xFFFFFFF;
    goto LABEL_51;
  }
  v194 = *(_QWORD *)(a5 + 40);
  v201 = 257;
  v141 = sub_1648A60(56, 1u);
  v142 = v141;
  if ( v141 )
    sub_15F8320((__int64)v141, v194, 0);
  if ( v204 )
  {
    v195 = v205;
    sub_157E9D0(v204 + 40, (__int64)v142);
    v143 = *v195;
    v144 = v142[3] & 7LL;
    v142[4] = v195;
    v143 &= 0xFFFFFFFFFFFFFFF8LL;
    v142[3] = v143 | v144;
    *(_QWORD *)(v143 + 8) = v142 + 3;
    *v195 = *v195 & 7 | (unsigned __int64)(v142 + 3);
  }
  sub_164B780((__int64)v142, &v199);
  v60 = v203;
  if ( v203 )
  {
    v197[0] = (__int64)v203;
    sub_1623A60((__int64)v197, (__int64)v203, 2);
    v145 = v142[6];
    v64 = v197;
    v62 = (__int64)(v142 + 6);
    if ( v145 )
    {
      sub_161E7C0((__int64)(v142 + 6), v145);
      v64 = v197;
      v62 = (__int64)(v142 + 6);
    }
    v60 = (unsigned __int8 *)v197[0];
    v142[6] = v197[0];
    if ( v60 )
      sub_1623210((__int64)v197, v60, v62);
  }
  v68 = *(_DWORD *)(a5 + 20) & 0xFFFFFFF;
  if ( v68 == *(_DWORD *)(a5 + 56) )
    goto LABEL_136;
LABEL_51:
  v69 = (v68 + 1) & 0xFFFFFFF;
  v70 = v69 | *(_DWORD *)(a5 + 20) & 0xF0000000;
  *(_DWORD *)(a5 + 20) = v70;
  if ( (v70 & 0x40000000) != 0 )
    v71 = *(_QWORD *)(a5 - 8);
  else
    v71 = a5 - 24 * v69;
  v72 = (_QWORD *)(v71 + 24LL * (unsigned int)(v69 - 1));
  if ( *v72 )
  {
    v73 = v72[1];
    v74 = v72[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v74 = v73;
    if ( v73 )
      *(_QWORD *)(v73 + 16) = *(_QWORD *)(v73 + 16) & 3LL | v74;
  }
  *v72 = v54;
  if ( v54 )
  {
    v75 = v54[1];
    v72[1] = v75;
    if ( v75 )
      *(_QWORD *)(v75 + 16) = (unsigned __int64)(v72 + 1) | *(_QWORD *)(v75 + 16) & 3LL;
    v72[2] = (unsigned __int64)(v54 + 1) | v72[2] & 3LL;
    v54[1] = v72;
  }
  v76 = *(_DWORD *)(a5 + 20) & 0xFFFFFFF;
  v77 = (unsigned int)(v76 - 1);
  if ( (*(_BYTE *)(a5 + 23) & 0x40) != 0 )
    v78 = *(_QWORD *)(a5 - 8);
  else
    v78 = a5 - 24 * v76;
  v79 = *(unsigned int *)(a5 + 56);
  v80 = v16 + 13;
  v81 = 1;
  *(_QWORD *)(v78 + 8 * v77 + 24 * v79 + 8) = v17;
  do
  {
    v82 = *v80;
    ++v81;
    v80 += 13;
    sub_164D160(v82, v17, a7, a8, a9, a10, v66, v67, a13, a14);
    result = sub_157F980(v82);
  }
  while ( v81 != a3 );
  if ( v203 )
    return sub_161E7C0((__int64)&v203, (__int64)v203);
  return result;
}
