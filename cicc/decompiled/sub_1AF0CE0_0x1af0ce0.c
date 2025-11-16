// Function: sub_1AF0CE0
// Address: 0x1af0ce0
//
__int64 __fastcall sub_1AF0CE0(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 *a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned int v15; // r13d
  const __m128i *v16; // rdi
  unsigned int v17; // eax
  __int64 v18; // rsi
  _QWORD *v19; // r15
  _QWORD *v20; // r12
  __int64 v21; // r14
  char v22; // al
  int v23; // eax
  __int64 v24; // rax
  char v25; // al
  __int64 v26; // r8
  _BYTE *v27; // rdx
  char v28; // al
  __int64 v29; // rax
  unsigned int v30; // r14d
  __int64 v31; // rax
  char v32; // al
  unsigned __int64 v33; // r15
  double v34; // xmm4_8
  double v35; // xmm5_8
  char v36; // al
  __int64 v37; // r12
  char v38; // al
  __int64 v39; // rax
  unsigned __int64 v40; // rdi
  int v41; // r12d
  unsigned __int64 v42; // r14
  unsigned int v43; // ebx
  int v44; // r8d
  int v45; // r9d
  char v46; // dl
  __int64 v47; // r15
  __int64 *v48; // rax
  __int64 *v49; // rsi
  __int64 v50; // rax
  __int64 v51; // rcx
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // r13
  __int64 v55; // r12
  unsigned __int64 v56; // rdi
  unsigned __int64 v57; // rax
  unsigned int v58; // r14d
  __int64 v59; // r13
  const __m128i *v60; // rsi
  __int64 v61; // rbx
  __int64 v62; // rdx
  __int64 v63; // rax
  __int64 v64; // rdx
  __int64 v65; // rax
  _BYTE *v66; // r14
  bool v67; // al
  __int64 v68; // rdx
  unsigned __int8 v69; // al
  unsigned __int8 v70; // cl
  __int64 v71; // rdi
  int v72; // eax
  bool v73; // al
  __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // rax
  _BYTE *v77; // r14
  __int64 v78; // rdx
  __int64 v79; // rcx
  double v80; // xmm4_8
  double v81; // xmm5_8
  __int64 v82; // rdx
  __int64 v83; // rax
  __int64 v84; // rax
  char v85; // al
  bool v86; // al
  __int64 *v87; // rax
  __int16 v88; // dx
  __int64 v89; // rax
  unsigned __int64 v90; // rcx
  __int64 *v91; // r12
  __int64 *v92; // r14
  __int64 *v93; // r15
  __int64 v94; // rax
  unsigned __int8 v95; // al
  __int64 *v96; // rdx
  unsigned __int32 v97; // eax
  unsigned int v98; // esi
  __int64 v99; // rax
  __int64 v100; // rax
  _QWORD *v101; // r14
  _QWORD *v102; // r12
  __int64 v103; // r13
  _QWORD *v104; // r13
  unsigned __int64 *v105; // rsi
  __int64 v106; // rdi
  unsigned __int64 v107; // rcx
  __m128i *v108; // rax
  __int64 v110; // rax
  unsigned int v111; // r14d
  __int64 v112; // rdx
  __int64 v113; // rcx
  int v114; // r8d
  int v115; // r9d
  double v116; // xmm4_8
  double v117; // xmm5_8
  __int64 v118; // r12
  unsigned int v119; // ecx
  __int64 v120; // rax
  unsigned int v121; // ecx
  char v122; // dl
  int v123; // eax
  bool v124; // al
  unsigned int v125; // r14d
  __int64 v126; // rax
  char v127; // cl
  bool v128; // al
  __int64 v129; // rax
  __int64 v130; // rdx
  __int64 v131; // r13
  __int64 v132; // rdx
  __int64 v133; // rax
  __int64 v134; // rax
  __int64 v135; // r13
  __int64 v136; // rax
  __int64 v137; // r13
  __int64 v138; // r12
  _QWORD *v139; // rdi
  __int64 v142; // [rsp+20h] [rbp-570h]
  __int64 v143; // [rsp+20h] [rbp-570h]
  __int64 v144; // [rsp+20h] [rbp-570h]
  __int64 v145; // [rsp+28h] [rbp-568h]
  unsigned int v146; // [rsp+28h] [rbp-568h]
  int v147; // [rsp+28h] [rbp-568h]
  _QWORD *v148; // [rsp+30h] [rbp-560h]
  unsigned __int8 v149; // [rsp+30h] [rbp-560h]
  unsigned int v150; // [rsp+30h] [rbp-560h]
  int v151; // [rsp+30h] [rbp-560h]
  __int64 v152; // [rsp+30h] [rbp-560h]
  __int64 v153; // [rsp+38h] [rbp-558h]
  __int64 v154; // [rsp+38h] [rbp-558h]
  __int64 v155; // [rsp+40h] [rbp-550h]
  int v156; // [rsp+40h] [rbp-550h]
  int v157; // [rsp+40h] [rbp-550h]
  int v158; // [rsp+40h] [rbp-550h]
  unsigned __int8 v159; // [rsp+40h] [rbp-550h]
  __int64 v160; // [rsp+40h] [rbp-550h]
  bool v161; // [rsp+40h] [rbp-550h]
  int v162; // [rsp+40h] [rbp-550h]
  int v163; // [rsp+40h] [rbp-550h]
  __int64 v164; // [rsp+48h] [rbp-548h]
  __int64 *v165; // [rsp+58h] [rbp-538h] BYREF
  __int64 *v166[2]; // [rsp+60h] [rbp-530h] BYREF
  __m128i v167; // [rsp+70h] [rbp-520h] BYREF
  __int64 v168; // [rsp+80h] [rbp-510h] BYREF
  unsigned int v169; // [rsp+88h] [rbp-508h]
  __int64 v170; // [rsp+A0h] [rbp-4F0h] BYREF
  __int64 *v171; // [rsp+A8h] [rbp-4E8h]
  __int64 *v172; // [rsp+B0h] [rbp-4E0h]
  unsigned int v173; // [rsp+B8h] [rbp-4D8h]
  unsigned int v174; // [rsp+BCh] [rbp-4D4h]
  int v175; // [rsp+C0h] [rbp-4D0h]
  _QWORD v176[17]; // [rsp+C8h] [rbp-4C8h] BYREF
  __m128i *v177; // [rsp+150h] [rbp-440h] BYREF
  __m128i *v178; // [rsp+158h] [rbp-438h]
  _QWORD v179[134]; // [rsp+160h] [rbp-430h] BYREF

  v171 = v176;
  v172 = v176;
  v177 = (__m128i *)v179;
  v13 = *(_QWORD *)(a1 + 80);
  v173 = 16;
  v175 = 0;
  HIDWORD(v178) = 128;
  if ( v13 )
  {
    v13 -= 24;
    v14 = v13;
  }
  else
  {
    v14 = 0;
  }
  v15 = 0;
  v179[0] = v14;
  v16 = (const __m128i *)v179;
  LODWORD(v178) = 1;
  v174 = 1;
  v170 = 1;
  v176[0] = v13;
  v17 = 1;
  do
  {
    v18 = v16->m128i_i64[v17 - 1];
    LODWORD(v178) = v17 - 1;
    v19 = *(_QWORD **)(v18 + 48);
    v20 = (_QWORD *)(v18 + 40);
    v153 = v18;
    if ( v19 == (_QWORD *)(v18 + 40) )
      goto LABEL_23;
    while ( 1 )
    {
      if ( !v19 )
LABEL_218:
        BUG();
      v25 = *((_BYTE *)v19 - 8);
      v26 = (__int64)(v19 - 3);
      if ( v25 == 78 )
        break;
      if ( v25 == 55 && (*((_BYTE *)v19 - 6) & 1) == 0 )
      {
        v27 = (_BYTE *)*(v19 - 6);
        v28 = v27[16];
        if ( v28 == 9 )
          goto LABEL_22;
        if ( v28 == 15 )
        {
          v29 = *(_QWORD *)v27;
          if ( *(_BYTE *)(*(_QWORD *)v27 + 8LL) == 16 )
            v29 = **(_QWORD **)(v29 + 16);
          v30 = *(_DWORD *)(v29 + 8);
          v31 = sub_15F2060((__int64)(v19 - 3));
          v18 = v30 >> 8;
          v32 = sub_15E4690(v31, v18);
          v26 = (__int64)(v19 - 3);
          if ( !v32 )
          {
LABEL_22:
            sub_1AEE6A0(v26, 1, 0, a3, a5, a6, a7, a8, a9, a10, a11, a12);
            v15 = 1;
            goto LABEL_23;
          }
        }
      }
LABEL_12:
      v19 = (_QWORD *)v19[1];
      if ( v20 == v19 )
        goto LABEL_23;
    }
    v21 = *(v19 - 6);
    v22 = *(_BYTE *)(v21 + 16);
    if ( v22 )
    {
      if ( v22 == 15 )
      {
        v84 = sub_15F2060((__int64)(v19 - 3));
        v85 = sub_15E4690(v84, 0);
        v26 = (__int64)(v19 - 3);
        if ( !v85 )
          goto LABEL_76;
        v22 = *(_BYTE *)(v21 + 16);
      }
      if ( v22 == 9 )
        goto LABEL_76;
      goto LABEL_9;
    }
    v23 = *(_DWORD *)(v21 + 36);
    if ( v23 == 4 )
    {
      v63 = *((_DWORD *)v19 - 1) & 0xFFFFFFF;
      v64 = 4 * v63;
      v65 = -3 * v63;
      v66 = (_BYTE *)v19[v65 - 3];
      if ( v66[16] > 0x10u )
        goto LABEL_9;
      v67 = sub_1593BB0(v19[v65 - 3], v18, v64, (__int64)a4);
      v26 = (__int64)(v19 - 3);
      if ( v67 )
        goto LABEL_76;
      v69 = v66[16];
      v70 = v69;
      if ( v69 == 13 )
      {
        if ( *((_DWORD *)v66 + 8) <= 0x40u )
        {
          if ( !*((_QWORD *)v66 + 3) )
            goto LABEL_76;
          goto LABEL_85;
        }
        v148 = v19 - 3;
        v71 = (__int64)(v66 + 24);
        v157 = *((_DWORD *)v66 + 8);
      }
      else
      {
        if ( *(_BYTE *)(*(_QWORD *)v66 + 8LL) != 16 )
        {
LABEL_86:
          if ( v70 == 9 )
            goto LABEL_76;
          goto LABEL_9;
        }
        v99 = sub_15A1020(v66, v18, v68, v69);
        v26 = (__int64)(v19 - 3);
        if ( !v99 || *(_BYTE *)(v99 + 16) != 13 )
        {
          v162 = *(_QWORD *)(*(_QWORD *)v66 + 32LL);
          if ( !v162 )
          {
LABEL_76:
            sub_1AEE6A0(v26, 0, 0, a3, a5, a6, a7, a8, a9, a10, a11, a12);
            v15 = 1;
            goto LABEL_23;
          }
          v119 = 0;
          while ( 1 )
          {
            v145 = v26;
            v150 = v119;
            v120 = sub_15A0A60((__int64)v66, v119);
            v121 = v150;
            v26 = v145;
            if ( !v120 )
              break;
            v122 = *(_BYTE *)(v120 + 16);
            if ( v122 != 9 )
            {
              if ( v122 != 13 )
                break;
              if ( *(_DWORD *)(v120 + 32) <= 0x40u )
              {
                v124 = *(_QWORD *)(v120 + 24) == 0;
              }
              else
              {
                v144 = v145;
                v146 = v150;
                v151 = *(_DWORD *)(v120 + 32);
                v123 = sub_16A57B0(v120 + 24);
                v121 = v146;
                v26 = v144;
                v124 = v151 == v123;
              }
              if ( !v124 )
                break;
            }
            v119 = v121 + 1;
            if ( v162 == v119 )
              goto LABEL_76;
          }
          v70 = v66[16];
          goto LABEL_86;
        }
        if ( *(_DWORD *)(v99 + 32) <= 0x40u )
        {
          v73 = *(_QWORD *)(v99 + 24) == 0;
          goto LABEL_83;
        }
        v148 = v19 - 3;
        v71 = v99 + 24;
        v157 = *(_DWORD *)(v99 + 32);
      }
      v72 = sub_16A57B0(v71);
      v26 = (__int64)v148;
      v73 = v157 == v72;
LABEL_83:
      if ( v73 )
        goto LABEL_76;
      v69 = v66[16];
LABEL_85:
      v70 = v69;
      goto LABEL_86;
    }
    if ( v23 != 79 )
      goto LABEL_9;
    v74 = *((_DWORD *)v19 - 1) & 0xFFFFFFF;
    v75 = 4 * v74;
    v76 = -3 * v74;
    v77 = (_BYTE *)v19[v76 - 3];
    if ( v77[16] > 0x10u )
      goto LABEL_9;
    if ( !sub_1593BB0(v19[v76 - 3], v18, v75, (__int64)a4) )
    {
      if ( v77[16] == 13 )
      {
        if ( *((_DWORD *)v77 + 8) <= 0x40u )
        {
          v86 = *((_QWORD *)v77 + 3) == 0;
        }
        else
        {
          v158 = *((_DWORD *)v77 + 8);
          v86 = v158 == (unsigned int)sub_16A57B0((__int64)(v77 + 24));
        }
      }
      else
      {
        if ( *(_BYTE *)(*(_QWORD *)v77 + 8LL) != 16 )
          goto LABEL_9;
        v110 = sub_15A1020(v77, v18, v78, v79);
        if ( !v110 || *(_BYTE *)(v110 + 16) != 13 )
        {
          v163 = *(_QWORD *)(*(_QWORD *)v77 + 32LL);
          if ( v163 )
          {
            v152 = (__int64)v77;
            v125 = 0;
            while ( 1 )
            {
              v126 = sub_15A0A60(v152, v125);
              if ( !v126 )
                goto LABEL_9;
              v127 = *(_BYTE *)(v126 + 16);
              if ( v127 != 9 )
              {
                if ( v127 != 13 )
                  goto LABEL_9;
                if ( *(_DWORD *)(v126 + 32) <= 0x40u )
                {
                  v128 = *(_QWORD *)(v126 + 24) == 0;
                }
                else
                {
                  v147 = *(_DWORD *)(v126 + 32);
                  v128 = v147 == (unsigned int)sub_16A57B0(v126 + 24);
                }
                if ( !v128 )
                  goto LABEL_9;
              }
              if ( v163 == ++v125 )
                goto LABEL_90;
            }
          }
          goto LABEL_90;
        }
        v111 = *(_DWORD *)(v110 + 32);
        if ( v111 <= 0x40 )
          v86 = *(_QWORD *)(v110 + 24) == 0;
        else
          v86 = v111 == (unsigned int)sub_16A57B0(v110 + 24);
      }
      if ( !v86 )
        goto LABEL_9;
    }
LABEL_90:
    v82 = v19[1];
    v83 = v19[2] + 40LL;
    LOBYTE(v83) = v82 != 0 && v82 != v83;
    if ( !(_BYTE)v83 )
      goto LABEL_218;
    if ( *(_BYTE *)(v82 - 8) != 31 )
    {
      v15 = v83;
      sub_1AEE6A0(v82 - 24, 0, 0, a3, a5, a6, a7, a8, v80, v81, a11, a12);
      goto LABEL_23;
    }
LABEL_9:
    v18 = 0xFFFFFFFFLL;
    if ( !(unsigned __int8)sub_1560260(v19 + 4, -1, 29) )
    {
      v24 = *(v19 - 6);
      if ( *(_BYTE *)(v24 + 16) )
        goto LABEL_12;
      v18 = 0xFFFFFFFFLL;
      v167.m128i_i64[0] = *(_QWORD *)(v24 + 112);
      if ( !(unsigned __int8)sub_1560260(&v167, -1, 29) )
        goto LABEL_12;
    }
    v62 = v19[1];
    if ( v62 == v19[2] + 40LL || !v62 )
      BUG();
    if ( *(_BYTE *)(v62 - 8) != 31 )
    {
      sub_1AEE6A0(v62 - 24, 0, 0, a3, a5, a6, a7, a8, a9, a10, a11, a12);
      v15 = 1;
    }
LABEL_23:
    v33 = sub_157EBA0(v153);
    v36 = *(_BYTE *)(v33 + 16);
    if ( v36 != 29 )
    {
      if ( v36 != 34 )
        goto LABEL_29;
      v87 = &v168;
      v167.m128i_i64[0] = 0;
      v167.m128i_i64[1] = 1;
      do
        *v87++ = -8;
      while ( v87 != &v170 );
      v88 = *(_WORD *)(v33 + 18);
      v89 = 24LL * (*(_DWORD *)(v33 + 20) & 0xFFFFFFF);
      if ( (*(_BYTE *)(v33 + 23) & 0x40) != 0 )
      {
        v90 = *(_QWORD *)(v33 - 8);
        v91 = (__int64 *)(v90 + v89);
        if ( (v88 & 1) != 0 )
          goto LABEL_108;
LABEL_164:
        v92 = (__int64 *)(v90 + 24);
      }
      else
      {
        v91 = (__int64 *)v33;
        v90 = v33 - v89;
        if ( (v88 & 1) == 0 )
          goto LABEL_164;
LABEL_108:
        v92 = (__int64 *)(v90 + 48);
      }
      if ( v92 == v91 )
      {
LABEL_120:
        if ( (v167.m128i_i8[8] & 1) == 0 )
          j___libc_free_0(v168);
        goto LABEL_29;
      }
      v143 = v33;
      v93 = v92;
      v149 = v15;
      while ( 2 )
      {
        while ( 1 )
        {
          v94 = sub_15A5110(*v93);
          v166[0] = (__int64 *)sub_157ED20(v94);
          v95 = sub_1AED800((__int64)&v167, v166, &v165);
          v96 = v165;
          v159 = v95;
          if ( !v95 )
            break;
          v91 -= 3;
          sub_15F7E90(v143, v93);
          v149 = v159;
          if ( v93 == v91 )
            goto LABEL_119;
        }
        ++v167.m128i_i64[0];
        v97 = ((unsigned __int32)v167.m128i_i32[2] >> 1) + 1;
        if ( (v167.m128i_i8[8] & 1) == 0 )
        {
          v98 = v169;
          if ( 3 * v169 > 4 * v97 )
            goto LABEL_115;
LABEL_150:
          v98 *= 2;
          goto LABEL_151;
        }
        v98 = 4;
        if ( 4 * v97 >= 0xC )
          goto LABEL_150;
LABEL_115:
        if ( v98 - (v97 + v167.m128i_i32[3]) <= v98 >> 3 )
        {
LABEL_151:
          sub_1AED960((__int64)&v167, v98);
          sub_1AED800((__int64)&v167, v166, &v165);
          v96 = v165;
          v97 = ((unsigned __int32)v167.m128i_i32[2] >> 1) + 1;
        }
        v167.m128i_i32[2] = v167.m128i_i8[8] & 1 | (2 * v97);
        if ( *v96 != -8 )
          --v167.m128i_i32[3];
        v93 += 3;
        *v96 = (__int64)v166[0];
        if ( v93 == v91 )
        {
LABEL_119:
          v15 = v149;
          goto LABEL_120;
        }
        continue;
      }
    }
    v37 = *(_QWORD *)(v33 - 72);
    v38 = *(_BYTE *)(v37 + 16);
    if ( v38 != 15 )
      goto LABEL_25;
    if ( !sub_15E4690(*(_QWORD *)(v153 + 56), 0) )
      goto LABEL_127;
    v38 = *(_BYTE *)(v37 + 16);
LABEL_25:
    if ( v38 == 9 )
    {
LABEL_127:
      sub_1AEE6A0(v33, 1, 0, a3, a5, a6, a7, a8, v34, v35, a11, a12);
      v15 = 1;
      goto LABEL_29;
    }
    if ( (unsigned __int8)sub_1560260((_QWORD *)(v33 + 56), -1, 30)
      || (v39 = *(_QWORD *)(v33 - 72), !*(_BYTE *)(v39 + 16))
      && (v167.m128i_i64[0] = *(_QWORD *)(v39 + 112), (unsigned __int8)sub_1560260(&v167, -1, 30)) )
    {
      v161 = sub_14DDD80(a1);
      if ( v161 )
      {
        v118 = *(_QWORD *)(v33 + 8);
        if ( v118 )
          goto LABEL_160;
        if ( !(unsigned __int8)sub_1560260((_QWORD *)(v33 + 56), -1, 36) )
        {
          if ( *(char *)(v33 + 23) < 0 )
          {
            v129 = sub_1648A40(v33);
            v131 = v129 + v130;
            v132 = 0;
            if ( *(char *)(v33 + 23) < 0 )
              v132 = sub_1648A40(v33);
            if ( (unsigned int)((v131 - v132) >> 4) )
              goto LABEL_221;
          }
          v133 = *(_QWORD *)(v33 - 72);
          if ( *(_BYTE *)(v133 + 16)
            || (v167.m128i_i64[0] = *(_QWORD *)(v133 + 112), !(unsigned __int8)sub_1560260(&v167, -1, 36)) )
          {
LABEL_221:
            if ( !(unsigned __int8)sub_1560260((_QWORD *)(v33 + 56), -1, 37) )
            {
              if ( *(char *)(v33 + 23) < 0 )
              {
                v134 = sub_1648A40(v33);
                v135 = v134 + v112;
                if ( *(char *)(v33 + 23) < 0 )
                  v118 = sub_1648A40(v33);
                if ( v118 != v135 )
                {
                  while ( *(_DWORD *)(*(_QWORD *)v118 + 8LL) <= 1u )
                  {
                    v118 += 16;
                    if ( v135 == v118 )
                      goto LABEL_206;
                  }
LABEL_160:
                  sub_1AF0320(v33, a3, a5, a6, a7, a8, v116, v117, a11, a12, v112, v113, v114, v115);
                  v15 = v161;
                  goto LABEL_29;
                }
              }
LABEL_206:
              v136 = *(_QWORD *)(v33 - 72);
              if ( *(_BYTE *)(v136 + 16) )
                goto LABEL_160;
              v167.m128i_i64[0] = *(_QWORD *)(v136 + 112);
              if ( !(unsigned __int8)sub_1560260(&v167, -1, 37) )
                goto LABEL_160;
            }
          }
        }
        v137 = *(_QWORD *)(v33 - 48);
        v138 = *(_QWORD *)(v33 - 24);
        v139 = sub_1648A60(56, 1u);
        if ( v139 )
          sub_15F8320((__int64)v139, v137, v33);
        sub_157F2D0(v138, *(_QWORD *)(v33 + 40), 0);
        sub_15F20C0((_QWORD *)v33);
        v15 = v161;
        if ( a3 )
          sub_15CDBF0(a3, v153, v138);
      }
    }
LABEL_29:
    v15 |= sub_1AEE9C0(v153, 1u, 0, a3);
    v40 = sub_157EBA0(v153);
    if ( v40 )
    {
      v41 = sub_15F4D60(v40);
      v42 = sub_157EBA0(v153);
      if ( v41 )
      {
        v155 = a3;
        v43 = 0;
        while ( 1 )
        {
          v47 = sub_15F4DF0(v42, v43);
          v48 = v171;
          if ( v172 == v171 )
          {
            v49 = &v171[v174];
            if ( v171 != v49 )
            {
              a4 = 0;
              while ( v47 != *v48 )
              {
                if ( *v48 == -2 )
                  a4 = v48;
                if ( v49 == ++v48 )
                {
                  if ( !a4 )
                    goto LABEL_95;
                  *a4 = v47;
                  --v175;
                  ++v170;
                  goto LABEL_43;
                }
              }
              goto LABEL_33;
            }
LABEL_95:
            if ( v174 < v173 )
              break;
          }
          sub_16CCBA0((__int64)&v170, v47);
          if ( v46 )
          {
LABEL_43:
            v50 = (unsigned int)v178;
            if ( (unsigned int)v178 < HIDWORD(v178) )
              goto LABEL_44;
LABEL_97:
            sub_16CD150((__int64)&v177, v179, 0, 8, v44, v45);
            v50 = (unsigned int)v178;
LABEL_44:
            ++v43;
            v177->m128i_i64[v50] = v47;
            LODWORD(v178) = (_DWORD)v178 + 1;
            if ( v43 == v41 )
            {
LABEL_45:
              a3 = v155;
              goto LABEL_46;
            }
            continue;
          }
LABEL_33:
          if ( ++v43 == v41 )
            goto LABEL_45;
        }
        ++v174;
        *v49 = v47;
        v50 = (unsigned int)v178;
        ++v170;
        if ( (unsigned int)v178 >= HIDWORD(v178) )
          goto LABEL_97;
        goto LABEL_44;
      }
    }
LABEL_46:
    v17 = (unsigned int)v178;
    v16 = v177;
  }
  while ( (_DWORD)v178 );
  if ( v177 != (__m128i *)v179 )
    _libc_free((unsigned __int64)v177);
  v51 = *(_QWORD *)(a1 + 80);
  v154 = a1 + 72;
  if ( v51 == a1 + 72 )
  {
    v53 = 0;
  }
  else
  {
    v52 = *(_QWORD *)(a1 + 80);
    v53 = 0;
    do
    {
      v52 = *(_QWORD *)(v52 + 8);
      ++v53;
    }
    while ( a1 + 72 != v52 );
  }
  if ( v174 - v175 != v53 )
  {
    v177 = 0;
    v178 = 0;
    v179[0] = 0;
    v54 = *(_QWORD *)(v51 + 8);
    if ( v154 == v54 )
    {
      v108 = 0;
      if ( a3 )
        goto LABEL_142;
    }
    else
    {
      v164 = a3;
      do
      {
        while ( 1 )
        {
          v55 = v54 - 24;
          if ( !v54 )
            v55 = 0;
          if ( !sub_183E920((__int64)&v170, v55) )
            break;
          v54 = *(_QWORD *)(v54 + 8);
          if ( v154 == v54 )
            goto LABEL_132;
        }
        v56 = sub_157EBA0(v55);
        if ( v56 )
        {
          v156 = sub_15F4D60(v56);
          v57 = sub_157EBA0(v55);
          if ( v156 )
          {
            v142 = v54;
            v58 = 0;
            v59 = v57;
            do
            {
              v61 = sub_15F4DF0(v59, v58);
              if ( sub_183E920((__int64)&v170, v61) )
                sub_157F2D0(v61, v55, 0);
              if ( v164 )
              {
                v167.m128i_i64[0] = v55;
                v60 = v178;
                v167.m128i_i64[1] = v61 | 4;
                if ( v178 == (__m128i *)v179[0] )
                {
                  sub_17F2860((const __m128i **)&v177, v178, &v167);
                }
                else
                {
                  if ( v178 )
                  {
                    *v178 = _mm_loadu_si128(&v167);
                    v60 = v178;
                  }
                  v178 = (__m128i *)&v60[1];
                }
              }
              ++v58;
            }
            while ( v156 != v58 );
            v54 = v142;
          }
        }
        if ( a2 )
          sub_13EB690(a2, v55);
        sub_157EE90(v55);
        v54 = *(_QWORD *)(v54 + 8);
      }
      while ( v154 != v54 );
LABEL_132:
      v160 = v54;
      a3 = v164;
      v100 = *(_QWORD *)(a1 + 80);
      if ( v154 != *(_QWORD *)(v100 + 8) )
      {
        v101 = *(_QWORD **)(v100 + 8);
        while ( 1 )
        {
          v102 = v101 - 3;
          v103 = 0;
          if ( v101 )
            v103 = (__int64)(v101 - 3);
          if ( sub_183E920((__int64)&v170, v103) )
            goto LABEL_135;
          if ( v164 )
          {
            sub_15CD5A0(v164, v103);
LABEL_135:
            v101 = (_QWORD *)v101[1];
            if ( (_QWORD *)v160 == v101 )
              break;
          }
          else
          {
            v104 = (_QWORD *)v101[1];
            sub_15E0220(v160, (__int64)(v101 - 3));
            v105 = (unsigned __int64 *)v101[1];
            v106 = (__int64)(v101 - 3);
            v107 = *v101 & 0xFFFFFFFFFFFFFFF8LL;
            *v105 = v107 | *v105 & 7;
            *(_QWORD *)(v107 + 8) = v105;
            *v101 &= 7uLL;
            v101[1] = 0;
            v101 = v104;
            sub_157EF40(v106);
            j_j___libc_free_0(v102, 64);
            if ( (_QWORD *)v160 == v104 )
              break;
          }
        }
      }
      v108 = v177;
      if ( v164 )
      {
LABEL_142:
        sub_15CD9D0(a3, v108->m128i_i64, v178 - v108);
        v108 = v177;
      }
      if ( v108 )
        j_j___libc_free_0(v108, v179[0] - (_QWORD)v108);
    }
    v15 = 1;
  }
  if ( v172 != v171 )
    _libc_free((unsigned __int64)v172);
  return v15;
}
