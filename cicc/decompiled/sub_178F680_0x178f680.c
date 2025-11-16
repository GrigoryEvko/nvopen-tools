// Function: sub_178F680
// Address: 0x178f680
//
__int64 __fastcall sub_178F680(
        __m128i *a1,
        __int64 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10,
        __int64 a11,
        __int64 a12,
        __int64 a13)
{
  __int64 *v13; // r13
  __int64 v14; // r12
  __m128 v15; // xmm0
  __m128i v16; // xmm1
  __int64 v17; // rax
  __int64 v18; // rbx
  __int64 v19; // r13
  __int64 v20; // r15
  _QWORD *v21; // rax
  double v22; // xmm4_8
  double v23; // xmm5_8
  __int64 v25; // rax
  _QWORD *v26; // rdx
  __int64 v27; // rcx
  unsigned __int8 v28; // al
  unsigned __int8 v29; // dl
  __int64 v30; // rax
  __int64 v31; // rax
  unsigned int v32; // eax
  __int64 v33; // rdi
  unsigned int v34; // edx
  __int64 v35; // rsi
  __int64 v36; // r15
  __int64 v37; // rcx
  __int64 v38; // r10
  __int64 v39; // rsi
  __int64 v40; // r15
  __int64 v41; // r11
  __int64 v42; // r8
  __int64 *v43; // r9
  __int64 v44; // rbx
  __int64 v45; // rdx
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rdi
  __int64 v49; // r13
  __int64 v50; // rax
  __int64 v51; // r8
  __int64 v52; // rcx
  __int64 v53; // rdx
  _QWORD *v54; // r14
  __int64 v55; // r8
  unsigned __int64 v56; // rdx
  __int64 v57; // rdx
  __int64 v58; // rcx
  __int64 v59; // rdx
  _QWORD *v60; // rax
  __int64 v61; // rcx
  unsigned __int64 v62; // rdx
  __int64 v63; // rdx
  int v64; // eax
  double v65; // xmm4_8
  double v66; // xmm5_8
  __int64 v67; // rbx
  int v68; // esi
  unsigned __int8 *v69; // rax
  unsigned __int8 *v70; // rcx
  __int64 v71; // rsi
  __int64 v72; // rax
  __int64 v73; // rbx
  _QWORD *v74; // rax
  __int64 v75; // rcx
  __int64 v76; // rdx
  _QWORD *v77; // r15
  _QWORD *v78; // rbx
  __int64 v79; // rax
  unsigned __int64 v80; // r12
  _QWORD *v81; // r15
  char v82; // dl
  _QWORD *v83; // rdx
  unsigned __int64 v84; // r8
  __int64 v85; // rdi
  _QWORD *v86; // rax
  int v87; // eax
  _BYTE *v88; // r15
  __int64 v89; // rdx
  __int64 v90; // rcx
  __int64 v91; // r11
  __int64 v92; // r10
  __int64 v93; // r15
  unsigned __int64 v94; // rax
  __int64 v95; // r11
  __int64 v96; // r10
  char v97; // al
  __int64 v98; // r11
  char v99; // cl
  __int64 v100; // rdx
  __int64 *v101; // r15
  __int64 v102; // rdx
  unsigned __int64 v103; // rcx
  __int64 v104; // rdx
  __int64 v105; // rax
  _QWORD *v106; // rdi
  __int64 v107; // rax
  __int64 v108; // rbx
  __int64 v109; // r14
  __int64 v110; // r13
  _QWORD *v111; // rax
  double v112; // xmm4_8
  double v113; // xmm5_8
  unsigned __int64 v114; // rdi
  __int64 v115; // rbx
  __int64 v116; // r13
  _QWORD *v117; // rax
  double v118; // xmm4_8
  double v119; // xmm5_8
  __int64 v120; // rax
  __int64 v121; // rdx
  __int64 v122; // rsi
  int v123; // eax
  bool v124; // al
  unsigned int v125; // r15d
  __int64 v126; // rax
  __int64 v127; // rbx
  __int64 v128; // r14
  __int64 v129; // r13
  _QWORD *v130; // rax
  double v131; // xmm4_8
  double v132; // xmm5_8
  __int64 v133; // rax
  bool v134; // al
  __int64 v135; // rax
  unsigned int v136; // r15d
  unsigned int v137; // ebx
  __int64 v138; // rax
  char v139; // cl
  bool v140; // al
  __int64 v141; // [rsp+0h] [rbp-110h]
  __int64 v142; // [rsp+8h] [rbp-108h]
  char v143; // [rsp+17h] [rbp-F9h]
  _QWORD *v144; // [rsp+18h] [rbp-F8h]
  __int64 v145; // [rsp+18h] [rbp-F8h]
  __int64 v146; // [rsp+18h] [rbp-F8h]
  int v147; // [rsp+18h] [rbp-F8h]
  __int64 *v148; // [rsp+20h] [rbp-F0h]
  __int64 v149; // [rsp+20h] [rbp-F0h]
  __int64 v150; // [rsp+20h] [rbp-F0h]
  __int64 v151; // [rsp+20h] [rbp-F0h]
  __int64 v152; // [rsp+20h] [rbp-F0h]
  __int64 v153; // [rsp+20h] [rbp-F0h]
  __int64 v154; // [rsp+28h] [rbp-E8h]
  unsigned __int8 v155; // [rsp+28h] [rbp-E8h]
  __int64 v156; // [rsp+28h] [rbp-E8h]
  __int64 v157; // [rsp+28h] [rbp-E8h]
  __int64 v158; // [rsp+28h] [rbp-E8h]
  int v159; // [rsp+28h] [rbp-E8h]
  int v160; // [rsp+28h] [rbp-E8h]
  __m128i v161; // [rsp+30h] [rbp-E0h] BYREF
  __m128i v162; // [rsp+40h] [rbp-D0h]
  __int64 v163; // [rsp+50h] [rbp-C0h]
  _QWORD v164[23]; // [rsp+58h] [rbp-B8h] BYREF

  v13 = (__int64 *)a1;
  v14 = a2;
  v15 = (__m128)_mm_loadu_si128(a1 + 167);
  v16 = _mm_loadu_si128(a1 + 168);
  v163 = a2;
  v161 = (__m128i)v15;
  v162 = v16;
  v17 = sub_13E3350(a2, &v161, 0, 1, a13);
  if ( v17 )
  {
    v18 = *(_QWORD *)(a2 + 8);
    if ( v18 )
    {
      v19 = a1->m128i_i64[0];
      v20 = v17;
      do
      {
        v21 = sub_1648700(v18);
        sub_170B990(v19, (__int64)v21);
        v18 = *(_QWORD *)(v18 + 8);
      }
      while ( v18 );
      if ( a2 == v20 )
        v20 = sub_1599EF0(*(__int64 ***)a2);
      sub_164D160(a2, v20, v15, *(double *)v16.m128i_i64, a5, a6, v22, v23, a9, a10);
      return v14;
    }
    return 0;
  }
  v25 = (__int64)sub_178A520(a1->m128i_i64, a2);
  if ( v25 )
    return v25;
  v26 = (*(_BYTE *)(a2 + 23) & 0x40) != 0
      ? *(_QWORD **)(a2 - 8)
      : (_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v27 = *v26;
  v28 = *(_BYTE *)(*v26 + 16LL);
  if ( v28 > 0x17u )
  {
    v29 = *(_BYTE *)(v26[3] + 16LL);
    if ( v29 == v28 && v29 > 0x17u )
    {
      v30 = *(_QWORD *)(v27 + 8);
      if ( v30 )
      {
        if ( !*(_QWORD *)(v30 + 8) )
        {
          v25 = sub_178C0A0(a1->m128i_i64, a2);
          if ( v25 )
            return v25;
        }
      }
    }
  }
  v31 = *(_QWORD *)(a2 + 8);
  if ( v31 && !*(_QWORD *)(v31 + 8) )
  {
    v72 = sub_178CF50(a1->m128i_i64, a2);
    v73 = v72;
    if ( v72 )
      return v72;
    v74 = sub_1648700(*(_QWORD *)(a2 + 8));
    v76 = *((unsigned __int8 *)v74 + 16);
    v77 = v74;
    if ( (_BYTE)v76 == 77 )
    {
      LODWORD(v163) = 0;
      v161.m128i_i64[1] = (__int64)v164;
      v162.m128i_i64[0] = (__int64)v164;
      v162.m128i_i64[1] = 0x100000010LL;
      v164[0] = a2;
      v161.m128i_i64[0] = 1;
      v144 = v74;
      v154 = a2;
      v78 = v74;
      while ( 1 )
      {
        v79 = v78[1];
        if ( !v79 )
          break;
        a2 = *(_QWORD *)(v79 + 8);
        v80 = v162.m128i_i64[0];
        v81 = (_QWORD *)v161.m128i_i64[1];
        if ( a2 )
          goto LABEL_95;
        if ( v162.m128i_i64[0] != v161.m128i_i64[1] )
          goto LABEL_92;
        v106 = (_QWORD *)(v161.m128i_i64[1] + 8LL * v162.m128i_u32[3]);
        if ( v106 != (_QWORD *)v161.m128i_i64[1] )
        {
          while ( v78 != (_QWORD *)*v81 )
          {
            if ( *v81 == -2 )
              a2 = (__int64)v81;
            if ( v106 == ++v81 )
            {
              if ( !a2 )
                goto LABEL_151;
              *(_QWORD *)a2 = v78;
              v80 = v162.m128i_i64[0];
              LODWORD(v163) = v163 - 1;
              v81 = (_QWORD *)v161.m128i_i64[1];
              ++v161.m128i_i64[0];
              goto LABEL_93;
            }
          }
          break;
        }
LABEL_151:
        if ( v162.m128i_i32[3] < (unsigned __int32)v162.m128i_i32[2] )
        {
          ++v162.m128i_i32[3];
          *v106 = v78;
          v81 = (_QWORD *)v161.m128i_i64[1];
          ++v161.m128i_i64[0];
          v80 = v162.m128i_i64[0];
        }
        else
        {
LABEL_92:
          a2 = (__int64)v78;
          sub_16CCBA0((__int64)&v161, (__int64)v78);
          v80 = v162.m128i_i64[0];
          v81 = (_QWORD *)v161.m128i_i64[1];
          if ( !v82 )
            break;
        }
LABEL_93:
        if ( v162.m128i_i32[3] - (_DWORD)v163 != 16 )
        {
          v78 = sub_1648700(v78[1]);
          if ( *((_BYTE *)v78 + 16) == 77 )
            continue;
        }
LABEL_95:
        v83 = v81;
        v84 = v80;
        v73 = 0;
        v77 = v144;
        v14 = v154;
        if ( (_QWORD *)v84 != v83 )
          _libc_free(v84);
        v76 = *((unsigned __int8 *)v144 + 16);
        goto LABEL_98;
      }
      v14 = v154;
      v107 = sub_1599EF0(*(__int64 ***)v154);
      v108 = *(_QWORD *)(v154 + 8);
      v109 = v107;
      if ( v108 )
      {
        v110 = *v13;
        do
        {
          v111 = sub_1648700(v108);
          sub_170B990(v110, (__int64)v111);
          v108 = *(_QWORD *)(v108 + 8);
        }
        while ( v108 );
        if ( v154 == v109 )
          v109 = sub_1599EF0(*(__int64 ***)v154);
        sub_164D160(v154, v109, v15, *(double *)v16.m128i_i64, a5, a6, v112, v113, a9, a10);
      }
      else
      {
        v14 = 0;
      }
      v114 = v162.m128i_i64[0];
      if ( v161.m128i_i64[1] == v162.m128i_i64[0] )
        return v14;
      goto LABEL_139;
    }
LABEL_98:
    v85 = v77[1];
    if ( v85 && !*(_QWORD *)(v85 + 8) && ((unsigned int)(unsigned __int8)v76 - 35 <= 0x11 || (_BYTE)v76 == 56) )
    {
      v155 = v76;
      v86 = sub_1648700(v85);
      v76 = v155;
      if ( (_QWORD *)v14 == v86 )
      {
        v126 = sub_1599EF0(*(__int64 ***)v14);
        v127 = *(_QWORD *)(v14 + 8);
        v128 = v126;
        if ( v127 )
        {
          v129 = *v13;
          do
          {
            v130 = sub_1648700(v127);
            sub_170B990(v129, (__int64)v130);
            v127 = *(_QWORD *)(v127 + 8);
          }
          while ( v127 );
          if ( v14 == v128 )
            v128 = sub_1599EF0(*(__int64 ***)v14);
          sub_164D160(v14, v128, v15, *(double *)v16.m128i_i64, a5, a6, v131, v132, a9, a10);
          return v14;
        }
        return 0;
      }
    }
    if ( (_BYTE)v76 == 75 && *(_BYTE *)(*(_QWORD *)v14 + 8LL) == 11 )
    {
      v87 = *((unsigned __int16 *)v77 + 9);
      BYTE1(v87) &= ~0x80u;
      if ( (unsigned int)(v87 - 32) <= 1 )
      {
        v88 = (_BYTE *)*(v77 - 3);
        if ( v88[16] <= 0x10u )
        {
          if ( sub_1593BB0((__int64)v88, a2, v76, v75) )
          {
LABEL_108:
            v91 = 0;
            v92 = 8LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF);
            if ( (*(_DWORD *)(v14 + 20) & 0xFFFFFFF) == 0 )
              goto LABEL_29;
            do
            {
              v150 = v92;
              v105 = 24LL * *(unsigned int *)(v14 + 56);
              if ( (*(_BYTE *)(v14 + 23) & 0x40) != 0 )
              {
                v93 = *(_QWORD *)(v14 - 8);
                v156 = v91;
                v94 = sub_157EBA0(*(_QWORD *)(v91 + v93 + v105 + 8));
                v95 = v156;
                v96 = v150;
              }
              else
              {
                v158 = v91;
                v93 = v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF);
                v94 = sub_157EBA0(*(_QWORD *)(v91 + v93 + v105 + 8));
                v96 = v150;
                v95 = v158;
              }
              v145 = v96;
              v149 = v95;
              v157 = 3 * v95;
              v97 = sub_14BFF20(*(_QWORD *)(v93 + 3 * v95), v13[333], 0, v13[330], v94, v13[332]);
              v98 = v149;
              v92 = v145;
              if ( v97 )
              {
                if ( v73 )
                {
                  v99 = *(_BYTE *)(v14 + 23);
                }
                else
                {
                  v99 = *(_BYTE *)(v14 + 23);
                  v120 = 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF);
                  v121 = v14 - v120;
                  if ( (v99 & 0x40) != 0 )
                    v121 = *(_QWORD *)(v14 - 8);
                  v122 = v121 + v120;
                  if ( v121 + v120 == v121 )
                  {
LABEL_171:
                    v146 = v98;
                    v152 = v92;
                    v133 = sub_159C470(*(_QWORD *)v14, 1, 0);
                    v99 = *(_BYTE *)(v14 + 23);
                    v98 = v146;
                    v92 = v152;
                    v73 = v133;
                  }
                  else
                  {
                    while ( 1 )
                    {
                      v73 = *(_QWORD *)v121;
                      if ( *(_BYTE *)(*(_QWORD *)v121 + 16LL) == 13 )
                      {
                        v125 = *(_DWORD *)(v73 + 32);
                        if ( v125 > 0x40 )
                        {
                          v141 = v98;
                          v142 = v92;
                          v143 = v99;
                          v151 = v121;
                          v123 = sub_16A57B0(v73 + 24);
                          v121 = v151;
                          v99 = v143;
                          v92 = v142;
                          v98 = v141;
                          v124 = v125 == v123;
                        }
                        else
                        {
                          v124 = *(_QWORD *)(v73 + 24) == 0;
                        }
                        if ( !v124 )
                          break;
                      }
                      v121 += 24;
                      if ( v122 == v121 )
                        goto LABEL_171;
                    }
                  }
                }
                if ( (v99 & 0x40) != 0 )
                  v100 = *(_QWORD *)(v14 - 8);
                else
                  v100 = v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF);
                v101 = (__int64 *)(v100 + v157);
                if ( *(_QWORD *)(v100 + v157) )
                {
                  v102 = v101[1];
                  v103 = v101[2] & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v103 = v102;
                  if ( v102 )
                    *(_QWORD *)(v102 + 16) = v103 | *(_QWORD *)(v102 + 16) & 3LL;
                }
                *v101 = v73;
                if ( v73 )
                {
                  v104 = *(_QWORD *)(v73 + 8);
                  v101[1] = v104;
                  if ( v104 )
                    *(_QWORD *)(v104 + 16) = (unsigned __int64)(v101 + 1) | *(_QWORD *)(v104 + 16) & 3LL;
                  v101[2] = (v73 + 8) | v101[2] & 3;
                  *(_QWORD *)(v73 + 8) = v101;
                }
              }
              v91 = v98 + 8;
            }
            while ( v92 != v91 );
            goto LABEL_21;
          }
          if ( v88[16] == 13 )
          {
            if ( *((_DWORD *)v88 + 8) <= 0x40u )
            {
              v134 = *((_QWORD *)v88 + 3) == 0;
            }
            else
            {
              v159 = *((_DWORD *)v88 + 8);
              v134 = v159 == (unsigned int)sub_16A57B0((__int64)(v88 + 24));
            }
          }
          else
          {
            if ( *(_BYTE *)(*(_QWORD *)v88 + 8LL) != 16 )
              goto LABEL_21;
            v135 = sub_15A1020(v88, a2, v89, v90);
            if ( !v135 || *(_BYTE *)(v135 + 16) != 13 )
            {
              v153 = v73;
              v137 = 0;
              v160 = *(_DWORD *)(*(_QWORD *)v88 + 32LL);
              while ( v160 != v137 )
              {
                v138 = sub_15A0A60((__int64)v88, v137);
                if ( !v138 )
                  goto LABEL_21;
                v139 = *(_BYTE *)(v138 + 16);
                if ( v139 != 9 )
                {
                  if ( v139 != 13 )
                    goto LABEL_21;
                  if ( *(_DWORD *)(v138 + 32) <= 0x40u )
                  {
                    v140 = *(_QWORD *)(v138 + 24) == 0;
                  }
                  else
                  {
                    v147 = *(_DWORD *)(v138 + 32);
                    v140 = v147 == (unsigned int)sub_16A57B0(v138 + 24);
                  }
                  if ( !v140 )
                    goto LABEL_21;
                }
                ++v137;
              }
              v73 = v153;
              goto LABEL_108;
            }
            v136 = *(_DWORD *)(v135 + 32);
            if ( v136 <= 0x40 )
              v134 = *(_QWORD *)(v135 + 24) == 0;
            else
              v134 = v136 == (unsigned int)sub_16A57B0(v135 + 24);
          }
          if ( !v134 )
            goto LABEL_21;
          goto LABEL_108;
        }
      }
    }
  }
LABEL_21:
  v32 = *(_DWORD *)(v14 + 20) & 0xFFFFFFF;
  if ( !v32 )
    goto LABEL_29;
  v33 = 0;
  v34 = 0;
  while ( 1 )
  {
    v35 = v14 - 24LL * v32;
    if ( (*(_BYTE *)(v14 + 23) & 0x40) != 0 )
      v35 = *(_QWORD *)(v14 - 8);
    v36 = *(_QWORD *)(v35 + v33);
    ++v34;
    if ( *(_BYTE *)(v36 + 16) != 77 )
      break;
    v33 += 24;
    if ( v32 == v34 )
      goto LABEL_29;
  }
  if ( v32 == v34 )
  {
LABEL_79:
    v71 = *(_QWORD *)(v35 + v33);
    v161.m128i_i64[0] = 0;
    v161.m128i_i64[1] = (__int64)v164;
    v162.m128i_i64[0] = (__int64)v164;
    v162.m128i_i64[1] = 16;
    LODWORD(v163) = 0;
    if ( !(unsigned __int8)sub_1789180(v14, v71, (__int64)&v161) )
    {
      if ( v162.m128i_i64[0] != v161.m128i_i64[1] )
        _libc_free(v162.m128i_u64[0]);
      goto LABEL_29;
    }
    v115 = *(_QWORD *)(v14 + 8);
    if ( v115 )
    {
      v116 = *v13;
      do
      {
        v117 = sub_1648700(v115);
        sub_170B990(v116, (__int64)v117);
        v115 = *(_QWORD *)(v115 + 8);
      }
      while ( v115 );
      if ( v14 == v36 )
        v36 = sub_1599EF0(*(__int64 ***)v14);
      sub_164D160(v14, v36, v15, *(double *)v16.m128i_i64, a5, a6, v118, v119, a9, a10);
    }
    else
    {
      v14 = 0;
    }
    v114 = v162.m128i_i64[0];
    if ( v162.m128i_i64[0] == v161.m128i_i64[1] )
      return v14;
LABEL_139:
    _libc_free(v114);
    return v14;
  }
  while ( 1 )
  {
    v37 = *(_QWORD *)(v35 + 24LL * v34);
    if ( v36 != v37 && *(_BYTE *)(v37 + 16) != 77 )
      break;
    if ( ++v34 == v32 )
      goto LABEL_79;
  }
LABEL_29:
  v38 = *(_QWORD *)(*(_QWORD *)(v14 + 40) + 48LL);
  if ( !v38 )
    BUG();
  if ( v14 != v38 - 24 && (*(_DWORD *)(v38 - 4) & 0xFFFFFFF) != 0 )
  {
    v148 = v13;
    v39 = 0;
    v40 = 8LL * (*(_DWORD *)(v38 - 4) & 0xFFFFFFF);
    do
    {
      if ( (*(_BYTE *)(v14 + 23) & 0x40) != 0 )
        v41 = *(_QWORD *)(v14 - 8);
      else
        v41 = v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF);
      v42 = 24LL * *(unsigned int *)(v14 + 56);
      v43 = (__int64 *)(v41 + v39 + v42 + 8);
      v44 = *v43;
      if ( (*(_BYTE *)(v38 - 1) & 0x40) != 0 )
        v45 = *(_QWORD *)(v38 - 32);
      else
        v45 = v38 - 24 - 24LL * (*(_DWORD *)(v38 - 4) & 0xFFFFFFF);
      v46 = *(_QWORD *)(v39 + v45 + 24LL * *(unsigned int *)(v38 + 32) + 8);
      if ( v46 != v44 )
      {
        v47 = 0x17FFFFFFE8LL;
        v48 = 0x7FFFFFFF8LL;
        v49 = *(_QWORD *)(v41 + 3 * v39);
        if ( (*(_DWORD *)(v14 + 20) & 0xFFFFFFF) != 0 )
        {
          v50 = 0;
          v51 = v41 + v42;
          do
          {
            v48 = 8 * v50;
            if ( v46 == *(_QWORD *)(v51 + 8 * v50 + 8) )
            {
              v47 = 24 * v50;
              goto LABEL_43;
            }
            ++v50;
          }
          while ( (*(_DWORD *)(v14 + 20) & 0xFFFFFFF) != (_DWORD)v50 );
          v47 = 0x17FFFFFFE8LL;
          v48 = 0x7FFFFFFF8LL;
        }
LABEL_43:
        v52 = *(_QWORD *)(v41 + v47);
        *v43 = v46;
        if ( (*(_BYTE *)(v14 + 23) & 0x40) != 0 )
          v53 = *(_QWORD *)(v14 - 8);
        else
          v53 = v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF);
        v54 = (_QWORD *)(v53 + 3 * v39);
        if ( *v54 )
        {
          v55 = v54[1];
          v56 = v54[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v56 = v55;
          if ( v55 )
            *(_QWORD *)(v55 + 16) = *(_QWORD *)(v55 + 16) & 3LL | v56;
        }
        *v54 = v52;
        if ( v52 )
        {
          v57 = *(_QWORD *)(v52 + 8);
          v54[1] = v57;
          if ( v57 )
            *(_QWORD *)(v57 + 16) = (unsigned __int64)(v54 + 1) | *(_QWORD *)(v57 + 16) & 3LL;
          v54[2] = v54[2] & 3LL | (v52 + 8);
          *(_QWORD *)(v52 + 8) = v54;
        }
        if ( (*(_BYTE *)(v14 + 23) & 0x40) != 0 )
          v58 = *(_QWORD *)(v14 - 8);
        else
          v58 = v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF);
        *(_QWORD *)(v48 + v58 + 24LL * *(unsigned int *)(v14 + 56) + 8) = v44;
        if ( (*(_BYTE *)(v14 + 23) & 0x40) != 0 )
          v59 = *(_QWORD *)(v14 - 8);
        else
          v59 = v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF);
        v60 = (_QWORD *)(v59 + v47);
        if ( *v60 )
        {
          v61 = v60[1];
          v62 = v60[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v62 = v61;
          if ( v61 )
            *(_QWORD *)(v61 + 16) = *(_QWORD *)(v61 + 16) & 3LL | v62;
        }
        *v60 = v49;
        if ( v49 )
        {
          v63 = *(_QWORD *)(v49 + 8);
          v60[1] = v63;
          if ( v63 )
            *(_QWORD *)(v63 + 16) = (unsigned __int64)(v60 + 1) | *(_QWORD *)(v63 + 16) & 3LL;
          v60[2] = (v49 + 8) | v60[2] & 3LL;
          *(_QWORD *)(v49 + 8) = v60;
        }
      }
      v39 += 8;
    }
    while ( v40 != v39 );
    v13 = v148;
  }
  if ( *(_BYTE *)(*(_QWORD *)v14 + 8LL) != 11 )
    return 0;
  v64 = sub_1643030(*(_QWORD *)v14);
  v67 = v13[333];
  v68 = v64;
  v69 = *(unsigned __int8 **)(v67 + 24);
  v70 = &v69[*(unsigned int *)(v67 + 32)];
  if ( v69 != v70 )
  {
    while ( v68 != *v69 )
    {
      if ( v70 == ++v69 )
        return sub_178E0E0(v13, v14, v15, *(double *)v16.m128i_i64, a5, a6, v65, v66, a9, a10);
    }
    return 0;
  }
  return sub_178E0E0(v13, v14, v15, *(double *)v16.m128i_i64, a5, a6, v65, v66, a9, a10);
}
