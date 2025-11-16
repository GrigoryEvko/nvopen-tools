// Function: sub_2A2E910
// Address: 0x2a2e910
//
__int64 __fastcall sub_2A2E910(__int64 **a1)
{
  unsigned int v1; // eax
  unsigned int v2; // r15d
  __int64 v3; // r13
  __int64 v4; // rax
  _QWORD *v5; // rax
  _QWORD *v6; // rbx
  _QWORD *v7; // rcx
  _QWORD *v8; // r13
  _BYTE *v9; // r12
  __int64 v10; // r15
  __int64 v11; // rsi
  unsigned int v12; // r14d
  __int16 v13; // ax
  unsigned __int16 v14; // r14
  __int64 *v15; // r15
  unsigned __int8 *v16; // r12
  __int64 *v17; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  char v22; // di
  __int64 v23; // rax
  __int64 v24; // rdx
  void *v25; // rax
  const void *v26; // rsi
  __int64 v27; // rax
  signed __int64 v28; // rcx
  char *v29; // rdi
  size_t v30; // rdx
  char *v31; // rax
  __m128i *v32; // rsi
  __int64 *v33; // rbx
  __int64 *v34; // rax
  __int64 v35; // rax
  __int64 *v36; // rax
  unsigned __int64 v37; // rax
  _BYTE *v38; // rdx
  __int64 v39; // rax
  char **v40; // rdx
  char v41; // al
  __m128i *v42; // rcx
  char v43; // dl
  char v44; // al
  __m128i *v45; // rsi
  void **v46; // rcx
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // r12
  __int64 *v50; // rax
  unsigned __int64 v51; // r13
  __int64 v52; // rbx
  __int64 v53; // r14
  void *v54; // r12
  __int64 v55; // r13
  _QWORD *v56; // rdi
  __int64 v57; // r9
  _QWORD *v58; // rdi
  char **v59; // rdx
  char v60; // al
  __m128i *v61; // rcx
  char v62; // dl
  char v63; // al
  __int64 v64; // rax
  __int64 v65; // r12
  __int64 v66; // rax
  __int64 v67; // r13
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // r14
  _QWORD *v71; // rax
  __int64 v72; // rbx
  _QWORD *v73; // r14
  __int64 v74; // rdx
  int v75; // eax
  _QWORD *v76; // rdi
  __int64 *v77; // rax
  __int64 v78; // rax
  _QWORD *v79; // rdi
  __int64 v80; // r9
  __int64 v81; // r14
  __int64 v82; // rax
  __int64 v83; // r13
  _QWORD *v84; // rdi
  __int64 v85; // r9
  _QWORD *v86; // rdi
  _QWORD *v87; // rdi
  __m128i v88; // xmm2
  __m128i v89; // xmm3
  __m128i v90; // xmm0
  __m128i v91; // xmm1
  __int128 v92; // rax
  __int128 v93; // rax
  __m128i *v94; // rsi
  void **v95; // rcx
  int v96; // r8d
  unsigned int v97; // eax
  __int64 v98; // rcx
  __int64 *v99; // rax
  __m128i v100; // xmm5
  __m128i v101; // xmm7
  _BYTE *v102; // rsi
  __int64 v103; // rdx
  __int64 v104; // rcx
  __int64 v105; // [rsp+10h] [rbp-2C0h]
  __int64 v106; // [rsp+18h] [rbp-2B8h]
  __int64 **v107; // [rsp+20h] [rbp-2B0h]
  unsigned __int64 v108; // [rsp+28h] [rbp-2A8h]
  _BYTE *v109; // [rsp+30h] [rbp-2A0h]
  __int64 v110; // [rsp+38h] [rbp-298h]
  __int64 v111; // [rsp+40h] [rbp-290h]
  __int64 v112; // [rsp+48h] [rbp-288h]
  __int64 *v113; // [rsp+50h] [rbp-280h]
  void *v114; // [rsp+58h] [rbp-278h]
  __int64 v115; // [rsp+60h] [rbp-270h]
  void *v116; // [rsp+68h] [rbp-268h]
  __int64 v117; // [rsp+70h] [rbp-260h]
  __int64 v118; // [rsp+78h] [rbp-258h]
  __int64 v119; // [rsp+80h] [rbp-250h]
  unsigned int v120; // [rsp+94h] [rbp-23Ch]
  __int64 v121; // [rsp+98h] [rbp-238h]
  __int64 v122; // [rsp+A0h] [rbp-230h]
  unsigned __int16 v123; // [rsp+ACh] [rbp-224h]
  void *v124; // [rsp+B0h] [rbp-220h]
  __m128i v125; // [rsp+B0h] [rbp-220h]
  __int64 v126; // [rsp+C0h] [rbp-210h]
  __int64 v127; // [rsp+E0h] [rbp-1F0h]
  __int64 *v128; // [rsp+E8h] [rbp-1E8h]
  _BYTE *v129; // [rsp+F0h] [rbp-1E0h]
  __int64 v131; // [rsp+100h] [rbp-1D0h]
  unsigned __int16 v132; // [rsp+100h] [rbp-1D0h]
  __int64 v133; // [rsp+100h] [rbp-1D0h]
  __int64 v134; // [rsp+100h] [rbp-1D0h]
  __int64 v135; // [rsp+100h] [rbp-1D0h]
  __int64 v136; // [rsp+108h] [rbp-1C8h]
  __int64 v137; // [rsp+108h] [rbp-1C8h]
  __int64 *v138; // [rsp+110h] [rbp-1C0h]
  unsigned __int8 v139; // [rsp+120h] [rbp-1B0h]
  __int64 v140; // [rsp+120h] [rbp-1B0h]
  __int64 v141; // [rsp+120h] [rbp-1B0h]
  __int64 v142; // [rsp+128h] [rbp-1A8h]
  signed __int64 v143; // [rsp+128h] [rbp-1A8h]
  size_t v144; // [rsp+128h] [rbp-1A8h]
  __int64 v145; // [rsp+128h] [rbp-1A8h]
  __int64 v146; // [rsp+128h] [rbp-1A8h]
  __int64 v147; // [rsp+128h] [rbp-1A8h]
  __int64 *v148; // [rsp+128h] [rbp-1A8h]
  __int64 **v149; // [rsp+130h] [rbp-1A0h] BYREF
  __int64 v150; // [rsp+138h] [rbp-198h] BYREF
  __int64 v151; // [rsp+140h] [rbp-190h] BYREF
  __int64 v152; // [rsp+148h] [rbp-188h]
  char *v153; // [rsp+150h] [rbp-180h] BYREF
  __int64 v154; // [rsp+158h] [rbp-178h]
  int v155; // [rsp+160h] [rbp-170h]
  __int16 v156; // [rsp+170h] [rbp-160h]
  __m128i v157; // [rsp+180h] [rbp-150h] BYREF
  __m128i v158; // [rsp+190h] [rbp-140h] BYREF
  __int64 v159; // [rsp+1A0h] [rbp-130h]
  char *v160; // [rsp+1B0h] [rbp-120h] BYREF
  __int64 *v161; // [rsp+1C0h] [rbp-110h]
  __int16 v162; // [rsp+1D0h] [rbp-100h]
  __m128i v163; // [rsp+1E0h] [rbp-F0h] BYREF
  __m128i v164; // [rsp+1F0h] [rbp-E0h] BYREF
  __int64 v165; // [rsp+200h] [rbp-D0h]
  void *src[2]; // [rsp+210h] [rbp-C0h] BYREF
  __int128 v167; // [rsp+220h] [rbp-B0h]
  __int64 v168; // [rsp+230h] [rbp-A0h]
  __m128i v169; // [rsp+240h] [rbp-90h] BYREF
  __m128i v170; // [rsp+250h] [rbp-80h]
  __int64 v171; // [rsp+260h] [rbp-70h]
  __int64 v172; // [rsp+270h] [rbp-60h] BYREF
  __int64 v173; // [rsp+278h] [rbp-58h] BYREF
  __int64 *v174; // [rsp+280h] [rbp-50h]
  __int64 *v175; // [rsp+288h] [rbp-48h]
  __int64 *v176; // [rsp+290h] [rbp-40h]
  __int64 v177; // [rsp+298h] [rbp-38h]

  v129 = sub_BA8CD0((__int64)a1, (__int64)"llvm.global_dtors", 0x11u, 0);
  if ( v129
    && (LOBYTE(v1) = sub_B2FC80((__int64)v129), v2 = v1, !(_BYTE)v1)
    && (v3 = *((_QWORD *)v129 - 4), *(_BYTE *)v3 == 9)
    && (v4 = *(_QWORD *)(*(_QWORD *)(v3 + 8) + 24LL), *(_BYTE *)(v4 + 8) == 15)
    && *(_DWORD *)(v4 + 12) == 3
    && (v5 = *(_QWORD **)(v4 + 16), *(_BYTE *)(*v5 + 8LL) == 12)
    && *(_BYTE *)(v5[1] + 8LL) == 14
    && *(_BYTE *)(v5[2] + 8LL) == 14 )
  {
    LODWORD(v173) = 0;
    v174 = 0;
    v175 = &v173;
    v176 = &v173;
    v177 = 0;
    if ( (*(_BYTE *)(v3 + 7) & 0x40) != 0 )
    {
      v6 = *(_QWORD **)(v3 - 8);
      v7 = &v6[4 * (*(_DWORD *)(v3 + 4) & 0x7FFFFFF)];
    }
    else
    {
      v7 = (_QWORD *)v3;
      v6 = (_QWORD *)(v3 - 32LL * (*(_DWORD *)(v3 + 4) & 0x7FFFFFF));
    }
    if ( v6 != v7 )
    {
      v139 = v2;
      v8 = v7;
      while ( 1 )
      {
        v9 = (_BYTE *)*v6;
        if ( *(_BYTE *)*v6 == 10 )
        {
          v10 = *((_DWORD *)v9 + 1) & 0x7FFFFFF;
          v11 = *(_QWORD *)&v9[-32 * v10];
          if ( *(_BYTE *)v11 == 17 )
            break;
        }
LABEL_47:
        v6 += 4;
        if ( v8 == v6 )
          goto LABEL_48;
      }
      v12 = *(_DWORD *)(v11 + 32);
      if ( v12 > 0x40 )
      {
        v147 = *(_QWORD *)&v9[-32 * (*((_DWORD *)v9 + 1) & 0x7FFFFFF)];
        v96 = sub_C444A0(v11 + 24);
        v97 = v12;
        v14 = -1;
        if ( v97 - v96 <= 0x40 )
        {
          v11 = v147;
          LOWORD(v98) = -1;
          v99 = *(__int64 **)(v147 + 24);
          if ( (unsigned __int64)*v99 <= 0xFFFF )
            v98 = *v99;
          v14 = v98;
        }
      }
      else
      {
        v13 = -1;
        if ( *(_QWORD *)(v11 + 24) <= 0xFFFFu )
          v13 = *(_QWORD *)(v11 + 24);
        v14 = v13;
      }
      v163.m128i_i64[0] = *(_QWORD *)&v9[32 * (1 - v10)];
      if ( sub_AC30F0(v163.m128i_i64[0]) )
      {
LABEL_48:
        v2 = v139;
        goto LABEL_49;
      }
      v15 = &v173;
      v16 = sub_BD3990(*(unsigned __int8 **)&v9[32 * (2LL - (*((_DWORD *)v9 + 1) & 0x7FFFFFF))], v11);
      v17 = v174;
      if ( !v174 )
        goto LABEL_30;
      do
      {
        if ( *((_WORD *)v17 + 16) < v14 )
        {
          v17 = (__int64 *)v17[3];
        }
        else
        {
          v15 = v17;
          v17 = (__int64 *)v17[2];
        }
      }
      while ( v17 );
      if ( v15 == &v173 || *((_WORD *)v15 + 16) > v14 )
      {
LABEL_30:
        v142 = (__int64)v15;
        v19 = sub_22077B0(0x40u);
        *(_WORD *)(v19 + 32) = v14;
        v15 = (__int64 *)v19;
        *(_QWORD *)(v19 + 40) = 0;
        *(_QWORD *)(v19 + 48) = 0;
        *(_QWORD *)(v19 + 56) = 0;
        v20 = sub_2A2E810(&v172, v142, (unsigned __int16 *)(v19 + 32));
        if ( v21 )
        {
          v22 = &v173 == (__int64 *)v21 || v20 || *(_WORD *)(v21 + 32) > v14;
          sub_220F040(v22, (__int64)v15, (_QWORD *)v21, &v173);
          ++v177;
        }
        else
        {
          v148 = (__int64 *)v20;
          j_j___libc_free_0((unsigned __int64)v15);
          v15 = v148;
        }
      }
      v23 = v15[6];
      if ( v23 != v15[5] && *(unsigned __int8 **)(v23 - 32) == v16 )
      {
        v102 = *(_BYTE **)(v23 - 16);
        if ( v102 == *(_BYTE **)(v23 - 8) )
        {
          sub_91DE00(v23 - 24, v102, &v163);
        }
        else
        {
          if ( v102 )
          {
            *(_QWORD *)v102 = v163.m128i_i64[0];
            v102 = *(_BYTE **)(v23 - 16);
          }
          *(_QWORD *)(v23 - 16) = v102 + 8;
        }
        goto LABEL_47;
      }
      src[0] = 0;
      src[1] = 0;
      *(_QWORD *)&v167 = 0;
      sub_91DE00((__int64)src, 0, &v163);
      v25 = src[1];
      v26 = src[0];
      v169 = (__m128i)(unsigned __int64)v16;
      v170 = 0u;
      if ( src[1] == src[0] )
      {
        v30 = 0;
        v28 = 0;
        v29 = 0;
      }
      else
      {
        if ( (void *)((char *)src[1] - (char *)src[0]) > (void *)0x7FFFFFFFFFFFFFF8LL )
          sub_4261EA(src, src[0], v24);
        v143 = (char *)src[1] - (char *)src[0];
        v27 = sub_22077B0((char *)src[1] - (char *)src[0]);
        v26 = src[0];
        v28 = v143;
        v29 = (char *)v27;
        v25 = src[1];
        v30 = (char *)src[1] - (char *)src[0];
      }
      v169.m128i_i64[1] = (__int64)v29;
      v170.m128i_i64[0] = (__int64)v29;
      v170.m128i_i64[1] = (__int64)&v29[v28];
      if ( v26 != v25 )
      {
        v144 = v30;
        v31 = (char *)memmove(v29, v26, v30);
        v30 = v144;
        v29 = v31;
      }
      v170.m128i_i64[0] = (__int64)&v29[v30];
      v32 = (__m128i *)v15[6];
      if ( v32 == (__m128i *)v15[7] )
      {
        sub_2A2E530((unsigned __int64 *)v15 + 5, v32->m128i_i8, &v169);
        v29 = (char *)v169.m128i_i64[1];
      }
      else
      {
        if ( v32 )
        {
          *v32 = v169;
          v32[1] = v170;
          v15[6] += 32;
LABEL_45:
          if ( src[0] )
            j_j___libc_free_0((unsigned __int64)src[0]);
          goto LABEL_47;
        }
        v15[6] = 32;
      }
      if ( v29 )
        j_j___libc_free_0((unsigned __int64)v29);
      goto LABEL_45;
    }
LABEL_49:
    if ( v177 )
    {
      v33 = *a1;
      v138 = *a1;
      v107 = (__int64 **)sub_BCE3C0(*a1, 0);
      v149 = v107;
      v34 = (__int64 *)sub_BCB120(v33);
      v108 = sub_BCF480(v34, &v149, 1, 0);
      v35 = sub_BCE3C0(v33, 0);
      v169.m128i_i64[1] = (__int64)v107;
      v169.m128i_i64[0] = v35;
      v170.m128i_i64[0] = (__int64)v107;
      v36 = (__int64 *)sub_BCB2D0(v33);
      v37 = sub_BCF480(v36, &v169, 3, 0);
      v121 = sub_BA8CA0((__int64)a1, (__int64)"__cxa_atexit", 0xCu, v37);
      v110 = (__int64)v38;
      if ( *v38 || sub_B2FC80((__int64)v38) || (unsigned __int8)sub_B2FC00((_BYTE *)v110) )
        goto LABEL_175;
      if ( (*(_BYTE *)(v110 + 2) & 1) != 0 )
        sub_B2C6D0(v110, (__int64)"__cxa_atexit", v103, v104);
      if ( (unsigned int)sub_BD3960(*(_QWORD *)(v110 + 96)) )
      {
LABEL_175:
        v150 = sub_BCB2B0(v138);
        v169.m128i_i64[0] = (__int64)a1;
        v169.m128i_i64[1] = (__int64)&v150;
        v109 = sub_BA8D20(
                 (__int64)a1,
                 (__int64)"__dso_handle",
                 0xCu,
                 v150,
                 (__int64 (__fastcall *)(__int64))sub_2A2E0D0,
                 (__int64)&v169);
        v126 = (__int64)v175;
        if ( v175 != &v173 )
        {
          while ( 1 )
          {
            v123 = *(_WORD *)(v126 + 32);
            v113 = *(__int64 **)(v126 + 48);
            if ( v113 != *(__int64 **)(v126 + 40) )
              break;
LABEL_122:
            v126 = sub_220EEE0(v126);
            if ( (__int64 *)v126 == &v173 )
              goto LABEL_123;
          }
          v128 = *(__int64 **)(v126 + 40);
          v122 = 0;
          while ( 1 )
          {
            v39 = v122++;
            v127 = *v128;
            v151 = v39;
            if ( sub_AC30F0(v127) )
            {
              LOWORD(v168) = 257;
            }
            else
            {
              *(_QWORD *)&v93 = sub_BD5D20(v127);
              LOWORD(v168) = 1283;
              src[0] = ".";
              v167 = v93;
            }
            if ( *(_QWORD *)(v126 + 48) - *(_QWORD *)(v126 + 40) > 0x20u )
            {
              v160 = "$";
              v161 = &v151;
              v162 = 2819;
            }
            else
            {
              v162 = 257;
            }
            if ( v123 == 0xFFFF )
            {
              v156 = 257;
            }
            else
            {
              v153 = ".";
              v156 = 2563;
              v155 = v123;
            }
            if ( 2 * (v123 != 0xFFFF) )
            {
              v40 = &v153;
              v41 = 2;
              if ( HIBYTE(v156) == 1 )
              {
                v40 = (char **)v153;
                v105 = v154;
                v41 = 2 * (v123 != 0xFFFF) + 1;
              }
              BYTE1(v159) = v41;
              v157.m128i_i64[0] = (__int64)"call_dtors";
              v158.m128i_i64[0] = (__int64)v40;
              v158.m128i_i64[1] = v105;
              LOBYTE(v159) = 3;
              if ( (_BYTE)v162 != 1 )
              {
                v42 = &v157;
                v43 = 2;
LABEL_66:
                v163.m128i_i64[0] = (__int64)v42;
                BYTE1(v165) = 2;
                v44 = v168;
                v164.m128i_i64[0] = (__int64)&v160;
                v163.m128i_i64[1] = v118;
                v164.m128i_i64[1] = v111;
                LOBYTE(v165) = v43;
                goto LABEL_67;
              }
            }
            else
            {
              v157.m128i_i64[0] = (__int64)"call_dtors";
              LOWORD(v159) = 259;
              if ( (_BYTE)v162 != 1 )
              {
                v42 = (__m128i *)"call_dtors";
                v118 = v157.m128i_i64[1];
                v43 = 3;
                goto LABEL_66;
              }
            }
            v43 = v159;
            v90 = _mm_loadu_si128(&v157);
            v91 = _mm_loadu_si128(&v158);
            v165 = v159;
            v163 = v90;
            v164 = v91;
            v44 = v168;
LABEL_67:
            if ( v44 == 1 )
            {
              v100 = _mm_loadu_si128(&v164);
              v169 = _mm_loadu_si128(&v163);
              v171 = v165;
              v170 = v100;
            }
            else
            {
              if ( BYTE1(v165) == 1 )
              {
                v115 = v163.m128i_i64[1];
                v45 = (__m128i *)v163.m128i_i64[0];
              }
              else
              {
                v45 = &v163;
                v43 = 2;
              }
              if ( BYTE1(v168) == 1 )
              {
                v46 = (void **)src[0];
                v114 = src[1];
              }
              else
              {
                v46 = src;
                v44 = 2;
              }
              v170.m128i_i64[0] = (__int64)v46;
              v169.m128i_i64[0] = (__int64)v45;
              v169.m128i_i64[1] = v115;
              v170.m128i_i64[1] = (__int64)v114;
              LOBYTE(v171) = v43;
              BYTE1(v171) = v44;
            }
            v47 = sub_BD2DA0(136);
            v124 = (void *)v47;
            if ( v47 )
              sub_B2C3B0(v47, v108, 8, 0xFFFFFFFF, (__int64)&v169, (__int64)a1);
            v169.m128i_i64[0] = (__int64)"body";
            LOWORD(v171) = 259;
            v48 = sub_22077B0(0x50u);
            v49 = v48;
            if ( v48 )
              sub_AA4D50(v48, (__int64)v138, (__int64)&v169, (__int64)v124, 0);
            v50 = (__int64 *)sub_BCB120(v138);
            v51 = sub_BCF640(v50, 0);
            v52 = v128[2];
            v131 = v128[1];
            if ( v131 != v52 )
            {
              v140 = v51;
              v145 = v49;
              do
              {
                v53 = *(_QWORD *)(v52 - 8);
                sub_B43C20((__int64)src, v145);
                v54 = src[0];
                LOWORD(v171) = 257;
                v55 = LOWORD(src[1]);
                v56 = sub_BD2C40(88, 1u);
                if ( v56 )
                  sub_B4A410((__int64)v56, v140, v53, (__int64)&v169, 1u, v57, (__int64)v54, v55);
                v52 -= 8;
              }
              while ( v131 != v52 );
              v51 = v140;
              v49 = v145;
            }
            sub_B43C20((__int64)&v169, v49);
            v58 = sub_BD2C40(72, 0);
            if ( v58 )
              sub_B4BB80((__int64)v58, (__int64)v138, 0, 0, v169.m128i_i64[0], v169.m128i_u16[4]);
            if ( sub_AC30F0(v127) )
            {
              LOWORD(v168) = 257;
            }
            else
            {
              *(_QWORD *)&v92 = sub_BD5D20(v127);
              src[0] = ".";
              v167 = v92;
              LOWORD(v168) = 1283;
            }
            if ( *(_QWORD *)(v126 + 48) - *(_QWORD *)(v126 + 40) > 0x20u )
            {
              v160 = "$";
              v161 = &v151;
              v162 = 2819;
            }
            else
            {
              v162 = 257;
            }
            if ( v123 == 0xFFFF )
            {
              v156 = 257;
            }
            else
            {
              v153 = ".";
              v155 = v123;
              v156 = 2563;
            }
            if ( 2 * (v123 != 0xFFFF) )
            {
              v59 = &v153;
              v60 = 2;
              if ( HIBYTE(v156) == 1 )
              {
                v59 = (char **)v153;
                v106 = v154;
                v60 = 2 * (v123 != 0xFFFF) + 1;
              }
              v158.m128i_i64[0] = (__int64)v59;
              BYTE1(v159) = v60;
              v157.m128i_i64[0] = (__int64)"register_call_dtors";
              v158.m128i_i64[1] = v106;
              LOBYTE(v159) = 3;
              if ( (_BYTE)v162 == 1 )
              {
LABEL_128:
                v62 = v159;
                v88 = _mm_loadu_si128(&v157);
                v89 = _mm_loadu_si128(&v158);
                v165 = v159;
                v163 = v88;
                v164 = v89;
                v63 = v168;
                goto LABEL_136;
              }
              v61 = &v157;
              v62 = 2;
            }
            else
            {
              v157.m128i_i64[0] = (__int64)"register_call_dtors";
              LOWORD(v159) = 259;
              if ( (_BYTE)v162 == 1 )
                goto LABEL_128;
              v61 = (__m128i *)"register_call_dtors";
              v119 = v157.m128i_i64[1];
              v62 = 3;
            }
            v163.m128i_i64[0] = (__int64)v61;
            BYTE1(v165) = 2;
            v63 = v168;
            v163.m128i_i64[1] = v119;
            v164.m128i_i64[0] = (__int64)&v160;
            v164.m128i_i64[1] = v112;
            LOBYTE(v165) = v62;
LABEL_136:
            if ( v63 == 1 )
            {
              v101 = _mm_loadu_si128(&v164);
              v169 = _mm_loadu_si128(&v163);
              v171 = v165;
              v170 = v101;
            }
            else
            {
              if ( BYTE1(v165) == 1 )
              {
                v117 = v163.m128i_i64[1];
                v94 = (__m128i *)v163.m128i_i64[0];
              }
              else
              {
                v94 = &v163;
                v62 = 2;
              }
              if ( BYTE1(v168) == 1 )
              {
                v116 = src[1];
                v95 = (void **)src[0];
              }
              else
              {
                v95 = src;
                v63 = 2;
              }
              v170.m128i_i64[0] = (__int64)v95;
              v169.m128i_i64[0] = (__int64)v94;
              v169.m128i_i64[1] = v117;
              v170.m128i_i64[1] = (__int64)v116;
              LOBYTE(v171) = v62;
              BYTE1(v171) = v63;
            }
            v64 = sub_BD2DA0(136);
            v65 = v64;
            if ( v64 )
              sub_B2C3B0(v64, v51, 8, 0xFFFFFFFF, (__int64)&v169, (__int64)a1);
            v169.m128i_i64[0] = (__int64)"entry";
            LOWORD(v171) = 259;
            v66 = sub_22077B0(0x50u);
            v67 = v66;
            if ( v66 )
              sub_AA4D50(v66, (__int64)v138, (__int64)&v169, v65, 0);
            v169.m128i_i64[0] = (__int64)"fail";
            LOWORD(v171) = 259;
            v68 = sub_22077B0(0x50u);
            v146 = v68;
            if ( v68 )
              sub_AA4D50(v68, (__int64)v138, (__int64)&v169, v65, 0);
            v169.m128i_i64[0] = (__int64)"return";
            LOWORD(v171) = 259;
            v69 = sub_22077B0(0x50u);
            v141 = v69;
            if ( v69 )
              sub_AA4D50(v69, (__int64)v138, (__int64)&v169, v65, 0);
            src[1] = (void *)sub_AC9EC0(v107);
            src[0] = v124;
            *(_QWORD *)&v167 = v109;
            sub_B43C20((__int64)&v163, v67);
            v169.m128i_i64[0] = (__int64)"call";
            v70 = v163.m128i_i64[0];
            LOWORD(v171) = 259;
            v132 = v163.m128i_u16[4];
            v71 = sub_BD2C40(88, 4u);
            v72 = (__int64)v71;
            if ( v71 )
            {
              v120 = v120 & 0xE0000000 | 4;
              sub_B44260((__int64)v71, **(_QWORD **)(v121 + 16), 56, v120, v70, v132);
              *(_QWORD *)(v72 + 72) = 0;
              sub_B4A290(v72, v121, v110, (__int64 *)src, 3, (__int64)&v169, 0, 0);
            }
            sub_B43C20((__int64)&v163, v67);
            v133 = sub_AD6530(*(_QWORD *)(v72 + 8), v67);
            LOWORD(v171) = 257;
            v73 = sub_BD2C40(72, unk_3F10FD0);
            if ( v73 )
            {
              v74 = *(_QWORD *)(v72 + 8);
              v125 = v163;
              if ( (unsigned int)*(unsigned __int8 *)(v74 + 8) - 17 > 1 )
              {
                v78 = sub_BCB2A0(*(_QWORD **)v74);
              }
              else
              {
                v75 = *(_DWORD *)(v74 + 32);
                v76 = *(_QWORD **)v74;
                BYTE4(v152) = *(_BYTE *)(v74 + 8) == 18;
                LODWORD(v152) = v75;
                v77 = (__int64 *)sub_BCB2A0(v76);
                v78 = sub_BCE1B0(v77, v152);
              }
              sub_B523C0((__int64)v73, v78, 53, 33, v72, v133, (__int64)&v169, v125.m128i_i64[0], v125.m128i_i64[1], 0);
            }
            sub_B43C20((__int64)&v169, v67);
            v134 = v169.m128i_i64[0];
            v136 = v169.m128i_u16[4];
            v79 = sub_BD2C40(72, 3u);
            if ( v79 )
              sub_B4C9A0((__int64)v79, v146, v141, (__int64)v73, 3u, v80, v134, v136);
            v81 = 0;
            sub_B43C20((__int64)&v163, v146);
            LOWORD(v171) = 257;
            v82 = sub_B6E160((__int64 *)a1, 0x162u, 0, 0);
            v83 = v82;
            if ( v82 )
              v81 = *(_QWORD *)(v82 + 24);
            v135 = v163.m128i_i64[0];
            v137 = v163.m128i_u16[4];
            v84 = sub_BD2C40(88, 1u);
            if ( v84 )
              sub_B4A410((__int64)v84, v81, v83, (__int64)&v169, 1u, v85, v135, v137);
            sub_B43C20((__int64)&v169, v146);
            v86 = sub_BD2C40(72, unk_3F148B8);
            if ( v86 )
              sub_B4C8A0((__int64)v86, (__int64)v138, v169.m128i_i64[0], v169.m128i_u16[4]);
            sub_B43C20((__int64)&v169, v141);
            v87 = sub_BD2C40(72, 0);
            if ( v87 )
              sub_B4BB80((__int64)v87, (__int64)v138, 0, 0, v169.m128i_i64[0], v169.m128i_u16[4]);
            sub_2A3ED40(a1, v65, v123, v127);
            v128 += 4;
            if ( v113 == v128 )
              goto LABEL_122;
          }
        }
      }
LABEL_123:
      v2 = 1;
      sub_B30290((__int64)v129);
    }
    sub_2A2E1E0(v174);
  }
  else
  {
    return 0;
  }
  return v2;
}
