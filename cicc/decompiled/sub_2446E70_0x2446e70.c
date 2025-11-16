// Function: sub_2446E70
// Address: 0x2446e70
//
__int64 __fastcall sub_2446E70(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        unsigned __int64 a6,
        int a7,
        __m128i *a8,
        __int64 a9,
        __m128i *a10)
{
  __int64 *i; // r15
  char v12; // al
  __int64 *v13; // r12
  __int64 v14; // rax
  __int64 *v15; // rbx
  char v16; // di
  __m128i *v17; // r9
  int v18; // esi
  int v19; // ecx
  __m128i *v20; // rax
  __int64 v21; // r10
  __int64 *v22; // rdx
  __int64 v23; // rax
  unsigned __int64 v24; // r12
  __int64 v25; // rbx
  unsigned __int64 v26; // r13
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rcx
  __int64 v29; // rax
  __int64 v30; // r13
  __int64 v31; // r12
  __int64 v32; // rax
  __int64 v33; // r12
  __int64 v34; // r12
  __int64 v35; // r12
  __int64 v36; // r12
  __int64 v37; // rax
  char v38; // al
  unsigned int v39; // edx
  char v40; // al
  __int64 v41; // r13
  __int64 *v42; // r8
  unsigned __int64 *v43; // rbx
  unsigned __int64 *v44; // r13
  unsigned __int64 v45; // r15
  __int64 *v46; // r14
  unsigned __int64 v47; // rcx
  __int64 *v48; // rax
  bool v49; // r9
  __int64 v50; // rax
  __int64 v51; // r14
  __int64 v52; // rsi
  __int64 v53; // rax
  int v54; // ecx
  __int64 v55; // rdi
  int v56; // ecx
  unsigned int v57; // edx
  __int64 *v58; // rax
  __int64 v59; // r10
  unsigned __int8 *v60; // rcx
  __int64 *v61; // rax
  unsigned int v62; // esi
  unsigned __int32 v63; // eax
  __m128i *v64; // rdx
  int v65; // ecx
  unsigned int v66; // r8d
  __int64 v67; // rax
  int v68; // eax
  int v69; // r11d
  __m128i *v70; // r8
  int v71; // ecx
  __int64 v72; // rax
  __int64 v73; // rdi
  __m128i *v74; // r8
  int v75; // ecx
  __int64 v76; // rax
  __int64 v77; // rdi
  int v78; // r10d
  __m128i *v79; // r9
  __int64 v80; // r8
  __int64 v81; // r9
  unsigned __int64 *v82; // rbx
  unsigned __int64 *v83; // r12
  unsigned __int64 v84; // rdi
  __int64 v85; // rax
  __int64 v86; // rbx
  unsigned __int64 v87; // rdx
  unsigned __int64 v88; // rdx
  __int64 v89; // rax
  __int64 *v90; // rbx
  __int64 v91; // rax
  __int64 v92; // r12
  _QWORD *v93; // r14
  __int64 v94; // r15
  __int64 v95; // rbx
  __int64 v96; // r13
  unsigned __int8 v97; // dl
  unsigned __int64 v98; // rax
  __int64 v99; // rsi
  _QWORD *v100; // rbx
  __int64 v101; // rdx
  __int64 v102; // r14
  __int64 v103; // r13
  __int64 v104; // rdi
  __int32 v105; // ecx
  __int32 v106; // ecx
  __int64 v107; // rax
  __int64 v108; // rax
  __int64 v109; // rax
  __int64 *v110; // r8
  __int64 v111; // rdx
  _BYTE *v112; // rdi
  unsigned int v113; // r12d
  _BYTE *v114; // r9
  __int64 *v115; // rsi
  __int64 v116; // rax
  __int64 *m128i_i64; // r8
  __int64 v118; // rbx
  __int64 v119; // r15
  __m128i *v120; // rax
  unsigned __int64 v121; // r12
  __m128i *v122; // rcx
  __m128i *v123; // rax
  __int64 v124; // r12
  __int64 v126; // rsi
  char v127; // dl
  unsigned __int64 *v128; // r8
  char v129; // dh
  char v130; // al
  __int64 v131; // rdx
  int v132; // r10d
  int v133; // r8d
  __int64 v134; // rax
  __int64 *v135; // [rsp+18h] [rbp-418h]
  __int64 *v136; // [rsp+28h] [rbp-408h]
  __int64 *v138; // [rsp+38h] [rbp-3F8h]
  unsigned __int8 v139; // [rsp+38h] [rbp-3F8h]
  int v140; // [rsp+40h] [rbp-3F0h]
  __int64 *v141; // [rsp+48h] [rbp-3E8h]
  __int64 *v142; // [rsp+48h] [rbp-3E8h]
  __int64 v143; // [rsp+58h] [rbp-3D8h]
  char v146; // [rsp+70h] [rbp-3C0h]
  __int64 v147; // [rsp+70h] [rbp-3C0h]
  _QWORD *v149; // [rsp+80h] [rbp-3B0h]
  unsigned __int64 v150; // [rsp+88h] [rbp-3A8h]
  _BYTE *v151; // [rsp+90h] [rbp-3A0h] BYREF
  __int64 v152; // [rsp+98h] [rbp-398h]
  _BYTE v153[32]; // [rsp+A0h] [rbp-390h] BYREF
  __int64 v154[2]; // [rsp+C0h] [rbp-370h] BYREF
  __int64 v155; // [rsp+D0h] [rbp-360h] BYREF
  __int64 *v156; // [rsp+E0h] [rbp-350h]
  __int64 v157; // [rsp+F0h] [rbp-340h] BYREF
  __int64 v158[2]; // [rsp+110h] [rbp-320h] BYREF
  __int64 v159; // [rsp+120h] [rbp-310h] BYREF
  __int64 *v160; // [rsp+130h] [rbp-300h]
  __int64 v161; // [rsp+140h] [rbp-2F0h] BYREF
  __int64 v162[2]; // [rsp+160h] [rbp-2D0h] BYREF
  __int64 v163; // [rsp+170h] [rbp-2C0h] BYREF
  __int64 *v164; // [rsp+180h] [rbp-2B0h]
  __int64 v165; // [rsp+190h] [rbp-2A0h] BYREF
  unsigned __int64 *v166; // [rsp+1B0h] [rbp-280h] BYREF
  __int64 v167; // [rsp+1B8h] [rbp-278h] BYREF
  __int64 *v168; // [rsp+1C0h] [rbp-270h] BYREF
  __int64 *v169; // [rsp+1C8h] [rbp-268h]
  __int64 *v170; // [rsp+1D0h] [rbp-260h]
  __int64 v171; // [rsp+1D8h] [rbp-258h]
  __int64 v172; // [rsp+1E0h] [rbp-250h] BYREF
  unsigned __int64 v173[2]; // [rsp+200h] [rbp-230h] BYREF
  _QWORD v174[2]; // [rsp+210h] [rbp-220h] BYREF
  _QWORD *v175; // [rsp+220h] [rbp-210h]
  _QWORD v176[4]; // [rsp+230h] [rbp-200h] BYREF
  _QWORD v177[10]; // [rsp+250h] [rbp-1E0h] BYREF
  unsigned __int64 *v178; // [rsp+2A0h] [rbp-190h]
  unsigned int v179; // [rsp+2A8h] [rbp-188h]
  char v180; // [rsp+2B0h] [rbp-180h] BYREF
  __m128i *v181; // [rsp+458h] [rbp+28h]
  __m128i *v182; // [rsp+458h] [rbp+28h]

  v151 = v153;
  v152 = 0x400000000LL;
  v136 = &a4[44 * a5];
  if ( a4 == v136 )
    return 0;
  for ( i = a4 + 4; ; i += 44 )
  {
    v141 = i - 4;
    v12 = *(_BYTE *)(i - 1) & 1;
    if ( *((_DWORD *)i - 2) >> 1 )
    {
      if ( v12 )
      {
        v13 = i;
        v14 = (__int64)(i + 32);
        goto LABEL_6;
      }
      v13 = (__int64 *)*i;
      v14 = *i + 16LL * *((unsigned int *)i + 2);
      if ( *i == v14 )
        goto LABEL_8;
      if ( (unsigned __int64)*v13 > 0xFFFFFFFFFFFFFFFDLL )
      {
        while ( 1 )
        {
          v13 += 2;
          if ( v13 == (__int64 *)v14 )
            break;
LABEL_6:
          if ( (unsigned __int64)*v13 <= 0xFFFFFFFFFFFFFFFDLL )
            goto LABEL_105;
        }
LABEL_8:
        v15 = v13;
        goto LABEL_9;
      }
LABEL_105:
      v15 = v13;
      v13 = (__int64 *)v14;
    }
    else
    {
      if ( v12 )
      {
        v90 = i;
        v91 = 32;
      }
      else
      {
        v90 = (__int64 *)*i;
        v91 = 2LL * *((unsigned int *)i + 2);
      }
      v15 = &v90[v91];
      v13 = v15;
    }
LABEL_9:
    if ( v15 != v13 )
    {
      v16 = a10->m128i_i8[8] & 1;
      if ( v16 )
      {
LABEL_11:
        v17 = a10 + 1;
        v18 = 15;
        goto LABEL_12;
      }
      while ( 1 )
      {
        v62 = a10[1].m128i_u32[2];
        v17 = (__m128i *)a10[1].m128i_i64[0];
        if ( v62 )
        {
          v18 = v62 - 1;
LABEL_12:
          v19 = v18 & (((0xBF58476D1CE4E5B9LL * *v15) >> 31) ^ (484763065 * *(_DWORD *)v15));
          v20 = &v17[v19];
          v21 = v20->m128i_i64[0];
          if ( v20->m128i_i64[0] != *v15 )
          {
            v69 = 1;
            v64 = 0;
            while ( 1 )
            {
              if ( v21 == -1 )
              {
                v66 = 48;
                v62 = 16;
                if ( !v64 )
                  v64 = v20;
                v63 = a10->m128i_u32[2];
                ++a10->m128i_i64[0];
                v65 = (v63 >> 1) + 1;
                if ( !v16 )
                {
                  v62 = a10[1].m128i_u32[2];
                  goto LABEL_83;
                }
                goto LABEL_84;
              }
              if ( v21 == -2 && !v64 )
                v64 = v20;
              v19 = v18 & (v69 + v19);
              v20 = &v17[v19];
              v21 = v20->m128i_i64[0];
              if ( *v15 == v20->m128i_i64[0] )
                break;
              ++v69;
            }
          }
          v22 = &v20->m128i_i64[1];
          v23 = v20->m128i_i64[1];
          goto LABEL_14;
        }
        v63 = a10->m128i_u32[2];
        ++a10->m128i_i64[0];
        v64 = 0;
        v65 = (v63 >> 1) + 1;
LABEL_83:
        v66 = 3 * v62;
LABEL_84:
        if ( 4 * v65 >= v66 )
          break;
        if ( v62 - a10->m128i_i32[3] - v65 <= v62 >> 3 )
        {
          sub_2446700((__int64)a10, v62);
          if ( (a10->m128i_i8[8] & 1) != 0 )
          {
            v74 = a10 + 1;
            v75 = 15;
          }
          else
          {
            v106 = a10[1].m128i_i32[2];
            v74 = (__m128i *)a10[1].m128i_i64[0];
            if ( !v106 )
            {
LABEL_231:
              a10->m128i_i32[2] = (2 * ((unsigned __int32)a10->m128i_i32[2] >> 1) + 2) | a10->m128i_i32[2] & 1;
              BUG();
            }
            v75 = v106 - 1;
          }
          LODWORD(v76) = v75 & (((0xBF58476D1CE4E5B9LL * *v15) >> 31) ^ (484763065 * *(_DWORD *)v15));
          v64 = &v74[(unsigned int)v76];
          v77 = v64->m128i_i64[0];
          if ( *v15 != v64->m128i_i64[0] )
          {
            v78 = 1;
            v79 = 0;
            while ( v77 != -1 )
            {
              if ( v77 == -2 && !v79 )
                v79 = v64;
              v76 = v75 & (unsigned int)(v76 + v78);
              v64 = &v74[v76];
              v77 = v64->m128i_i64[0];
              if ( *v15 == v64->m128i_i64[0] )
                goto LABEL_109;
              ++v78;
            }
            goto LABEL_115;
          }
          goto LABEL_109;
        }
LABEL_86:
        a10->m128i_i32[2] = (2 * (v63 >> 1) + 2) | v63 & 1;
        if ( v64->m128i_i64[0] != -1 )
          --a10->m128i_i32[3];
        v67 = *v15;
        v22 = &v64->m128i_i64[1];
        *v22 = 0;
        *(v22 - 1) = v67;
        v23 = 0;
LABEL_14:
        *v22 = v23 - v15[1];
        do
        {
          v15 += 2;
          if ( v15 == v13 )
            goto LABEL_17;
        }
        while ( (unsigned __int64)*v15 > 0xFFFFFFFFFFFFFFFDLL );
        if ( v13 == v15 )
          goto LABEL_17;
        v16 = a10->m128i_i8[8] & 1;
        if ( v16 )
          goto LABEL_11;
      }
      sub_2446700((__int64)a10, 2 * v62);
      if ( (a10->m128i_i8[8] & 1) != 0 )
      {
        v70 = a10 + 1;
        v71 = 15;
      }
      else
      {
        v105 = a10[1].m128i_i32[2];
        v70 = (__m128i *)a10[1].m128i_i64[0];
        if ( !v105 )
          goto LABEL_231;
        v71 = v105 - 1;
      }
      LODWORD(v72) = v71 & (((0xBF58476D1CE4E5B9LL * *v15) >> 31) ^ (484763065 * *(_DWORD *)v15));
      v64 = &v70[(unsigned int)v72];
      v73 = v64->m128i_i64[0];
      if ( *v15 != v64->m128i_i64[0] )
      {
        v132 = 1;
        v79 = 0;
        while ( v73 != -1 )
        {
          if ( v73 == -2 && !v79 )
            v79 = v64;
          v72 = v71 & (unsigned int)(v72 + v132);
          v64 = &v70[v72];
          v73 = v64->m128i_i64[0];
          if ( *v15 == v64->m128i_i64[0] )
            goto LABEL_109;
          ++v132;
        }
LABEL_115:
        if ( v79 )
          v64 = v79;
      }
LABEL_109:
      v63 = a10->m128i_u32[2];
      goto LABEL_86;
    }
LABEL_17:
    v24 = *(i - 3);
    v25 = *(_QWORD *)(a2 + 40);
    v26 = a6 - v24;
    v177[0] = sub_BD5C60(a2);
    v27 = a6 - v24;
    if ( v24 >= a6 - v24 )
      v27 = v24;
    if ( v27 > 0xFFFFFFFE )
    {
      v28 = v27 / 0xFFFFFFFF + 1;
      v26 /= v28;
      v24 /= v28;
    }
    v29 = sub_B8C2F0(v177, v24, v26, 0);
    sub_29A5E20(a2, a3, *(i - 4), i[32], *((unsigned int *)i + 66), v29);
    v30 = *(_QWORD *)(a2 + 40);
    if ( v25 == sub_AA5510(v30) )
    {
      v140 = 0;
      v147 = v25 + 48;
      v149 = (_QWORD *)(*(_QWORD *)(*(_QWORD *)(v25 + 48) & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (_QWORD *)(v25 + 48) != v149 )
      {
        v135 = i;
        v92 = v30;
        v182 = a10;
        while ( 1 )
        {
          v93 = v149;
          v94 = *(v149 - 1);
          v150 = *v149 & 0xFFFFFFFFFFFFFFF8LL;
          v149 = (_QWORD *)v150;
          if ( !v94 )
            goto LABEL_145;
          v95 = 0;
          do
          {
            v96 = *(_QWORD *)(v94 + 24);
            if ( !sub_B46AA0(v96) )
            {
              if ( *(_BYTE *)v96 == 84 )
              {
                v95 = *(_QWORD *)(*(_QWORD *)(v96 - 8)
                                + 32LL * *(unsigned int *)(v96 + 72)
                                + 8LL * (unsigned int)((v94 - *(_QWORD *)(v96 - 8)) >> 5));
                if ( v92 != v95 )
                  goto LABEL_145;
              }
              else
              {
                v95 = *(_QWORD *)(v96 + 40);
                if ( v92 != v95 )
                  goto LABEL_145;
              }
            }
            v94 = *(_QWORD *)(v94 + 8);
          }
          while ( v94 );
          if ( !v95 )
            goto LABEL_145;
          v97 = *((_BYTE *)v93 - 24);
          if ( v97 == 84 )
            goto LABEL_145;
          v98 = (unsigned int)v97 - 39;
          if ( (unsigned int)v98 <= 0x38 )
          {
            v99 = 0x100060000000001LL;
            if ( _bittest64(&v99, v98) )
              goto LABEL_145;
          }
          v100 = v93 - 3;
          v139 = *((_BYTE *)v93 - 24);
          if ( (unsigned __int8)sub_B46790((unsigned __int8 *)v93 - 24, 0) )
            goto LABEL_145;
          if ( !(unsigned __int8)sub_B46900((unsigned __int8 *)v93 - 24) )
            goto LABEL_145;
          if ( v139 == 60 )
            goto LABEL_145;
          if ( (unsigned __int8)(v139 - 34) <= 0x33u )
          {
            v109 = 0x8000000000041LL;
            if ( _bittest64(&v109, (unsigned int)v139 - 34) )
            {
              if ( *(_BYTE *)*(v93 - 7) == 25
                || (unsigned __int8)sub_A73ED0(v93 + 6, 32)
                || (unsigned __int8)sub_B49560((__int64)(v93 - 3), 32)
                || (unsigned __int8)sub_A73ED0(v93 + 6, 6)
                || (unsigned __int8)sub_B49560((__int64)(v93 - 3), 6) )
              {
                goto LABEL_145;
              }
            }
          }
          if ( (unsigned __int8)sub_B46490((__int64)(v93 - 3)) )
            goto LABEL_145;
          if ( !(unsigned __int8)sub_B46420((__int64)(v93 - 3)) || (v101 = v93[1], v102 = v93[2] + 48LL, v102 == v101) )
          {
LABEL_209:
            v126 = v143;
            v128 = (unsigned __int64 *)sub_AA5190(v92);
            v130 = v129;
            if ( !v128 )
            {
              v127 = 0;
              v130 = 0;
            }
            LOBYTE(v126) = v127;
            v131 = v126;
            BYTE1(v131) = v130;
            sub_B44550(v100, v92, v128, v131);
            ++v140;
            if ( v147 == v150 )
            {
LABEL_146:
              i = v135;
              a10 = v182;
              goto LABEL_23;
            }
          }
          else
          {
            v103 = v101;
            while ( 1 )
            {
              v104 = v103 - 24;
              if ( !v103 )
                v104 = 0;
              if ( (unsigned __int8)sub_B46490(v104) )
                break;
              v103 = *(_QWORD *)(v103 + 8);
              if ( v102 == v103 )
                goto LABEL_209;
            }
LABEL_145:
            if ( v147 == v150 )
              goto LABEL_146;
          }
        }
      }
    }
    v140 = 0;
LABEL_23:
    v31 = **(_QWORD **)(a1 + 48);
    v138 = *(__int64 **)(a1 + 48);
    v32 = sub_B2BE50(v31);
    if ( sub_B6EA50(v32)
      || (v107 = sub_B2BE50(v31),
          v108 = sub_B6F970(v107),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v108 + 48LL))(v108)) )
    {
      sub_B174A0((__int64)v177, (__int64)"pgo-icall-prom", (__int64)"Promoted", 8, a2);
      sub_B18290((__int64)v177, "Promote indirect call to ", 0x19u);
      sub_B16080((__int64)v173, "DirectCallee", 12, (unsigned __int8 *)*(i - 4));
      v33 = sub_23FD640((__int64)v177, (__int64)v173);
      sub_B18290(v33, " with count ", 0xCu);
      sub_B16B10((__int64 *)&v166, "Count", 5, *(i - 3));
      v34 = sub_23FD640(v33, (__int64)&v166);
      sub_B18290(v34, " out of ", 8u);
      sub_B16B10(v162, "TotalCount", 10, a6);
      v35 = sub_23FD640(v34, (__int64)v162);
      sub_B18290(v35, ", sink ", 7u);
      sub_B16530(v158, "SinkCount", 9, v140);
      v36 = sub_23FD640(v35, (__int64)v158);
      sub_B18290(v36, " instruction(s) and compare ", 0x1Cu);
      sub_B169E0(v154, "VTable", 6, *((_DWORD *)v141 + 6) >> 1);
      v37 = sub_23FD640(v36, (__int64)v154);
      sub_B18290(v37, " vtable(s): {", 0xDu);
      if ( v156 != &v157 )
        j_j___libc_free_0((unsigned __int64)v156);
      if ( (__int64 *)v154[0] != &v155 )
        j_j___libc_free_0(v154[0]);
      if ( v160 != &v161 )
        j_j___libc_free_0((unsigned __int64)v160);
      if ( (__int64 *)v158[0] != &v159 )
        j_j___libc_free_0(v158[0]);
      if ( v164 != &v165 )
        j_j___libc_free_0((unsigned __int64)v164);
      if ( (__int64 *)v162[0] != &v163 )
        j_j___libc_free_0(v162[0]);
      if ( v170 != &v172 )
        j_j___libc_free_0((unsigned __int64)v170);
      if ( v166 != (unsigned __int64 *)&v168 )
        j_j___libc_free_0((unsigned __int64)v166);
      if ( v175 != v176 )
        j_j___libc_free_0((unsigned __int64)v175);
      if ( (_QWORD *)v173[0] != v174 )
        j_j___libc_free_0(v173[0]);
      LODWORD(v167) = 0;
      v168 = 0;
      v38 = *((_BYTE *)v141 + 24);
      v39 = *((_DWORD *)v141 + 6);
      v169 = &v167;
      v170 = &v167;
      v40 = v38 & 1;
      v171 = 0;
      if ( !(v39 >> 1) )
      {
        if ( v40 )
        {
          v110 = i;
          v111 = 32;
        }
        else
        {
          v110 = (__int64 *)*i;
          v111 = 2LL * *((unsigned int *)i + 2);
        }
        v42 = &v110[v111];
        v41 = (__int64)v42;
LABEL_49:
        if ( v40 )
          goto LABEL_118;
LABEL_50:
        if ( v42 == (__int64 *)v41 )
          goto LABEL_118;
        v181 = a10;
        v43 = (unsigned __int64 *)v41;
        v44 = (unsigned __int64 *)v42;
        v142 = i;
        v45 = *v42;
LABEL_80:
        v46 = &v167;
        if ( v169 == &v167 )
        {
          v49 = 1;
          goto LABEL_59;
        }
LABEL_90:
        if ( *(_QWORD *)(sub_220EF80((__int64)v46) + 32) < v45 && v46 )
        {
          v49 = 1;
          if ( v46 == &v167 )
            goto LABEL_59;
          goto LABEL_93;
        }
        while ( 1 )
        {
          v44 += 2;
          if ( v44 == v43 )
            break;
          if ( *v44 <= 0xFFFFFFFFFFFFFFFDLL )
          {
            if ( v43 == v44 )
              break;
            v46 = v168;
            v45 = *v44;
            if ( !v168 )
              goto LABEL_80;
            while ( 1 )
            {
              v47 = v46[4];
              v48 = (__int64 *)v46[3];
              if ( v45 < v47 )
                v48 = (__int64 *)v46[2];
              if ( !v48 )
                break;
              v46 = v48;
            }
            if ( v45 < v47 )
            {
              if ( v169 != v46 )
                goto LABEL_90;
LABEL_58:
              v49 = 1;
              if ( v46 == &v167 )
              {
LABEL_59:
                v146 = v49;
                v50 = sub_22077B0(0x28u);
                *(_QWORD *)(v50 + 32) = v45;
                sub_220F040(v146, v50, v46, &v167);
                ++v171;
                continue;
              }
LABEL_93:
              v49 = v45 < v46[4];
              goto LABEL_59;
            }
            if ( v45 > v47 )
              goto LABEL_58;
          }
        }
        i = v142;
        a10 = v181;
        if ( v169 == &v167 )
          goto LABEL_118;
        v51 = (__int64)v169;
        while ( 2 )
        {
          v52 = *(_QWORD *)(v51 + 32);
          v53 = *(_QWORD *)(a1 + 16);
          v54 = *(_DWORD *)(v53 + 144);
          v55 = *(_QWORD *)(v53 + 128);
          if ( v54 )
          {
            v56 = v54 - 1;
            v57 = v56 & (((0xBF58476D1CE4E5B9LL * v52) >> 31) ^ (484763065 * v52));
            v58 = (__int64 *)(v55 + 16LL * v57);
            v59 = *v58;
            if ( *v58 == v52 )
            {
LABEL_66:
              v60 = (unsigned __int8 *)v58[1];
LABEL_67:
              sub_B16080((__int64)v173, "VTable", 6, v60);
              sub_23FD640((__int64)v177, (__int64)v173);
              if ( v175 != v176 )
                j_j___libc_free_0((unsigned __int64)v175);
              if ( (_QWORD *)v173[0] != v174 )
                j_j___libc_free_0(v173[0]);
              v61 = (__int64 *)sub_220EF30(v51);
              v51 = (__int64)v61;
              if ( v61 == &v167 )
              {
                a10 = v181;
                goto LABEL_118;
              }
              if ( v169 != v61 )
                sub_B18290((__int64)v177, ", ", 2u);
              continue;
            }
            v68 = 1;
            while ( v59 != -1 )
            {
              v133 = v68 + 1;
              v134 = v56 & (v57 + v68);
              v57 = v134;
              v58 = (__int64 *)(v55 + 16 * v134);
              v59 = *v58;
              if ( v52 == *v58 )
                goto LABEL_66;
              v68 = v133;
            }
          }
          break;
        }
        v60 = 0;
        goto LABEL_67;
      }
      if ( v40 )
      {
        v41 = (__int64)(i + 32);
        v42 = i;
        goto LABEL_47;
      }
      v42 = (__int64 *)*i;
      v41 = *i + 16LL * *((unsigned int *)i + 2);
      if ( *i != v41 )
      {
LABEL_47:
        while ( (unsigned __int64)*v42 > 0xFFFFFFFFFFFFFFFDLL )
        {
          v42 += 2;
          if ( v42 == (__int64 *)v41 )
            goto LABEL_49;
        }
        goto LABEL_50;
      }
LABEL_118:
      sub_B18290((__int64)v177, "}", 1u);
      sub_24442B0((unsigned __int64)v168);
      sub_1049740(v138, (__int64)v177);
      v82 = v178;
      v177[0] = &unk_49D9D40;
      v83 = &v178[10 * v179];
      if ( v178 != v83 )
      {
        do
        {
          v83 -= 10;
          v84 = v83[4];
          if ( (unsigned __int64 *)v84 != v83 + 6 )
            j_j___libc_free_0(v84);
          if ( (unsigned __int64 *)*v83 != v83 + 2 )
            j_j___libc_free_0(*v83);
        }
        while ( v82 != v83 );
        v83 = v178;
      }
      if ( v83 != (unsigned __int64 *)&v180 )
        _libc_free((unsigned __int64)v83);
    }
    v85 = (unsigned int)v152;
    v86 = *(i - 3);
    v87 = (unsigned int)v152 + 1LL;
    if ( v87 > HIDWORD(v152) )
    {
      sub_C8D5F0((__int64)&v151, v153, v87, 8u, v80, v81);
      v85 = (unsigned int)v152;
    }
    *(_QWORD *)&v151[8 * v85] = v86;
    v88 = a6;
    v89 = (unsigned int)(v152 + 1);
    if ( *(i - 3) <= a6 )
      v88 = *(i - 3);
    LODWORD(v152) = v152 + 1;
    a6 -= v88;
    if ( v136 == i + 40 )
      break;
  }
  v112 = v151;
  v113 = 0;
  if ( (_DWORD)v89 )
  {
    v114 = &v151[8 * v89];
    v115 = &a8->m128i_i64[1];
    do
    {
      v116 = *v115 - *(_QWORD *)v112;
      if ( *(_QWORD *)v112 < (unsigned __int64)*v115 )
        v116 = 0;
      v112 += 8;
      v115 += 2;
      *(v115 - 2) = v116;
    }
    while ( v112 != v114 );
    m128i_i64 = a8[a9].m128i_i64;
    v118 = (16 * a9) >> 4;
    if ( 16 * a9 > 0 )
    {
      v119 = (16 * a9) >> 4;
      do
      {
        v120 = (__m128i *)sub_2207800(16 * v119);
        v121 = (unsigned __int64)v120;
        if ( v120 )
        {
          sub_2445140(a8, &a8[a9], v120, v119);
          goto LABEL_192;
        }
        v119 >>= 1;
      }
      while ( v119 );
      m128i_i64 = a8[a9].m128i_i64;
    }
    v121 = 0;
    sub_24447C0(a8->m128i_i64, m128i_i64);
LABEL_192:
    j_j___libc_free_0(v121);
    v122 = a8;
    while ( v118 > 0 )
    {
      while ( 1 )
      {
        v123 = &v122[v118 >> 1];
        if ( !v123->m128i_i64[1] )
          break;
        v122 = v123 + 1;
        v118 = v118 - (v118 >> 1) - 1;
        if ( v118 <= 0 )
          goto LABEL_196;
      }
      v118 >>= 1;
    }
LABEL_196:
    v124 = v122 - a8;
    sub_B99FD0(a2, 2u, 0);
    if ( a6 )
      sub_ED2230(*(__int64 ***)(a1 + 8), a2, a8->m128i_i64, v124, a6, 0, a7);
    v113 = 1;
    sub_2445FE0(a1, a3, a10);
    v112 = v151;
  }
  if ( v112 != v153 )
    _libc_free((unsigned __int64)v112);
  return v113;
}
