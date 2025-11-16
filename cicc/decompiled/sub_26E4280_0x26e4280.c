// Function: sub_26E4280
// Address: 0x26e4280
//
void __fastcall sub_26E4280(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5, _QWORD *a6)
{
  __int64 *v6; // r12
  bool v7; // zf
  const __m128i *v8; // r15
  int *v9; // r14
  int *v10; // r13
  size_t v11; // r12
  _QWORD **v12; // r13
  _QWORD *v13; // rbx
  _QWORD *v14; // r14
  unsigned __int64 v15; // r13
  const __m128i *v16; // rcx
  size_t v17; // r15
  unsigned __int64 v18; // rsi
  size_t v19; // rdx
  const void *v20; // rdi
  const void *v21; // rsi
  int v22; // eax
  _QWORD *v23; // r13
  __int64 v24; // r14
  __int64 i; // rbx
  __int64 v26; // r13
  _QWORD *v27; // r15
  __int64 v28; // rdx
  __int128 v29; // rax
  int *v30; // rax
  size_t v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rcx
  __int64 *v34; // r13
  __int64 v35; // rdi
  _BYTE *v36; // rax
  __int64 v37; // rdi
  _BYTE *v38; // rax
  __int64 v39; // rax
  __m128i *v40; // rdx
  __int64 v41; // rdi
  __m128i v42; // xmm0
  __int64 v43; // rdi
  _BYTE *v44; // rax
  __int64 v45; // rax
  __m128i *v46; // rdx
  __m128i v47; // xmm0
  __int64 v48; // rdi
  _BYTE *v49; // rax
  __int64 v50; // rdi
  _BYTE *v51; // rax
  __int64 v52; // rax
  __m128i *v53; // rdx
  __int64 v54; // rdi
  __m128i v55; // xmm0
  __int64 v56; // rdi
  _BYTE *v57; // rax
  __int64 v58; // rax
  __m128i *v59; // rdx
  __m128i v60; // xmm0
  __int64 v61; // rax
  __m128i *v62; // rcx
  unsigned int v63; // eax
  unsigned __int64 v64; // rdx
  __int64 *v65; // r13
  unsigned int v66; // eax
  __int64 v67; // rcx
  const char **v68; // rsi
  unsigned int v69; // eax
  __int64 v70; // rcx
  const char **v71; // rsi
  unsigned int v72; // eax
  __int64 v73; // rcx
  const char **v74; // rsi
  unsigned int v75; // eax
  __int64 v76; // rcx
  const char **v77; // rsi
  unsigned int v78; // eax
  __int64 v79; // r13
  __int64 v80; // rax
  _QWORD *v81; // rbx
  unsigned __int64 v82; // rdi
  __int64 v83; // rax
  __int64 v84; // rdx
  __int64 v85; // rsi
  char v86; // al
  unsigned __int64 v87; // rdx
  unsigned __int64 v88; // rbx
  _QWORD *v89; // r11
  _QWORD **v90; // rax
  size_t v91; // r13
  void *v92; // rax
  _QWORD *v93; // rax
  _QWORD *v94; // rsi
  unsigned __int64 v95; // rdi
  _QWORD *v96; // rcx
  unsigned __int64 v97; // rdx
  _QWORD **v98; // rax
  _QWORD *v99; // rdx
  _QWORD *v100; // rax
  _QWORD *v101; // rax
  _QWORD *v102; // rax
  const __m128i *v103; // r13
  __int64 v104; // rax
  __m128i *v105; // rax
  unsigned __int64 v106; // rsi
  __int64 v107; // r8
  int v108; // eax
  __int64 v109; // rcx
  _QWORD *v110; // rcx
  unsigned __int64 v111; // r8
  unsigned int v112; // eax
  __int64 v113; // rcx
  const char **v114; // r8
  __int64 v115; // rdi
  _BYTE *v116; // rax
  __int64 v117; // rdi
  _BYTE *v118; // rax
  __int64 v119; // rax
  __m128i *v120; // rdx
  __int64 v121; // rdi
  __m128i v122; // xmm0
  __int64 v123; // rdi
  _BYTE *v124; // rax
  __int64 v125; // rax
  __m128i *v126; // rdx
  __int64 v127; // rdi
  _BYTE *v128; // rax
  __int64 v129; // rdi
  _BYTE *v130; // rax
  __int64 v131; // rax
  __m128i *v132; // rdx
  __int64 v133; // rdi
  __m128i si128; // xmm0
  __int64 v135; // rdi
  _BYTE *v136; // rax
  __int64 v137; // rax
  __m128i *v138; // rdx
  __m128i v139; // xmm0
  _QWORD *v140; // rax
  unsigned __int64 v141; // r8
  const __m128i *v142; // rax
  __m128i *v143; // rdx
  _QWORD *v144; // rax
  unsigned __int64 v145; // r8
  const __m128i *v146; // r15
  _QWORD *v147; // rax
  unsigned __int64 v148; // r8
  const __m128i *v149; // rax
  __m128i *v150; // rdx
  _QWORD *v151; // rax
  unsigned __int64 v152; // r8
  const __m128i *v153; // rax
  __m128i *v154; // rdx
  _QWORD *v155; // rax
  unsigned __int64 v156; // r8
  const __m128i *v157; // rax
  __m128i *v158; // rdx
  _QWORD *v159; // rax
  unsigned __int64 v160; // rdx
  const __m128i *v161; // rax
  __m128i *v162; // rdx
  char *v163; // r14
  char *v164; // r14
  char *v165; // r14
  char *v166; // r14
  _QWORD *v167; // rax
  const __m128i *v168; // rdx
  __int64 v169; // rax
  __m128i *v170; // rax
  _QWORD *v171; // rax
  unsigned __int64 v172; // rdx
  const __m128i *v173; // rax
  __m128i *v174; // rdx
  char *v175; // r14
  char *v176; // r14
  const __m128i *v178; // [rsp+20h] [rbp-160h]
  __int64 v179; // [rsp+28h] [rbp-158h]
  _QWORD *v180; // [rsp+28h] [rbp-158h]
  __int64 v181; // [rsp+28h] [rbp-158h]
  int *v182; // [rsp+30h] [rbp-150h]
  __int64 v183; // [rsp+30h] [rbp-150h]
  _QWORD *v184; // [rsp+30h] [rbp-150h]
  _QWORD *v185; // [rsp+30h] [rbp-150h]
  __int64 v186; // [rsp+38h] [rbp-148h]
  __int64 v187; // [rsp+48h] [rbp-138h] BYREF
  const char *v188; // [rsp+50h] [rbp-130h] BYREF
  __int64 v189; // [rsp+58h] [rbp-128h]
  _QWORD *v190; // [rsp+60h] [rbp-120h]
  void *s; // [rsp+70h] [rbp-110h] BYREF
  unsigned __int64 v192; // [rsp+78h] [rbp-108h]
  _QWORD *v193; // [rsp+80h] [rbp-100h] BYREF
  __int64 v194; // [rsp+88h] [rbp-F8h]
  int v195; // [rsp+90h] [rbp-F0h] BYREF
  __int64 v196; // [rsp+98h] [rbp-E8h]
  _QWORD v197[2]; // [rsp+A0h] [rbp-E0h] BYREF
  unsigned __int64 v198; // [rsp+B0h] [rbp-D0h] BYREF
  __int64 v199; // [rsp+B8h] [rbp-C8h]
  _QWORD v200[6]; // [rsp+C0h] [rbp-C0h] BYREF
  char v201; // [rsp+F0h] [rbp-90h] BYREF

  v6 = (__int64 *)a1;
  if ( !LOBYTE(qword_4FF8040[17]) && !LOBYTE(qword_4FF7F60[17]) )
    return;
  v192 = 1;
  s = v197;
  v193 = 0;
  v7 = LOBYTE(qword_4FF8120[17]) == 0;
  v194 = 0;
  v195 = 1065353216;
  v196 = 0;
  v197[0] = 0;
  if ( !v7 )
  {
    v8 = *(const __m128i **)(a1 + 216);
    if ( v8 )
    {
      v9 = (int *)&v198;
      while ( 1 )
      {
        v10 = (int *)v8[1].m128i_i64[0];
        v11 = v8[1].m128i_u64[1];
        if ( v10 )
        {
          sub_C7D030(v9);
          sub_C7D280(v9, v10, v11);
          sub_C7D290(v9, &v188);
          v11 = (size_t)v188;
        }
        v179 = v11 % v192;
        v12 = *(_QWORD ***)((char *)s + v179 * 8);
        if ( v12 )
        {
          v13 = *v12;
          v182 = v9;
          v14 = (_QWORD *)*((_QWORD *)s + v11 % v192);
          v15 = v192;
          v16 = v8;
          v17 = v11 % v192;
          v18 = v13[3];
          while ( 1 )
          {
            if ( v11 == v18 )
            {
              v19 = v16[1].m128i_u64[1];
              if ( v19 == v13[2] )
              {
                v20 = (const void *)v16[1].m128i_i64[0];
                v21 = (const void *)v13[1];
                if ( v20 == v21 )
                  break;
                if ( v21 )
                {
                  if ( v20 )
                  {
                    v178 = v16;
                    v22 = memcmp(v20, v21, v19);
                    v16 = v178;
                    if ( !v22 )
                      break;
                  }
                }
              }
            }
            if ( !*v13 || (v18 = *(_QWORD *)(*v13 + 24LL), v14 = v13, v17 != v18 % v15) )
            {
              v9 = v182;
              v8 = v16;
              goto LABEL_91;
            }
            v13 = (_QWORD *)*v13;
          }
          v23 = v14;
          v8 = v16;
          v9 = v182;
          if ( !*v23 )
            goto LABEL_91;
          goto LABEL_20;
        }
LABEL_91:
        v83 = sub_22077B0(0x20u);
        if ( v83 )
          *(_QWORD *)v83 = 0;
        v84 = v194;
        v85 = v192;
        v184 = (_QWORD *)v83;
        *(__m128i *)(v83 + 8) = _mm_loadu_si128(v8 + 1);
        v86 = sub_222DA10((__int64)&v195, v85, v84, 1);
        a6 = v184;
        v88 = v87;
        if ( v86 )
          break;
        v89 = s;
LABEL_95:
        a6[3] = v11;
        v90 = (_QWORD **)&v89[v179];
        if ( v89[v179] )
        {
          *a6 = **v90;
          **v90 = a6;
        }
        else
        {
          v99 = v193;
          v193 = a6;
          *a6 = v99;
          if ( v99 )
          {
            v89[v99[3] % v192] = a6;
            v90 = (_QWORD **)((char *)s + v179 * 8);
          }
          *v90 = &v193;
        }
        ++v194;
LABEL_20:
        if ( (*(_BYTE *)(v8->m128i_i64[1] + 32) & 0xF) != 1 )
          ++*(_QWORD *)(a1 + 416);
        v8 = (const __m128i *)v8->m128i_i64[0];
        if ( !v8 )
        {
          v6 = (__int64 *)a1;
          goto LABEL_24;
        }
      }
      if ( v87 == 1 )
      {
        v197[0] = 0;
        v89 = v197;
      }
      else
      {
        if ( v87 > 0xFFFFFFFFFFFFFFFLL )
          sub_4261EA(&v195, v85, v87);
        v91 = 8 * v87;
        v92 = (void *)sub_22077B0(8 * v87);
        v93 = memset(v92, 0, v91);
        a6 = v184;
        v89 = v93;
      }
      v94 = v193;
      v193 = 0;
      if ( !v94 )
      {
LABEL_108:
        if ( s != v197 )
        {
          v180 = v89;
          v185 = a6;
          j_j___libc_free_0((unsigned __int64)s);
          v89 = v180;
          a6 = v185;
        }
        v192 = v88;
        s = v89;
        v179 = v11 % v88;
        goto LABEL_95;
      }
      a5 = &v193;
      v95 = 0;
      while ( 1 )
      {
        while ( 1 )
        {
          v96 = v94;
          v94 = (_QWORD *)*v94;
          v97 = v96[3] % v88;
          v98 = (_QWORD **)&v89[v97];
          if ( !*v98 )
            break;
          *v96 = **v98;
          **v98 = v96;
LABEL_104:
          if ( !v94 )
            goto LABEL_108;
        }
        *v96 = v193;
        v193 = v96;
        *v98 = &v193;
        if ( !*v96 )
        {
          v95 = v97;
          goto LABEL_104;
        }
        v89[v95] = v96;
        v95 = v97;
        if ( !v94 )
          goto LABEL_108;
      }
    }
  }
LABEL_24:
  v24 = *(_QWORD *)(*v6 + 32);
  for ( i = *v6 + 24; i != v24; v24 = *(_QWORD *)(v24 + 8) )
  {
    while ( 1 )
    {
      v26 = v24 - 56;
      if ( !v24 )
        v26 = 0;
      if ( !sub_B2FC80(v26) )
      {
        if ( (unsigned __int8)sub_B2D620(v26, "use-sample-profile", 0x12u) )
        {
          if ( (*(_BYTE *)(v26 + 32) & 0xF) != 1 )
          {
            v27 = (_QWORD *)v6[1];
            v198 = sub_B2D7E0(v26, "sample-profile-suffix-elision-policy", 0x24u);
            v183 = sub_A72240((__int64 *)&v198);
            v186 = v28;
            *(_QWORD *)&v29 = sub_BD5D20(v26);
            v30 = (int *)sub_C16140(v29, v183, v186);
            v32 = sub_26C7880(v27, v30, v31);
            v34 = (__int64 *)v32;
            if ( v32 )
              break;
          }
        }
      }
      v24 = *(_QWORD *)(v24 + 8);
      if ( i == v24 )
        goto LABEL_39;
    }
    ++v6[43];
    v6[48] += *(_QWORD *)(v32 + 56);
    if ( LOBYTE(qword_4FF8120[17]) && v194 )
      sub_26E0DD0((__int64)v6, v32, &s, v33);
    if ( unk_4F838D4 )
      sub_26E0A90(v6, v34, 1);
    sub_26E0C10((__int64)v6, v34);
    sub_26E3EF0((__int64)v6, v34);
  }
LABEL_39:
  if ( !LOBYTE(qword_4FF8040[17]) )
  {
LABEL_62:
    if ( LOBYTE(qword_4FF7F60[17]) )
      goto LABEL_63;
    goto LABEL_85;
  }
  if ( unk_4F838D4 )
  {
    v127 = (__int64)sub_CB72A0();
    v128 = *(_BYTE **)(v127 + 32);
    if ( *(_BYTE **)(v127 + 24) == v128 )
    {
      v127 = sub_CB6200(v127, (unsigned __int8 *)"(", 1u);
    }
    else
    {
      *v128 = 40;
      ++*(_QWORD *)(v127 + 32);
    }
    v129 = sub_CB59D0(v127, v6[44]);
    v130 = *(_BYTE **)(v129 + 32);
    if ( *(_BYTE **)(v129 + 24) == v130 )
    {
      v129 = sub_CB6200(v129, (unsigned __int8 *)"/", 1u);
    }
    else
    {
      *v130 = 47;
      ++*(_QWORD *)(v129 + 32);
    }
    v131 = sub_CB59D0(v129, v6[43]);
    v132 = *(__m128i **)(v131 + 32);
    v133 = v131;
    if ( *(_QWORD *)(v131 + 24) - (_QWORD)v132 <= 0x28u )
    {
      v133 = sub_CB6200(v131, ") of functions' profile are invalid and (", 0x29u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_43918E0);
      v132[2].m128i_i8[8] = 40;
      v132[2].m128i_i64[0] = 0x20646E612064696CLL;
      *v132 = si128;
      v132[1] = _mm_load_si128((const __m128i *)&xmmword_43918F0);
      *(_QWORD *)(v131 + 32) += 41LL;
    }
    v135 = sub_CB59D0(v133, v6[49]);
    v136 = *(_BYTE **)(v135 + 32);
    if ( *(_BYTE **)(v135 + 24) == v136 )
    {
      v135 = sub_CB6200(v135, (unsigned __int8 *)"/", 1u);
    }
    else
    {
      *v136 = 47;
      ++*(_QWORD *)(v135 + 32);
    }
    v137 = sub_CB59D0(v135, v6[48]);
    v138 = *(__m128i **)(v137 + 32);
    if ( *(_QWORD *)(v137 + 24) - (_QWORD)v138 <= 0x39u )
    {
      sub_CB6200(v137, ") of samples are discarded due to function hash mismatch.\n", 0x3Au);
    }
    else
    {
      v139 = _mm_load_si128((const __m128i *)&xmmword_4391900);
      qmemcpy(&v138[3], "mismatch.\n", 10);
      *v138 = v139;
      v138[1] = _mm_load_si128((const __m128i *)&xmmword_4391910);
      v138[2] = _mm_load_si128((const __m128i *)&xmmword_4391920);
      *(_QWORD *)(v137 + 32) += 58LL;
    }
  }
  if ( LOBYTE(qword_4FF8120[17]) )
  {
    v115 = (__int64)sub_CB72A0();
    v116 = *(_BYTE **)(v115 + 32);
    if ( *(_BYTE **)(v115 + 24) == v116 )
    {
      v115 = sub_CB6200(v115, (unsigned __int8 *)"(", 1u);
    }
    else
    {
      *v116 = 40;
      ++*(_QWORD *)(v115 + 32);
    }
    v117 = sub_CB59D0(v115, v6[52]);
    v118 = *(_BYTE **)(v117 + 32);
    if ( *(_BYTE **)(v117 + 24) == v118 )
    {
      v117 = sub_CB6200(v117, (unsigned __int8 *)"/", 1u);
    }
    else
    {
      *v118 = 47;
      ++*(_QWORD *)(v117 + 32);
    }
    v119 = sub_CB59D0(v117, v6[43]);
    v120 = *(__m128i **)(v119 + 32);
    v121 = v119;
    if ( *(_QWORD *)(v119 + 24) - (_QWORD)v120 <= 0x28u )
    {
      v121 = sub_CB6200(v119, ") of functions' profile are matched and (", 0x29u);
    }
    else
    {
      v122 = _mm_load_si128((const __m128i *)&xmmword_43918E0);
      v120[2].m128i_i8[8] = 40;
      v120[2].m128i_i64[0] = 0x20646E6120646568LL;
      *v120 = v122;
      v120[1] = _mm_load_si128((const __m128i *)&xmmword_4391930);
      *(_QWORD *)(v119 + 32) += 41LL;
    }
    v123 = sub_CB59D0(v121, v6[53]);
    v124 = *(_BYTE **)(v123 + 32);
    if ( *(_BYTE **)(v123 + 24) == v124 )
    {
      v123 = sub_CB6200(v123, (unsigned __int8 *)"/", 1u);
    }
    else
    {
      *v124 = 47;
      ++*(_QWORD *)(v123 + 32);
    }
    v125 = sub_CB59D0(v123, v6[48]);
    v126 = *(__m128i **)(v125 + 32);
    if ( *(_QWORD *)(v125 + 24) - (_QWORD)v126 <= 0x2Fu )
    {
      sub_CB6200(v125, ") of samples are reused by call graph matching.\n", 0x30u);
    }
    else
    {
      *v126 = _mm_load_si128((const __m128i *)&xmmword_4391900);
      v126[1] = _mm_load_si128((const __m128i *)&xmmword_4391940);
      v126[2] = _mm_load_si128((const __m128i *)&xmmword_4391950);
      *(_QWORD *)(v125 + 32) += 48LL;
    }
  }
  v35 = (__int64)sub_CB72A0();
  v36 = *(_BYTE **)(v35 + 32);
  if ( *(_BYTE **)(v35 + 24) == v36 )
  {
    v35 = sub_CB6200(v35, (unsigned __int8 *)"(", 1u);
  }
  else
  {
    *v36 = 40;
    ++*(_QWORD *)(v35 + 32);
  }
  v37 = sub_CB59D0(v35, v6[46] + v6[47]);
  v38 = *(_BYTE **)(v37 + 32);
  if ( *(_BYTE **)(v37 + 24) == v38 )
  {
    v37 = sub_CB6200(v37, (unsigned __int8 *)"/", 1u);
  }
  else
  {
    *v38 = 47;
    ++*(_QWORD *)(v37 + 32);
  }
  v39 = sub_CB59D0(v37, v6[45]);
  v40 = *(__m128i **)(v39 + 32);
  v41 = v39;
  if ( *(_QWORD *)(v39 + 24) - (_QWORD)v40 <= 0x28u )
  {
    v41 = sub_CB6200(v39, ") of callsites' profile are invalid and (", 0x29u);
  }
  else
  {
    v42 = _mm_load_si128((const __m128i *)&xmmword_4391960);
    v40[2].m128i_i8[8] = 40;
    v40[2].m128i_i64[0] = 0x20646E612064696CLL;
    *v40 = v42;
    v40[1] = _mm_load_si128((const __m128i *)&xmmword_43918F0);
    *(_QWORD *)(v39 + 32) += 41LL;
  }
  v43 = sub_CB59D0(v41, v6[50] + v6[51]);
  v44 = *(_BYTE **)(v43 + 32);
  if ( *(_BYTE **)(v43 + 24) == v44 )
  {
    v43 = sub_CB6200(v43, (unsigned __int8 *)"/", 1u);
  }
  else
  {
    *v44 = 47;
    ++*(_QWORD *)(v43 + 32);
  }
  v45 = sub_CB59D0(v43, v6[48]);
  v46 = *(__m128i **)(v45 + 32);
  if ( *(_QWORD *)(v45 + 24) - (_QWORD)v46 <= 0x3Du )
  {
    sub_CB6200(v45, ") of samples are discarded due to callsite location mismatch.\n", 0x3Eu);
  }
  else
  {
    v47 = _mm_load_si128((const __m128i *)&xmmword_4391900);
    qmemcpy(&v46[3], "ion mismatch.\n", 14);
    *v46 = v47;
    v46[1] = _mm_load_si128((const __m128i *)&xmmword_4391910);
    v46[2] = _mm_load_si128((const __m128i *)&xmmword_4391970);
    *(_QWORD *)(v45 + 32) += 62LL;
  }
  v48 = (__int64)sub_CB72A0();
  v49 = *(_BYTE **)(v48 + 32);
  if ( *(_BYTE **)(v48 + 24) == v49 )
  {
    v48 = sub_CB6200(v48, (unsigned __int8 *)"(", 1u);
  }
  else
  {
    *v49 = 40;
    ++*(_QWORD *)(v48 + 32);
  }
  v50 = sub_CB59D0(v48, v6[47]);
  v51 = *(_BYTE **)(v50 + 32);
  if ( *(_BYTE **)(v50 + 24) == v51 )
  {
    v50 = sub_CB6200(v50, (unsigned __int8 *)"/", 1u);
  }
  else
  {
    *v51 = 47;
    ++*(_QWORD *)(v50 + 32);
  }
  v52 = sub_CB59D0(v50, v6[47] + v6[46]);
  v53 = *(__m128i **)(v52 + 32);
  v54 = v52;
  if ( *(_QWORD *)(v52 + 24) - (_QWORD)v53 <= 0x13u )
  {
    v54 = sub_CB6200(v52, ") of callsites and (", 0x14u);
  }
  else
  {
    v55 = _mm_load_si128((const __m128i *)&xmmword_4391980);
    v53[1].m128i_i32[0] = 673211502;
    *v53 = v55;
    *(_QWORD *)(v52 + 32) += 20LL;
  }
  v56 = sub_CB59D0(v54, v6[51]);
  v57 = *(_BYTE **)(v56 + 32);
  if ( *(_BYTE **)(v56 + 24) == v57 )
  {
    v56 = sub_CB6200(v56, (unsigned __int8 *)"/", 1u);
  }
  else
  {
    *v57 = 47;
    ++*(_QWORD *)(v56 + 32);
  }
  v58 = sub_CB59D0(v56, v6[51] + v6[50]);
  v59 = *(__m128i **)(v58 + 32);
  if ( *(_QWORD *)(v58 + 24) - (_QWORD)v59 > 0x35u )
  {
    v60 = _mm_load_si128((const __m128i *)&xmmword_4391900);
    v59[3].m128i_i32[0] = 1735289192;
    v59[3].m128i_i16[2] = 2606;
    *v59 = v60;
    v59[1] = _mm_load_si128((const __m128i *)&xmmword_4391990);
    v59[2] = _mm_load_si128((const __m128i *)&xmmword_43919A0);
    *(_QWORD *)(v58 + 32) += 54LL;
    goto LABEL_62;
  }
  sub_CB6200(v58, ") of samples are recovered by stale profile matching.\n", 0x36u);
  if ( LOBYTE(qword_4FF7F60[17]) )
  {
LABEL_63:
    v61 = *(_QWORD *)*v6;
    v198 = (unsigned __int64)v200;
    v187 = v61;
    v199 = 0x200000000LL;
    if ( unk_4F838D4 )
    {
      v200[1] = 19;
      v200[0] = "NumStaleProfileFunc";
      v100 = (_QWORD *)v6[44];
      v200[4] = 17;
      v200[2] = v100;
      v200[3] = "TotalProfiledFunc";
      v101 = (_QWORD *)v6[43];
      LODWORD(v199) = 2;
      v200[5] = v101;
      v102 = (_QWORD *)v6[49];
      v188 = "MismatchedFunctionSamples";
      v189 = 25;
      v190 = v102;
      if ( &v188 >= v200 && &v188 < (const char **)&v201 )
      {
        sub_C8D5F0((__int64)&v198, v200, 3u, 0x18u, (__int64)a5, (__int64)a6);
        v104 = v198;
        v103 = (const __m128i *)(v198 + (char *)&v188 - (char *)v200);
      }
      else
      {
        v103 = (const __m128i *)&v188;
        sub_C8D5F0((__int64)&v198, v200, 3u, 0x18u, (__int64)a5, (__int64)a6);
        v104 = v198;
      }
      v105 = (__m128i *)(v104 + 24LL * (unsigned int)v199);
      *v105 = _mm_loadu_si128(v103);
      v105[1].m128i_i64[0] = v103[1].m128i_i64[0];
      v106 = HIDWORD(v199);
      LODWORD(v199) = v199 + 1;
      v107 = (unsigned int)v199;
      v108 = v199;
      v64 = HIDWORD(v199);
      v109 = 24LL * (unsigned int)v199;
      if ( (unsigned int)v199 >= (unsigned __int64)HIDWORD(v199) )
      {
        v167 = (_QWORD *)v6[48];
        a6 = (_QWORD *)((unsigned int)v199 + 1LL);
        v189 = 20;
        v188 = "TotalFunctionSamples";
        v168 = (const __m128i *)&v188;
        v190 = v167;
        v169 = v198;
        if ( HIDWORD(v199) < (unsigned __int64)a6 )
        {
          if ( v198 > (unsigned __int64)&v188 || (v181 = v198, (unsigned __int64)&v188 >= v198 + v109) )
          {
            sub_C8D5F0((__int64)&v198, v200, (unsigned int)v199 + 1LL, 0x18u, (unsigned int)v199, (__int64)a6);
            v107 = (unsigned int)v199;
            v169 = v198;
            v168 = (const __m128i *)&v188;
          }
          else
          {
            sub_C8D5F0((__int64)&v198, v200, (unsigned int)v199 + 1LL, 0x18u, (unsigned int)v199, (__int64)a6);
            v107 = (unsigned int)v199;
            v169 = v198;
            v168 = (const __m128i *)((char *)&v188 + v198 - v181);
          }
        }
        v170 = (__m128i *)(v169 + 24 * v107);
        *v170 = _mm_loadu_si128(v168);
        v170[1].m128i_i64[0] = v168[1].m128i_i64[0];
        v106 = HIDWORD(v199);
        v65 = (__int64 *)v198;
        v63 = v199 + 1;
        LODWORD(v199) = v199 + 1;
        v64 = HIDWORD(v199);
      }
      else
      {
        v65 = (__int64 *)v198;
        v110 = (_QWORD *)(v198 + v109);
        if ( v110 )
        {
          v110[1] = 20;
          *v110 = "TotalFunctionSamples";
          v110[2] = v6[48];
          v106 = HIDWORD(v199);
          v108 = v199;
          v65 = (__int64 *)v198;
          v64 = HIDWORD(v199);
        }
        v63 = v108 + 1;
        LODWORD(v199) = v63;
      }
      v111 = v63;
      v62 = (__m128i *)&v65[3 * v63];
      if ( !LOBYTE(qword_4FF8120[17]) )
      {
LABEL_131:
        if ( v64 <= v111 )
        {
          v144 = (_QWORD *)v6[46];
          v145 = v111 + 1;
          v189 = 22;
          v188 = "NumMismatchedCallsites";
          v146 = (const __m128i *)&v188;
          v190 = v144;
          if ( v64 < v145 )
          {
            if ( v65 > (__int64 *)&v188 || v62 <= (__m128i *)&v188 )
            {
              sub_C8D5F0((__int64)&v198, v200, v145, 0x18u, v145, (__int64)a6);
              v62 = (__m128i *)(v198 + 24LL * (unsigned int)v199);
            }
            else
            {
              sub_C8D5F0((__int64)&v198, v200, v145, 0x18u, v145, (__int64)a6);
              v146 = (const __m128i *)(v198 + (char *)&v188 - (char *)v65);
              v62 = (__m128i *)(v198 + 24LL * (unsigned int)v199);
            }
          }
          *v62 = _mm_loadu_si128(v146);
          v62[1].m128i_i64[0] = v146[1].m128i_i64[0];
          v64 = HIDWORD(v199);
          v65 = (__int64 *)v198;
          v66 = v199 + 1;
          LODWORD(v199) = v199 + 1;
          goto LABEL_67;
        }
        if ( !v62 )
        {
LABEL_66:
          v66 = v63 + 1;
          LODWORD(v199) = v66;
LABEL_67:
          v67 = v66;
          v68 = (const char **)&v65[3 * v66];
          if ( v66 >= v64 )
          {
            v155 = (_QWORD *)v6[47];
            v156 = v67 + 1;
            v189 = 21;
            v188 = "NumRecoveredCallsites";
            v190 = v155;
            v157 = (const __m128i *)&v188;
            if ( v67 + 1 > v64 )
            {
              if ( v65 > (__int64 *)&v188 || v68 <= &v188 )
              {
                sub_C8D5F0((__int64)&v198, v200, v156, 0x18u, v156, (__int64)a6);
                v65 = (__int64 *)v198;
                v67 = (unsigned int)v199;
                v157 = (const __m128i *)&v188;
              }
              else
              {
                v165 = (char *)((char *)&v188 - (char *)v65);
                sub_C8D5F0((__int64)&v198, v200, v156, 0x18u, v156, (__int64)a6);
                v65 = (__int64 *)v198;
                v67 = (unsigned int)v199;
                v157 = (const __m128i *)&v165[v198];
              }
            }
            v158 = (__m128i *)&v65[3 * v67];
            *v158 = _mm_loadu_si128(v157);
            v158[1].m128i_i64[0] = v157[1].m128i_i64[0];
            v64 = HIDWORD(v199);
            v65 = (__int64 *)v198;
            v69 = v199 + 1;
            LODWORD(v199) = v199 + 1;
          }
          else
          {
            if ( v68 )
            {
              v68[1] = (const char *)21;
              *v68 = "NumRecoveredCallsites";
              v68[2] = (const char *)v6[47];
              v66 = v199;
              v64 = HIDWORD(v199);
              v65 = (__int64 *)v198;
            }
            v69 = v66 + 1;
            LODWORD(v199) = v69;
          }
          v70 = v69;
          v71 = (const char **)&v65[3 * v69];
          if ( v64 <= v69 )
          {
            v147 = (_QWORD *)v6[45];
            v148 = v70 + 1;
            v189 = 22;
            v188 = "TotalProfiledCallsites";
            v190 = v147;
            v149 = (const __m128i *)&v188;
            if ( v64 < v70 + 1 )
            {
              if ( v65 > (__int64 *)&v188 || v71 <= &v188 )
              {
                sub_C8D5F0((__int64)&v198, v200, v148, 0x18u, v148, (__int64)a6);
                v65 = (__int64 *)v198;
                v70 = (unsigned int)v199;
                v149 = (const __m128i *)&v188;
              }
              else
              {
                v164 = (char *)((char *)&v188 - (char *)v65);
                sub_C8D5F0((__int64)&v198, v200, v148, 0x18u, v148, (__int64)a6);
                v65 = (__int64 *)v198;
                v70 = (unsigned int)v199;
                v149 = (const __m128i *)&v164[v198];
              }
            }
            v150 = (__m128i *)&v65[3 * v70];
            *v150 = _mm_loadu_si128(v149);
            v150[1].m128i_i64[0] = v149[1].m128i_i64[0];
            v64 = HIDWORD(v199);
            v65 = (__int64 *)v198;
            v72 = v199 + 1;
            LODWORD(v199) = v199 + 1;
          }
          else
          {
            if ( v71 )
            {
              v71[1] = (const char *)22;
              *v71 = "TotalProfiledCallsites";
              v71[2] = (const char *)v6[45];
              v69 = v199;
              v64 = HIDWORD(v199);
              v65 = (__int64 *)v198;
            }
            v72 = v69 + 1;
            LODWORD(v199) = v72;
          }
          v73 = v72;
          v74 = (const char **)&v65[3 * v72];
          if ( v64 <= v72 )
          {
            v151 = (_QWORD *)v6[50];
            v152 = v73 + 1;
            v189 = 25;
            v188 = "MismatchedCallsiteSamples";
            v190 = v151;
            v153 = (const __m128i *)&v188;
            if ( v64 < v73 + 1 )
            {
              if ( v65 > (__int64 *)&v188 || v74 <= &v188 )
              {
                sub_C8D5F0((__int64)&v198, v200, v152, 0x18u, v152, (__int64)a6);
                v65 = (__int64 *)v198;
                v73 = (unsigned int)v199;
                v153 = (const __m128i *)&v188;
              }
              else
              {
                v166 = (char *)((char *)&v188 - (char *)v65);
                sub_C8D5F0((__int64)&v198, v200, v152, 0x18u, v152, (__int64)a6);
                v65 = (__int64 *)v198;
                v73 = (unsigned int)v199;
                v153 = (const __m128i *)&v166[v198];
              }
            }
            v154 = (__m128i *)&v65[3 * v73];
            *v154 = _mm_loadu_si128(v153);
            v154[1].m128i_i64[0] = v153[1].m128i_i64[0];
            v64 = HIDWORD(v199);
            v65 = (__int64 *)v198;
            v75 = v199 + 1;
            LODWORD(v199) = v199 + 1;
          }
          else
          {
            if ( v74 )
            {
              v74[1] = (const char *)25;
              *v74 = "MismatchedCallsiteSamples";
              v74[2] = (const char *)v6[50];
              v72 = v199;
              v64 = HIDWORD(v199);
              v65 = (__int64 *)v198;
            }
            v75 = v72 + 1;
            LODWORD(v199) = v75;
          }
          v76 = v75;
          v77 = (const char **)&v65[3 * v75];
          if ( v64 <= v75 )
          {
            v140 = (_QWORD *)v6[51];
            v141 = v76 + 1;
            v189 = 24;
            v188 = "RecoveredCallsiteSamples";
            v190 = v140;
            v142 = (const __m128i *)&v188;
            if ( v64 < v76 + 1 )
            {
              if ( v65 > (__int64 *)&v188 || v77 <= &v188 )
              {
                sub_C8D5F0((__int64)&v198, v200, v141, 0x18u, v141, (__int64)a6);
                v65 = (__int64 *)v198;
                v76 = (unsigned int)v199;
                v142 = (const __m128i *)&v188;
              }
              else
              {
                v163 = (char *)((char *)&v188 - (char *)v65);
                sub_C8D5F0((__int64)&v198, v200, v141, 0x18u, v141, (__int64)a6);
                v65 = (__int64 *)v198;
                v76 = (unsigned int)v199;
                v142 = (const __m128i *)&v163[v198];
              }
            }
            v143 = (__m128i *)&v65[3 * v76];
            *v143 = _mm_loadu_si128(v142);
            v143[1].m128i_i64[0] = v142[1].m128i_i64[0];
            v65 = (__int64 *)v198;
            v78 = v199 + 1;
            LODWORD(v199) = v199 + 1;
          }
          else
          {
            if ( v77 )
            {
              v77[1] = (const char *)24;
              *v77 = "RecoveredCallsiteSamples";
              v77[2] = (const char *)v6[51];
              v75 = v199;
              v65 = (__int64 *)v198;
            }
            v78 = v75 + 1;
            LODWORD(v199) = v78;
          }
          v79 = sub_B8D360(&v187, v65, v78);
          v80 = sub_BA8E40(*v6, "llvm.stats", 0xAu);
          sub_B979A0(v80, v79);
          if ( (_QWORD *)v198 != v200 )
            _libc_free(v198);
          goto LABEL_85;
        }
LABEL_65:
        v62->m128i_i64[1] = 22;
        v62->m128i_i64[0] = (__int64)"NumMismatchedCallsites";
        v62[1].m128i_i64[0] = v6[46];
        v63 = v199;
        v64 = HIDWORD(v199);
        v65 = (__int64 *)v198;
        goto LABEL_66;
      }
      if ( v106 <= v63 )
      {
        v171 = (_QWORD *)v6[52];
        v189 = 33;
        v188 = "NumCallGraphRecoveredProfiledFunc";
        v172 = v111 + 1;
        v190 = v171;
        v173 = (const __m128i *)&v188;
        if ( v106 < v111 + 1 )
        {
          if ( v65 > (__int64 *)&v188 || v62 <= (__m128i *)&v188 )
          {
            sub_C8D5F0((__int64)&v198, v200, v172, 0x18u, v111, (__int64)a6);
            v65 = (__int64 *)v198;
            v111 = (unsigned int)v199;
            v173 = (const __m128i *)&v188;
          }
          else
          {
            v175 = (char *)((char *)&v188 - (char *)v65);
            sub_C8D5F0((__int64)&v198, v200, v172, 0x18u, v111, (__int64)a6);
            v65 = (__int64 *)v198;
            v111 = (unsigned int)v199;
            v173 = (const __m128i *)&v175[v198];
          }
        }
        v174 = (__m128i *)&v65[3 * v111];
        *v174 = _mm_loadu_si128(v173);
        v174[1].m128i_i64[0] = v173[1].m128i_i64[0];
        v65 = (__int64 *)v198;
        v106 = HIDWORD(v199);
        v112 = v199 + 1;
        LODWORD(v199) = v199 + 1;
        goto LABEL_127;
      }
      if ( !v62 )
      {
LABEL_126:
        v112 = v199 + 1;
        LODWORD(v199) = v199 + 1;
LABEL_127:
        v113 = v112;
        v64 = v106;
        v114 = (const char **)&v65[3 * v112];
        if ( v112 >= v106 )
        {
          v159 = (_QWORD *)v6[53];
          v189 = 32;
          v188 = "NumCallGraphRecoveredFuncSamples";
          v160 = v113 + 1;
          v190 = v159;
          v161 = (const __m128i *)&v188;
          if ( v106 < v113 + 1 )
          {
            if ( v65 > (__int64 *)&v188 || v114 <= &v188 )
            {
              sub_C8D5F0((__int64)&v198, v200, v160, 0x18u, (__int64)v114, (__int64)a6);
              v65 = (__int64 *)v198;
              v113 = (unsigned int)v199;
              v161 = (const __m128i *)&v188;
            }
            else
            {
              v176 = (char *)((char *)&v188 - (char *)v65);
              sub_C8D5F0((__int64)&v198, v200, v160, 0x18u, (__int64)v114, (__int64)a6);
              v65 = (__int64 *)v198;
              v113 = (unsigned int)v199;
              v161 = (const __m128i *)&v176[v198];
            }
          }
          v162 = (__m128i *)&v65[3 * v113];
          *v162 = _mm_loadu_si128(v161);
          v162[1].m128i_i64[0] = v161[1].m128i_i64[0];
          v65 = (__int64 *)v198;
          v64 = HIDWORD(v199);
          LODWORD(v199) = v199 + 1;
          v111 = (unsigned int)v199;
          v63 = v199;
          v62 = (__m128i *)(v198 + 24LL * (unsigned int)v199);
        }
        else
        {
          if ( v114 )
          {
            v114[1] = (const char *)32;
            *v114 = "NumCallGraphRecoveredFuncSamples";
            v114[2] = (const char *)v6[53];
            v112 = v199;
            v64 = HIDWORD(v199);
            v65 = (__int64 *)v198;
          }
          LODWORD(v199) = v112 + 1;
          v63 = v112 + 1;
          v111 = v63;
          v62 = (__m128i *)&v65[3 * v63];
        }
        goto LABEL_131;
      }
    }
    else
    {
      v62 = (__m128i *)v200;
      if ( !LOBYTE(qword_4FF8120[17]) )
        goto LABEL_65;
    }
    v62->m128i_i64[1] = 33;
    v62->m128i_i64[0] = (__int64)"NumCallGraphRecoveredProfiledFunc";
    v62[1].m128i_i64[0] = v6[52];
    v65 = (__int64 *)v198;
    v106 = HIDWORD(v199);
    goto LABEL_126;
  }
LABEL_85:
  v81 = v193;
  while ( v81 )
  {
    v82 = (unsigned __int64)v81;
    v81 = (_QWORD *)*v81;
    j_j___libc_free_0(v82);
  }
  memset(s, 0, 8 * v192);
  v194 = 0;
  v193 = 0;
  if ( s != v197 )
    j_j___libc_free_0((unsigned __int64)s);
}
