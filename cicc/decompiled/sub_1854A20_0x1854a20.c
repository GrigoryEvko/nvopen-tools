// Function: sub_1854A20
// Address: 0x1854a20
//
__int64 __fastcall sub_1854A20(
        __int64 a1,
        char **a2,
        __int64 a3,
        __int64 **a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  int v13; // ecx
  bool v14; // dl
  char v15; // al
  _QWORD *v16; // rbx
  _QWORD *v17; // r12
  __int64 v18; // rsi
  __int64 v19; // r13
  __int64 v20; // rax
  __int64 *v21; // rbx
  __int64 v22; // r14
  __int64 *v23; // r12
  __int64 *v24; // rsi
  __int64 v25; // rax
  __int64 *v26; // rax
  __int64 v27; // rdx
  __int64 *v28; // r14
  __int64 *v29; // r12
  __m128i *v30; // rbx
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 *v33; // rdx
  __int64 *v34; // rax
  __int64 v35; // rdx
  __int64 *v36; // r14
  unsigned __int8 *v37; // rsi
  int v38; // eax
  __int64 v39; // rdx
  __int64 *v40; // r12
  __m128 v41; // xmm0
  bool v42; // zf
  char v43; // dl
  __int64 *v44; // r15
  __int64 v45; // rsi
  __int64 v46; // rdx
  unsigned __int64 v47; // rax
  unsigned __int64 v48; // r12
  __int64 v49; // r12
  __int64 *v50; // r15
  __int64 *v51; // rdx
  unsigned __int64 v52; // rax
  __int64 *v53; // r15
  __int64 *v54; // rdx
  __int64 *v55; // r14
  __int64 v56; // r13
  size_t v57; // r15
  __int64 *v58; // r15
  unsigned __int64 v59; // r15
  __int64 v60; // rax
  __int64 v61; // rsi
  char v62; // al
  unsigned __int64 v63; // rdx
  __int64 v64; // r15
  char v65; // al
  _QWORD *v66; // rax
  _QWORD *i; // rdx
  char v68; // cl
  __int64 v69; // r15
  int v70; // eax
  char v71; // cl
  char v72; // dl
  __int64 v73; // rax
  double v74; // xmm4_8
  double v75; // xmm5_8
  __int64 v76; // rax
  unsigned __int64 v77; // rsi
  _QWORD *v78; // r12
  __int64 v79; // rdx
  __int64 v80; // rax
  __int64 v82; // rdi
  char *v83; // rsi
  __int64 v84; // rdx
  double v85; // xmm4_8
  double v86; // xmm5_8
  char **v87; // rcx
  __int64 v88; // rdx
  __int64 v89; // rax
  __int64 *v90; // rdi
  __int64 v91; // rax
  _QWORD *v92; // r12
  _QWORD *v93; // rbx
  __int64 v94; // rsi
  __int64 *v95; // r12
  unsigned __int64 v96; // r12
  __int64 *v97; // r13
  const char *v98; // rax
  size_t v99; // rdx
  _WORD *v100; // rdi
  char *v101; // rsi
  size_t v102; // r15
  unsigned __int64 v103; // rax
  _BYTE *v104; // rax
  __int64 v105; // r15
  __int64 v106; // rax
  __int64 v107; // rax
  __int64 v108; // rdx
  __int64 v109; // r14
  __int64 v110; // rax
  __int64 v111; // rax
  unsigned __int64 v112; // rax
  char v113; // al
  __int64 v114; // [rsp-8h] [rbp-2A8h]
  int v115; // [rsp+24h] [rbp-27Ch]
  __int64 *v120; // [rsp+50h] [rbp-250h]
  const __m128i *v121; // [rsp+58h] [rbp-248h]
  size_t v122; // [rsp+60h] [rbp-240h]
  size_t v123; // [rsp+60h] [rbp-240h]
  __int64 *v124; // [rsp+60h] [rbp-240h]
  __int64 *v125; // [rsp+60h] [rbp-240h]
  int *v126; // [rsp+68h] [rbp-238h]
  __int64 *v127; // [rsp+68h] [rbp-238h]
  int *v128; // [rsp+68h] [rbp-238h]
  __int64 *v129; // [rsp+68h] [rbp-238h]
  int *v130; // [rsp+68h] [rbp-238h]
  __m128i *v131; // [rsp+68h] [rbp-238h]
  __int64 *v132; // [rsp+70h] [rbp-230h]
  __int64 *v133; // [rsp+70h] [rbp-230h]
  __int64 *v134; // [rsp+70h] [rbp-230h]
  unsigned __int64 v135; // [rsp+80h] [rbp-220h] BYREF
  char v136; // [rsp+88h] [rbp-218h]
  __int64 *v137; // [rsp+90h] [rbp-210h] BYREF
  _QWORD v138[2]; // [rsp+98h] [rbp-208h] BYREF
  __int64 v139; // [rsp+A8h] [rbp-1F8h]
  __int64 v140; // [rsp+B0h] [rbp-1F0h]
  unsigned __int64 v141; // [rsp+C0h] [rbp-1E0h] BYREF
  __int64 v142; // [rsp+C8h] [rbp-1D8h] BYREF
  __int64 v143; // [rsp+D0h] [rbp-1D0h] BYREF
  __int64 v144; // [rsp+D8h] [rbp-1C8h]
  __int64 v145; // [rsp+E0h] [rbp-1C0h]
  __int64 v146; // [rsp+F0h] [rbp-1B0h] BYREF
  int v147; // [rsp+F8h] [rbp-1A8h] BYREF
  __int64 v148; // [rsp+100h] [rbp-1A0h]
  const __m128i *v149; // [rsp+108h] [rbp-198h]
  int *v150; // [rsp+110h] [rbp-190h]
  __int64 v151; // [rsp+118h] [rbp-188h]
  __int64 v152; // [rsp+120h] [rbp-180h] BYREF
  __int64 v153; // [rsp+128h] [rbp-178h]
  __int64 v154; // [rsp+130h] [rbp-170h]
  __int64 v155; // [rsp+138h] [rbp-168h]
  char **v156; // [rsp+140h] [rbp-160h]
  __int64 *v157; // [rsp+148h] [rbp-158h]
  __int64 v158; // [rsp+150h] [rbp-150h]
  _BYTE v159[16]; // [rsp+160h] [rbp-140h] BYREF
  __int64 v160; // [rsp+170h] [rbp-130h]
  __int64 *v161; // [rsp+190h] [rbp-110h]
  int v162; // [rsp+1A0h] [rbp-100h]
  _QWORD *v163; // [rsp+1B0h] [rbp-F0h]
  unsigned int v164; // [rsp+1C0h] [rbp-E0h]
  __m128i v165; // [rsp+1D0h] [rbp-D0h] BYREF
  __int64 (__fastcall *v166)(__m128i *, __int64, int); // [rsp+1E0h] [rbp-C0h]
  _DWORD v167[4]; // [rsp+1E8h] [rbp-B8h]
  _QWORD *v168; // [rsp+1F8h] [rbp-A8h]
  unsigned int v169; // [rsp+208h] [rbp-98h]
  char v170; // [rsp+210h] [rbp-90h]
  char v171; // [rsp+219h] [rbp-87h]

  sub_1674380((__int64)v159, (_QWORD *)a3);
  v13 = *((_DWORD *)a4 + 2);
  v147 = 0;
  v148 = 0;
  v149 = (const __m128i *)&v147;
  v150 = &v147;
  v151 = 0;
  if ( !v13 )
    goto LABEL_2;
  v24 = *a4;
  v25 = **a4;
  if ( v25 != -8 && v25 )
  {
    v28 = *a4;
  }
  else
  {
    v26 = v24 + 1;
    do
    {
      do
      {
        v27 = *v26;
        v28 = v26++;
      }
      while ( !v27 );
    }
    while ( v27 == -8 );
  }
  v29 = &v24[v13];
  if ( v29 == v28 )
    goto LABEL_2;
  v30 = &v165;
  do
  {
    while ( 1 )
    {
      v31 = *(_QWORD *)*v28;
      v165.m128i_i64[0] = *v28 + 64;
      v165.m128i_i64[1] = v31;
      sub_1852CC0(&v146, &v165);
      v32 = v28[1];
      v33 = v28 + 1;
      if ( !v32 || v32 == -8 )
        break;
      ++v28;
      if ( v33 == v29 )
        goto LABEL_30;
    }
    v34 = v28 + 2;
    do
    {
      do
      {
        v35 = *v34;
        v28 = v34++;
      }
      while ( !v35 );
    }
    while ( v35 == -8 );
  }
  while ( v28 != v29 );
LABEL_30:
  v121 = v149;
  if ( v149 == (const __m128i *)&v147 )
  {
LABEL_2:
    v14 = 0;
    goto LABEL_3;
  }
  v115 = 0;
  v36 = (__int64 *)&v141;
  do
  {
    v37 = (unsigned __int8 *)v121[2].m128i_i64[0];
    v38 = sub_16D1B30((__int64 *)a4, v37, v121[2].m128i_u64[1]);
    if ( v38 == -1 )
    {
      v39 = *((unsigned int *)a4 + 2);
      v40 = &(*a4)[v39];
    }
    else
    {
      v39 = (__int64)*a4;
      v40 = &(*a4)[v38];
    }
    v41 = (__m128)_mm_loadu_si128(v121 + 2);
    v42 = a2[3] == 0;
    v165 = (__m128i)v41;
    if ( v42 )
      sub_4263D6(a4, v37, v39);
    ((void (__fastcall *)(unsigned __int64 *, char **, __m128i *))a2[4])(&v135, a2 + 1, v30);
    v43 = v136 & 1;
    v136 = (2 * (v136 & 1)) | v136 & 0xFD;
    if ( v43 )
    {
      v112 = v135 & 0xFFFFFFFFFFFFFFFELL;
      *(_BYTE *)(a1 + 8) |= 3u;
      *(_QWORD *)a1 = v112;
      goto LABEL_4;
    }
    v44 = (__int64 *)v135;
    v135 = 0;
    v45 = (__int64)v44;
    v120 = v44;
    sub_16330F0(v30, (__int64)v44);
    v47 = v165.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v165.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      *(_BYTE *)(a1 + 8) |= 3u;
      *(_QWORD *)a1 = v47;
      if ( !v44 )
        goto LABEL_38;
      goto LABEL_78;
    }
    v49 = *v40;
    v156 = 0;
    v157 = 0;
    v158 = 0;
    v50 = (__int64 *)v44[4];
    v152 = 0;
    v153 = 0;
    v154 = 0;
    v155 = 0;
    v132 = v120 + 3;
    if ( v50 != v120 + 3 )
    {
      while ( v50 )
      {
        if ( (*((_BYTE *)v50 - 33) & 0x20) == 0 )
          goto LABEL_44;
        sub_15E4EB0(v36, (__int64)(v50 - 7));
        v122 = v142;
        v126 = (int *)v141;
        sub_16C1840(v30);
        sub_16C1A90(v30->m128i_i32, v126, v122);
        sub_16C1AA0(v30, &v137);
        v51 = v137;
        if ( (__int64 *)v141 != &v143 )
        {
          v127 = v137;
          j_j___libc_free_0(v141, v143 + 1);
          v51 = v127;
        }
        if ( sub_18518A0(*(_QWORD *)(v49 + 8), *(_QWORD *)(v49 + 16), (unsigned __int64)v51) )
        {
          sub_15E4B20((__int64)v30, (__int64)(v50 - 7));
          v52 = v165.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
          if ( (v165.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          {
LABEL_74:
            *(_BYTE *)(a1 + 8) |= 3u;
            *(_QWORD *)a1 = v52;
            goto LABEL_75;
          }
          if ( byte_4FAAA20 )
          {
            v165.m128i_i64[0] = sub_161FF10(*(__int64 **)a3, (void *)v120[26], v120[27]);
            v60 = sub_1627350(*(__int64 **)a3, v30->m128i_i64, (__int64 *)1, 0, 1);
            sub_1627100((__int64)(v50 - 7), "thinlto_src_module", 0x12u, v60);
          }
          v165.m128i_i64[0] = (__int64)(v50 - 7);
          sub_18547C0((__int64)&v152, (char **)v30);
          v50 = (__int64 *)v50[1];
          if ( v132 == v50 )
            goto LABEL_54;
        }
        else
        {
LABEL_44:
          v50 = (__int64 *)v50[1];
          if ( v132 == v50 )
            goto LABEL_54;
        }
      }
LABEL_179:
      BUG();
    }
LABEL_54:
    v53 = (__int64 *)v120[2];
    v133 = v120 + 1;
    if ( v120 + 1 != v53 )
    {
      while ( v53 )
      {
        if ( (*((_BYTE *)v53 - 33) & 0x20) == 0 )
          goto LABEL_56;
        sub_15E4EB0(v36, (__int64)(v53 - 7));
        v123 = v142;
        v128 = (int *)v141;
        sub_16C1840(v30);
        sub_16C1A90(v30->m128i_i32, v128, v123);
        sub_16C1AA0(v30, &v137);
        v54 = v137;
        if ( (__int64 *)v141 != &v143 )
        {
          v129 = v137;
          j_j___libc_free_0(v141, v143 + 1);
          v54 = v129;
        }
        if ( sub_18518A0(*(_QWORD *)(v49 + 8), *(_QWORD *)(v49 + 16), (unsigned __int64)v54) )
        {
          sub_15E4B20((__int64)v30, (__int64)(v53 - 7));
          v52 = v165.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
          if ( (v165.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
            goto LABEL_74;
          v165.m128i_i64[0] = (__int64)(v53 - 7);
          sub_18547C0((__int64)&v152, (char **)v30);
          v53 = (__int64 *)v53[1];
          if ( v133 == v53 )
            goto LABEL_64;
        }
        else
        {
LABEL_56:
          v53 = (__int64 *)v53[1];
          if ( v133 == v53 )
            goto LABEL_64;
        }
      }
      goto LABEL_179;
    }
LABEL_64:
    if ( v120 + 5 == (__int64 *)v120[6] )
      goto LABEL_128;
    v124 = v36;
    v55 = (__int64 *)v120[6];
    v56 = v49;
    do
    {
      if ( !v55 )
        goto LABEL_179;
      if ( (*((_BYTE *)v55 - 25) & 0x20) != 0 )
      {
        sub_15E4EB0(v124, (__int64)(v55 - 6));
        v57 = v142;
        v130 = (int *)v141;
        sub_16C1840(v30);
        sub_16C1A90(v30->m128i_i32, v130, v57);
        sub_16C1AA0(v30, &v137);
        v58 = v137;
        if ( (__int64 *)v141 != &v143 )
          j_j___libc_free_0(v141, v143 + 1);
        if ( sub_18518A0(*(_QWORD *)(v56 + 8), *(_QWORD *)(v56 + 16), (unsigned __int64)v58) )
        {
          sub_15E4B20((__int64)v30, (__int64)(v55 - 6));
          v59 = v165.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
          if ( (v165.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          {
            *(_BYTE *)(a1 + 8) |= 3u;
            *(_QWORD *)a1 = v59;
            goto LABEL_75;
          }
          v61 = sub_164A820(*(v55 - 9));
          v62 = *(_BYTE *)(v61 + 16);
          if ( v62 && v62 != 3 )
            v61 = 0;
          sub_15E4B20((__int64)v30, v61);
          v63 = v165.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
          if ( (v165.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          {
            *(_BYTE *)(a1 + 8) |= 3u;
            *(_QWORD *)a1 = v63;
            goto LABEL_75;
          }
          v64 = sub_164A820(*(v55 - 9));
          v65 = *(_BYTE *)(v64 + 16);
          if ( v65 && v65 != 3 )
            v64 = 0;
          v165.m128i_i64[0] = 0;
          v167[0] = 128;
          v66 = (_QWORD *)sub_22077B0(0x2000);
          v166 = 0;
          v165.m128i_i64[1] = (__int64)v66;
          v142 = 2;
          v145 = 0;
          for ( i = &v66[8 * (unsigned __int64)v167[0]]; i != v66; v66 += 8 )
          {
            if ( v66 )
            {
              v68 = v142;
              v66[2] = 0;
              v66[3] = -8;
              v66[1] = v68 & 6;
              *v66 = &unk_49E6B50;
              v66[4] = v145;
            }
          }
          v170 = 0;
          v171 = 1;
          v69 = sub_1AB6FF0(v64, v30, 0);
          v70 = *(_BYTE *)(v55 - 2) & 0xF;
          v71 = *(_BYTE *)(v55 - 2) & 0xF;
          v72 = *(_BYTE *)(v69 + 32);
          if ( (unsigned int)(v70 - 7) > 1 )
          {
            *(_BYTE *)(v69 + 32) = v71 | v72 & 0xF0;
          }
          else
          {
            *(_BYTE *)(v69 + 32) = v71 | v72 & 0xC0;
            if ( v70 == 7 )
            {
LABEL_95:
              *(_BYTE *)(v69 + 33) |= 0x40u;
LABEL_96:
              v73 = sub_15A4510((__int64 ***)v69, (__int64 **)*(v55 - 6), 0);
              sub_164D160((__int64)(v55 - 6), v73, v41, a6, a7, a8, v74, v75, a11, a12);
              sub_164B7C0(v69, (__int64)(v55 - 6));
              if ( v170 )
              {
                if ( v169 )
                {
                  v92 = v168;
                  v131 = v30;
                  v93 = &v168[2 * v169];
                  do
                  {
                    if ( *v92 != -8 && *v92 != -4 )
                    {
                      v94 = v92[1];
                      if ( v94 )
                        sub_161E7C0((__int64)(v92 + 1), v94);
                    }
                    v92 += 2;
                  }
                  while ( v93 != v92 );
                  v30 = v131;
                }
                j___libc_free_0(v168);
              }
              if ( v167[0] )
              {
                v76 = -8;
                v77 = v165.m128i_i64[1] + ((unsigned __int64)v167[0] << 6);
                v138[0] = 2;
                v137 = (__int64 *)&unk_49E6B50;
                v141 = (unsigned __int64)&unk_49E6B50;
                v138[1] = 0;
                v139 = -8;
                v140 = 0;
                v78 = (_QWORD *)v165.m128i_i64[1];
                v142 = 2;
                v143 = 0;
                v144 = -16;
                v145 = 0;
                while ( 1 )
                {
                  v79 = v78[3];
                  if ( v76 != v79 )
                  {
                    v76 = v144;
                    if ( v79 != v144 )
                    {
                      v80 = v78[7];
                      if ( v80 != 0 && v80 != -8 && v80 != -16 )
                      {
                        sub_1649B30(v78 + 5);
                        v79 = v78[3];
                      }
                      v76 = v79;
                    }
                  }
                  *v78 = &unk_49EE2B0;
                  if ( v76 != 0 && v76 != -8 && v76 != -16 )
                    sub_1649B30(v78 + 1);
                  v78 += 8;
                  if ( (_QWORD *)v77 == v78 )
                    break;
                  v76 = v139;
                }
                v141 = (unsigned __int64)&unk_49EE2B0;
                if ( v144 != -8 && v144 != 0 && v144 != -16 )
                  sub_1649B30(&v142);
                v137 = (__int64 *)&unk_49EE2B0;
                if ( v139 != 0 && v139 != -8 && v139 != -16 )
                  sub_1649B30(v138);
              }
              j___libc_free_0(v165.m128i_i64[1]);
              if ( byte_4FAAA20 )
              {
                v89 = sub_161FF10(*(__int64 **)a3, (void *)v120[26], v120[27]);
                v90 = *(__int64 **)a3;
                v165.m128i_i64[0] = v89;
                v91 = sub_1627350(v90, v30->m128i_i64, (__int64 *)1, 0, 1);
                sub_1627100(v69, "thinlto_src_module", 0x12u, v91);
              }
              v165.m128i_i64[0] = v69;
              sub_18547C0((__int64)&v152, (char **)v30);
              goto LABEL_66;
            }
          }
          if ( v70 != 8 && ((*(_BYTE *)(v69 + 32) & 0x30) == 0 || v71 == 9) )
            goto LABEL_96;
          goto LABEL_95;
        }
      }
LABEL_66:
      v55 = (__int64 *)v55[1];
    }
    while ( v120 + 5 != v55 );
    v36 = v124;
LABEL_128:
    sub_157E370(v120);
    v82 = (__int64)v120;
    v83 = *a2;
    if ( (unsigned __int8)sub_1ACEF80(v120, *a2, &v152) )
    {
      v113 = *(_BYTE *)(a1 + 8);
      *(_BYTE *)a1 = 1;
      *(_BYTE *)(a1 + 8) = v113 & 0xFC | 2;
LABEL_75:
      if ( v156 )
        j_j___libc_free_0(v156, v158 - (_QWORD)v156);
      j___libc_free_0(v153);
LABEL_78:
      sub_1633490((_QWORD **)v120);
      v45 = 736;
      j_j___libc_free_0(v120, 736);
LABEL_38:
      if ( (v136 & 2) == 0 )
      {
        v48 = v135;
        if ( (v136 & 1) != 0 )
        {
          if ( v135 )
            (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v135 + 8LL))(v135);
        }
        else if ( v135 )
        {
          sub_1633490((_QWORD **)v135);
          j_j___libc_free_0(v48, 736);
        }
        goto LABEL_4;
      }
LABEL_153:
      sub_1264230(&v135, v45, v46);
    }
    v87 = v156;
    v134 = v157;
    if ( byte_4FAABE0 && v157 != (__int64 *)v156 )
    {
      v97 = (__int64 *)v156;
      v125 = v36;
      do
      {
        v105 = *v97;
        v106 = sub_16BA580(v82, (__int64)v83, v84);
        v107 = sub_16E7EE0(v106, *(char **)(a3 + 208), *(_QWORD *)(a3 + 216));
        v108 = *(_QWORD *)(v107 + 24);
        v109 = v107;
        if ( (unsigned __int64)(*(_QWORD *)(v107 + 16) - v108) > 8 )
        {
          *(_BYTE *)(v108 + 8) = 32;
          *(_QWORD *)v108 = 0x74726F706D49203ALL;
          *(_QWORD *)(v107 + 24) += 9LL;
        }
        else
        {
          v109 = sub_16E7EE0(v107, ": Import ", 9u);
        }
        v98 = sub_1649960(v105);
        v100 = *(_WORD **)(v109 + 24);
        v101 = (char *)v98;
        v102 = v99;
        v103 = *(_QWORD *)(v109 + 16) - (_QWORD)v100;
        if ( v99 > v103 )
        {
          v110 = sub_16E7EE0(v109, v101, v99);
          v100 = *(_WORD **)(v110 + 24);
          v109 = v110;
          v103 = *(_QWORD *)(v110 + 16) - (_QWORD)v100;
        }
        else if ( v99 )
        {
          memcpy(v100, v101, v99);
          v111 = *(_QWORD *)(v109 + 16);
          v100 = (_WORD *)(v102 + *(_QWORD *)(v109 + 24));
          *(_QWORD *)(v109 + 24) = v100;
          v103 = v111 - (_QWORD)v100;
        }
        if ( v103 <= 5 )
        {
          v109 = sub_16E7EE0(v109, " from ", 6u);
        }
        else
        {
          *(_DWORD *)v100 = 1869768224;
          v100[2] = 8301;
          *(_QWORD *)(v109 + 24) += 6LL;
        }
        v83 = (char *)v120[26];
        v82 = sub_16E7EE0(v109, v83, v120[27]);
        v104 = *(_BYTE **)(v82 + 24);
        if ( *(_BYTE **)(v82 + 16) == v104 )
        {
          v83 = "\n";
          sub_16E7EE0(v82, "\n", 1u);
        }
        else
        {
          *v104 = 10;
          ++*(_QWORD *)(v82 + 24);
        }
        ++v97;
      }
      while ( v134 != v97 );
      v36 = v125;
      v87 = v156;
      v134 = v157;
    }
    *(_QWORD *)v167 = sub_1851A40;
    v166 = (__int64 (__fastcall *)(__m128i *, __int64, int))sub_18512F0;
    v137 = v120;
    sub_16786A0(
      v36,
      (__int64)v159,
      &v137,
      v87,
      ((char *)v134 - (char *)v87) >> 3,
      v30,
      *(double *)v41.m128_u64,
      a6,
      a7,
      a8,
      v85,
      v86,
      a11,
      a12,
      1,
      0);
    v45 = v114;
    if ( (v141 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      v141 = v141 & 0xFFFFFFFFFFFFFFFELL | 1;
      sub_16BCAE0(v36, v114, v88);
    }
    v95 = v137;
    if ( v137 )
    {
      sub_1633490((_QWORD **)v137);
      v45 = 736;
      j_j___libc_free_0(v95, 736);
    }
    if ( v166 )
    {
      v45 = (__int64)v30;
      v166(v30, (__int64)v30, 3);
    }
    v115 += ((char *)v157 - (char *)v156) >> 3;
    if ( v156 )
    {
      v45 = v158 - (_QWORD)v156;
      j_j___libc_free_0(v156, v158 - (_QWORD)v156);
    }
    j___libc_free_0(v153);
    if ( (v136 & 2) != 0 )
      goto LABEL_153;
    v96 = v135;
    if ( (v136 & 1) != 0 )
    {
      if ( v135 )
        (*(void (__fastcall **)(unsigned __int64, __int64))(*(_QWORD *)v135 + 8LL))(v135, v45);
    }
    else if ( v135 )
    {
      sub_1633490((_QWORD **)v135);
      j_j___libc_free_0(v96, 736);
    }
    v121 = (const __m128i *)sub_220EF30(v121);
  }
  while ( v121 != (const __m128i *)&v147 );
  v14 = v115 != 0;
LABEL_3:
  v15 = *(_BYTE *)(a1 + 8);
  *(_BYTE *)a1 = v14;
  *(_BYTE *)(a1 + 8) = v15 & 0xFC | 2;
LABEL_4:
  sub_1851C60(v148);
  if ( v164 )
  {
    v16 = v163;
    v17 = &v163[2 * v164];
    do
    {
      if ( *v16 != -8 && *v16 != -4 )
      {
        v18 = v16[1];
        if ( v18 )
          sub_161E7C0((__int64)(v16 + 1), v18);
      }
      v16 += 2;
    }
    while ( v17 != v16 );
  }
  j___libc_free_0(v163);
  if ( v162 )
  {
    v19 = sub_16704E0();
    v20 = sub_16704F0();
    v21 = v161;
    v22 = v20;
    v23 = &v161[v162];
    if ( v161 == v23 )
      goto LABEL_126;
    do
    {
      if ( !sub_1670560(*v21, v19) )
        sub_1670560(*v21, v22);
      ++v21;
    }
    while ( v23 != v21 );
  }
  v23 = v161;
LABEL_126:
  j___libc_free_0(v23);
  j___libc_free_0(v160);
  return a1;
}
