// Function: sub_2067630
// Address: 0x2067630
//
__int64 *__fastcall sub_2067630(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  _BYTE *v5; // r12
  __int64 v6; // rbx
  _QWORD *v7; // r13
  unsigned __int8 v8; // al
  __int64 v9; // r15
  __int64 v10; // r14
  __int64 v11; // rax
  unsigned int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  unsigned int v17; // r10d
  unsigned __int8 v18; // al
  const void **v19; // r14
  int v20; // edx
  __int64 v21; // rax
  __int64 v22; // r15
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 *v26; // r12
  int v28; // edx
  __int64 v29; // rax
  _QWORD *v30; // r15
  __int64 v31; // rsi
  __int64 v32; // rdi
  int v33; // r14d
  __int64 v34; // r12
  __int64 v35; // rax
  __int64 *v36; // rsi
  __int32 v37; // edx
  __m128i *v38; // rax
  __int64 v39; // r8
  unsigned __int64 v40; // rdi
  __int64 *v41; // r15
  __int64 *v42; // rax
  __int64 v43; // rax
  __int64 v44; // rcx
  int v45; // r9d
  __int64 v46; // rdx
  __int64 *v47; // r14
  __int64 v48; // r8
  __int64 v49; // rax
  unsigned __int8 v50; // al
  __int64 v51; // rax
  __int64 v52; // r14
  unsigned int v53; // r12d
  __int64 v54; // rax
  unsigned int v55; // r12d
  int v56; // eax
  unsigned int v57; // r10d
  unsigned int v58; // r14d
  unsigned int v59; // r13d
  __int64 v60; // rax
  __int64 v61; // rax
  int v62; // r8d
  int v63; // r15d
  __int64 v64; // r9
  __int64 v65; // rax
  __int64 v66; // r12
  __int64 v67; // rbx
  __int64 v68; // r9
  __int64 *v69; // rax
  __int64 *v70; // r15
  int v71; // edx
  __int64 v72; // rsi
  __int64 v73; // r12
  __int64 *v74; // r14
  int v75; // edx
  int v76; // r15d
  __int64 *v77; // rax
  __int64 v78; // r14
  _QWORD *v79; // r13
  __int64 v80; // rax
  int v81; // r8d
  int v82; // r9d
  __int64 v83; // r14
  int v84; // r12d
  __int64 v85; // rax
  unsigned int v86; // r15d
  __int64 v87; // r10
  __int64 *v88; // rax
  __int64 v89; // r12
  int v90; // r8d
  __int64 v91; // rax
  __int64 v92; // rax
  __int64 v93; // rcx
  int v94; // r8d
  int v95; // r9d
  int v96; // r8d
  __int64 v97; // rax
  __int64 v98; // r13
  __int64 v99; // r15
  __int64 v100; // rdx
  __int64 v101; // rdx
  __int64 v102; // rcx
  int v103; // r8d
  int v104; // r9d
  __int64 *v105; // r12
  __int64 v106; // r15
  __int64 v107; // r13
  __int64 *v108; // r13
  int v109; // edx
  int v110; // r14d
  __int64 *v111; // rax
  __int64 v112; // r15
  __int64 v113; // rcx
  __int64 v114; // r8
  __int64 v115; // r9
  __int64 v116; // r15
  bool v117; // al
  int v118; // edx
  __int64 v119; // rsi
  int v120; // edx
  __int64 v121; // rsi
  __int64 v122; // rax
  bool v123; // zf
  _QWORD *v124; // r13
  __m128i v125; // xmm1
  _QWORD *v126; // rsi
  __int64 v127; // rax
  int v128; // edx
  __int64 v129; // r12
  int v130; // r8d
  __int64 v131; // r12
  __int64 v132; // rax
  __int32 v133; // eax
  unsigned int v134; // r10d
  __int64 v135; // rdx
  bool v136; // al
  __int64 v137; // r12
  int v138; // edx
  __int64 v139; // rsi
  __int32 v140; // edx
  __int64 v141; // rcx
  int v142; // r8d
  int v143; // r9d
  __int32 v144; // edx
  int v145; // [rsp-10h] [rbp-270h]
  unsigned int v146; // [rsp+0h] [rbp-260h]
  const void **v147; // [rsp+8h] [rbp-258h]
  int v148; // [rsp+10h] [rbp-250h]
  unsigned int v149; // [rsp+10h] [rbp-250h]
  unsigned int v150; // [rsp+18h] [rbp-248h]
  const void **v151; // [rsp+18h] [rbp-248h]
  unsigned int v152; // [rsp+18h] [rbp-248h]
  unsigned int v153; // [rsp+18h] [rbp-248h]
  unsigned int v154; // [rsp+20h] [rbp-240h]
  _BYTE *v155; // [rsp+20h] [rbp-240h]
  __int64 v156; // [rsp+20h] [rbp-240h]
  _QWORD *v157; // [rsp+20h] [rbp-240h]
  __int64 v158; // [rsp+20h] [rbp-240h]
  unsigned int v159; // [rsp+20h] [rbp-240h]
  unsigned int v160; // [rsp+28h] [rbp-238h]
  unsigned int v161; // [rsp+28h] [rbp-238h]
  unsigned int v162; // [rsp+28h] [rbp-238h]
  __int64 v163; // [rsp+28h] [rbp-238h]
  __int64 *v164; // [rsp+A8h] [rbp-1B8h] BYREF
  __m128i v165; // [rsp+B0h] [rbp-1B0h] BYREF
  __m128i v166; // [rsp+C0h] [rbp-1A0h] BYREF
  __m128i v167; // [rsp+D0h] [rbp-190h] BYREF
  _BYTE v168[64]; // [rsp+E0h] [rbp-180h] BYREF
  _BYTE *v169; // [rsp+120h] [rbp-140h] BYREF
  __int64 v170; // [rsp+128h] [rbp-138h]
  _BYTE v171[64]; // [rsp+130h] [rbp-130h] BYREF
  char *v172; // [rsp+170h] [rbp-F0h]
  char v173; // [rsp+180h] [rbp-E0h] BYREF
  char *v174; // [rsp+188h] [rbp-D8h]
  char v175; // [rsp+198h] [rbp-C8h] BYREF
  char *v176; // [rsp+1A8h] [rbp-B8h]
  char v177; // [rsp+1B8h] [rbp-A8h] BYREF

  v5 = (_BYTE *)a2;
  v6 = a1;
  v7 = *(_QWORD **)(a1 + 552);
  v8 = *(_BYTE *)(a2 + 16);
  v164 = (__int64 *)a2;
  v9 = v7[2];
  if ( v8 > 0x10u )
  {
    v32 = *(_QWORD *)(a1 + 712);
    if ( v8 == 53 )
    {
      v43 = *(unsigned int *)(v32 + 360);
      if ( (_DWORD)v43 )
      {
        v44 = *(_QWORD *)(v32 + 344);
        v45 = 1;
        LODWORD(v46) = (v43 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v47 = (__int64 *)(v44 + 16LL * (unsigned int)v46);
        v48 = *v47;
        if ( a2 == *v47 )
        {
LABEL_41:
          if ( v47 != (__int64 *)(v44 + 16 * v43) )
          {
            v49 = sub_1E0A0C0(v7[4]);
            v50 = sub_2046180(v49, *(_DWORD *)(v49 + 4));
            return sub_1D299D0(v7, *((_DWORD *)v47 + 2), v50, 0, 0);
          }
        }
        else
        {
          while ( v48 != -8 )
          {
            v46 = ((_DWORD)v43 - 1) & (unsigned int)(v46 + v45);
            v47 = (__int64 *)(v44 + 16 * v46);
            v48 = *v47;
            if ( a2 == *v47 )
              goto LABEL_41;
            ++v45;
          }
        }
      }
    }
    v33 = sub_1FD4520(v32, (__int64 *)a2);
    sub_2043DE0((__int64)&v167, (__int64)v164);
    v34 = *(_QWORD *)a2;
    v35 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(v6 + 552) + 32LL));
    sub_204E3C0((__int64)&v169, *(_QWORD *)(*(_QWORD *)(v6 + 552) + 48LL), v9, v35, v33, v34, (unsigned int *)&v167);
    v36 = *(__int64 **)(v6 + 552);
    v166.m128i_i32[2] = 0;
    v37 = *(_DWORD *)(v6 + 536);
    v167.m128i_i64[0] = 0;
    v166.m128i_i64[0] = (__int64)(v36 + 11);
    v38 = *(__m128i **)v6;
    v167.m128i_i32[2] = v37;
    if ( v38 )
    {
      if ( &v167 != &v38[3] )
      {
        v39 = v38[3].m128i_i64[0];
        v167.m128i_i64[0] = v39;
        if ( v39 )
        {
          sub_1623A60((__int64)&v167, v39, 2);
          v36 = *(__int64 **)(v6 + 552);
        }
      }
    }
    v26 = sub_204EDE0(
            (__int64)&v169,
            v36,
            *(_QWORD *)(v6 + 712),
            (__int64)&v167,
            v166.m128i_i64,
            0,
            a3,
            a4,
            a5,
            (__int64)v164);
    if ( v167.m128i_i64[0] )
      sub_161E7C0((__int64)&v167, v167.m128i_i64[0]);
    if ( v176 != &v177 )
      _libc_free((unsigned __int64)v176);
    if ( v174 != &v175 )
      _libc_free((unsigned __int64)v174);
    if ( v172 != &v173 )
      _libc_free((unsigned __int64)v172);
    v40 = (unsigned __int64)v169;
    if ( v169 == v171 )
      return v26;
    goto LABEL_31;
  }
  v10 = *(_QWORD *)a2;
  v11 = sub_1E0A0C0(v7[4]);
  LOBYTE(v12) = sub_204D4D0(v9, v11, v10);
  v17 = v12;
  v18 = *(_BYTE *)(a2 + 16);
  v19 = (const void **)v13;
  if ( v18 == 13 )
  {
    v20 = *(_DWORD *)(a1 + 536);
    v21 = *(_QWORD *)a1;
    v169 = 0;
    v22 = *(_QWORD *)(a1 + 552);
    LODWORD(v170) = v20;
    if ( v21 )
    {
      if ( &v169 != (_BYTE **)(v21 + 48) )
      {
        v23 = *(_QWORD *)(v21 + 48);
        v169 = (_BYTE *)v23;
        if ( v23 )
        {
          v160 = v17;
          sub_1623A60((__int64)&v169, v23, 2);
          v17 = v160;
        }
      }
    }
    v24 = sub_1D37E40(v22, (__int64)v5, (__int64)&v169, v17, v19, 0, a3, *(double *)a4.m128i_i64, a5, 0);
    goto LABEL_8;
  }
  if ( v18 <= 3u )
  {
    v28 = *(_DWORD *)(a1 + 536);
    v29 = *(_QWORD *)a1;
    v169 = 0;
    v30 = *(_QWORD **)(a1 + 552);
    LODWORD(v170) = v28;
    if ( v29 )
    {
      if ( &v169 != (_BYTE **)(v29 + 48) )
      {
        v31 = *(_QWORD *)(v29 + 48);
        v169 = (_BYTE *)v31;
        if ( v31 )
        {
          v161 = v17;
          sub_1623A60((__int64)&v169, v31, 2);
          v17 = v161;
        }
      }
    }
    v24 = (__int64)sub_1D29600(v30, (__int64)v5, (__int64)&v169, v17, (__int64)v19, 0, 0, 0);
    goto LABEL_8;
  }
  if ( v18 != 15 )
  {
    switch ( v18 )
    {
      case 0xEu:
        v162 = v17;
        v41 = *(__int64 **)(a1 + 552);
        sub_204D410((__int64)&v169, *(_QWORD *)a1, *(_DWORD *)(a1 + 536));
        v42 = sub_1D360F0(v41, a2, (__int64)&v169, v162, v19, 0, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
        v25 = (__int64)v169;
        v26 = v42;
        if ( !v169 )
          return v26;
        goto LABEL_9;
      case 9u:
        if ( (unsigned int)*(unsigned __int8 *)(*v164 + 8) - 13 > 1 )
          return sub_1D2B530(*(_QWORD **)(a1 + 552), v17, v13, v14, v15, v16);
        break;
      case 5u:
        sub_2067240((__int64 *)a1, *(unsigned __int16 *)(a2 + 18), a2, a3, *(double *)a4.m128i_i64, a5);
        return (__int64 *)sub_205F5C0(a1 + 8, (__int64 *)&v164)[1];
      default:
        if ( (unsigned __int8)(v18 - 6) <= 1u )
        {
          v169 = v171;
          v170 = 0x400000000LL;
          v78 = sub_13CF970(a2);
          if ( v78 + 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != v78 )
          {
            v157 = (_QWORD *)(v78 + 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
            v79 = (_QWORD *)v78;
            do
            {
              v80 = sub_20685E0(a1, *v79);
              v83 = v80;
              if ( v80 )
              {
                v84 = *(_DWORD *)(v80 + 60);
                if ( v84 )
                {
                  v85 = (unsigned int)v170;
                  v86 = 0;
                  do
                  {
                    v87 = v86;
                    if ( HIDWORD(v170) <= (unsigned int)v85 )
                    {
                      sub_16CD150((__int64)&v169, v171, 0, 16, v81, v82);
                      v85 = (unsigned int)v170;
                      v87 = v86;
                    }
                    v88 = (__int64 *)&v169[16 * v85];
                    ++v86;
                    *v88 = v83;
                    v88[1] = v87;
                    v85 = (unsigned int)(v170 + 1);
                    LODWORD(v170) = v170 + 1;
                  }
                  while ( v86 != v84 );
                }
              }
              v79 += 3;
            }
            while ( v79 != v157 );
          }
          v89 = *(_QWORD *)(a1 + 552);
          sub_204D410((__int64)&v167, *(_QWORD *)a1, *(_DWORD *)(a1 + 536));
          v26 = sub_1D37190(
                  v89,
                  (__int64)v169,
                  (unsigned int)v170,
                  (__int64)&v167,
                  v90,
                  *(double *)a3.m128i_i64,
                  *(double *)a4.m128i_i64,
                  a5);
          goto LABEL_60;
        }
        break;
    }
    if ( (unsigned int)v18 - 11 <= 1 )
    {
      v154 = v17;
      v169 = v171;
      v170 = 0x400000000LL;
      v56 = sub_15958F0(a2);
      v57 = v154;
      v148 = v56;
      if ( v56 )
      {
        v146 = v154;
        v147 = v19;
        v58 = 0;
        do
        {
          v59 = 0;
          v60 = sub_15A0940((__int64)v5, v58);
          v61 = sub_20685E0(v6, v60);
          v63 = *(_DWORD *)(v61 + 60);
          v64 = v61;
          v65 = (unsigned int)v170;
          if ( v63 )
          {
            v155 = v5;
            v66 = v6;
            v67 = v64;
            do
            {
              v68 = v59;
              if ( HIDWORD(v170) <= (unsigned int)v65 )
              {
                sub_16CD150((__int64)&v169, v171, 0, 16, v62, v59);
                v65 = (unsigned int)v170;
                v68 = v59;
              }
              v69 = (__int64 *)&v169[16 * v65];
              ++v59;
              *v69 = v67;
              v69[1] = v68;
              v65 = (unsigned int)(v170 + 1);
              LODWORD(v170) = v170 + 1;
            }
            while ( v59 != v63 );
            v6 = v66;
            v5 = v155;
          }
          ++v58;
        }
        while ( v148 != v58 );
        v19 = v147;
        v57 = v146;
      }
      v70 = *(__int64 **)(v6 + 552);
      v71 = *(_DWORD *)(v6 + 536);
      v72 = *(_QWORD *)v6;
      if ( *(_BYTE *)(*(_QWORD *)v5 + 8LL) == 14 )
      {
        sub_204D410((__int64)&v167, v72, v71);
        v26 = sub_1D37190(
                (__int64)v70,
                (__int64)v169,
                (unsigned int)v170,
                (__int64)&v167,
                v96,
                *(double *)a3.m128i_i64,
                *(double *)a4.m128i_i64,
                a5);
        sub_17CD270(v167.m128i_i64);
LABEL_61:
        v40 = (unsigned __int64)v169;
        if ( v169 == v171 )
          return v26;
        goto LABEL_31;
      }
      v73 = (unsigned int)v170;
      v150 = v57;
      v156 = (__int64)v169;
      sub_204D410((__int64)&v167, v72, v71);
      v74 = sub_204D450(v70, v150, v19, (__int64)&v167, v156, v73, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
      v76 = v75;
      v77 = sub_205F5C0(v6 + 8, (__int64 *)&v164);
      v26 = v74;
      v77[1] = (__int64)v74;
      *((_DWORD *)v77 + 4) = v76;
LABEL_60:
      sub_17CD270(v167.m128i_i64);
      goto LABEL_61;
    }
    if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)a2 + 8LL) - 13) <= 1u )
    {
      v167.m128i_i64[0] = (__int64)v168;
      v91 = *(_QWORD *)(a1 + 552);
      v167.m128i_i64[1] = 0x400000000LL;
      v163 = *(_QWORD *)a2;
      v92 = sub_1E0A0C0(*(_QWORD *)(v91 + 32));
      sub_20C7CE0(v9, v92, v163, &v167, 0, 0);
      if ( v167.m128i_i32[2] )
      {
        v170 = 0x400000000LL;
        v112 = 16LL * v167.m128i_u32[2];
        v166.m128i_i64[0] = 0;
        v166.m128i_i32[2] = 0;
        v169 = v171;
        sub_202F910((__int64)&v169, v167.m128i_u32[2], &v166, v93, v94, v95);
        v158 = v112;
        v116 = 0;
        do
        {
          v123 = v5[16] == 9;
          v124 = *(_QWORD **)(a1 + 552);
          v125 = _mm_loadu_si128((const __m128i *)(v167.m128i_i64[0] + v116));
          v165 = v125;
          if ( v123 )
          {
            v126 = sub_1D2B530(v124, v165.m128i_u32[0], v165.m128i_i64[1], v113, v114, v115);
            v127 = (__int64)v169;
            *(_QWORD *)&v169[v116] = v126;
            *(_DWORD *)(v127 + v116 + 8) = v128;
          }
          else
          {
            if ( v165.m128i_i8[0] )
              v117 = (unsigned __int8)(v165.m128i_i8[0] - 86) <= 0x17u || (unsigned __int8)(v165.m128i_i8[0] - 8) <= 5u;
            else
              v117 = sub_1F58CD0((__int64)&v165);
            v118 = *(_DWORD *)(a1 + 536);
            v119 = *(_QWORD *)a1;
            if ( v117 )
            {
              sub_204D410((__int64)&v166, v119, v118);
              a3 = 0;
              v121 = (__int64)sub_1D364E0(
                                (__int64)v124,
                                (__int64)&v166,
                                v165.m128i_u32[0],
                                (const void **)v165.m128i_i64[1],
                                0,
                                0.0,
                                *(double *)v125.m128i_i64,
                                a5);
            }
            else
            {
              sub_204D410((__int64)&v166, v119, v118);
              v121 = sub_1D38BB0(
                       (__int64)v124,
                       0,
                       (__int64)&v166,
                       v165.m128i_u32[0],
                       (const void **)v165.m128i_i64[1],
                       0,
                       a3,
                       *(double *)v125.m128i_i64,
                       a5,
                       0);
            }
            v122 = (__int64)v169;
            *(_QWORD *)&v169[v116] = v121;
            *(_DWORD *)(v122 + v116 + 8) = v120;
            sub_17CD270(v166.m128i_i64);
          }
          v116 += 16;
        }
        while ( v158 != v116 );
        v129 = *(_QWORD *)(a1 + 552);
        sub_204D410((__int64)&v166, *(_QWORD *)a1, *(_DWORD *)(a1 + 536));
        v26 = sub_1D37190(
                v129,
                (__int64)v169,
                (unsigned int)v170,
                (__int64)&v166,
                v130,
                *(double *)a3.m128i_i64,
                *(double *)v125.m128i_i64,
                a5);
        sub_17CD270(v166.m128i_i64);
        if ( v169 != v171 )
          _libc_free((unsigned __int64)v169);
      }
      else
      {
        v26 = 0;
      }
      v40 = v167.m128i_i64[0];
      if ( (_BYTE *)v167.m128i_i64[0] == v168 )
        return v26;
    }
    else
    {
      if ( v18 == 4 )
        return sub_1D2AB00(*(_QWORD **)(a1 + 552), a2, v17, v13, 0, 0, 0);
      v97 = *v164;
      v98 = *(_QWORD *)(*v164 + 32);
      v169 = v171;
      v170 = 0x1000000000LL;
      if ( *(_BYTE *)(a2 + 16) == 8 )
      {
        if ( (_DWORD)v98 )
        {
          v149 = v17;
          v99 = 0;
          v151 = (const void **)v13;
          do
          {
            v100 = v99++;
            v167.m128i_i64[0] = sub_20685E0(a1, *(_QWORD *)(a2 + 24 * (v100 - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))));
            v167.m128i_i64[1] = v101;
            sub_1D23890((__int64)&v169, &v167, v101, v102, v103, v104);
          }
          while ( (unsigned int)v98 != v99 );
          v19 = v151;
          v17 = v149;
          v6 = a1;
        }
      }
      else
      {
        v131 = *(_QWORD *)(v97 + 24);
        v159 = v17;
        v132 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 552) + 32LL));
        LOBYTE(v133) = sub_204D4D0(v9, v132, v131);
        v134 = v159;
        v166.m128i_i64[0] = 0;
        v165.m128i_i32[0] = v133;
        v165.m128i_i64[1] = v135;
        v166.m128i_i32[2] = 0;
        if ( (_BYTE)v133 )
        {
          v136 = (unsigned __int8)(v133 - 86) <= 0x17u || (unsigned __int8)(v133 - 8) <= 5u;
        }
        else
        {
          v136 = sub_1F58CD0((__int64)&v165);
          v134 = v159;
        }
        v153 = v134;
        v137 = *(_QWORD *)(a1 + 552);
        v138 = *(_DWORD *)(a1 + 536);
        v139 = *(_QWORD *)a1;
        if ( v136 )
        {
          sub_204D410((__int64)&v167, v139, v138);
          a3.m128i_i64[0] = 0;
          v166.m128i_i64[0] = (__int64)sub_1D364E0(
                                         v137,
                                         (__int64)&v167,
                                         v165.m128i_u32[0],
                                         (const void **)v165.m128i_i64[1],
                                         0,
                                         0.0,
                                         *(double *)a4.m128i_i64,
                                         a5);
          v166.m128i_i32[2] = v140;
          sub_17CD270(v167.m128i_i64);
        }
        else
        {
          sub_204D410((__int64)&v167, v139, v138);
          v166.m128i_i64[0] = sub_1D38BB0(
                                v137,
                                0,
                                (__int64)&v167,
                                v165.m128i_u32[0],
                                (const void **)v165.m128i_i64[1],
                                0,
                                a3,
                                *(double *)a4.m128i_i64,
                                a5,
                                0);
          v166.m128i_i32[2] = v144;
          sub_17CD270(v167.m128i_i64);
          v143 = v145;
        }
        sub_202F910((__int64)&v169, (unsigned int)v98, &v166, v141, v142, v143);
        v17 = v153;
      }
      v152 = v17;
      v105 = *(__int64 **)(v6 + 552);
      v106 = (__int64)v169;
      v107 = (unsigned int)v170;
      sub_204D410((__int64)&v167, *(_QWORD *)v6, *(_DWORD *)(v6 + 536));
      v108 = sub_204D450(
               v105,
               v152,
               v19,
               (__int64)&v167,
               v106,
               v107,
               *(double *)a3.m128i_i64,
               *(double *)a4.m128i_i64,
               a5);
      v110 = v109;
      v111 = sub_205F5C0(v6 + 8, (__int64 *)&v164);
      v26 = v108;
      v111[1] = (__int64)v108;
      *((_DWORD *)v111 + 4) = v110;
      sub_17CD270(v167.m128i_i64);
      v40 = (unsigned __int64)v169;
      if ( v169 == v171 )
        return v26;
    }
LABEL_31:
    _libc_free(v40);
    return v26;
  }
  v51 = *v164;
  if ( *(_BYTE *)(*v164 + 8) == 16 )
    v51 = **(_QWORD **)(v51 + 16);
  v52 = *(_QWORD *)(a1 + 552);
  v53 = *(_DWORD *)(v51 + 8) >> 8;
  v54 = sub_1E0A0C0(*(_QWORD *)(v52 + 32));
  v55 = (unsigned __int8)sub_2046180(v54, v53);
  sub_204D410((__int64)&v169, *(_QWORD *)a1, *(_DWORD *)(a1 + 536));
  v24 = sub_1D38BB0(v52, 0, (__int64)&v169, v55, 0, 0, a3, *(double *)a4.m128i_i64, a5, 0);
LABEL_8:
  v25 = (__int64)v169;
  v26 = (__int64 *)v24;
  if ( v169 )
LABEL_9:
    sub_161E7C0((__int64)&v169, v25);
  return v26;
}
