// Function: sub_1DB1BC0
// Address: 0x1db1bc0
//
__int64 __fastcall sub_1DB1BC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __m128i *a6)
{
  __int64 v7; // rbx
  unsigned __int64 v8; // r12
  _BYTE *v9; // rdi
  __int64 v10; // rax
  __int64 v11; // r12
  __int8 v12; // r14
  const __m128i *v13; // rax
  __int64 v14; // rax
  __m128i v15; // xmm1
  __m128i v16; // xmm0
  unsigned int v17; // esi
  _BYTE *v18; // r14
  int v19; // eax
  __int64 v20; // rcx
  __int64 v21; // r8
  __m128i *v22; // r12
  int v23; // r11d
  unsigned int v24; // edx
  _BYTE *v25; // r15
  unsigned int i; // r13d
  __m128i *v27; // r14
  const __m128i *v28; // rdx
  const __m128i *v29; // rax
  __m128i *v30; // r13
  const __m128i *v31; // rax
  unsigned __int64 v32; // r12
  unsigned __int64 v33; // r15
  __int64 v34; // rdx
  __int64 v35; // rcx
  unsigned int v36; // r8d
  const __m128i *j; // r12
  __int64 v38; // rax
  __m128i *v39; // rax
  __int64 v40; // rax
  unsigned __int64 v41; // rdi
  _BYTE *v42; // rbx
  _BYTE *v43; // r12
  _BYTE *v44; // rdi
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rax
  __int64 v49; // rax
  int *v50; // rcx
  __int64 v51; // rdx
  __int64 v52; // rax
  int v53; // edx
  unsigned int v54; // r15d
  __m128i *v55; // r15
  int v56; // eax
  __m128i *v57; // rsi
  const __m128i *v58; // rsi
  char v59; // al
  __int64 v60; // rsi
  unsigned int v61; // eax
  unsigned __int64 v62; // rax
  unsigned int v63; // r15d
  unsigned __int64 v64; // r13
  __int64 v65; // rax
  int v66; // ecx
  __int64 v67; // rdx
  unsigned __int64 v68; // r12
  unsigned int v69; // r13d
  __int64 v70; // rax
  __int64 v71; // rsi
  unsigned int v72; // r15d
  _BYTE *v73; // r14
  int v74; // eax
  unsigned int v75; // edx
  __m128i *v76; // r11
  unsigned int v77; // r13d
  __int64 v78; // rcx
  __int8 v79; // si
  _BYTE *v80; // r14
  int v81; // eax
  unsigned int v82; // edx
  int v83; // r10d
  unsigned int v84; // r13d
  __int8 v85; // si
  char v86; // al
  char v87; // al
  unsigned int v88; // r13d
  unsigned __int64 v89; // r13
  unsigned __int64 v90; // rax
  char v91; // al
  unsigned int v92; // r13d
  char v93; // al
  char v94; // al
  char v95; // al
  char v96; // al
  unsigned int v97; // r13d
  char v98; // al
  __m128i *v99; // [rsp+0h] [rbp-210h]
  __m128i *v100; // [rsp+0h] [rbp-210h]
  __int64 v102; // [rsp+18h] [rbp-1F8h]
  __int64 v103; // [rsp+28h] [rbp-1E8h]
  __int64 v104; // [rsp+28h] [rbp-1E8h]
  __int64 v105; // [rsp+28h] [rbp-1E8h]
  __int64 v106; // [rsp+28h] [rbp-1E8h]
  __int64 v107; // [rsp+28h] [rbp-1E8h]
  __int64 v108; // [rsp+28h] [rbp-1E8h]
  __int64 v109; // [rsp+28h] [rbp-1E8h]
  int v110; // [rsp+30h] [rbp-1E0h]
  int v111; // [rsp+30h] [rbp-1E0h]
  int v112; // [rsp+30h] [rbp-1E0h]
  __m128i *v113; // [rsp+30h] [rbp-1E0h]
  int v114; // [rsp+30h] [rbp-1E0h]
  __m128i *v115; // [rsp+30h] [rbp-1E0h]
  __m128i *v116; // [rsp+30h] [rbp-1E0h]
  unsigned int v117; // [rsp+38h] [rbp-1D8h]
  unsigned int v118; // [rsp+38h] [rbp-1D8h]
  __m128i *v119; // [rsp+38h] [rbp-1D8h]
  unsigned int v120; // [rsp+38h] [rbp-1D8h]
  __m128i *v121; // [rsp+38h] [rbp-1D8h]
  unsigned int v122; // [rsp+38h] [rbp-1D8h]
  unsigned int v123; // [rsp+38h] [rbp-1D8h]
  _QWORD *v124; // [rsp+40h] [rbp-1D0h]
  unsigned int v125; // [rsp+50h] [rbp-1C0h]
  __int64 v126; // [rsp+50h] [rbp-1C0h]
  int v127; // [rsp+50h] [rbp-1C0h]
  __int64 v128; // [rsp+50h] [rbp-1C0h]
  unsigned int v129; // [rsp+50h] [rbp-1C0h]
  unsigned int v130; // [rsp+50h] [rbp-1C0h]
  unsigned int v131; // [rsp+50h] [rbp-1C0h]
  __int64 v132; // [rsp+58h] [rbp-1B8h]
  unsigned int v133; // [rsp+58h] [rbp-1B8h]
  unsigned int v134; // [rsp+58h] [rbp-1B8h]
  unsigned int v135; // [rsp+58h] [rbp-1B8h]
  _BYTE *v136; // [rsp+60h] [rbp-1B0h] BYREF
  __int64 v137; // [rsp+68h] [rbp-1A8h]
  _BYTE s[16]; // [rsp+70h] [rbp-1A0h] BYREF
  __m128i v139; // [rsp+80h] [rbp-190h] BYREF
  __m128i v140; // [rsp+90h] [rbp-180h] BYREF
  __int64 v141; // [rsp+A0h] [rbp-170h]
  _QWORD v142[2]; // [rsp+B0h] [rbp-160h] BYREF
  __int64 v143; // [rsp+C0h] [rbp-150h]
  __m128i v144; // [rsp+E0h] [rbp-130h] BYREF
  __m128i v145; // [rsp+F0h] [rbp-120h] BYREF
  __m128i v146; // [rsp+100h] [rbp-110h] BYREF
  __m128i v147; // [rsp+110h] [rbp-100h] BYREF
  __m128i v148; // [rsp+120h] [rbp-F0h] BYREF
  __int64 v149; // [rsp+130h] [rbp-E0h]
  __int32 v150; // [rsp+138h] [rbp-D8h]
  __int64 v151; // [rsp+140h] [rbp-D0h] BYREF
  _BYTE *v152; // [rsp+148h] [rbp-C8h]
  __int64 v153; // [rsp+150h] [rbp-C0h]
  unsigned int v154; // [rsp+158h] [rbp-B8h]
  const __m128i *v155; // [rsp+160h] [rbp-B0h] BYREF
  __m128i *v156; // [rsp+168h] [rbp-A8h]
  const __m128i *v157; // [rsp+170h] [rbp-A0h]
  __m128i v158; // [rsp+180h] [rbp-90h] BYREF
  __m128i v159; // [rsp+190h] [rbp-80h] BYREF
  __int64 v160; // [rsp+1A0h] [rbp-70h]

  v7 = a1;
  v8 = *(unsigned int *)(a1 + 48);
  v9 = s;
  v124 = (_QWORD *)a2;
  v151 = 0;
  v152 = 0;
  v153 = 0;
  v154 = 0;
  v155 = 0;
  v156 = 0;
  v157 = 0;
  v136 = s;
  v137 = 0x400000000LL;
  if ( (unsigned int)v8 > 4 )
  {
    a2 = (__int64)s;
    sub_16CD150((__int64)&v136, s, v8, 4, (int)&v136, (int)a6);
    v9 = v136;
  }
  LODWORD(v137) = v8;
  if ( 4 * v8 )
  {
    a2 = 0;
    memset(v9, 0, 4 * v8);
  }
  v10 = *(unsigned int *)(v7 + 48);
  if ( (_DWORD)v10 )
  {
    v102 = a4;
    v11 = 0;
    v132 = 4 * v10;
    while ( 1 )
    {
      v12 = 0;
      v13 = (const __m128i *)(*(_QWORD *)(v7 + 40) + 10 * v11);
      v139 = _mm_loadu_si128(v13);
      v140 = _mm_loadu_si128(v13 + 1);
      v141 = v13[2].m128i_i64[0];
      if ( v139.m128i_i8[0] || v139.m128i_i32[2] >= 0 )
        goto LABEL_12;
      v14 = v139.m128i_i32[2] & 0x7FFFFFFF;
      if ( *(_DWORD *)(v124[36] + 4 * v14) == 0x3FFFFFFF )
        break;
      if ( *(_DWORD *)(v124[39] + 4 * v14) )
      {
        v60 = *(unsigned int *)(v124[33] + 4 * v14);
        if ( (int)v60 > 0 )
          goto LABEL_75;
      }
      v140.m128i_i32[2] = *(_DWORD *)(v124[36] + 4 * v14);
      v12 = 1;
      v140.m128i_i64[0] = 0;
      v139.m128i_i32[0] = v139.m128i_i32[0] & 0xFFF00000 | 5;
LABEL_12:
      v15 = _mm_loadu_si128(&v139);
      v16 = _mm_loadu_si128(&v140);
      v146.m128i_i8[8] = v12;
      v17 = v154;
      v150 = 0;
      v144 = v15;
      v146.m128i_i64[0] = v141;
      v160 = v141;
      v149 = v141;
      v145 = v16;
      v158 = v15;
      v159 = v16;
      v147 = v15;
      v148 = v16;
      if ( !v154 )
      {
        ++v151;
        goto LABEL_52;
      }
      v125 = v154;
      v18 = v152;
      v142[0] = 19;
      v143 = 0;
      v158.m128i_i64[0] = 20;
      v159.m128i_i64[0] = 0;
      v19 = sub_1E36300(&v147);
      v20 = v11;
      v21 = v7;
      v22 = 0;
      v23 = 1;
      v24 = v125 - 1;
      v25 = v18;
      LODWORD(a6) = (v125 - 1) & v19;
      for ( i = (unsigned int)a6; ; i = v24 & v88 )
      {
        v27 = (__m128i *)&v25[48 * i];
        if ( (unsigned __int8)(v147.m128i_i8[0] - 19) > 1u )
        {
          a2 = (__int64)&v25[48 * i];
          v103 = v21;
          v110 = v23;
          v117 = v24;
          v126 = v20;
          v59 = sub_1E31610(&v147, a2);
          v20 = v126;
          v24 = v117;
          v23 = v110;
          v21 = v103;
          if ( v59 )
          {
LABEL_16:
            v28 = v155;
            v11 = v20;
            v7 = v21;
            v29 = &v155[3 * v27[2].m128i_u32[2]];
            goto LABEL_17;
          }
          LOBYTE(a2) = v27->m128i_i8[0];
        }
        else
        {
          a2 = v27->m128i_u8[0];
          if ( v147.m128i_i8[0] == (_BYTE)a2 )
            goto LABEL_16;
        }
        if ( (unsigned __int8)(a2 - 19) <= 1u )
        {
          if ( LOBYTE(v142[0]) == (_BYTE)a2 )
            break;
LABEL_138:
          if ( v158.m128i_i8[0] != (_BYTE)a2 )
            goto LABEL_120;
          goto LABEL_118;
        }
        v104 = v21;
        v111 = v23;
        v118 = v24;
        v128 = v20;
        v86 = sub_1E31610(&v25[48 * i], v142);
        v20 = v128;
        v24 = v118;
        v23 = v111;
        v21 = v104;
        if ( v86 )
          break;
        LOBYTE(a2) = v27->m128i_i8[0];
        if ( (unsigned __int8)(v27->m128i_i8[0] - 19) <= 1u )
          goto LABEL_138;
        v87 = sub_1E31610(&v25[48 * i], &v158);
        v20 = v128;
        v24 = v118;
        v23 = v111;
        v21 = v104;
        if ( !v87 )
          goto LABEL_120;
LABEL_118:
        if ( !v22 )
          v22 = (__m128i *)&v25[48 * i];
LABEL_120:
        v88 = v23 + i;
        ++v23;
      }
      v55 = v22;
      v11 = v20;
      v17 = v154;
      v7 = v21;
      if ( !v55 )
        v55 = v27;
      v56 = v153 + 1;
      ++v151;
      if ( 4 * ((int)v153 + 1) < 3 * v154 )
      {
        if ( v154 - (v56 + HIDWORD(v153)) > v154 >> 3 )
        {
          a6 = v55;
          goto LABEL_55;
        }
        sub_1DB1790((__int64)&v151, v154);
        v72 = v154;
        if ( !v154 )
        {
LABEL_53:
          a6 = 0;
          v55 = 0;
          goto LABEL_54;
        }
        v142[0] = 19;
        v73 = v152;
        v143 = 0;
        v158.m128i_i64[0] = 20;
        v159.m128i_i64[0] = 0;
        v74 = sub_1E36300(&v147);
        v75 = v72 - 1;
        v76 = 0;
        v127 = 1;
        v77 = (v72 - 1) & v74;
        v78 = v7;
        while ( 2 )
        {
          v55 = (__m128i *)&v73[48 * v77];
          a6 = v55;
          if ( (unsigned __int8)(v147.m128i_i8[0] - 19) > 1u )
          {
            v106 = v78;
            v113 = v76;
            v120 = v75;
            v93 = sub_1E31610(&v147, &v73[48 * v77]);
            v75 = v120;
            v76 = v113;
            v78 = v106;
            a6 = (__m128i *)&v73[48 * v77];
            if ( v93 )
              goto LABEL_110;
            v79 = v55->m128i_i8[0];
          }
          else
          {
            v79 = v55->m128i_i8[0];
            if ( v147.m128i_i8[0] == v55->m128i_i8[0] )
              goto LABEL_110;
          }
          if ( (unsigned __int8)(v79 - 19) > 1u )
          {
            v100 = a6;
            v108 = v78;
            v115 = v76;
            v122 = v75;
            v96 = sub_1E31610(&v73[48 * v77], v142);
            v75 = v122;
            v76 = v115;
            v78 = v108;
            a6 = v100;
            if ( v96 )
              goto LABEL_103;
            v79 = v55->m128i_i8[0];
          }
          else if ( v79 == LOBYTE(v142[0]) )
          {
            goto LABEL_103;
          }
          if ( (unsigned __int8)(v79 - 19) > 1u )
          {
            v109 = v78;
            v116 = v76;
            v123 = v75;
            v98 = sub_1E31610(&v73[48 * v77], &v158);
            v75 = v123;
            v76 = v116;
            v78 = v109;
            if ( v98 )
              goto LABEL_148;
          }
          else if ( v79 == v158.m128i_i8[0] )
          {
LABEL_148:
            if ( !v76 )
              v76 = (__m128i *)&v73[48 * v77];
          }
          v97 = v127 + v77;
          ++v127;
          v77 = v75 & v97;
          continue;
        }
      }
LABEL_52:
      sub_1DB1790((__int64)&v151, 2 * v17);
      v54 = v154;
      if ( !v154 )
        goto LABEL_53;
      v142[0] = 19;
      v80 = v152;
      v159.m128i_i64[0] = 0;
      v143 = 0;
      v158.m128i_i64[0] = 20;
      v81 = sub_1E36300(&v147);
      v82 = v54 - 1;
      v83 = 1;
      v76 = 0;
      v84 = (v54 - 1) & v81;
      v78 = v7;
      while ( 2 )
      {
        v55 = (__m128i *)&v80[48 * v84];
        a6 = v55;
        if ( (unsigned __int8)(v147.m128i_i8[0] - 19) > 1u )
        {
          v105 = v78;
          v112 = v83;
          v119 = v76;
          v130 = v82;
          v91 = sub_1E31610(&v147, &v80[48 * v84]);
          v82 = v130;
          v76 = v119;
          v83 = v112;
          v78 = v105;
          a6 = (__m128i *)&v80[48 * v84];
          if ( v91 )
          {
LABEL_110:
            v7 = v78;
            goto LABEL_54;
          }
          v85 = v55->m128i_i8[0];
        }
        else
        {
          v85 = v55->m128i_i8[0];
          if ( v147.m128i_i8[0] == v55->m128i_i8[0] )
            goto LABEL_110;
        }
        if ( (unsigned __int8)(v85 - 19) <= 1u )
        {
          if ( v85 == LOBYTE(v142[0]) )
            break;
LABEL_132:
          if ( v85 == v158.m128i_i8[0] )
            goto LABEL_133;
          goto LABEL_135;
        }
        v99 = a6;
        v107 = v78;
        v114 = v83;
        v121 = v76;
        v131 = v82;
        v94 = sub_1E31610(&v80[48 * v84], v142);
        v82 = v131;
        v76 = v121;
        v83 = v114;
        v78 = v107;
        a6 = v99;
        if ( !v94 )
        {
          v85 = v55->m128i_i8[0];
          if ( (unsigned __int8)(v55->m128i_i8[0] - 19) <= 1u )
            goto LABEL_132;
          v95 = sub_1E31610(&v80[48 * v84], &v158);
          v82 = v131;
          v76 = v121;
          v83 = v114;
          v78 = v107;
          if ( v95 )
          {
LABEL_133:
            if ( !v76 )
              v76 = (__m128i *)&v80[48 * v84];
          }
LABEL_135:
          v92 = v83 + v84;
          ++v83;
          v84 = v82 & v92;
          continue;
        }
        break;
      }
LABEL_103:
      v7 = v78;
      if ( v76 )
      {
        a6 = v76;
        v55 = v76;
      }
LABEL_54:
      v56 = v153 + 1;
LABEL_55:
      LODWORD(v153) = v56;
      v158.m128i_i64[0] = 19;
      v159.m128i_i64[0] = 0;
      if ( (unsigned __int8)(v55->m128i_i8[0] - 19) > 1u )
      {
        if ( !(unsigned __int8)sub_1E31610(a6, &v158) )
LABEL_57:
          --HIDWORD(v153);
      }
      else if ( v55->m128i_i8[0] != 19 )
      {
        goto LABEL_57;
      }
      *v55 = _mm_loadu_si128(&v147);
      v55[1] = _mm_loadu_si128(&v148);
      v55[2].m128i_i64[0] = v149;
      v55[2].m128i_i32[2] = v150;
      v57 = v156;
      if ( v156 == v157 )
      {
        sub_1DAB900(&v155, v156, &v144);
        v58 = v156;
      }
      else
      {
        if ( v156 )
        {
          *v156 = _mm_loadu_si128(&v144);
          v57[1] = _mm_loadu_si128(&v145);
          v57[2] = _mm_loadu_si128(&v146);
          v57 = v156;
        }
        v58 = v57 + 3;
        v156 = (__m128i *)v58;
      }
      v28 = v155;
      a2 = -1431655765 * (unsigned int)(v58 - v155) - 1;
      v55[2].m128i_i32[2] = a2;
      v29 = v156 - 3;
LABEL_17:
      *(_DWORD *)&v136[v11] = -1431655765 * (v29 - v28);
      v11 += 4;
      if ( v11 == v132 )
      {
        a4 = v102;
        goto LABEL_19;
      }
    }
    v60 = *(unsigned int *)(v124[33] + 4 * v14);
    if ( (int)v60 <= 0 )
    {
      sub_1E310D0(&v139, 0);
      v139.m128i_i32[0] &= 0xFFF000FF;
      goto LABEL_12;
    }
LABEL_75:
    v12 = 0;
    sub_1E311F0(&v139, v60, a3);
    goto LABEL_12;
  }
LABEL_19:
  v30 = v156;
  v31 = v155;
  *(_DWORD *)(v7 + 48) = 0;
  v32 = *(_QWORD *)(a4 + 8);
  *(_DWORD *)(a4 + 16) = 0;
  v33 = 0xAAAAAAAAAAAAAAABLL * (v30 - v31);
  v34 = v32 << 6;
  v35 = (unsigned int)v33;
  v36 = -1431655765 * (v30 - v31);
  if ( (unsigned int)v33 > v32 << 6 )
  {
    v64 = (unsigned int)(v33 + 63) >> 6;
    if ( v64 < 2 * v32 )
      v64 = 2 * v32;
    v65 = (__int64)realloc(*(_QWORD *)a4, 8 * v64, 8 * (int)v64, v33, v36, (int)a6);
    v36 = v33;
    if ( !v65 )
    {
      if ( 8 * v64 )
      {
        sub_16BD1C0("Allocation failed", 1u);
        v65 = 0;
        v36 = v33;
      }
      else
      {
        v65 = malloc(1u);
        v36 = v33;
        if ( !v65 )
        {
          sub_16BD1C0("Allocation failed", 1u);
          v36 = v33;
          v65 = 0;
        }
      }
    }
    v66 = *(_DWORD *)(a4 + 16);
    *(_QWORD *)a4 = v65;
    *(_QWORD *)(a4 + 8) = v64;
    LODWORD(a6) = (unsigned int)(v66 + 63) >> 6;
    a2 = (unsigned int)a6;
    if ( v64 > (unsigned int)a6 )
    {
      v89 = v64 - (unsigned int)a6;
      if ( v89 )
      {
        a2 = 0;
        v129 = (unsigned int)(v66 + 63) >> 6;
        v135 = v36;
        memset((void *)(v65 + 8LL * (unsigned int)a6), 0, 8 * v89);
        v66 = *(_DWORD *)(a4 + 16);
        v65 = *(_QWORD *)a4;
        LODWORD(a6) = v129;
        v36 = v135;
      }
    }
    v35 = v66 & 0x3F;
    if ( (_DWORD)v35 )
    {
      a2 = (unsigned int)((_DWORD)a6 - 1);
      v32 = (unsigned int)v32;
      *(_QWORD *)(v65 + 8 * a2) &= ~(-1LL << v35);
      v65 = *(_QWORD *)a4;
      v67 = *(_QWORD *)(a4 + 8) - (unsigned int)v32;
      if ( !v67 )
        goto LABEL_91;
    }
    else
    {
      v32 = (unsigned int)v32;
      v67 = *(_QWORD *)(a4 + 8) - (unsigned int)v32;
      if ( !v67 )
        goto LABEL_91;
    }
    a2 = 0;
    v133 = v36;
    memset((void *)(v65 + 8 * v32), 0, 8 * v67);
    v36 = v133;
LABEL_91:
    v34 = *(unsigned int *)(a4 + 16);
    v61 = v34;
    if ( (unsigned int)v33 <= (unsigned int)v34 )
      goto LABEL_77;
    v68 = *(_QWORD *)(a4 + 8);
    v69 = (unsigned int)(v34 + 63) >> 6;
    v70 = v69;
    if ( v69 >= v68 || (v32 = v68 - v69) == 0 )
    {
LABEL_93:
      v61 = v34;
      v35 = v34 & 0x3F;
      if ( (v34 & 0x3F) != 0 )
      {
        v34 = *(_QWORD *)a4;
        a2 = v69 - 1;
        *(_QWORD *)(*(_QWORD *)a4 + 8 * a2) &= ~(-1LL << v35);
        v61 = *(_DWORD *)(a4 + 16);
      }
      goto LABEL_77;
    }
LABEL_112:
    a2 = 0;
    v134 = v36;
    memset((void *)(*(_QWORD *)a4 + 8 * v70), 0, 8 * v32);
    v34 = *(unsigned int *)(a4 + 16);
    v36 = v134;
    goto LABEL_93;
  }
  if ( !(_DWORD)v33 )
    goto LABEL_21;
  v61 = 0;
  if ( v32 )
  {
    v70 = 0;
    v69 = 0;
    goto LABEL_112;
  }
LABEL_77:
  *(_DWORD *)(a4 + 16) = v33;
  if ( (unsigned int)v33 < v61 )
  {
    v62 = *(_QWORD *)(a4 + 8);
    v63 = (unsigned int)(v33 + 63) >> 6;
    v35 = v63;
    if ( v62 > v63 )
    {
      v90 = v62 - v63;
      if ( v90 )
      {
        a2 = 0;
        memset((void *)(*(_QWORD *)a4 + 8LL * v63), 0, 8 * v90);
        v36 = *(_DWORD *)(a4 + 16);
      }
    }
    v36 &= 0x3Fu;
    if ( v36 )
    {
      v34 = *(_QWORD *)a4;
      v35 = v36;
      a2 = v63 - 1;
      *(_QWORD *)(*(_QWORD *)a4 + 8 * a2) &= ~(-1LL << v36);
    }
  }
  v30 = v156;
  v31 = v155;
LABEL_21:
  for ( j = v31; v30 != j; j += 3 )
  {
    v38 = *(unsigned int *)(v7 + 48);
    if ( (unsigned int)v38 >= *(_DWORD *)(v7 + 52) )
    {
      a2 = v7 + 56;
      sub_16CD150(v7 + 40, (const void *)(v7 + 56), 0, 40, v36, (int)a6);
      v38 = *(unsigned int *)(v7 + 48);
    }
    v39 = (__m128i *)(*(_QWORD *)(v7 + 40) + 40 * v38);
    *v39 = _mm_loadu_si128(j);
    v39[1] = _mm_loadu_si128(j + 1);
    v34 = j[2].m128i_i64[0];
    v39[2].m128i_i64[0] = v34;
    ++*(_DWORD *)(v7 + 48);
    if ( j[2].m128i_i8[8] )
    {
      a2 = *(_QWORD *)a4;
      v35 = 0xAAAAAAAAAAAAAAABLL * (j - v155);
      v34 = 1LL << v35;
      *(_QWORD *)(*(_QWORD *)a4 + 8LL * ((unsigned int)v35 >> 6)) |= 1LL << v35;
    }
  }
  v158.m128i_i64[0] = v7 + 216;
  v159.m128i_i64[0] = 0x400000000LL;
  v158.m128i_i64[1] = (__int64)&v159.m128i_i64[1];
  sub_1DA9720(v158.m128i_i64, a2, v34, v35, v36, (int)a6);
  v40 = v159.m128i_u32[0];
  if ( v159.m128i_i32[0] )
  {
    v41 = v158.m128i_u64[1];
    do
    {
      if ( *(_DWORD *)(v41 + 12) >= *(_DWORD *)(v41 + 8) )
        break;
      v46 = v41 + 16 * v40 - 16;
      v47 = *(_QWORD *)v46;
      v48 = *(unsigned int *)(v46 + 12);
      if ( *(_DWORD *)(v158.m128i_i64[0] + 80) )
        v49 = v48 + 36;
      else
        v49 = v48 + 16;
      v50 = (int *)(v47 + 4 * v49);
      v51 = *v50 & 0x7FFFFFFF;
      if ( (_DWORD)v51 != 0x7FFFFFFF )
      {
        *v50 = *(_DWORD *)&v136[4 * v51] & 0x7FFFFFFF | ((unsigned int)*v50 >> 31 << 31);
        sub_1DAB7B0(
          (__int64)&v158,
          *(_QWORD *)(*(_QWORD *)(v158.m128i_i64[1] + 16LL * v159.m128i_u32[0] - 16)
                    + 16LL * *(unsigned int *)(v158.m128i_i64[1] + 16LL * v159.m128i_u32[0] - 16 + 12)));
        v41 = v158.m128i_u64[1];
      }
      v52 = v41 + 16LL * v159.m128i_u32[0] - 16;
      v53 = *(_DWORD *)(v52 + 12) + 1;
      *(_DWORD *)(v52 + 12) = v53;
      v40 = v159.m128i_u32[0];
      v41 = v158.m128i_u64[1];
      if ( v53 == *(_DWORD *)(v158.m128i_i64[1] + 16LL * v159.m128i_u32[0] - 8) )
      {
        v71 = *(unsigned int *)(v158.m128i_i64[0] + 80);
        if ( (_DWORD)v71 )
        {
          sub_39460A0(&v158.m128i_u64[1], v71);
          v40 = v159.m128i_u32[0];
          v41 = v158.m128i_u64[1];
        }
      }
    }
    while ( (_DWORD)v40 );
  }
  else
  {
    v41 = v158.m128i_u64[1];
  }
  if ( (unsigned __int64 *)v41 != &v159.m128i_u64[1] )
    _libc_free(v41);
  if ( v136 != s )
    _libc_free((unsigned __int64)v136);
  if ( v155 )
    j_j___libc_free_0(v155, (char *)v157 - (char *)v155);
  if ( v154 )
  {
    v42 = v152;
    v147.m128i_i64[0] = 19;
    v159.m128i_i64[0] = 0;
    v148.m128i_i64[0] = 0;
    v43 = &v152[48 * v154];
    v158.m128i_i64[0] = 20;
    do
    {
      while ( (unsigned __int8)(*v42 - 19) <= 1u
           || (unsigned __int8)sub_1E31610(v42, &v147)
           || (unsigned __int8)(*v42 - 19) <= 1u )
      {
        v42 += 48;
        if ( v43 == v42 )
          return j___libc_free_0(v152);
      }
      v44 = v42;
      v42 += 48;
      sub_1E31610(v44, &v158);
    }
    while ( v43 != v42 );
  }
  return j___libc_free_0(v152);
}
