// Function: sub_20912B0
// Address: 0x20912b0
//
void __fastcall sub_20912B0(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        int a5,
        int a6,
        __m128i a7,
        __m128i a8,
        __m128i a9,
        __int64 a10,
        __m128i *src,
        unsigned __int64 a12,
        int a13,
        int a14,
        unsigned int a15)
{
  __int64 v15; // rbx
  __int64 v17; // rdx
  __int64 *v18; // r13
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  unsigned int v22; // edi
  unsigned int v23; // ebx
  __m128i *v24; // rax
  __int64 v25; // rdx
  __m128i *v26; // r12
  __int64 *v27; // r14
  __int64 v28; // rbx
  __int64 v29; // rdx
  __int64 v30; // rax
  unsigned __int32 v31; // r13d
  unsigned int v32; // eax
  __int32 v33; // eax
  __int64 v34; // r13
  __int64 v35; // r15
  __int64 v36; // r12
  __int64 v37; // rbx
  __int64 v38; // r15
  __int64 v39; // rdx
  __int64 v40; // rax
  bool v41; // zf
  unsigned int v42; // eax
  unsigned int v43; // eax
  __int64 v44; // r13
  __int64 v45; // r15
  __int64 v46; // rdx
  __int64 v47; // rax
  _QWORD *v48; // rsi
  _QWORD *v49; // rax
  unsigned int v50; // r9d
  unsigned __int64 v51; // rax
  int v52; // ecx
  unsigned int v53; // r9d
  int v54; // ecx
  __int64 v55; // r15
  __int64 v56; // rcx
  __int64 v57; // r11
  __int32 v58; // r10d
  __int64 v59; // rax
  __int64 v60; // rsi
  __int64 v61; // rax
  __m128i *v62; // r13
  __int64 v63; // rsi
  __int64 v64; // rax
  const __m128i *v65; // r12
  unsigned __int64 v66; // rax
  __m128i *v67; // r15
  const __m128i *v68; // rdi
  unsigned __int64 v69; // rax
  unsigned __int64 v70; // rdx
  __int64 v71; // r12
  __int64 v72; // r10
  __int64 v73; // rax
  unsigned int v74; // edx
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 *v77; // rax
  unsigned __int64 v78; // rdx
  unsigned __int8 *v79; // rax
  unsigned int v80; // ebx
  __int64 *v81; // r13
  __int128 v82; // rax
  __int16 *v83; // rdx
  __int128 v84; // rax
  __int64 v85; // rdx
  __int64 v86; // rsi
  __int64 v87; // rdx
  int v88; // ecx
  int v89; // eax
  __int64 v90; // rdx
  __int64 v91; // rcx
  __int64 v92; // r8
  __int64 v93; // r9
  __int64 v94; // rsi
  __int64 *v95; // rbx
  __int64 v97; // rdx
  __int64 v98; // rcx
  __int64 v99; // r8
  int v100; // r9d
  __int64 *v101; // rax
  __int64 *v102; // rdi
  __int64 v103; // rbx
  __int16 *v104; // rdx
  __int64 *v105; // rax
  __int64 *v106; // r14
  __int64 v107; // rdx
  unsigned __int64 v108; // r13
  __int64 v109; // r12
  __int64 v110; // rcx
  __int64 v111; // r8
  __int64 v112; // r9
  __int128 v113; // rax
  __int64 *v114; // rax
  int v115; // edx
  __m128i v116; // xmm7
  int v117; // eax
  int v118; // eax
  __int32 v119; // [rsp+14h] [rbp-15Ch]
  unsigned int v120; // [rsp+18h] [rbp-158h]
  __int64 v121; // [rsp+20h] [rbp-150h]
  __int64 v122; // [rsp+20h] [rbp-150h]
  __m128i *v123; // [rsp+28h] [rbp-148h]
  __m128i v124; // [rsp+28h] [rbp-148h]
  const void **v125; // [rsp+28h] [rbp-148h]
  __int64 v126; // [rsp+30h] [rbp-140h]
  unsigned int v127; // [rsp+30h] [rbp-140h]
  unsigned int v128; // [rsp+30h] [rbp-140h]
  __int64 *v129; // [rsp+30h] [rbp-140h]
  __int64 v130; // [rsp+38h] [rbp-138h]
  int v131; // [rsp+38h] [rbp-138h]
  int v132; // [rsp+38h] [rbp-138h]
  __int64 v133; // [rsp+40h] [rbp-130h]
  __int64 v134; // [rsp+40h] [rbp-130h]
  __int64 *v137; // [rsp+50h] [rbp-120h]
  __int16 *v138; // [rsp+58h] [rbp-118h]
  __int64 v139; // [rsp+60h] [rbp-110h]
  __int64 v140; // [rsp+60h] [rbp-110h]
  __int64 *v141; // [rsp+60h] [rbp-110h]
  unsigned __int64 v142; // [rsp+68h] [rbp-108h]
  _QWORD *v145; // [rsp+80h] [rbp-F0h]
  __int64 v146; // [rsp+88h] [rbp-E8h]
  unsigned int v147; // [rsp+90h] [rbp-E0h]
  __int64 v148; // [rsp+90h] [rbp-E0h]
  __int64 v149; // [rsp+A0h] [rbp-D0h]
  const void **v150; // [rsp+A0h] [rbp-D0h]
  __int128 v151; // [rsp+A0h] [rbp-D0h]
  __int64 v152; // [rsp+C0h] [rbp-B0h] BYREF
  __int32 v153; // [rsp+C8h] [rbp-A8h]
  __int64 v154[2]; // [rsp+D0h] [rbp-A0h] BYREF
  __int64 v155; // [rsp+E0h] [rbp-90h] BYREF
  unsigned __int32 v156; // [rsp+E8h] [rbp-88h]
  __m128i v157; // [rsp+F0h] [rbp-80h] BYREF
  __m128i v158; // [rsp+100h] [rbp-70h] BYREF
  __int64 v159; // [rsp+110h] [rbp-60h]
  __int64 v160; // [rsp+118h] [rbp-58h]
  __int64 v161; // [rsp+120h] [rbp-50h]
  __int64 v162; // [rsp+128h] [rbp-48h] BYREF
  unsigned __int32 v163; // [rsp+130h] [rbp-40h]
  __int32 v164; // [rsp+138h] [rbp-38h]
  __int32 v165; // [rsp+13Ch] [rbp-34h]

  v15 = 0;
  v17 = *(_QWORD *)(a1 + 712);
  v18 = *(__int64 **)(a10 + 8);
  v133 = *(_QWORD *)(v17 + 8);
  v149 = a10;
  if ( (__int64 *)(v133 + 320) != v18 )
    v15 = *(_QWORD *)(a10 + 8);
  v139 = v133 + 320;
  if ( -858993459 * (unsigned int)((__int64)(a12 - (_QWORD)src) >> 3) != 1 )
    goto LABEL_5;
  if ( a10 != a3 )
    goto LABEL_5;
  v71 = src->m128i_i64[1];
  if ( v71 != src[1].m128i_i64[0] )
    goto LABEL_5;
  v72 = *(_QWORD *)(a12 + 8);
  if ( v72 != *(_QWORD *)(a12 + 16) || src[1].m128i_i64[1] != *(_QWORD *)(a12 + 24) )
    goto LABEL_5;
  v73 = *(_QWORD *)(v17 + 32);
  v74 = *(_DWORD *)(v72 + 32);
  v122 = v73;
  v129 = (__int64 *)(v71 + 24);
  v125 = (const void **)(v72 + 24);
  v157.m128i_i32[2] = v74;
  if ( v74 <= 0x40 )
  {
    v75 = *(_QWORD *)(v72 + 24);
LABEL_84:
    v76 = *(_QWORD *)(v71 + 24) ^ v75;
    v153 = v74;
    v152 = v76;
    goto LABEL_85;
  }
  v148 = v72;
  sub_16A4FD0((__int64)&v157, v125);
  v74 = v157.m128i_u32[2];
  v72 = v148;
  if ( v157.m128i_i32[2] <= 0x40u )
  {
    v75 = v157.m128i_i64[0];
    goto LABEL_84;
  }
  sub_16A8F00(v157.m128i_i64, v129);
  v76 = v157.m128i_i64[0];
  v72 = v148;
  v153 = v157.m128i_i32[2];
  v152 = v157.m128i_i64[0];
  if ( v157.m128i_i32[2] > 0x40u )
  {
    v118 = sub_16A5940((__int64)&v152);
    v72 = v148;
    if ( v118 == 1 )
      goto LABEL_87;
    goto LABEL_100;
  }
LABEL_85:
  if ( !v76 || (v76 & (v76 - 1)) != 0 )
  {
LABEL_100:
    sub_135E100(&v152);
LABEL_5:
    if ( !(unsigned int)sub_1700720(*(_QWORD *)(a1 + 544)) )
      goto LABEL_6;
    v65 = (const __m128i *)(a12 + 40);
    if ( src != (__m128i *)(a12 + 40) )
    {
      _BitScanReverse64(&v66, 0xCCCCCCCCCCCCCCCDLL * (((char *)v65 - (char *)src) >> 3));
      sub_2046F10((__int64)src, (__m128i *)(a12 + 40), 2LL * (int)(63 - (v66 ^ 0x3F)), v19, v20, v21, a7);
      if ( (char *)v65 - (char *)src <= 640 )
      {
        sub_2045C10(src, v65);
      }
      else
      {
        v67 = src + 40;
        sub_2045C10(src, src + 40);
        if ( v65 != &src[40] )
        {
          do
          {
            v68 = v67;
            v67 = (__m128i *)((char *)v67 + 40);
            sub_20453E0(v68);
          }
          while ( v65 != v67 );
        }
      }
    }
    v69 = a12;
    if ( a12 > (unsigned __int64)src )
    {
      while ( 1 )
      {
        v70 = v69;
        v69 -= 40LL;
        if ( *(_DWORD *)(v69 + 32) > *(_DWORD *)(a12 + 32) )
          goto LABEL_77;
        if ( !*(_DWORD *)v69 && *(_QWORD *)(v69 + 24) == v15 )
          break;
        if ( (unsigned __int64)src >= v69 )
          goto LABEL_77;
      }
      a7 = _mm_loadu_si128((const __m128i *)(v70 - 40));
      v157 = a7;
      a8 = _mm_loadu_si128((const __m128i *)(v70 - 24));
      a9 = _mm_loadu_si128((const __m128i *)a12);
      v158 = a8;
      v159 = *(_QWORD *)(v70 - 8);
      *(__m128i *)(v70 - 40) = a9;
      *(__m128i *)(v70 - 24) = _mm_loadu_si128((const __m128i *)(a12 + 16));
      *(_DWORD *)(v70 - 8) = *(_DWORD *)(a12 + 32);
      v116 = _mm_loadu_si128(&v158);
      v117 = v159;
      *(__m128i *)a12 = _mm_loadu_si128(&v157);
      *(_DWORD *)(a12 + 32) = v117;
      *(__m128i *)(a12 + 16) = v116;
LABEL_77:
      v22 = a15;
    }
    else
    {
LABEL_6:
      v22 = a15;
      if ( a12 < (unsigned __int64)src )
        return;
    }
    v23 = v22;
    v24 = src;
    do
    {
      v25 = v23;
      v23 += v24[2].m128i_u32[0];
      if ( (unsigned __int64)v24[2].m128i_u32[0] + v25 > 0x80000000 )
        v23 = 0x80000000;
      v24 = (__m128i *)((char *)v24 + 40);
    }
    while ( a12 >= (unsigned __int64)v24 );
    v26 = src;
    v147 = v23;
    v27 = v18;
    v120 = v22 >> 1;
    v121 = v22 >> 1;
    while ( 1 )
    {
      v28 = a4;
      if ( (__m128i *)a12 != v26 )
      {
        v28 = (__int64)sub_1E0B6F0(v133, *(_QWORD *)(v149 + 40));
        sub_1DD8DC0(v139, v28);
        v29 = *v27;
        v30 = *(_QWORD *)v28;
        *(_QWORD *)(v28 + 8) = v27;
        v29 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)v28 = v29 | v30 & 7;
        *(_QWORD *)(v29 + 8) = v28;
        *v27 = v28 | *v27 & 7;
        sub_2090460(a1, (__int64)a2, a7, a8, a9);
      }
      v31 = v26[2].m128i_u32[0];
      v32 = 0;
      if ( v31 <= v147 )
        v32 = v147 - v31;
      v147 = v32;
      v33 = v26->m128i_i32[0];
      if ( v26->m128i_i32[0] == 1 )
      {
        v44 = *(_QWORD *)(a1 + 608) + 80LL * v26[1].m128i_u32[2];
        v45 = *(_QWORD *)(v44 + 64);
        sub_1DD8DC0(v139, v45);
        v46 = *v27;
        v47 = *(_QWORD *)v45;
        *(_QWORD *)(v45 + 8) = v27;
        v46 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)v45 = v46 | v47 & 7;
        *(_QWORD *)(v46 + 8) = v45;
        *v27 = v45 | *v27 & 7;
        v48 = *(_QWORD **)(v45 + 88);
        v49 = *(_QWORD **)(v45 + 96);
        v50 = v26[2].m128i_u32[0];
        if ( v49 == v48 )
        {
LABEL_62:
          v54 = v147;
        }
        else
        {
          while ( *v48 != a4 )
          {
            if ( v49 == ++v48 )
              goto LABEL_62;
          }
          v51 = v121 + v50;
          v52 = 0;
          v53 = v120 + v50;
          if ( v51 > 0x80000000 )
            v53 = 0x80000000;
          v128 = v53;
          if ( v147 >= v120 )
            v52 = v147 - v120;
          v131 = v52;
          sub_1DD76A0(v45, (__int64)v48, v120);
          sub_1D96570(*(unsigned int **)(v45 + 112), *(unsigned int **)(v45 + 120));
          v54 = v131;
          v50 = v128;
        }
        v132 = v50;
        sub_2052F00(a1, v149, v28, v54);
        sub_2052F00(a1, v149, v45, v132);
        sub_1D96570(*(unsigned int **)(v149 + 112), *(unsigned int **)(v149 + 120));
        *(_QWORD *)(v44 + 40) = v149;
        *(_QWORD *)(v44 + 72) = v28;
        if ( a3 == v149 )
        {
          sub_206A770(a1, v44 + 56, v44, a3, a7, a8, a9);
          *(_BYTE *)(v44 + 48) = 1;
        }
        goto LABEL_13;
      }
      if ( v33 != 2 )
        break;
      v130 = *(_QWORD *)(a1 + 632) + 184LL * v26[1].m128i_u32[2];
      v34 = *(_QWORD *)(v130 + 64);
      v35 = 32LL * *(unsigned int *)(v130 + 72);
      if ( v34 != v34 + v35 )
      {
        v123 = v26;
        v36 = *(_QWORD *)(v130 + 64);
        v126 = v28;
        v37 = v34 + v35;
        do
        {
          v38 = *(_QWORD *)(v36 + 8);
          v36 += 32;
          sub_1DD8DC0(v139, v38);
          v39 = *v27;
          v40 = *(_QWORD *)v38;
          *(_QWORD *)(v38 + 8) = v27;
          v39 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)v38 = v39 | v40 & 7;
          *(_QWORD *)(v39 + 8) = v38;
          *v27 = *v27 & 7 | v38;
        }
        while ( v36 != v37 );
        v28 = v126;
        v26 = v123;
      }
      *(_QWORD *)(v130 + 48) = v149;
      v41 = *(_BYTE *)(v130 + 46) == 0;
      *(_QWORD *)(v130 + 56) = v28;
      *(_DWORD *)(v130 + 180) = v147;
      if ( v41 )
      {
        v127 = *(_DWORD *)(v130 + 176);
        v42 = v120 + v127;
        if ( v121 + (unsigned __int64)v127 > 0x80000000 )
          v42 = 0x80000000;
        *(_DWORD *)(v130 + 176) = v42;
        v43 = v147 - v120;
        if ( v147 < v120 )
          v43 = 0;
        *(_DWORD *)(v130 + 180) = v43;
      }
      if ( a3 != v149 )
        goto LABEL_13;
      v26 = (__m128i *)((char *)v26 + 40);
      sub_206BB00(a1, v130, a3, a7, a8, a9);
      *(_BYTE *)(v130 + 45) = 1;
      if ( a12 < (unsigned __int64)v26 )
        return;
LABEL_14:
      v149 = v28;
    }
    if ( v33 )
      goto LABEL_13;
    v55 = v26->m128i_i64[1];
    v56 = v26[1].m128i_i64[0];
    if ( v55 == v56 )
    {
      v55 = (__int64)a2;
      v58 = 17;
      v57 = 0;
    }
    else
    {
      v57 = (__int64)a2;
      v58 = 21;
    }
    v59 = *(_QWORD *)a1;
    v156 = *(_DWORD *)(a1 + 536);
    if ( v59 && &v155 != (__int64 *)(v59 + 48) && (v60 = *(_QWORD *)(v59 + 48), (v155 = v60) != 0) )
    {
      v119 = v58;
      v124.m128i_i64[0] = v57;
      v124.m128i_i64[1] = v56;
      sub_1623A60((__int64)&v155, v60, 2);
      v61 = v26[1].m128i_i64[1];
      v157.m128i_i64[1] = v55;
      v160 = v28;
      v159 = v61;
      v31 = v26[2].m128i_u32[0];
      v157.m128i_i32[0] = v119;
      v158 = v124;
      v161 = v149;
      v162 = v155;
      if ( v155 )
      {
        sub_1623A60((__int64)&v162, v155, 2);
        v164 = v31;
        v163 = v156;
        v165 = v147;
        if ( v155 )
          sub_161E7C0((__int64)&v155, v155);
        if ( a3 != v149 )
        {
LABEL_54:
          v62 = *(__m128i **)(a1 + 592);
          if ( v62 == *(__m128i **)(a1 + 600) )
          {
            sub_2055B00((__int64 *)(a1 + 584), *(_QWORD *)(a1 + 592), (__int64)&v157);
          }
          else
          {
            if ( v62 )
            {
              v62->m128i_i32[0] = v157.m128i_i32[0];
              v62->m128i_i64[1] = v157.m128i_i64[1];
              v62[1] = v158;
              v62[2].m128i_i64[0] = v159;
              v62[2].m128i_i64[1] = v160;
              v62[3].m128i_i64[0] = v161;
              v63 = v162;
              v62[3].m128i_i64[1] = v162;
              if ( v63 )
                sub_1623A60((__int64)&v62[3].m128i_i64[1], v63, 2);
              v62[4].m128i_i32[0] = v163;
              v62[4].m128i_i32[2] = v164;
              v62[4].m128i_i32[3] = v165;
              v62 = *(__m128i **)(a1 + 592);
            }
            *(_QWORD *)(a1 + 592) = v62 + 5;
          }
LABEL_60:
          if ( v162 )
            sub_161E7C0((__int64)&v162, v162);
LABEL_13:
          v26 = (__m128i *)((char *)v26 + 40);
          if ( a12 < (unsigned __int64)v26 )
            return;
          goto LABEL_14;
        }
LABEL_65:
        sub_2069F40((__int64 *)a1, (__int64)&v157, a3, a7, a8, a9);
        goto LABEL_60;
      }
    }
    else
    {
      v64 = v26[1].m128i_i64[1];
      v157.m128i_i32[0] = v58;
      v157.m128i_i64[1] = v55;
      v159 = v64;
      v158.m128i_i64[0] = v57;
      v158.m128i_i64[1] = v56;
      v160 = v28;
      v161 = v149;
      v162 = 0;
    }
    v164 = v31;
    v163 = v156;
    v165 = v147;
    if ( a3 != v149 )
      goto LABEL_54;
    goto LABEL_65;
  }
LABEL_87:
  v134 = v72;
  v77 = sub_20685E0(a1, a2, a7, a8, a9);
  v142 = v78;
  v140 = (__int64)v77;
  v79 = (unsigned __int8 *)(v77[5] + 16LL * (unsigned int)v78);
  v80 = *v79;
  v150 = (const void **)*((_QWORD *)v79 + 1);
  sub_204D410((__int64)v154, *(_QWORD *)a1, *(_DWORD *)(a1 + 536));
  v81 = *(__int64 **)(a1 + 552);
  *(_QWORD *)&v82 = sub_1D38970(
                      (__int64)v81,
                      (__int64)&v152,
                      (__int64)v154,
                      v80,
                      v150,
                      0,
                      a7,
                      *(double *)a8.m128i_i64,
                      a9,
                      0);
  v137 = sub_1D332F0(
           v81,
           119,
           (__int64)v154,
           v80,
           v150,
           0,
           *(double *)a7.m128i_i64,
           *(double *)a8.m128i_i64,
           a9,
           v140,
           v142,
           v82);
  v138 = v83;
  v141 = *(__int64 **)(a1 + 552);
  v156 = *(_DWORD *)(v134 + 32);
  if ( v156 > 0x40 )
  {
    sub_16A4FD0((__int64)&v155, v125);
    if ( v156 > 0x40 )
    {
      sub_16A89F0(&v155, v129);
      goto LABEL_90;
    }
  }
  else
  {
    v155 = *(_QWORD *)(v134 + 24);
  }
  v155 |= *(_QWORD *)(v71 + 24);
LABEL_90:
  v157.m128i_i32[2] = v156;
  v156 = 0;
  v157.m128i_i64[0] = v155;
  *(_QWORD *)&v84 = sub_1D38970(
                      (__int64)v141,
                      (__int64)&v157,
                      (__int64)v154,
                      v80,
                      v150,
                      0,
                      a7,
                      *(double *)a8.m128i_i64,
                      a9,
                      0);
  *(_QWORD *)&v151 = sub_1F81070(
                       v141,
                       (__int64)v154,
                       2u,
                       0,
                       (unsigned __int64)v137,
                       v138,
                       (__m128)a7,
                       *(double *)a8.m128i_i64,
                       a9,
                       v84,
                       0x11u);
  *((_QWORD *)&v151 + 1) = v85;
  sub_135E100(v157.m128i_i64);
  sub_135E100(&v155);
  v86 = src[2].m128i_u32[0];
  v87 = *(unsigned int *)(a12 + 32);
  v88 = 0x80000000;
  if ( (unsigned __int64)(v86 + v87) <= 0x80000000 )
    v88 = v86 + v87;
  sub_2052F00(a1, a3, src[1].m128i_i64[1], v88);
  if ( v122 )
  {
    v89 = sub_1377370(v122, *(_QWORD *)(a3 + 40), 0);
    sub_2052F00(a1, a3, a4, v89);
  }
  else
  {
    sub_2052F00(a1, a3, a4, -1);
  }
  v94 = src[1].m128i_i64[1];
  v95 = *(__int64 **)(a1 + 552);
  v145 = sub_1D2A490(v95, v94, v90, v91, v92, v93);
  v146 = v97;
  v101 = sub_2051DF0((__int64 *)a1, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9, v94, v97, v98, v99, v100);
  v102 = v95;
  v103 = a1;
  v105 = sub_1D3A900(
           v102,
           0xBFu,
           (__int64)v154,
           1u,
           0,
           0,
           (__m128)a7,
           *(double *)a8.m128i_i64,
           a9,
           (unsigned __int64)v101,
           v104,
           v151,
           (__int64)v145,
           v146);
  v106 = *(__int64 **)(a1 + 552);
  v108 = v107;
  v109 = (__int64)v105;
  *(_QWORD *)&v113 = sub_1D2A490(v106, a4, v107, v110, v111, v112);
  v114 = sub_1D332F0(
           v106,
           188,
           (__int64)v154,
           1,
           0,
           0,
           *(double *)a7.m128i_i64,
           *(double *)a8.m128i_i64,
           a9,
           v109,
           v108,
           v113);
  sub_2045100(*(_QWORD *)(v103 + 552), (__int64)v114, v115);
  sub_17CD270(v154);
  sub_135E100(&v152);
}
