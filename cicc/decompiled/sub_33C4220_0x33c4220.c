// Function: sub_33C4220
// Address: 0x33c4220
//
unsigned __int64 __fastcall sub_33C4220(
        __int64 a1,
        _BYTE *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __int64 a8,
        __m128i *src,
        unsigned __int64 a10,
        int a11,
        int a12,
        unsigned int a13)
{
  __int64 *v14; // r15
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rbx
  unsigned __int64 result; // rax
  unsigned int v19; // r9d
  __m128i *v20; // rdx
  unsigned int v21; // eax
  __int64 v22; // rcx
  __int64 v23; // r13
  __int64 *v24; // r12
  __int64 v25; // rbx
  __int64 v26; // rdx
  __int64 v27; // rax
  unsigned __int32 v28; // r14d
  unsigned int v29; // eax
  __int32 v30; // eax
  __int64 v31; // r15
  __int64 v32; // r14
  __int64 *v33; // r13
  __int64 v34; // r12
  __int64 v35; // rbx
  __int64 v36; // r14
  __int64 v37; // rdx
  __int64 v38; // rax
  bool v39; // zf
  unsigned int v40; // eax
  unsigned int v41; // eax
  __int64 v42; // r14
  __int64 v43; // r15
  __int64 v44; // r8
  __int64 v45; // rdi
  __int64 v46; // rdx
  _QWORD *v47; // rsi
  __int64 v48; // r9
  _QWORD *v49; // rax
  unsigned __int64 v50; // rax
  unsigned int v51; // r9d
  unsigned int v52; // ecx
  __int64 v53; // rcx
  __int64 v54; // rcx
  __int64 v55; // r9
  __int64 v56; // r11
  __int32 v57; // r15d
  __int64 v58; // rax
  __int64 v59; // rsi
  __int64 v60; // rax
  _QWORD *v61; // r14
  __m128i *v62; // r13
  __int64 v63; // rsi
  __int64 v64; // rsi
  __int64 v65; // rax
  char v66; // al
  __int64 v67; // r13
  __int64 v68; // r14
  unsigned int v69; // eax
  __int64 v70; // rdx
  __int64 v71; // rdx
  __int64 v72; // rdx
  __int64 v73; // rax
  int v74; // r15d
  __int128 v75; // rax
  int v76; // r9d
  __int64 v77; // rdx
  __int64 v78; // r14
  __int64 v79; // rdx
  __int64 v80; // r15
  __int128 v81; // rax
  int v82; // r9d
  __int64 v83; // r14
  __int64 v84; // rdx
  __int64 v85; // r15
  __int64 v86; // r8
  __int64 v87; // r9
  __int64 v88; // rsi
  __int64 v89; // rdx
  __int64 v90; // rcx
  __int64 v91; // r8
  __int64 v92; // r9
  unsigned int v93; // eax
  __int64 v94; // r8
  __int64 v95; // r9
  __int64 v96; // rsi
  __int64 v97; // r13
  __int64 v98; // rdx
  __int64 v99; // rcx
  __int64 v100; // r8
  __int64 v101; // r9
  __int128 v102; // rax
  __int64 v103; // r12
  __int64 v104; // rdx
  __int64 v105; // r13
  __int64 v106; // r14
  __int128 v107; // rax
  int v108; // r9d
  __int64 v109; // rax
  int v110; // edx
  __int64 v111; // rax
  const __m128i *v112; // r13
  unsigned __int64 v113; // rax
  __m128i *v114; // r14
  const __m128i *v115; // rdi
  unsigned __int64 v116; // rdx
  __m128i v117; // xmm2
  __m128i v118; // xmm7
  int v119; // eax
  int v120; // eax
  __int128 v121; // [rsp-30h] [rbp-190h]
  __int128 v122; // [rsp-20h] [rbp-180h]
  __int128 v123; // [rsp-20h] [rbp-180h]
  unsigned int v124; // [rsp+4h] [rbp-15Ch]
  __int64 v125; // [rsp+8h] [rbp-158h]
  unsigned __int64 v126; // [rsp+8h] [rbp-158h]
  __int64 v127; // [rsp+10h] [rbp-150h]
  __m128i v128; // [rsp+10h] [rbp-150h]
  __int64 v129; // [rsp+10h] [rbp-150h]
  __int64 v130; // [rsp+18h] [rbp-148h]
  unsigned int v131; // [rsp+18h] [rbp-148h]
  unsigned int v132; // [rsp+18h] [rbp-148h]
  const void **v133; // [rsp+18h] [rbp-148h]
  __int64 v135; // [rsp+28h] [rbp-138h]
  unsigned int v136; // [rsp+28h] [rbp-138h]
  __int64 v137; // [rsp+28h] [rbp-138h]
  unsigned int v138; // [rsp+28h] [rbp-138h]
  __int64 *v139; // [rsp+28h] [rbp-138h]
  __int64 *v140; // [rsp+30h] [rbp-130h]
  __int128 v143; // [rsp+40h] [rbp-120h]
  __int128 v144; // [rsp+40h] [rbp-120h]
  __int64 v145; // [rsp+50h] [rbp-110h]
  __int64 v146; // [rsp+50h] [rbp-110h]
  __int64 v147; // [rsp+50h] [rbp-110h]
  bool v148; // [rsp+58h] [rbp-108h]
  unsigned int v149; // [rsp+58h] [rbp-108h]
  unsigned int v150; // [rsp+58h] [rbp-108h]
  unsigned int v152; // [rsp+70h] [rbp-F0h]
  __m128i *v153; // [rsp+80h] [rbp-E0h]
  __int64 v154; // [rsp+80h] [rbp-E0h]
  __int128 v155; // [rsp+80h] [rbp-E0h]
  __int64 v156; // [rsp+A0h] [rbp-C0h] BYREF
  __int32 v157; // [rsp+A8h] [rbp-B8h]
  _QWORD v158[2]; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v159; // [rsp+C0h] [rbp-A0h] BYREF
  unsigned __int32 v160; // [rsp+C8h] [rbp-98h]
  __m128i v161; // [rsp+D0h] [rbp-90h] BYREF
  __m128i v162; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v163; // [rsp+F0h] [rbp-70h]
  __int64 v164; // [rsp+F8h] [rbp-68h]
  __int64 v165; // [rsp+100h] [rbp-60h]
  __int64 v166; // [rsp+108h] [rbp-58h] BYREF
  unsigned __int32 v167; // [rsp+110h] [rbp-50h]
  __int64 v168; // [rsp+118h] [rbp-48h] BYREF
  __int32 v169; // [rsp+120h] [rbp-40h]
  __int32 v170; // [rsp+124h] [rbp-3Ch]
  __int8 v171; // [rsp+128h] [rbp-38h]

  v14 = *(__int64 **)(a8 + 8);
  v15 = *(_QWORD *)(a1 + 960);
  v16 = *(_QWORD *)(v15 + 8) + 320LL;
  v140 = *(__int64 **)(v15 + 8);
  v17 = 0;
  v145 = v16;
  if ( v14 != (__int64 *)v16 )
    v17 = *(_QWORD *)(a8 + 8);
  if ( -858993459 * (unsigned int)((__int64)(a10 - (_QWORD)src) >> 3) != 1 )
    goto LABEL_5;
  if ( a8 != a3 )
    goto LABEL_5;
  v67 = src->m128i_i64[1];
  if ( v67 != src[1].m128i_i64[0] )
    goto LABEL_5;
  v68 = *(_QWORD *)(a10 + 8);
  if ( v68 != *(_QWORD *)(a10 + 16) || src[1].m128i_i64[1] != *(_QWORD *)(a10 + 24) )
    goto LABEL_5;
  v129 = *(_QWORD *)(v15 + 32);
  v139 = (__int64 *)(v67 + 24);
  v133 = (const void **)(v68 + 24);
  v69 = *(_DWORD *)(v68 + 32);
  v161.m128i_i32[2] = v69;
  if ( v69 <= 0x40 )
  {
    v70 = *(_QWORD *)(v68 + 24);
LABEL_81:
    v71 = *(_QWORD *)(v67 + 24) ^ v70;
    v157 = v69;
    v156 = v71;
    a5 = v71;
    goto LABEL_82;
  }
  sub_C43780((__int64)&v161, v133);
  v69 = v161.m128i_u32[2];
  if ( v161.m128i_i32[2] <= 0x40u )
  {
    v70 = v161.m128i_i64[0];
    goto LABEL_81;
  }
  sub_C43C10(&v161, v139);
  a5 = v161.m128i_i64[0];
  v157 = v161.m128i_i32[2];
  v156 = v161.m128i_i64[0];
  if ( v161.m128i_i32[2] > 0x40u )
  {
    v126 = v161.m128i_i64[0];
    v120 = sub_C44630((__int64)&v156);
    a5 = v126;
    if ( v120 == 1 )
      goto LABEL_84;
    if ( v126 )
      j_j___libc_free_0_0(v126);
    goto LABEL_5;
  }
LABEL_82:
  if ( !a5 || (a5 & (a5 - 1)) != 0 )
  {
LABEL_5:
    result = *(_QWORD *)(a1 + 856);
    if ( !*(_DWORD *)(result + 648) )
      goto LABEL_6;
    v112 = (const __m128i *)(a10 + 40);
    if ( src != (__m128i *)(a10 + 40) )
    {
      _BitScanReverse64(&v113, 0xCCCCCCCCCCCCCCCDLL * (((char *)v112 - (char *)src) >> 3));
      sub_3366E50((__int64)src, (__m128i *)(a10 + 40), 2LL * (int)(63 - (v113 ^ 0x3F)), v16, a5, a6, a7);
      if ( (char *)v112 - (char *)src <= 640 )
      {
        sub_33664C0(src, v112);
      }
      else
      {
        v114 = src + 40;
        sub_33664C0(src, src + 40);
        if ( v112 != &src[40] )
        {
          do
          {
            v115 = v114;
            v114 = (__m128i *)((char *)v114 + 40);
            sub_3365BE0(v115);
          }
          while ( v112 != v114 );
        }
      }
    }
    result = a10;
    if ( a10 > (unsigned __int64)src )
    {
      while ( 1 )
      {
        v116 = result;
        result -= 40LL;
        if ( *(_DWORD *)(result + 32) > *(_DWORD *)(a10 + 32) )
          goto LABEL_107;
        if ( !*(_DWORD *)result && *(_QWORD *)(result + 24) == v17 )
          break;
        if ( (unsigned __int64)src >= result )
          goto LABEL_107;
      }
      v161 = _mm_loadu_si128((const __m128i *)(v116 - 40));
      v117 = _mm_loadu_si128((const __m128i *)a10);
      v162 = _mm_loadu_si128((const __m128i *)(v116 - 24));
      v163 = *(_QWORD *)(v116 - 8);
      *(__m128i *)(v116 - 40) = v117;
      *(__m128i *)(v116 - 24) = _mm_loadu_si128((const __m128i *)(a10 + 16));
      *(_DWORD *)(v116 - 8) = *(_DWORD *)(a10 + 32);
      v118 = _mm_loadu_si128(&v162);
      v119 = v163;
      *(__m128i *)a10 = _mm_loadu_si128(&v161);
      *(_DWORD *)(a10 + 32) = v119;
      *(__m128i *)(a10 + 16) = v118;
LABEL_107:
      v19 = a13;
    }
    else
    {
LABEL_6:
      v19 = a13;
      if ( a10 < (unsigned __int64)src )
        return result;
    }
    v20 = src;
    v21 = v19;
    do
    {
      v22 = v21;
      v21 += v20[2].m128i_u32[0];
      if ( (unsigned __int64)v20[2].m128i_u32[0] + v22 > 0x80000000 )
        v21 = 0x80000000;
      v20 = (__m128i *)((char *)v20 + 40);
    }
    while ( a10 >= (unsigned __int64)v20 );
    v152 = v21;
    v23 = a8;
    v153 = src;
    v24 = v14;
    v124 = v19 >> 1;
    v125 = v19 >> 1;
    while ( 1 )
    {
      if ( (__m128i *)a10 == v153 )
      {
        v25 = a4;
        v65 = sub_AA5030(*(_QWORD *)(a4 + 16), 1);
        if ( !v65 )
          BUG();
        v148 = *(_BYTE *)(v65 - 24) == 36;
      }
      else
      {
        v161.m128i_i8[8] = 0;
        v25 = sub_2E7AAE0((__int64)v140, *(_QWORD *)(v23 + 16), v161.m128i_i64[0], 0);
        sub_2E33BD0(v145, v25);
        v26 = *v24;
        v27 = *(_QWORD *)v25;
        *(_QWORD *)(v25 + 8) = v24;
        v26 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)v25 = v26 | v27 & 7;
        *(_QWORD *)(v26 + 8) = v25;
        *v24 = v25 | *v24 & 7;
        sub_33C4170(a1, a2);
        v148 = 0;
      }
      v28 = v153[2].m128i_u32[0];
      v29 = 0;
      if ( v28 <= v152 )
        v29 = v152 - v28;
      v152 = v29;
      v30 = v153->m128i_i32[0];
      if ( v153->m128i_i32[0] != 1 )
      {
        if ( v30 == 2 )
        {
          v135 = *(_QWORD *)(*(_QWORD *)(a1 + 896) + 56LL) + 192LL * v153[1].m128i_u32[2];
          v31 = *(_QWORD *)(v135 + 64);
          v32 = 32LL * *(unsigned int *)(v135 + 72);
          if ( v31 + v32 != v31 )
          {
            v130 = v23;
            v33 = v24;
            v34 = *(_QWORD *)(v135 + 64);
            v127 = v25;
            v35 = v31 + v32;
            do
            {
              v36 = *(_QWORD *)(v34 + 8);
              v34 += 32;
              sub_2E33BD0(v145, v36);
              v37 = *v33;
              v38 = *(_QWORD *)v36;
              *(_QWORD *)(v36 + 8) = v33;
              v37 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)v36 = v37 | v38 & 7;
              *(_QWORD *)(v37 + 8) = v36;
              *v33 = *v33 & 7 | v36;
            }
            while ( v34 != v35 );
            v24 = v33;
            v25 = v127;
            v23 = v130;
          }
          v39 = *(_BYTE *)(v135 + 47) == 0;
          *(_QWORD *)(v135 + 48) = v23;
          *(_QWORD *)(v135 + 56) = v25;
          *(_DWORD *)(v135 + 180) = v152;
          if ( v39 )
          {
            v131 = *(_DWORD *)(v135 + 176);
            v40 = v124 + v131;
            if ( v125 + (unsigned __int64)v131 > 0x80000000 )
              v40 = 0x80000000;
            *(_DWORD *)(v135 + 176) = v40;
            v41 = v152 - v124;
            if ( v152 < v124 )
              v41 = 0;
            *(_DWORD *)(v135 + 180) = v41;
          }
          if ( v148 )
            *(_BYTE *)(v135 + 184) = 1;
          if ( a3 == v23 )
          {
            sub_3393440(a1, v135, a3);
            *(_BYTE *)(v135 + 46) = 1;
          }
          goto LABEL_13;
        }
        if ( v30 )
          goto LABEL_13;
        v54 = v153->m128i_i64[1];
        v55 = v153[1].m128i_i64[0];
        if ( v54 == v55 )
        {
          v54 = (__int64)a2;
          v57 = 17;
          v56 = 0;
        }
        else
        {
          v56 = (__int64)a2;
          v57 = 21;
        }
        if ( v148 )
          v57 = 15;
        v58 = *(_QWORD *)a1;
        v160 = *(_DWORD *)(a1 + 848);
        if ( v58 && &v159 != (__int64 *)(v58 + 48) && (v59 = *(_QWORD *)(v58 + 48), (v159 = v59) != 0) )
        {
          v128.m128i_i64[0] = v56;
          v128.m128i_i64[1] = v55;
          v137 = v54;
          sub_B96E90((__int64)&v159, v59, 1);
          v164 = v25;
          v161.m128i_i32[0] = v57;
          v28 = v153[2].m128i_u32[0];
          v165 = v23;
          v60 = v153[1].m128i_i64[1];
          v161.m128i_i64[1] = v137;
          v162 = v128;
          v163 = v60;
          v166 = v159;
          if ( v159 )
          {
            sub_B96E90((__int64)&v166, v159, 1);
            v168 = 0;
            v169 = v28;
            v167 = v160;
            v171 = 0;
            v170 = v152;
            if ( v159 )
              sub_B91220((__int64)&v159, v159);
            if ( a3 != v23 )
            {
LABEL_59:
              v61 = *(_QWORD **)(a1 + 896);
              v62 = (__m128i *)v61[2];
              if ( v62 == (__m128i *)v61[3] )
              {
                sub_3376950(v61 + 1, v61[2], (__int64)&v161);
              }
              else
              {
                if ( v62 )
                {
                  *v62 = v161;
                  v62[1] = v162;
                  v62[2].m128i_i64[0] = v163;
                  v62[2].m128i_i64[1] = v164;
                  v62[3].m128i_i64[0] = v165;
                  v63 = v166;
                  v62[3].m128i_i64[1] = v166;
                  if ( v63 )
                    sub_B96E90((__int64)&v62[3].m128i_i64[1], v63, 1);
                  v62[4].m128i_i32[0] = v167;
                  v64 = v168;
                  v62[4].m128i_i64[1] = v168;
                  if ( v64 )
                    sub_B96E90((__int64)&v62[4].m128i_i64[1], v64, 1);
                  v62[5].m128i_i32[0] = v169;
                  v62[5].m128i_i32[1] = v170;
                  v62[5].m128i_i8[8] = v171;
                  v62 = (__m128i *)v61[2];
                }
                v61[2] = v62 + 6;
              }
              goto LABEL_67;
            }
LABEL_95:
            sub_3391190(a1, (unsigned int *)&v161, a3);
LABEL_67:
            if ( v168 )
              sub_B91220((__int64)&v168, v168);
            if ( v166 )
              sub_B91220((__int64)&v166, v166);
            goto LABEL_13;
          }
        }
        else
        {
          v161.m128i_i32[0] = v57;
          v161.m128i_i64[1] = v54;
          v111 = v153[1].m128i_i64[1];
          v162.m128i_i64[0] = v56;
          v162.m128i_i64[1] = v55;
          v163 = v111;
          v164 = v25;
          v165 = v23;
          v166 = 0;
        }
        v168 = 0;
        v169 = v28;
        v167 = v160;
        v171 = 0;
        v170 = v152;
        if ( a3 != v23 )
          goto LABEL_59;
        goto LABEL_95;
      }
      v42 = *(_QWORD *)(*(_QWORD *)(a1 + 896) + 32LL) + 104LL * v153[1].m128i_u32[2];
      v43 = *(_QWORD *)(v42 + 64);
      sub_2E33BD0(v145, v43);
      v45 = *(_QWORD *)v43;
      v46 = *v24;
      *(_QWORD *)(v43 + 8) = v24;
      v46 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v43 = v46 | v45 & 7;
      *(_QWORD *)(v46 + 8) = v43;
      *v24 = v43 | *v24 & 7;
      v47 = *(_QWORD **)(v43 + 112);
      v48 = v153[2].m128i_u32[0];
      v49 = &v47[*(unsigned int *)(v43 + 120)];
      if ( v49 == v47 )
      {
LABEL_73:
        v53 = v152;
        if ( !v148 )
          goto LABEL_44;
      }
      else
      {
        while ( *v47 != a4 )
        {
          if ( ++v47 == v49 )
            goto LABEL_73;
        }
        v50 = v125 + (unsigned int)v48;
        v51 = v124 + v48;
        if ( v50 > 0x80000000 )
          v51 = 0x80000000;
        v52 = 0;
        v132 = v51;
        if ( v152 >= v124 )
          v52 = v152 - v124;
        v136 = v52;
        sub_2E32F90(v43, (__int64)v47, v124);
        sub_2E33470(*(unsigned int **)(v43 + 144), *(unsigned int **)(v43 + 152));
        v53 = v136;
        v48 = v132;
        if ( !v148 )
        {
LABEL_44:
          if ( !*(_BYTE *)(v42 + 49) )
          {
            v149 = v48;
            sub_3373E10(a1, v23, v25, v53, v44, v48);
            v48 = v149;
          }
          goto LABEL_46;
        }
      }
      v138 = v48;
      v150 = v53;
      v66 = sub_B2D620(*v140, "branch-target-enforcement", 0x19u);
      v53 = v150;
      v48 = v138;
      if ( v66 )
        goto LABEL_44;
      *(_BYTE *)(v42 + 49) = 1;
LABEL_46:
      sub_3373E10(a1, v23, v43, (unsigned int)v48, v44, v48);
      sub_2E33470(*(unsigned int **)(v23 + 144), *(unsigned int **)(v23 + 152));
      *(_QWORD *)(v42 + 40) = v23;
      *(_QWORD *)(v42 + 72) = v25;
      if ( a3 == v23 )
      {
        sub_3391DE0(a1, v42 + 56, v42, a3);
        *(_BYTE *)(v42 + 48) = 1;
      }
LABEL_13:
      v153 = (__m128i *)((char *)v153 + 40);
      result = (unsigned __int64)v153;
      if ( a10 < (unsigned __int64)v153 )
        return result;
      v23 = v25;
    }
  }
LABEL_84:
  HIWORD(v74) = 0;
  *(_QWORD *)&v143 = sub_338B750(a1, (__int64)a2);
  v73 = *(_QWORD *)(v143 + 48) + 16LL * (unsigned int)v72;
  *((_QWORD *)&v143 + 1) = v72;
  LOWORD(v74) = *(_WORD *)v73;
  v154 = *(_QWORD *)(v73 + 8);
  sub_336E8F0((__int64)v158, *(_QWORD *)a1, *(_DWORD *)(a1 + 848));
  v146 = *(_QWORD *)(a1 + 864);
  *(_QWORD *)&v75 = sub_34007B0(v146, (unsigned int)&v156, (unsigned int)v158, (unsigned __int16)v74, v154, 0, 0);
  *(_QWORD *)&v144 = sub_3406EB0(v146, 187, (unsigned int)v158, (unsigned __int16)v74, v154, v76, v143, v75);
  *((_QWORD *)&v144 + 1) = v77;
  v147 = *(_QWORD *)(a1 + 864);
  v160 = *(_DWORD *)(v68 + 32);
  if ( v160 > 0x40 )
  {
    sub_C43780((__int64)&v159, v133);
    if ( v160 > 0x40 )
    {
      sub_C43BD0(&v159, v139);
      goto LABEL_87;
    }
  }
  else
  {
    v159 = *(_QWORD *)(v68 + 24);
  }
  v159 |= *(_QWORD *)(v67 + 24);
LABEL_87:
  v161.m128i_i32[2] = v160;
  v161.m128i_i64[0] = v159;
  v160 = 0;
  v78 = sub_34007B0(v147, (unsigned int)&v161, (unsigned int)v158, v74, v154, 0, 0);
  v80 = v79;
  *(_QWORD *)&v81 = sub_33ED040(v147, 17);
  *((_QWORD *)&v121 + 1) = v80;
  *(_QWORD *)&v121 = v78;
  v83 = sub_340F900(v147, 208, (unsigned int)v158, 2, 0, v82, v144, v121, v81);
  v85 = v84;
  sub_969240(v161.m128i_i64);
  sub_969240(&v159);
  v88 = src[2].m128i_u32[0];
  v89 = *(unsigned int *)(a10 + 32);
  v90 = 0x80000000LL;
  if ( (unsigned __int64)(v88 + v89) <= 0x80000000 )
    v90 = (unsigned int)(v88 + v89);
  sub_3373E10(a1, a3, src[1].m128i_i64[1], v90, v86, v87);
  if ( v129 )
  {
    v93 = sub_FF0300(v129, *(_QWORD *)(a3 + 16), 0);
    sub_3373E10(a1, a3, a4, v93, v94, v95);
  }
  else
  {
    sub_3373E10(a1, a3, a4, 0xFFFFFFFFLL, v91, v92);
  }
  v96 = src[1].m128i_i64[1];
  v97 = *(_QWORD *)(a1 + 864);
  *(_QWORD *)&v155 = sub_33EEAD0(v97, v96);
  *((_QWORD *)&v155 + 1) = v98;
  *(_QWORD *)&v102 = sub_3373A60(a1, v96, v98, v99, v100, v101);
  *((_QWORD *)&v122 + 1) = v85;
  *(_QWORD *)&v122 = v83;
  v103 = sub_340F900(v97, 305, (unsigned int)v158, 1, 0, DWORD2(v155), v102, v122, v155);
  v105 = v104;
  v106 = *(_QWORD *)(a1 + 864);
  *(_QWORD *)&v107 = sub_33EEAD0(v106, a4);
  *((_QWORD *)&v123 + 1) = v105;
  *(_QWORD *)&v123 = v103;
  v109 = sub_3406EB0(v106, 301, (unsigned int)v158, 1, 0, v108, v123, v107);
  sub_3365B50(*(_QWORD *)(a1 + 864), v109, v110);
  sub_9C6650(v158);
  return sub_969240(&v156);
}
