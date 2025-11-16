// Function: sub_2DFD970
// Address: 0x2dfd970
//
__int64 __fastcall sub_2DFD970(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // r12
  unsigned int v5; // r15d
  __int16 v7; // dx
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // r14
  __int64 v11; // r13
  __int64 v12; // rcx
  int v13; // r12d
  int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r15
  __int64 *v19; // rax
  __int64 *v20; // rbx
  __int64 v21; // r15
  bool v22; // al
  const __m128i *v23; // r14
  __int64 v24; // r13
  __int64 v25; // rax
  unsigned __int8 v26; // dl
  __int64 v27; // rdx
  __m128i v28; // xmm0
  __int64 v29; // rdi
  bool v30; // zf
  __m128i *v31; // rax
  int v32; // ecx
  unsigned int v33; // esi
  int v34; // edx
  __int64 v35; // rdx
  __int64 v36; // r11
  __int64 v37; // r9
  __int16 v38; // si
  const __m128i *v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // rax
  int v42; // edx
  __int64 v43; // r8
  __int64 v44; // r14
  int v45; // r12d
  int v46; // edi
  const __m128i *v47; // rdx
  __m128i v48; // xmm4
  __int64 v49; // rsi
  __int64 v50; // r9
  __m128i v51; // xmm0
  __int64 v52; // rax
  unsigned __int8 *v53; // rsi
  _QWORD *v54; // rax
  __int64 v55; // rdx
  int v56; // eax
  __int64 *v57; // rcx
  __int64 *v58; // rdx
  __int64 v59; // rax
  __int64 v60; // rcx
  unsigned __int64 v61; // r8
  unsigned __int64 v62; // r9
  unsigned __int64 v63; // rdi
  unsigned __int64 *v64; // rbx
  unsigned __int64 *v65; // r12
  unsigned __int64 v66; // r14
  unsigned __int64 v67; // rdi
  unsigned __int64 v68; // r9
  __int64 v69; // rsi
  __int8 *v70; // rdx
  __int64 v71; // rax
  __m128i *v72; // rdx
  __int8 *v73; // rsi
  __int64 v74; // rax
  __int64 *v75; // rax
  __int64 *v76; // rdx
  __int64 v77; // rcx
  unsigned __int64 v78; // rdi
  __int64 *v79; // rsi
  unsigned __int64 v80; // r14
  unsigned __int64 v81; // r13
  unsigned __int64 v82; // r12
  unsigned __int64 v83; // rbx
  unsigned __int64 v84; // rdi
  unsigned __int64 v85; // rdi
  unsigned __int64 v86; // r8
  _QWORD *v87; // rax
  _QWORD *v88; // rbx
  _QWORD *v89; // r15
  unsigned __int64 v90; // rdi
  unsigned __int64 v91; // rdi
  __int64 v92; // rsi
  __int32 v93; // r14d
  void *v94; // r14
  _QWORD *v95; // rdx
  _QWORD *v96; // rax
  _QWORD *v97; // rax
  _QWORD *v98; // rdx
  __int64 v99; // [rsp+8h] [rbp-1E8h]
  __int64 v100; // [rsp+10h] [rbp-1E0h]
  unsigned __int8 v101; // [rsp+1Fh] [rbp-1D1h]
  __int64 v102; // [rsp+20h] [rbp-1D0h]
  __int64 v103; // [rsp+28h] [rbp-1C8h]
  __int64 v104; // [rsp+28h] [rbp-1C8h]
  __int64 v105; // [rsp+38h] [rbp-1B8h]
  __int64 v106; // [rsp+38h] [rbp-1B8h]
  __int64 v107; // [rsp+40h] [rbp-1B0h]
  __int64 v108; // [rsp+48h] [rbp-1A8h]
  char v109; // [rsp+48h] [rbp-1A8h]
  __int64 v110; // [rsp+50h] [rbp-1A0h]
  __int64 v111; // [rsp+50h] [rbp-1A0h]
  __int64 *v112; // [rsp+50h] [rbp-1A0h]
  unsigned __int64 v113; // [rsp+50h] [rbp-1A0h]
  __int64 v114; // [rsp+50h] [rbp-1A0h]
  unsigned __int64 v115; // [rsp+50h] [rbp-1A0h]
  unsigned __int64 v116; // [rsp+50h] [rbp-1A0h]
  unsigned __int64 v117; // [rsp+60h] [rbp-190h]
  bool v118; // [rsp+60h] [rbp-190h]
  __int64 v119; // [rsp+68h] [rbp-188h]
  bool v120; // [rsp+68h] [rbp-188h]
  __int64 *v121; // [rsp+70h] [rbp-180h]
  __int64 v122; // [rsp+70h] [rbp-180h]
  __int64 v124; // [rsp+80h] [rbp-170h] BYREF
  unsigned __int8 *v125; // [rsp+88h] [rbp-168h] BYREF
  __m128i v126; // [rsp+90h] [rbp-160h] BYREF
  __int64 v127; // [rsp+A0h] [rbp-150h]
  __m128i v128; // [rsp+B0h] [rbp-140h] BYREF
  __int64 v129; // [rsp+C0h] [rbp-130h]
  __m128i v130; // [rsp+D0h] [rbp-120h] BYREF
  __int64 v131; // [rsp+E0h] [rbp-110h]
  __m128i v132; // [rsp+F0h] [rbp-100h] BYREF
  __int64 v133; // [rsp+100h] [rbp-F0h]
  const __m128i *v134; // [rsp+110h] [rbp-E0h] BYREF
  __m128i v135; // [rsp+118h] [rbp-D8h] BYREF
  __int64 v136; // [rsp+128h] [rbp-C8h]
  __int64 v137; // [rsp+130h] [rbp-C0h]

  if ( (unsigned __int16)(*(_WORD *)(a2 + 68) - 14) > 1u )
    return 0;
  v3 = a1;
  v4 = a2;
  if ( *(_BYTE *)sub_2E89150(a2) != 14 )
    return 0;
  v7 = *(_WORD *)(a2 + 68);
  v8 = *(_DWORD *)(a2 + 40) & 0xFFFFFF;
  if ( v7 == 14 )
  {
    if ( (_DWORD)v8 != 4 )
      return 0;
    v11 = *(_QWORD *)(a2 + 32);
    v10 = v11 + 40;
    if ( *(_BYTE *)(v11 + 40) > 1u )
      return 0;
  }
  else
  {
    v9 = *(_QWORD *)(a2 + 32);
    v10 = v9 + 40 * v8;
    v11 = v9 + 80;
    if ( v10 == v11 )
    {
      v118 = 0;
      v5 = 0;
      goto LABEL_22;
    }
  }
  v12 = a1;
  v13 = 0;
  do
  {
    while ( 1 )
    {
      if ( !*(_BYTE *)v11 )
      {
        v14 = *(_DWORD *)(v11 + 8);
        if ( v14 < 0 )
        {
          v15 = *(_QWORD *)(v12 + 112);
          v16 = v14 & 0x7FFFFFFF;
          if ( (unsigned int)v16 >= *(_DWORD *)(v15 + 160) )
            break;
          v17 = *(_QWORD *)(v15 + 152);
          v18 = *(_QWORD *)(v17 + 8 * v16);
          if ( !v18 )
            break;
          v119 = v12;
          v117 = a3 & 0xFFFFFFFFFFFFFFF8LL;
          v19 = (__int64 *)sub_2E09D00(*(_QWORD *)(v17 + 8 * v16), a3 & 0xFFFFFFFFFFFFFFF8LL);
          v12 = v119;
          v20 = v19;
          v21 = *(_QWORD *)v18 + 24LL * *(unsigned int *)(v18 + 8);
          if ( v19 == (__int64 *)v21 )
            break;
          v22 = sub_2DF8330(v19, v117);
          v12 = v119;
          if ( v22 && v117 == (v20[1] & 0xFFFFFFFFFFFFFFF8LL) )
          {
            v20 += 3;
            if ( (__int64 *)v21 == v20 )
              break;
          }
          if ( *(_DWORD *)(v117 + 24) < *(_DWORD *)((*v20 & 0xFFFFFFFFFFFFFFF8LL) + 24) || !v20[2] )
            break;
        }
      }
      v11 += 40;
      if ( v10 == v11 )
        goto LABEL_20;
    }
    v11 += 40;
    v13 = 1;
  }
  while ( v10 != v11 );
LABEL_20:
  v5 = v13;
  v4 = a2;
  v118 = 0;
  v3 = v12;
  v7 = *(_WORD *)(a2 + 68);
  if ( v7 == 14 )
    v118 = *(_BYTE *)(*(_QWORD *)(a2 + 32) + 40LL) == 1;
LABEL_22:
  v120 = v7 == 15;
  v23 = (const __m128i *)sub_2E89170(v4);
  v24 = sub_2E891C0(v4);
  sub_AF47B0((__int64)&v126, *(unsigned __int64 **)(v24 + 16), *(unsigned __int64 **)(v24 + 24));
  v128 = _mm_loadu_si128(&v126);
  v129 = v127;
  v25 = sub_B10CD0(v4 + 56);
  v26 = *(_BYTE *)(v25 - 16);
  if ( (v26 & 2) != 0 )
  {
    if ( *(_DWORD *)(v25 - 24) != 2 )
    {
LABEL_24:
      v27 = 0;
      goto LABEL_25;
    }
    v41 = *(_QWORD *)(v25 - 32);
  }
  else
  {
    if ( ((*(_WORD *)(v25 - 16) >> 6) & 0xF) != 2 )
      goto LABEL_24;
    v41 = v25 - 16 - 8LL * ((v26 >> 2) & 0xF);
  }
  v27 = *(_QWORD *)(v41 + 8);
LABEL_25:
  v28 = _mm_loadu_si128(&v128);
  v137 = v27;
  v133 = v129;
  v136 = v129;
  v134 = v23;
  v132 = v28;
  v135 = v28;
  v29 = v3 + 1144;
  v30 = (unsigned __int8)sub_2DF9F10(v3 + 1144, (__int64)&v134, &v130) == 0;
  v31 = (__m128i *)v130.m128i_i64[0];
  if ( !v30 )
    goto LABEL_31;
  v32 = *(_DWORD *)(v3 + 1160);
  v33 = *(_DWORD *)(v3 + 1168);
  v132.m128i_i64[0] = v130.m128i_i64[0];
  ++*(_QWORD *)(v3 + 1144);
  v34 = v32 + 1;
  if ( 4 * (v32 + 1) >= 3 * v33 )
  {
    v33 *= 2;
  }
  else if ( v33 - *(_DWORD *)(v3 + 1164) - v34 > v33 >> 3 )
  {
    goto LABEL_28;
  }
  sub_2DFA0F0(v29, v33);
  sub_2DF9F10(v29, (__int64)&v134, &v132);
  v34 = *(_DWORD *)(v3 + 1160) + 1;
  v31 = (__m128i *)v132.m128i_i64[0];
LABEL_28:
  *(_DWORD *)(v3 + 1160) = v34;
  if ( v31->m128i_i64[0] || v31[1].m128i_i8[8] || v31[2].m128i_i64[0] )
    --*(_DWORD *)(v3 + 1164);
  *v31 = _mm_loadu_si128((const __m128i *)&v134);
  v31[1] = _mm_loadu_si128((const __m128i *)&v135.m128i_u64[1]);
  v35 = v137;
  v31[2].m128i_i64[1] = 0;
  v31[2].m128i_i64[0] = v35;
LABEL_31:
  v121 = &v31[2].m128i_i64[1];
  v36 = v31[2].m128i_i64[1];
  if ( !v36 )
  {
    v48 = _mm_loadu_si128(&v128);
    v49 = *(_QWORD *)(v4 + 56);
    v131 = v129;
    v125 = (unsigned __int8 *)v49;
    v130 = v48;
    if ( v49 )
      sub_B96E90((__int64)&v125, v49, 1);
    v50 = sub_22077B0(0x1E8u);
    if ( v50 )
    {
      v51 = _mm_loadu_si128(&v130);
      v52 = v131;
      *(_QWORD *)v50 = v23;
      v53 = v125;
      v133 = v52;
      *(_QWORD *)(v50 + 24) = v52;
      *(_QWORD *)(v50 + 32) = v53;
      v132 = v51;
      *(__m128i *)(v50 + 8) = v51;
      if ( v53 )
      {
        v110 = v50;
        sub_B976B0((__int64)&v125, v53, v50 + 32);
        v50 = v110;
        v125 = 0;
      }
      *(_QWORD *)(v50 + 40) = v50;
      *(_QWORD *)(v50 + 56) = v50 + 72;
      *(_QWORD *)(v50 + 64) = 0x400000000LL;
      *(_QWORD *)(v50 + 48) = 0;
      *(_QWORD *)(v50 + 400) = v3;
      memset((void *)(v50 + 232), 0, 0xA0u);
      v54 = (_QWORD *)(v50 + 232);
      *(_QWORD *)(v50 + 392) = 0;
      do
      {
        *v54 = 0;
        v54 += 2;
        *(v54 - 1) = 0;
      }
      while ( v54 != (_QWORD *)(v50 + 296) );
      do
      {
        *v54 = 0;
        v54 += 3;
        *((_BYTE *)v54 - 16) = 0;
        *(v54 - 1) = 0;
      }
      while ( (_QWORD *)(v50 + 392) != v54 );
      *(_DWORD *)(v50 + 448) = 0;
      *(_QWORD *)(v50 + 408) = v50 + 424;
      *(_QWORD *)(v50 + 416) = 0x200000000LL;
      *(_QWORD *)(v50 + 456) = 0;
      *(_QWORD *)(v50 + 464) = v50 + 448;
      *(_QWORD *)(v50 + 472) = v50 + 448;
      *(_QWORD *)(v50 + 480) = 0;
    }
    v124 = v50;
    if ( v125 )
    {
      v111 = v50;
      sub_B91220((__int64)&v125, (__int64)v125);
      v50 = v111;
    }
    v55 = *(unsigned int *)(v3 + 1008);
    v112 = *(__int64 **)(v3 + 1000);
    v56 = *(_DWORD *)(v3 + 1008);
    v57 = &v124;
    if ( v55 + 1 > (unsigned __int64)*(unsigned int *)(v3 + 1012) )
    {
      if ( v112 > &v124 || &v124 >= &v112[v55] )
      {
        v107 = -1;
        v109 = 0;
      }
      else
      {
        v109 = 1;
        v107 = &v124 - v112;
      }
      v103 = v50;
      v105 = v3 + 1016;
      v75 = (__int64 *)sub_C8D7D0(v3 + 1000, v3 + 1016, v55 + 1, 8u, (unsigned __int64 *)&v132, v50);
      v76 = *(__int64 **)(v3 + 1000);
      v112 = v75;
      v50 = v103;
      v77 = 8LL * *(unsigned int *)(v3 + 1008);
      v78 = (unsigned __int64)v76;
      v79 = (__int64 *)((char *)v75 + v77);
      if ( v77 )
      {
        do
        {
          if ( v75 )
          {
            v77 = *v76;
            *v75 = *v76;
            *v76 = 0;
          }
          ++v75;
          ++v76;
        }
        while ( v79 != v75 );
        v80 = *(_QWORD *)(v3 + 1000);
        v78 = v80;
        if ( v80 != v80 + 8LL * *(unsigned int *)(v3 + 1008) )
        {
          v102 = v103;
          v101 = v5;
          v100 = v3;
          v99 = v4;
          v104 = v24;
          v81 = v80 + 8LL * *(unsigned int *)(v3 + 1008);
          do
          {
            v82 = *(_QWORD *)(v81 - 8);
            v81 -= 8LL;
            if ( v82 )
            {
              v83 = *(_QWORD *)(v82 + 456);
              while ( v83 )
              {
                sub_2DF5850(*(_QWORD *)(v83 + 24));
                v84 = v83;
                v83 = *(_QWORD *)(v83 + 16);
                j_j___libc_free_0(v84);
              }
              v85 = *(_QWORD *)(v82 + 408);
              if ( v85 != v82 + 424 )
                _libc_free(v85);
              v86 = *(unsigned int *)(v82 + 392);
              if ( (_DWORD)v86 )
              {
                sub_2DF5350(v82 + 232, (__int64)sub_2DF57F0, 0, v77, v86, v50);
                *(_DWORD *)(v82 + 392) = 0;
                memset((void *)(v82 + 232), 0, 0xA0u);
                v97 = (_QWORD *)(v82 + 232);
                v88 = (_QWORD *)(v82 + 296);
                do
                {
                  *v97 = 0;
                  v97 += 2;
                  *(v97 - 1) = 0;
                }
                while ( v88 != v97 );
                v98 = (_QWORD *)(v82 + 296);
                v87 = (_QWORD *)(v82 + 392);
                do
                {
                  *v98 = 0;
                  v98 += 3;
                  *((_BYTE *)v98 - 16) = 0;
                  *(v98 - 1) = 0;
                }
                while ( v87 != v98 );
              }
              else
              {
                v87 = (_QWORD *)(v82 + 392);
                v88 = (_QWORD *)(v82 + 296);
              }
              *(_DWORD *)(v82 + 396) = 0;
              v89 = v87;
              do
              {
                v90 = *(v89 - 3);
                v89 -= 3;
                if ( v90 )
                  j_j___libc_free_0_0(v90);
              }
              while ( v89 != v88 );
              v91 = *(_QWORD *)(v82 + 56);
              if ( v91 != v82 + 72 )
                _libc_free(v91);
              v92 = *(_QWORD *)(v82 + 32);
              if ( v92 )
                sub_B91220(v82 + 32, v92);
              j_j___libc_free_0(v82);
            }
          }
          while ( v80 != v81 );
          v3 = v100;
          v24 = v104;
          v50 = v102;
          v5 = v101;
          v4 = v99;
          v78 = *(_QWORD *)(v100 + 1000);
        }
      }
      v93 = v132.m128i_i32[0];
      if ( v105 != v78 )
      {
        v106 = v50;
        _libc_free(v78);
        v50 = v106;
      }
      v55 = *(unsigned int *)(v3 + 1008);
      *(_DWORD *)(v3 + 1012) = v93;
      *(_QWORD *)(v3 + 1000) = v112;
      v56 = v55;
      v57 = &v112[v107];
      if ( !v109 )
        v57 = &v124;
    }
    v58 = &v112[v55];
    if ( v58 )
    {
      *v58 = *v57;
      *v57 = 0;
      v50 = v124;
      v56 = *(_DWORD *)(v3 + 1008);
    }
    v59 = (unsigned int)(v56 + 1);
    *(_DWORD *)(v3 + 1008) = v59;
    if ( v50 )
    {
      v113 = v50;
      sub_2DF5850(*(_QWORD *)(v50 + 456));
      v62 = v113;
      v63 = *(_QWORD *)(v113 + 408);
      if ( v63 != v113 + 424 )
      {
        _libc_free(v63);
        v62 = v113;
      }
      if ( *(_DWORD *)(v62 + 392) )
      {
        v94 = (void *)(v62 + 232);
        v116 = v62;
        sub_2DF5350(v62 + 232, (__int64)sub_2DF57F0, 0, v60, v61, v62);
        v62 = v116;
        *(_DWORD *)(v116 + 392) = 0;
        v95 = (_QWORD *)(v116 + 296);
        memset(v94, 0, 0xA0u);
        v96 = v94;
        do
        {
          *v96 = 0;
          v96 += 2;
          *(v96 - 1) = 0;
        }
        while ( v95 != v96 );
        do
        {
          *v95 = 0;
          v95 += 3;
          *((_BYTE *)v95 - 16) = 0;
          *(v95 - 1) = 0;
        }
        while ( (_QWORD *)(v116 + 392) != v95 );
      }
      *(_DWORD *)(v62 + 396) = 0;
      v114 = v3;
      v64 = (unsigned __int64 *)(v62 + 272);
      v108 = v4;
      v65 = (unsigned __int64 *)(v62 + 368);
      v66 = v62;
      do
      {
        if ( *v65 )
          j_j___libc_free_0_0(*v65);
        v65 -= 3;
      }
      while ( v64 != v65 );
      v67 = *(_QWORD *)(v66 + 56);
      v68 = v66;
      v3 = v114;
      v4 = v108;
      if ( v67 != v66 + 72 )
      {
        _libc_free(v67);
        v68 = v66;
      }
      v69 = *(_QWORD *)(v68 + 32);
      if ( v69 )
      {
        v115 = v68;
        sub_B91220(v68 + 32, v69);
        v68 = v115;
      }
      j_j___libc_free_0(v68);
      v59 = *(unsigned int *)(v3 + 1008);
    }
    v36 = *(_QWORD *)(*(_QWORD *)(v3 + 1000) + 8 * v59 - 8);
    *v121 = v36;
  }
  v37 = v120;
  v38 = *(_WORD *)(v4 + 68);
  v39 = *(const __m128i **)(v4 + 32);
  if ( !(_BYTE)v5 )
  {
    v40 = 1;
    if ( v38 != 14 )
    {
      v39 += 5;
      v40 = 0xCCCCCCCCCCCCCCCDLL * ((40LL * (*(_DWORD *)(v4 + 40) & 0xFFFFFF) - 80) >> 3);
    }
    v5 = 1;
    sub_2DFD480(v36, a3, v39, v40, v118, v120, v24);
    return v5;
  }
  if ( v38 == 14 )
  {
    v46 = 1;
    v135.m128i_i32[1] = 4;
    v44 = 1;
    v134 = (const __m128i *)&v135.m128i_u64[1];
LABEL_78:
    v70 = &v135.m128i_i8[8];
    do
    {
      v71 = *(_QWORD *)v70;
      *((_DWORD *)v70 + 2) = 0;
      v70 += 40;
      *((_QWORD *)v70 - 3) = 0;
      *((_QWORD *)v70 - 2) = 0;
      *((_QWORD *)v70 - 1) = 0;
      *((_QWORD *)v70 - 5) = v71 & 0xFFFFFFF000000000LL | 0x800000000LL;
    }
    while ( v70 != (__int8 *)&v135.m128i_u64[5 * v44 + 1] );
    v47 = v134;
    goto LABEL_44;
  }
  v42 = *(_DWORD *)(v4 + 40);
  v134 = (const __m128i *)&v135.m128i_u64[1];
  v135.m128i_i64[0] = 0x400000000LL;
  v43 = 0xCCCCCCCCCCCCCCCDLL * ((40LL * (v42 & 0xFFFFFF) - 80) >> 3);
  v44 = (unsigned int)v43;
  v45 = -858993459 * ((40LL * (v42 & 0xFFFFFF) - 80) >> 3);
  v46 = v45;
  if ( (unsigned int)v43 > 4uLL )
  {
    v122 = v36;
    sub_C8D5F0((__int64)&v134, &v135.m128i_u64[1], (unsigned int)v43, 0x28u, v43, v120);
    v72 = (__m128i *)v134;
    v36 = v122;
    v37 = v120;
    v73 = &v134->m128i_i8[40 * v44];
    do
    {
      if ( v72 )
      {
        v72->m128i_i8[0] = 0;
        v74 = v72->m128i_i64[0];
        v72->m128i_i32[2] = 0;
        v72[1].m128i_i64[0] = 0;
        v72[1].m128i_i64[1] = 0;
        v72->m128i_i64[0] = v74 & 0xFFFFFFF0000000FFLL | 0x800000000LL;
        v72[2].m128i_i64[0] = 0;
      }
      v72 = (__m128i *)((char *)v72 + 40);
    }
    while ( v72 != (__m128i *)v73 );
    v135.m128i_i32[0] = v45;
    v47 = v134;
    goto LABEL_45;
  }
  v47 = (const __m128i *)&v135.m128i_u64[1];
  if ( (_DWORD)v43 )
    goto LABEL_78;
LABEL_44:
  v135.m128i_i32[0] = v46;
LABEL_45:
  sub_2DFD480(v36, a3, v47, v44, 0, v37, v24);
  if ( v134 != (const __m128i *)&v135.m128i_u64[1] )
    _libc_free((unsigned __int64)v134);
  return v5;
}
