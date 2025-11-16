// Function: sub_256C1D0
// Address: 0x256c1d0
//
__int64 __fastcall sub_256C1D0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned int a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v9; // r14
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // r9
  int v16; // r15d
  __int64 v17; // rsi
  __int64 v18; // rcx
  __int64 *v19; // rdx
  __int64 v20; // rdi
  __int64 v21; // rdx
  unsigned int *v22; // rax
  __int64 v23; // r9
  unsigned int v24; // esi
  __int64 v25; // rcx
  const __m128i *v26; // r14
  __m128i v27; // xmm0
  __int64 v28; // r9
  __m128i v29; // xmm4
  __int64 v30; // rcx
  __int64 v31; // rdx
  _QWORD *v32; // rax
  __int64 v33; // rax
  __int64 *v34; // r13
  __m128i *v35; // rsi
  __int64 *v36; // r12
  __m128i *v37; // rax
  _QWORD *v38; // rdx
  __int32 v39; // eax
  __int64 v40; // rdx
  int v41; // eax
  const __m128i *v42; // rdi
  const __m128i *v43; // rdx
  __int64 v44; // rcx
  char v45; // al
  unsigned __int64 *v46; // r12
  unsigned __int64 *v47; // r13
  unsigned __int64 *v48; // r13
  unsigned __int64 *v49; // r15
  __int64 v50; // r14
  __int64 v51; // rax
  int v52; // ecx
  int v53; // esi
  __m128i v54; // xmm0
  __int64 v55; // rdx
  unsigned int v56; // r15d
  __int64 v57; // r14
  __m128i v58; // xmm5
  __int64 v59; // rax
  unsigned int v60; // r12d
  __int64 v61; // rax
  bool v63; // cc
  int v64; // ecx
  unsigned int v65; // esi
  unsigned __int64 *v66; // rax
  int v67; // edx
  __m128i v68; // xmm3
  _DWORD *v69; // rdi
  __int64 v70; // rdx
  int v71; // ecx
  __int64 v72; // rdi
  __int64 v73; // rcx
  __int64 v74; // rdx
  __int64 v75; // rdx
  unsigned __int64 *v76; // rbx
  int *v77; // rdi
  int *v78; // rax
  const __m128i *v79; // rax
  const __m128i *v80; // r9
  const __m128i *v81; // r8
  __int8 v82; // al
  __int64 v83; // r15
  __int64 v84; // rax
  __int64 v85; // rcx
  __m128i v86; // xmm0
  __int64 v87; // r14
  __int64 v88; // rax
  __int64 v89; // rdx
  __m128i *v90; // rax
  __m128i v91; // xmm7
  const __m128i *v92; // rax
  unsigned __int64 v93; // r13
  __m128i *v94; // r12
  const __m128i *v95; // rbx
  const __m128i *v96; // r13
  __m128i v97; // xmm6
  int v98; // esi
  __int64 v99; // r12
  __int64 v100; // rbx
  unsigned __int64 v101; // rdi
  __int32 v102; // r12d
  const __m128i *v103; // [rsp+8h] [rbp-228h]
  const __m128i *v104; // [rsp+30h] [rbp-200h]
  __int64 v105; // [rsp+38h] [rbp-1F8h]
  __int64 v106; // [rsp+40h] [rbp-1F0h]
  __int64 v107; // [rsp+40h] [rbp-1F0h]
  __int64 v108; // [rsp+40h] [rbp-1F0h]
  const __m128i *v109; // [rsp+58h] [rbp-1D8h]
  __int64 v110; // [rsp+58h] [rbp-1D8h]
  __int64 v111; // [rsp+58h] [rbp-1D8h]
  __int64 v112; // [rsp+58h] [rbp-1D8h]
  __int64 v113; // [rsp+58h] [rbp-1D8h]
  __int64 v114; // [rsp+58h] [rbp-1D8h]
  __int64 v115; // [rsp+58h] [rbp-1D8h]
  __m128i v116; // [rsp+60h] [rbp-1D0h] BYREF
  unsigned int v117; // [rsp+7Ch] [rbp-1B4h] BYREF
  unsigned __int64 *v118; // [rsp+80h] [rbp-1B0h] BYREF
  unsigned __int64 *v119; // [rsp+88h] [rbp-1A8h] BYREF
  __int64 v120; // [rsp+90h] [rbp-1A0h] BYREF
  unsigned int *v121; // [rsp+98h] [rbp-198h]
  const __m128i *v122; // [rsp+A0h] [rbp-190h] BYREF
  __int64 v123; // [rsp+A8h] [rbp-188h]
  _BYTE v124[48]; // [rsp+B0h] [rbp-180h] BYREF
  __m128i v125; // [rsp+E0h] [rbp-150h] BYREF
  _BYTE v126[48]; // [rsp+F0h] [rbp-140h] BYREF
  __m128i v127; // [rsp+120h] [rbp-110h] BYREF
  __m128i v128; // [rsp+130h] [rbp-100h] BYREF
  __int64 *v129; // [rsp+140h] [rbp-F0h] BYREF
  __int64 v130; // [rsp+148h] [rbp-E8h]
  _BYTE v131[48]; // [rsp+150h] [rbp-E0h] BYREF
  unsigned int v132; // [rsp+180h] [rbp-B0h]
  __int64 v133; // [rsp+188h] [rbp-A8h]
  __m128i v134; // [rsp+190h] [rbp-A0h] BYREF
  __m128i v135; // [rsp+1A0h] [rbp-90h]
  const __m128i *v136; // [rsp+1B0h] [rbp-80h] BYREF
  __int64 v137; // [rsp+1B8h] [rbp-78h]
  _BYTE v138[48]; // [rsp+1C0h] [rbp-70h] BYREF
  __int32 v139; // [rsp+1F0h] [rbp-40h]
  __int64 v140; // [rsp+1F8h] [rbp-38h]

  v9 = a1 + 168;
  v12 = a1;
  v13 = a9;
  v116.m128i_i64[0] = a5;
  v14 = *(unsigned int *)(a1 + 192);
  v116.m128i_i64[1] = a6;
  if ( !a9 )
    v13 = a4;
  a9 = v13;
  if ( !(_DWORD)v14 )
  {
    ++*(_QWORD *)(a1 + 168);
    v134.m128i_i64[0] = 0;
    goto LABEL_141;
  }
  v15 = *(_QWORD *)(a1 + 176);
  v16 = 1;
  LODWORD(v17) = (v14 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
  v18 = v15 + 72LL * (unsigned int)v17;
  v19 = 0;
  v20 = *(_QWORD *)v18;
  if ( v13 != *(_QWORD *)v18 )
  {
    while ( v20 != -4096 )
    {
      if ( v20 == -8192 && !v19 )
        v19 = (__int64 *)v18;
      v17 = ((_DWORD)v14 - 1) & (unsigned int)(v17 + v16);
      v18 = v15 + 72 * v17;
      v20 = *(_QWORD *)v18;
      if ( v13 == *(_QWORD *)v18 )
        goto LABEL_5;
      ++v16;
    }
    if ( !v19 )
      v19 = (__int64 *)v18;
    v52 = *(_DWORD *)(v12 + 184);
    ++*(_QWORD *)(v12 + 168);
    v53 = v52 + 1;
    v134.m128i_i64[0] = (__int64)v19;
    if ( 4 * (v52 + 1) < (unsigned int)(3 * v14) )
    {
      v25 = (unsigned int)(v14 - *(_DWORD *)(v12 + 188) - v53);
      if ( (unsigned int)v25 > (unsigned int)v14 >> 3 )
      {
LABEL_54:
        *(_DWORD *)(v12 + 184) = v53;
        if ( *v19 != -4096 )
          --*(_DWORD *)(v12 + 188);
        *v19 = v13;
        v23 = (__int64)(v19 + 1);
        v19[1] = (__int64)(v19 + 3);
        v19[2] = 0xC00000000LL;
        v24 = *(_DWORD *)(v12 + 16);
        v117 = v24;
        goto LABEL_57;
      }
      v98 = v14;
LABEL_142:
      sub_2569A70(v9, v98);
      sub_255DA40(v9, &a9, &v134);
      v25 = *(unsigned int *)(v12 + 184);
      v13 = a9;
      v19 = (__int64 *)v134.m128i_i64[0];
      v53 = v25 + 1;
      goto LABEL_54;
    }
LABEL_141:
    v98 = 2 * v14;
    goto LABEL_142;
  }
LABEL_5:
  v21 = *(unsigned int *)(v18 + 16);
  v22 = *(unsigned int **)(v18 + 8);
  v23 = v18 + 8;
  v24 = *(_DWORD *)(v12 + 16);
  v25 = (__int64)&v22[v21];
  v117 = v24;
  if ( v22 != (unsigned int *)v25 )
  {
    while ( 1 )
    {
      v14 = *v22;
      v26 = (const __m128i *)(*(_QWORD *)(v12 + 8) + 112 * v14);
      if ( a4 == v26->m128i_i64[0] )
        break;
      if ( (unsigned int *)v25 == ++v22 )
        goto LABEL_57;
    }
    v27 = _mm_loadu_si128(&v116);
    v28 = *(unsigned int *)(a3 + 8);
    v117 = *v22;
    v121 = &v117;
    v120 = v12;
    v127.m128i_i64[1] = a9;
    v129 = (__int64 *)v131;
    v127.m128i_i64[0] = a4;
    v130 = 0x300000000LL;
    v134 = v27;
    v128 = v27;
    if ( (_DWORD)v28 )
    {
      sub_2538710((__int64)&v129, a3, v21, v25, v14, v28);
      v26 = (const __m128i *)(*(_QWORD *)(v12 + 8) + 112LL * v117);
      v63 = *(_DWORD *)(a3 + 8) <= 1u;
      v132 = a7;
      v133 = a8;
      if ( !v63 )
        v132 = a7 & 0xFFFFFFFC | 2;
    }
    else
    {
      v132 = a7;
      v133 = a8;
    }
    v134 = *v26;
    v29 = _mm_loadu_si128(v26 + 1);
    v136 = (const __m128i *)v138;
    v137 = 0x300000000LL;
    v135 = v29;
    v30 = v26[2].m128i_u32[2];
    if ( (_DWORD)v30 )
      sub_2538710((__int64)&v136, (__int64)v26[2].m128i_i64, v21, v30, v14, v28);
    v139 = v26[6].m128i_i32[0];
    v140 = v26[6].m128i_i64[1];
    v31 = v26[2].m128i_u32[2];
    if ( (_DWORD)v31 )
    {
      v32 = (_QWORD *)v26[2].m128i_i64[0];
      if ( *v32 == 0x7FFFFFFF )
        goto LABEL_25;
      if ( v32[1] == 0x7FFFFFFF )
        goto LABEL_25;
      v33 = (unsigned int)v130;
      if ( !(_DWORD)v130 )
        goto LABEL_25;
    }
    else
    {
      v33 = (unsigned int)v130;
      if ( !(_DWORD)v130 )
        goto LABEL_112;
    }
    v34 = v129;
    if ( *v129 == 0x7FFFFFFF || v129[1] == 0x7FFFFFFF )
    {
      v26[2].m128i_i32[2] = 0;
      sub_2555810((__int64)v26[2].m128i_i64, 0x7FFFFFFF, 0x7FFFFFFF, v30, v14, v28);
      goto LABEL_25;
    }
    if ( (_DWORD)v31 )
    {
      v35 = (__m128i *)v26[2].m128i_i64[0];
      v36 = &v129[2 * v33];
      do
      {
        v37 = sub_2555870((const __m128i *)v26[2].m128i_i32, v35, v34, v30, v14, v28);
        if ( v26[2].m128i_i32[2] )
        {
          v38 = (_QWORD *)v26[2].m128i_i64[0];
          if ( *v38 == 0x7FFFFFFF || v38[1] == 0x7FFFFFFF )
            break;
        }
        v34 += 2;
        v35 = v37;
      }
      while ( v36 != v34 );
LABEL_25:
      v26[1].m128i_i64[0] = sub_250C590(v26 + 1, (unsigned __int8 **)&v128, v26[6].m128i_i64[1]);
      v39 = v26[6].m128i_i32[0];
      v26[1].m128i_i64[1] = v40;
      v41 = v132 | v39;
      v26[6].m128i_i32[0] = v41;
      if ( (v41 & 2) != 0 || v26[2].m128i_i32[2] > 1u )
        v26[6].m128i_i32[0] = v41 & 0xFFFFFFFC | 2;
      v42 = v136;
      if ( *v26 == *(_OWORD *)&v134 )
      {
        v44 = v26[2].m128i_u32[2];
        v43 = (const __m128i *)v26[2].m128i_i64[0];
        if ( v44 == (unsigned int)v137 )
        {
          v79 = (const __m128i *)v26[2].m128i_i64[0];
          v80 = &v43[v44];
          if ( v43 == v80 )
          {
LABEL_122:
            v82 = v26[1].m128i_i8[8];
            if ( v82 == v135.m128i_i8[8] && (!v82 || v26[1].m128i_i64[0] == v135.m128i_i64[0]) )
            {
              v60 = 1;
              if ( v26[6].m128i_i32[0] == v139 )
              {
LABEL_89:
                if ( v42 != (const __m128i *)v138 )
                  _libc_free((unsigned __int64)v42);
                if ( v129 != (__int64 *)v131 )
                  _libc_free((unsigned __int64)v129);
                return v60;
              }
            }
          }
          else
          {
            v81 = v136;
            while ( v79->m128i_i64[0] == v81->m128i_i64[0] && v79->m128i_i64[1] == v81->m128i_i64[1] )
            {
              ++v79;
              ++v81;
              if ( v80 == v79 )
                goto LABEL_122;
            }
          }
        }
      }
      else
      {
        v43 = (const __m128i *)v26[2].m128i_i64[0];
        v44 = v26[2].m128i_u32[2];
      }
      v122 = (const __m128i *)v124;
      v123 = 0x300000000LL;
      sub_255D850(
        v136,
        &v136[(unsigned int)v137],
        v43,
        (__int64)v43[v44].m128i_i64,
        (__int64)&v122,
        (__int64)sub_2534890);
      v104 = &v122[(unsigned int)v123];
      if ( v104 != v122 )
      {
        v109 = v122;
        v105 = v12 + 136;
        v103 = v26;
        while ( 1 )
        {
          v125 = _mm_loadu_si128(v109);
          v45 = sub_255DB00(v105, v125.m128i_i64, &v118);
          if ( !v45 )
          {
            v64 = *(_DWORD *)(v12 + 152);
            v65 = *(_DWORD *)(v12 + 160);
            v66 = v118;
            ++*(_QWORD *)(v12 + 136);
            v67 = v64 + 1;
            v119 = v66;
            if ( 4 * (v64 + 1) >= 3 * v65 )
            {
              v65 *= 2;
            }
            else if ( v65 - *(_DWORD *)(v12 + 156) - v67 > v65 >> 3 )
            {
              goto LABEL_71;
            }
            sub_2569C80(v105, v65);
            sub_255DB00(v105, v125.m128i_i64, &v119);
            v67 = *(_DWORD *)(v12 + 152) + 1;
            v66 = v119;
LABEL_71:
            *(_DWORD *)(v12 + 152) = v67;
            if ( *v66 != 0x7FFFFFFFFFFFFFFFLL || v66[1] != 0x7FFFFFFFFFFFFFFFLL )
              --*(_DWORD *)(v12 + 156);
            v68 = _mm_loadu_si128(&v125);
            v66[2] = (unsigned __int64)(v66 + 4);
            v47 = v66 + 2;
            *((_OWORD *)v66 + 3) = 0;
            v66[3] = 0x400000000LL;
            *((_DWORD *)v66 + 14) = 0;
            v66[8] = 0;
            v66[9] = (unsigned __int64)(v66 + 7);
            v66[10] = (unsigned __int64)(v66 + 7);
            v66[11] = 0;
            *(__m128i *)v66 = v68;
            *((_OWORD *)v66 + 2) = 0;
LABEL_74:
            v69 = (_DWORD *)*v47;
            v70 = *v47 + 4LL * *((unsigned int *)v47 + 2);
            v71 = *((_DWORD *)v47 + 2);
            if ( *v47 != v70 )
            {
              while ( *v69 != v117 )
              {
                if ( (_DWORD *)v70 == ++v69 )
                  goto LABEL_82;
              }
              if ( (_DWORD *)v70 != v69 )
              {
                if ( (_DWORD *)v70 != v69 + 1 )
                {
                  memmove(v69, v69 + 1, v70 - (_QWORD)(v69 + 1));
                  v71 = *((_DWORD *)v47 + 2);
                }
                *((_DWORD *)v47 + 2) = v71 - 1;
              }
            }
            goto LABEL_82;
          }
          v46 = v118;
          v47 = v118 + 2;
          if ( !v118[11] )
            goto LABEL_74;
          v48 = v118 + 7;
          if ( v118[8] )
          {
            v49 = v118 + 7;
            v50 = v118[8];
            while ( 1 )
            {
              while ( *(_DWORD *)(v50 + 32) < v117 )
              {
                v50 = *(_QWORD *)(v50 + 24);
                if ( !v50 )
                  goto LABEL_40;
              }
              v51 = *(_QWORD *)(v50 + 16);
              if ( *(_DWORD *)(v50 + 32) <= v117 )
                break;
              v49 = (unsigned __int64 *)v50;
              v50 = *(_QWORD *)(v50 + 16);
              if ( !v51 )
              {
LABEL_40:
                v45 = v49 == v48;
                goto LABEL_41;
              }
            }
            v72 = *(_QWORD *)(v50 + 24);
            if ( v72 )
            {
              do
              {
                while ( 1 )
                {
                  v73 = *(_QWORD *)(v72 + 16);
                  v74 = *(_QWORD *)(v72 + 24);
                  if ( v117 < *(_DWORD *)(v72 + 32) )
                    break;
                  v72 = *(_QWORD *)(v72 + 24);
                  if ( !v74 )
                    goto LABEL_100;
                }
                v49 = (unsigned __int64 *)v72;
                v72 = *(_QWORD *)(v72 + 16);
              }
              while ( v73 );
            }
LABEL_100:
            while ( v51 )
            {
              while ( 1 )
              {
                v75 = *(_QWORD *)(v51 + 24);
                if ( v117 <= *(_DWORD *)(v51 + 32) )
                  break;
                v51 = *(_QWORD *)(v51 + 24);
                if ( !v75 )
                  goto LABEL_103;
              }
              v50 = v51;
              v51 = *(_QWORD *)(v51 + 16);
            }
LABEL_103:
            if ( v118[9] != v50 || v48 != v49 )
            {
              if ( (unsigned __int64 *)v50 != v49 )
              {
                v106 = v12;
                v76 = v118;
                do
                {
                  v77 = (int *)v50;
                  v50 = sub_220EF30(v50);
                  v78 = sub_220F330(v77, v48);
                  j_j___libc_free_0((unsigned __int64)v78);
                  --v76[11];
                }
                while ( (unsigned __int64 *)v50 != v49 );
                v12 = v106;
              }
              goto LABEL_82;
            }
          }
          else
          {
            v49 = v118 + 7;
LABEL_41:
            if ( (unsigned __int64 *)v118[9] != v49 || !v45 )
              goto LABEL_82;
          }
          sub_253AEE0(v118[8]);
          v46[9] = (unsigned __int64)v48;
          v46[8] = 0;
          v46[10] = (unsigned __int64)v48;
          v46[11] = 0;
LABEL_82:
          if ( v104 == ++v109 )
          {
            v26 = v103;
            break;
          }
        }
      }
      v125.m128i_i64[0] = (__int64)v126;
      v125.m128i_i64[1] = 0x300000000LL;
      sub_255D850(
        (const __m128i *)v26[2].m128i_i64[0],
        (const __m128i *)(v26[2].m128i_i64[0] + 16LL * v26[2].m128i_u32[2]),
        v136,
        (__int64)v136[(unsigned int)v137].m128i_i64,
        (__int64)&v125,
        (__int64)sub_2534890);
      sub_256BDC0(&v120, (__int64)&v125);
      if ( (_BYTE *)v125.m128i_i64[0] != v126 )
        _libc_free(v125.m128i_u64[0]);
      if ( v122 != (const __m128i *)v124 )
        _libc_free((unsigned __int64)v122);
      v42 = v136;
      v60 = 0;
      goto LABEL_89;
    }
LABEL_112:
    sub_2538710((__int64)v26[2].m128i_i64, (__int64)&v129, v31, v30, v14, v28);
    goto LABEL_25;
  }
LABEL_57:
  v120 = v12;
  v121 = &v117;
  if ( *(_DWORD *)(v12 + 20) <= v24 )
  {
    v83 = v12 + 24;
    v111 = v23;
    v84 = sub_C8D7D0(v12 + 8, v12 + 24, 0, 0x70u, (unsigned __int64 *)&v125, v23);
    v85 = a9;
    v86 = _mm_loadu_si128(&v116);
    v87 = v84;
    v88 = *(unsigned int *)(v12 + 16);
    v23 = v111;
    v134 = v86;
    v89 = 7 * v88;
    v90 = (__m128i *)(v89 * 16 + v87);
    if ( v89 * 16 + v87 )
    {
      v127 = v86;
      v90->m128i_i64[1] = a9;
      v90->m128i_i64[0] = a4;
      v91 = _mm_loadu_si128(&v127);
      v90[2].m128i_i64[0] = (__int64)v90[3].m128i_i64;
      v90[2].m128i_i64[1] = 0x300000000LL;
      v90[1] = v91;
      v14 = *(unsigned int *)(a3 + 8);
      if ( (_DWORD)v14 )
      {
        v108 = v111;
        v115 = v89 * 16 + v87;
        sub_2538710((__int64)v90[2].m128i_i64, a3, (__int64)v90[3].m128i_i64, 0x300000000LL, v14, v23);
        v63 = *(_DWORD *)(a3 + 8) <= 1u;
        v23 = v108;
        *(_DWORD *)(v115 + 96) = a7;
        v85 = a8;
        *(_QWORD *)(v115 + 104) = a8;
        if ( !v63 )
          *(_DWORD *)(v115 + 96) = a7 & 0xFFFFFFFC | 2;
      }
      else
      {
        v90[6].m128i_i32[0] = a7;
        v85 = a8;
        v90[6].m128i_i64[1] = a8;
      }
      v89 = 7LL * *(unsigned int *)(v12 + 16);
    }
    v92 = *(const __m128i **)(v12 + 8);
    v93 = (unsigned __int64)&v92[v89];
    if ( v92 != &v92[v89] )
    {
      v107 = v12;
      v94 = (__m128i *)v87;
      v95 = &v92[v89];
      v96 = v92;
      v112 = v23;
      do
      {
        if ( v94 )
        {
          v94->m128i_i64[0] = v96->m128i_i64[0];
          v94->m128i_i64[1] = v96->m128i_i64[1];
          v97 = _mm_loadu_si128(v96 + 1);
          v94[2].m128i_i64[0] = (__int64)v94[3].m128i_i64;
          v94[2].m128i_i32[2] = 0;
          v94[2].m128i_i32[3] = 3;
          v94[1] = v97;
          if ( v96[2].m128i_i32[2] )
            sub_2538710((__int64)v94[2].m128i_i64, (__int64)v96[2].m128i_i64, (__int64)v94[3].m128i_i64, v85, v14, v23);
          v94[6].m128i_i32[0] = v96[6].m128i_i32[0];
          v94[6].m128i_i64[1] = v96[6].m128i_i64[1];
        }
        v96 += 7;
        v94 += 7;
      }
      while ( v95 != v96 );
      v12 = v107;
      v23 = v112;
      v99 = *(_QWORD *)(v107 + 8);
      v93 = v99 + 112LL * *(unsigned int *)(v107 + 16);
      if ( v99 != v93 )
      {
        v100 = v99 + 112LL * *(unsigned int *)(v107 + 16);
        do
        {
          v100 -= 112;
          v101 = *(_QWORD *)(v100 + 32);
          if ( v101 != v100 + 48 )
            _libc_free(v101);
        }
        while ( v100 != v99 );
        v12 = v107;
        v23 = v112;
        v93 = *(_QWORD *)(v107 + 8);
      }
    }
    v102 = v125.m128i_i32[0];
    if ( v83 != v93 )
    {
      v113 = v23;
      _libc_free(v93);
      v23 = v113;
    }
    ++*(_DWORD *)(v12 + 16);
    v56 = v117;
    *(_QWORD *)(v12 + 8) = v87;
    *(_DWORD *)(v12 + 20) = v102;
  }
  else
  {
    v54 = _mm_loadu_si128(&v116);
    v55 = a9;
    v56 = v24;
    v134 = v54;
    v57 = *(_QWORD *)(v12 + 8) + 112LL * v24;
    if ( v57 )
    {
      v127 = v54;
      *(_QWORD *)v57 = a4;
      *(_QWORD *)(v57 + 8) = v55;
      v58 = _mm_loadu_si128(&v127);
      *(_QWORD *)(v57 + 32) = v57 + 48;
      *(_QWORD *)(v57 + 40) = 0x300000000LL;
      *(__m128i *)(v57 + 16) = v58;
      if ( *(_DWORD *)(a3 + 8) )
      {
        v110 = v23;
        sub_2538710(v57 + 32, a3, v55, v25, v14, v23);
        v63 = *(_DWORD *)(a3 + 8) <= 1u;
        v23 = v110;
        *(_DWORD *)(v57 + 96) = a7;
        *(_QWORD *)(v57 + 104) = a8;
        if ( !v63 )
          *(_DWORD *)(v57 + 96) = a7 & 0xFFFFFFFC | 2;
      }
      else
      {
        *(_DWORD *)(v57 + 96) = a7;
        *(_QWORD *)(v57 + 104) = a8;
      }
      v24 = *(_DWORD *)(v12 + 16);
      v56 = v117;
    }
    *(_DWORD *)(v12 + 16) = v24 + 1;
  }
  v59 = *(unsigned int *)(v23 + 8);
  if ( v59 + 1 > (unsigned __int64)*(unsigned int *)(v23 + 12) )
  {
    v114 = v23;
    sub_C8D5F0(v23, (const void *)(v23 + 16), v59 + 1, 4u, v14, v23);
    v23 = v114;
    v59 = *(unsigned int *)(v114 + 8);
  }
  v60 = 0;
  *(_DWORD *)(*(_QWORD *)v23 + 4 * v59) = v56;
  v61 = v117;
  ++*(_DWORD *)(v23 + 8);
  sub_256BDC0(&v120, *(_QWORD *)(v12 + 8) + 112 * v61 + 32);
  return v60;
}
