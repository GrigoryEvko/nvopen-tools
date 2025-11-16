// Function: sub_C9A3C0
// Address: 0xc9a3c0
//
__int64 __fastcall sub_C9A3C0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // r9
  __int64 v6; // rcx
  __int64 v7; // rdx
  _QWORD *v8; // r15
  _QWORD *v9; // r12
  const void *v10; // r14
  __int64 v11; // rsi
  size_t v12; // r13
  _QWORD *v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rsi
  _QWORD *v16; // rbx
  _QWORD *v17; // rdx
  _QWORD *v18; // rax
  size_t v19; // r12
  const void *v20; // r13
  _QWORD *v21; // r14
  __int64 v22; // rax
  _QWORD *v23; // r10
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  _QWORD *v27; // rax
  _QWORD *v28; // rax
  _QWORD *v29; // r15
  __int64 v30; // r12
  __int64 v31; // rdx
  _QWORD *v32; // r13
  _QWORD *v33; // r14
  _QWORD *v34; // rbx
  _QWORD *v35; // rdi
  _QWORD *v36; // rdi
  _QWORD *v37; // rdi
  _QWORD *v38; // rdi
  _QWORD *v39; // rdi
  _QWORD *v40; // rdi
  int v41; // eax
  _QWORD *v42; // rax
  int v43; // eax
  unsigned int v44; // r9d
  _QWORD *v45; // r8
  __int64 v46; // rax
  int v47; // eax
  int v48; // eax
  _QWORD *v49; // rax
  __int64 v50; // rax
  int v51; // eax
  __int64 result; // rax
  __int64 v53; // r13
  _QWORD *v54; // r13
  _QWORD *v55; // rbx
  _QWORD *v56; // r12
  _QWORD *v57; // rdi
  _QWORD *v58; // rdi
  _QWORD *v59; // rdi
  _QWORD *v60; // rdi
  _QWORD *v61; // rdi
  _QWORD *v62; // rdi
  __int64 v63; // rdx
  int v64; // eax
  __int64 v65; // rdx
  bool v66; // zf
  _QWORD *v67; // rdx
  _QWORD *v68; // r14
  __int64 v69; // r14
  __int64 v70; // r12
  __int64 v71; // rbx
  __int64 v72; // rax
  int v73; // edx
  __int64 v74; // rax
  _QWORD *v75; // rax
  _QWORD *v76; // r13
  __m128i *v77; // r13
  unsigned __int64 v78; // rax
  _QWORD *v79; // rax
  _QWORD *v80; // r15
  int v81; // r15d
  __int64 v82; // rdi
  __int64 v83; // rax
  unsigned int v84; // r9d
  _QWORD *v85; // r8
  _QWORD *v86; // rcx
  __int64 *v87; // rdx
  __int64 *v88; // rdx
  __int64 v89; // rax
  __int64 v90; // rax
  _QWORD *v91; // rax
  int v92; // eax
  __m128i *v93; // r14
  int v94; // r14d
  __int64 v95; // rdi
  int v96; // eax
  __int64 v97; // [rsp+8h] [rbp-78h]
  __int64 v98; // [rsp+10h] [rbp-70h]
  unsigned int v99; // [rsp+18h] [rbp-68h]
  _QWORD *v100; // [rsp+18h] [rbp-68h]
  _QWORD *v101; // [rsp+20h] [rbp-60h]
  _QWORD *v102; // [rsp+20h] [rbp-60h]
  _QWORD *v103; // [rsp+28h] [rbp-58h]
  _QWORD *v104; // [rsp+28h] [rbp-58h]
  _QWORD *v105; // [rsp+28h] [rbp-58h]
  __int64 v106; // [rsp+28h] [rbp-58h]
  __int64 v107; // [rsp+30h] [rbp-50h]
  __int64 v108; // [rsp+30h] [rbp-50h]
  unsigned int v109; // [rsp+30h] [rbp-50h]
  __m128i *v110; // [rsp+30h] [rbp-50h]
  _QWORD *v111; // [rsp+38h] [rbp-48h]
  __int64 v112; // [rsp+38h] [rbp-48h]
  _QWORD *v113; // [rsp+38h] [rbp-48h]
  __int64 v114; // [rsp+38h] [rbp-48h]
  _QWORD *v115; // [rsp+38h] [rbp-48h]
  _QWORD *v116; // [rsp+38h] [rbp-48h]
  unsigned __int64 v117[7]; // [rsp+48h] [rbp-38h] BYREF

  v2 = a2;
  v3 = a1;
  v4 = sub_220F880();
  *(_QWORD *)(a2 + 8) = v4;
  v98 = v4 - *(_QWORD *)a2;
  v99 = *(_DWORD *)(a1 + 8);
  v103 = *(_QWORD **)a1;
  v97 = *(_QWORD *)a1 + 8LL * v99;
  v6 = 8LL * v99;
  v7 = v6 >> 3;
  if ( v6 >> 5 )
  {
    v8 = *(_QWORD **)a1;
    while ( a2 != *v8 )
    {
      if ( a2 == v8[1] )
      {
        v9 = ++v8 + 1;
        goto LABEL_9;
      }
      if ( a2 == v8[2] )
      {
        v8 += 2;
        v9 = v8 + 1;
        goto LABEL_9;
      }
      if ( a2 == v8[3] )
      {
        v8 += 3;
        v9 = v8 + 1;
        goto LABEL_9;
      }
      v8 += 4;
      if ( (_QWORD *)(*(_QWORD *)a1 + 32 * (v6 >> 5)) == v8 )
      {
        v7 = (v97 - (__int64)v8) >> 3;
        goto LABEL_120;
      }
    }
    goto LABEL_8;
  }
  v8 = *(_QWORD **)a1;
LABEL_120:
  if ( v7 == 2 )
  {
    v91 = v8;
    goto LABEL_125;
  }
  if ( v7 != 3 )
  {
    if ( v7 != 1 )
    {
LABEL_123:
      v8 = (_QWORD *)(*(_QWORD *)a1 + 8LL * v99);
      v9 = (_QWORD *)(v97 + 8);
      goto LABEL_9;
    }
    goto LABEL_126;
  }
  v9 = v8 + 1;
  v91 = v8 + 1;
  if ( a2 != *v8 )
  {
LABEL_125:
    v8 = v91 + 1;
    if ( a2 == *v91 )
    {
      v8 = v91;
      goto LABEL_8;
    }
LABEL_126:
    if ( a2 != *v8 )
      goto LABEL_123;
LABEL_8:
    v9 = v8 + 1;
  }
LABEL_9:
  if ( *(unsigned int *)(a1 + 16660) <= v98 / 1000 )
  {
    v63 = *(unsigned int *)(a1 + 152);
    v114 = a1 + 144;
    v64 = v63;
    if ( *(_DWORD *)(a1 + 156) <= (unsigned int)v63 )
    {
      v106 = a1 + 160;
      v110 = (__m128i *)sub_C8D7D0(v114, a1 + 160, 0, 0x80u, v117, v5);
      v93 = &v110[8 * (unsigned __int64)*(unsigned int *)(a1 + 152)];
      if ( v93 )
      {
        v93->m128i_i64[0] = *(_QWORD *)a2;
        v93->m128i_i64[1] = *(_QWORD *)(a2 + 8);
        v93[1].m128i_i64[0] = (__int64)v93[2].m128i_i64;
        sub_C95D30(v93[1].m128i_i64, *(_BYTE **)(a2 + 16), *(_QWORD *)(a2 + 16) + *(_QWORD *)(a2 + 24));
        v93[3].m128i_i64[0] = (__int64)v93[4].m128i_i64;
        sub_C95D30(v93[3].m128i_i64, *(_BYTE **)(a2 + 48), *(_QWORD *)(a2 + 48) + *(_QWORD *)(a2 + 56));
        v93[5].m128i_i64[0] = (__int64)v93[6].m128i_i64;
        sub_C95D30(v93[5].m128i_i64, *(_BYTE **)(a2 + 80), *(_QWORD *)(a2 + 80) + *(_QWORD *)(a2 + 88));
        v93[7].m128i_i32[0] = *(_DWORD *)(a2 + 112);
        v93[7].m128i_i32[2] = *(_DWORD *)(a2 + 120);
      }
      sub_C9A200(v114, v110);
      v94 = v117[0];
      v95 = *(_QWORD *)(a1 + 144);
      if ( v106 != v95 )
        _libc_free(v95, v110);
      ++*(_DWORD *)(v3 + 152);
      *(_DWORD *)(v3 + 156) = v94;
      *(_QWORD *)(v3 + 144) = v110;
    }
    else
    {
      v65 = v63 << 7;
      v66 = *(_QWORD *)(a1 + 144) + v65 == 0;
      v67 = (_QWORD *)(*(_QWORD *)(a1 + 144) + v65);
      v68 = v67;
      if ( !v66 )
      {
        *v67 = *(_QWORD *)a2;
        v67[1] = *(_QWORD *)(a2 + 8);
        v67[2] = v67 + 4;
        sub_C95D30(v67 + 2, *(_BYTE **)(a2 + 16), *(_QWORD *)(a2 + 16) + *(_QWORD *)(a2 + 24));
        v68[6] = v68 + 8;
        sub_C95D30(v68 + 6, *(_BYTE **)(a2 + 48), *(_QWORD *)(a2 + 48) + *(_QWORD *)(a2 + 56));
        v68[10] = v68 + 12;
        sub_C95D30(v68 + 10, *(_BYTE **)(a2 + 80), *(_QWORD *)(a2 + 80) + *(_QWORD *)(a2 + 88));
        *((_DWORD *)v68 + 28) = *(_DWORD *)(a2 + 112);
        *((_DWORD *)v68 + 30) = *(_DWORD *)(a2 + 120);
        v64 = *(_DWORD *)(a1 + 152);
      }
      *(_DWORD *)(a1 + 152) = v64 + 1;
    }
    v108 = v3 + 160;
    if ( *(_QWORD *)(*v8 + 136LL) != *(_QWORD *)(*v8 + 128LL) )
    {
      v104 = v8;
      v69 = *(_QWORD *)(*v8 + 136LL);
      v100 = v9;
      v70 = v3;
      v71 = *(_QWORD *)(*v8 + 128LL);
      do
      {
        v72 = *(unsigned int *)(v70 + 152);
        v73 = v72;
        if ( *(_DWORD *)(v70 + 156) <= (unsigned int)v72 )
        {
          v77 = (__m128i *)sub_C8D7D0(v114, v108, 0, 0x80u, v117, v5);
          v78 = (unsigned __int64)*(unsigned int *)(v70 + 152) << 7;
          v66 = &v77->m128i_i8[v78] == 0;
          v79 = (__int64 *)((char *)v77->m128i_i64 + v78);
          v80 = v79;
          if ( !v66 )
          {
            *v79 = *(_QWORD *)v71;
            v79[1] = *(_QWORD *)(v71 + 8);
            v79[2] = v79 + 4;
            sub_C95D30(v79 + 2, *(_BYTE **)(v71 + 16), *(_QWORD *)(v71 + 16) + *(_QWORD *)(v71 + 24));
            v80[6] = v80 + 8;
            sub_C95D30(v80 + 6, *(_BYTE **)(v71 + 48), *(_QWORD *)(v71 + 48) + *(_QWORD *)(v71 + 56));
            v80[10] = v80 + 12;
            sub_C95D30(v80 + 10, *(_BYTE **)(v71 + 80), *(_QWORD *)(v71 + 80) + *(_QWORD *)(v71 + 88));
            *((_DWORD *)v80 + 28) = *(_DWORD *)(v71 + 112);
            *((_DWORD *)v80 + 30) = *(_DWORD *)(v71 + 120);
          }
          sub_C9A200(v114, v77);
          v81 = v117[0];
          v82 = *(_QWORD *)(v70 + 144);
          if ( v108 != v82 )
            _libc_free(v82, v77);
          ++*(_DWORD *)(v70 + 152);
          *(_QWORD *)(v70 + 144) = v77;
          *(_DWORD *)(v70 + 156) = v81;
        }
        else
        {
          v74 = v72 << 7;
          v66 = *(_QWORD *)(v70 + 144) + v74 == 0;
          v75 = (_QWORD *)(*(_QWORD *)(v70 + 144) + v74);
          v76 = v75;
          if ( !v66 )
          {
            *v75 = *(_QWORD *)v71;
            v75[1] = *(_QWORD *)(v71 + 8);
            v75[2] = v75 + 4;
            sub_C95D30(v75 + 2, *(_BYTE **)(v71 + 16), *(_QWORD *)(v71 + 16) + *(_QWORD *)(v71 + 24));
            v76[6] = v76 + 8;
            sub_C95D30(v76 + 6, *(_BYTE **)(v71 + 48), *(_QWORD *)(v71 + 48) + *(_QWORD *)(v71 + 56));
            v76[10] = v76 + 12;
            sub_C95D30(v76 + 10, *(_BYTE **)(v71 + 80), *(_QWORD *)(v71 + 80) + *(_QWORD *)(v71 + 88));
            *((_DWORD *)v76 + 28) = *(_DWORD *)(v71 + 112);
            *((_DWORD *)v76 + 30) = *(_DWORD *)(v71 + 120);
            v73 = *(_DWORD *)(v70 + 152);
          }
          *(_DWORD *)(v70 + 152) = v73 + 1;
        }
        v71 += 128;
      }
      while ( v69 != v71 );
      v3 = v70;
      v8 = v104;
      v2 = a2;
      v9 = v100;
    }
    v6 = 8LL * *(unsigned int *)(v3 + 8);
    v99 = *(_DWORD *)(v3 + 8);
    v103 = *(_QWORD **)v3;
    v97 = *(_QWORD *)v3 + v6;
  }
  v10 = *(const void **)(v2 + 16);
  v11 = v6 - 8;
  v12 = *(_QWORD *)(v2 + 24);
  v13 = &v103[(unsigned __int64)v6 / 8 - 1];
  v14 = v11 >> 5;
  v15 = v11 >> 3;
  if ( v14 <= 0 )
  {
LABEL_57:
    if ( v15 != 2 )
    {
      if ( v15 != 3 )
      {
        if ( v15 != 1 )
          goto LABEL_47;
LABEL_60:
        v50 = *(v13 - 1);
        if ( v12 != *(_QWORD *)(v50 + 24) )
          goto LABEL_47;
        if ( v12 )
        {
          v113 = v13;
          v51 = memcmp(*(const void **)(v50 + 16), v10, v12);
          v13 = v113;
          if ( v51 )
            goto LABEL_47;
        }
        goto LABEL_20;
      }
      v89 = *(v13 - 1);
      if ( v12 == *(_QWORD *)(v89 + 24) )
      {
        if ( !v12 || (v115 = v13, v92 = memcmp(*(const void **)(v89 + 16), v10, v12), v13 = v115, !v92) )
        {
LABEL_20:
          if ( v103 == v13 )
            goto LABEL_47;
          goto LABEL_21;
        }
      }
      --v13;
    }
    v90 = *(v13 - 1);
    if ( v12 != *(_QWORD *)(v90 + 24)
      || v12 && (v116 = v13, v96 = memcmp(*(const void **)(v90 + 16), v10, v12), v13 = v116, v96) )
    {
      --v13;
      goto LABEL_60;
    }
    goto LABEL_20;
  }
  v107 = v3;
  v16 = v13;
  v17 = &v13[-4 * v14];
  v18 = v9;
  v19 = v12;
  v20 = v10;
  v111 = v17;
  v21 = v18;
  while ( 1 )
  {
    v26 = *(v16 - 1);
    if ( v19 == *(_QWORD *)(v26 + 24) && (!v19 || !memcmp(*(const void **)(v26 + 16), v20, v19)) )
    {
      v27 = v21;
      v13 = v16;
      v3 = v107;
      v10 = v20;
      v12 = v19;
      v9 = v27;
      goto LABEL_20;
    }
    v22 = *(v16 - 2);
    v23 = v16 - 1;
    if ( v19 == *(_QWORD *)(v22 + 24) )
    {
      if ( !v19 )
        break;
      v41 = memcmp(*(const void **)(v22 + 16), v20, v19);
      v23 = v16 - 1;
      if ( !v41 )
        break;
    }
    v24 = *(v16 - 3);
    v23 = v16 - 2;
    if ( v19 == *(_QWORD *)(v24 + 24) )
    {
      if ( !v19 )
        break;
      v47 = memcmp(*(const void **)(v24 + 16), v20, v19);
      v23 = v16 - 2;
      if ( !v47 )
        break;
      v25 = *(v16 - 4);
      v23 = v16 - 3;
      if ( v19 == *(_QWORD *)(v25 + 24) )
        goto LABEL_53;
LABEL_15:
      v16 -= 4;
      if ( v111 == v16 )
        goto LABEL_56;
    }
    else
    {
      v25 = *(v16 - 4);
      v23 = v16 - 3;
      if ( v19 != *(_QWORD *)(v25 + 24) )
        goto LABEL_15;
LABEL_53:
      if ( !v19 )
        break;
      v101 = v23;
      v48 = memcmp(*(const void **)(v25 + 16), v20, v19);
      v23 = v101;
      if ( !v48 )
        break;
      v16 -= 4;
      if ( v111 == v16 )
      {
LABEL_56:
        v13 = v16;
        v49 = v21;
        v3 = v107;
        v10 = v20;
        v12 = v19;
        v9 = v49;
        v15 = v13 - v103;
        goto LABEL_57;
      }
    }
  }
  v42 = v21;
  v3 = v107;
  v10 = v20;
  v12 = v19;
  v9 = v42;
  if ( v103 != v23 )
    goto LABEL_21;
LABEL_47:
  v43 = sub_C92610();
  v44 = sub_C92740(v3 + 16544, v10, v12, v43);
  v45 = (_QWORD *)(*(_QWORD *)(v3 + 16544) + 8LL * v44);
  v46 = *v45;
  if ( *v45 )
  {
    if ( v46 != -8 )
      goto LABEL_49;
    --*(_DWORD *)(v3 + 16560);
  }
  v105 = v45;
  v109 = v44;
  v83 = sub_C7D670(v12 + 25, 8);
  v84 = v109;
  v85 = v105;
  v86 = (_QWORD *)v83;
  if ( v12 )
  {
    v102 = (_QWORD *)v83;
    memcpy((void *)(v83 + 24), v10, v12);
    v84 = v109;
    v85 = v105;
    v86 = v102;
  }
  *((_BYTE *)v86 + v12 + 24) = 0;
  *v86 = v12;
  v86[1] = 0;
  v86[2] = 0;
  *v85 = v86;
  ++*(_DWORD *)(v3 + 16556);
  v87 = (__int64 *)(*(_QWORD *)(v3 + 16544) + 8LL * (unsigned int)sub_C929D0((__int64 *)(v3 + 16544), v84));
  v46 = *v87;
  if ( *v87 == -8 || !v46 )
  {
    v88 = v87 + 1;
    do
    {
      do
        v46 = *v88++;
      while ( !v46 );
    }
    while ( v46 == -8 );
  }
LABEL_49:
  ++*(_QWORD *)(v46 + 8);
  *(_QWORD *)(v46 + 16) += v98;
  v103 = *(_QWORD **)v3;
  v99 = *(_DWORD *)(v3 + 8);
  v97 = *(_QWORD *)v3 + 8LL * v99;
LABEL_21:
  if ( v97 - (__int64)v9 > 0 )
  {
    v112 = v3;
    v28 = v8;
    v29 = v9;
    v30 = (v97 - (__int64)v9) >> 3;
    while ( 1 )
    {
      v31 = v28[1];
      v32 = (_QWORD *)*v28;
      v28[1] = 0;
      *v28 = v31;
      if ( v32 )
      {
        v33 = (_QWORD *)v32[17];
        v34 = (_QWORD *)v32[16];
        if ( v33 != v34 )
        {
          do
          {
            v35 = (_QWORD *)v34[10];
            if ( v35 != v34 + 12 )
              j_j___libc_free_0(v35, v34[12] + 1LL);
            v36 = (_QWORD *)v34[6];
            if ( v36 != v34 + 8 )
              j_j___libc_free_0(v36, v34[8] + 1LL);
            v37 = (_QWORD *)v34[2];
            if ( v37 != v34 + 4 )
              j_j___libc_free_0(v37, v34[4] + 1LL);
            v34 += 16;
          }
          while ( v33 != v34 );
          v34 = (_QWORD *)v32[16];
        }
        if ( v34 )
          j_j___libc_free_0(v34, v32[18] - (_QWORD)v34);
        v38 = (_QWORD *)v32[10];
        if ( v38 != v32 + 12 )
          j_j___libc_free_0(v38, v32[12] + 1LL);
        v39 = (_QWORD *)v32[6];
        if ( v39 != v32 + 8 )
          j_j___libc_free_0(v39, v32[8] + 1LL);
        v40 = (_QWORD *)v32[2];
        if ( v40 != v32 + 4 )
          j_j___libc_free_0(v40, v32[4] + 1LL);
        j_j___libc_free_0(v32, 152);
      }
      v28 = v29;
      if ( !--v30 )
        break;
      ++v29;
    }
    v3 = v112;
    v99 = *(_DWORD *)(v112 + 8);
    v103 = *(_QWORD **)v112;
  }
  result = (__int64)v103;
  v53 = v99 - 1;
  *(_DWORD *)(v3 + 8) = v53;
  v54 = (_QWORD *)v103[v53];
  if ( v54 )
  {
    v55 = (_QWORD *)v54[17];
    v56 = (_QWORD *)v54[16];
    if ( v55 != v56 )
    {
      do
      {
        v57 = (_QWORD *)v56[10];
        if ( v57 != v56 + 12 )
          j_j___libc_free_0(v57, v56[12] + 1LL);
        v58 = (_QWORD *)v56[6];
        if ( v58 != v56 + 8 )
          j_j___libc_free_0(v58, v56[8] + 1LL);
        v59 = (_QWORD *)v56[2];
        if ( v59 != v56 + 4 )
          j_j___libc_free_0(v59, v56[4] + 1LL);
        v56 += 16;
      }
      while ( v55 != v56 );
      v56 = (_QWORD *)v54[16];
    }
    if ( v56 )
      j_j___libc_free_0(v56, v54[18] - (_QWORD)v56);
    v60 = (_QWORD *)v54[10];
    if ( v60 != v54 + 12 )
      j_j___libc_free_0(v60, v54[12] + 1LL);
    v61 = (_QWORD *)v54[6];
    if ( v61 != v54 + 8 )
      j_j___libc_free_0(v61, v54[8] + 1LL);
    v62 = (_QWORD *)v54[2];
    if ( v62 != v54 + 4 )
      j_j___libc_free_0(v62, v54[4] + 1LL);
    return j_j___libc_free_0(v54, 152);
  }
  return result;
}
