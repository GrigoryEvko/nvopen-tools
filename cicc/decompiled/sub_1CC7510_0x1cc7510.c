// Function: sub_1CC7510
// Address: 0x1cc7510
//
__int64 __fastcall sub_1CC7510(_QWORD *a1, unsigned __int64 a2)
{
  _QWORD *v4; // rbx
  _QWORD *v5; // rax
  _QWORD *v6; // rsi
  __int64 v7; // rcx
  __int64 v8; // rdx
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rsi
  unsigned int v12; // ecx
  __int64 *v13; // rdx
  __int64 v14; // r9
  unsigned __int64 v15; // rax
  unsigned int i; // r12d
  unsigned __int64 v17; // rax
  _QWORD *v18; // rax
  _QWORD *v19; // rdi
  __int64 v20; // rsi
  __int64 v21; // rcx
  __int64 v22; // rbx
  _QWORD *j; // r13
  __int64 v24; // r12
  _QWORD *v25; // r14
  char v26; // al
  __int64 v27; // rax
  int v28; // eax
  unsigned __int8 v29; // al
  __int64 v30; // rcx
  __int64 *v31; // rax
  __int64 *v32; // rbx
  __int64 v33; // r15
  _QWORD *v34; // rax
  unsigned __int8 v35; // dl
  __int64 v36; // rdi
  __int64 v38; // rsi
  __int64 v39; // r15
  __m128i *v40; // rax
  unsigned __int64 v41; // r12
  __m128i *v42; // rdx
  __m128i *v43; // rsi
  __int64 v44; // rax
  unsigned __int64 v45; // rbx
  _QWORD *v46; // rax
  _QWORD *v47; // rsi
  __int64 v48; // rcx
  __int64 v49; // rdx
  __m128i *v50; // rax
  __m128i *v51; // r12
  __int64 *v52; // r13
  __m128i v53; // xmm0
  __m128i v54; // xmm1
  __int64 *v55; // rax
  __int64 v56; // rax
  __m128i *v57; // rax
  __m128i *v58; // rdx
  _BOOL8 v59; // rdi
  __int64 v60; // rax
  __m128i *v61; // rax
  __m128i *v62; // rdx
  __m128i *v63; // r13
  _BOOL8 v64; // rdi
  __m128i *v65; // r9
  __int64 *v66; // rax
  __int64 *v67; // rsi
  __int64 *v68; // rcx
  __int64 v69; // r14
  __int64 v70; // rax
  __int64 v71; // r14
  _QWORD *v72; // rax
  __m128i *v73; // rdx
  _BOOL8 v74; // rdi
  __int64 v75; // rax
  __m128i *v76; // r9
  __m128i *v77; // rax
  int v78; // edx
  int v79; // r8d
  __int64 v80; // rax
  __m128i *v81; // rax
  __m128i *v82; // rdx
  _BOOL8 v83; // rdi
  __int64 v84; // rax
  __m128i *v85; // rdi
  _QWORD *v86; // [rsp+8h] [rbp-178h]
  _QWORD *v87; // [rsp+10h] [rbp-170h]
  __int64 *v88; // [rsp+18h] [rbp-168h]
  __int64 *v89; // [rsp+18h] [rbp-168h]
  _QWORD *v90; // [rsp+20h] [rbp-160h]
  _QWORD *v91; // [rsp+28h] [rbp-158h]
  __m128i *v92; // [rsp+28h] [rbp-158h]
  _QWORD *v93; // [rsp+28h] [rbp-158h]
  __m128i *v94; // [rsp+28h] [rbp-158h]
  __m128i *v95; // [rsp+28h] [rbp-158h]
  _QWORD *v96; // [rsp+30h] [rbp-150h]
  int v97; // [rsp+44h] [rbp-13Ch]
  __int64 v98; // [rsp+48h] [rbp-138h]
  __int64 *v99; // [rsp+48h] [rbp-138h]
  __m128i *v100; // [rsp+48h] [rbp-138h]
  __int64 v101; // [rsp+50h] [rbp-130h]
  _QWORD *v102; // [rsp+50h] [rbp-130h]
  unsigned __int64 v103; // [rsp+58h] [rbp-128h]
  unsigned __int8 v104; // [rsp+60h] [rbp-120h]
  __int64 v105; // [rsp+60h] [rbp-120h]
  int v106; // [rsp+68h] [rbp-118h]
  unsigned __int64 v107; // [rsp+68h] [rbp-118h]
  _QWORD *v108; // [rsp+78h] [rbp-108h] BYREF
  __m128i v109; // [rsp+80h] [rbp-100h] BYREF
  __m128i v110; // [rsp+90h] [rbp-F0h] BYREF
  __int64 *v111; // [rsp+A0h] [rbp-E0h]
  __m128i v112; // [rsp+B0h] [rbp-D0h] BYREF
  __m128i v113; // [rsp+C0h] [rbp-C0h]
  __int64 *v114; // [rsp+D0h] [rbp-B0h]
  __int64 v115; // [rsp+D8h] [rbp-A8h]
  unsigned __int64 v116; // [rsp+E0h] [rbp-A0h] BYREF
  __int64 *v117; // [rsp+E8h] [rbp-98h]
  __int64 *v118; // [rsp+F0h] [rbp-90h]
  __int64 v119; // [rsp+F8h] [rbp-88h]
  int v120; // [rsp+100h] [rbp-80h]
  _BYTE v121[120]; // [rsp+108h] [rbp-78h] BYREF

  v4 = a1 + 9;
  v5 = (_QWORD *)a1[10];
  if ( v5 )
  {
    v6 = a1 + 9;
    do
    {
      while ( 1 )
      {
        v7 = v5[2];
        v8 = v5[3];
        if ( v5[4] >= a2 )
          break;
        v5 = (_QWORD *)v5[3];
        if ( !v8 )
          goto LABEL_6;
      }
      v6 = v5;
      v5 = (_QWORD *)v5[2];
    }
    while ( v7 );
LABEL_6:
    if ( v4 != v6 && v6[4] <= a2 )
      return 0;
  }
  v9 = sub_157EBA0(a2);
  if ( (unsigned int)sub_15F4D60(v9) <= 1 )
    return 0;
  if ( a2 + 40 == (*(_QWORD *)(a2 + 40) & 0xFFFFFFFFFFFFFFF8LL) )
    return 0;
  v10 = *(unsigned int *)(*a1 + 48LL);
  if ( !(_DWORD)v10 )
    return 0;
  v11 = *(_QWORD *)(*a1 + 32LL);
  v12 = (v10 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v13 = (__int64 *)(v11 + 16LL * v12);
  v14 = *v13;
  if ( a2 != *v13 )
  {
    v78 = 1;
    while ( v14 != -8 )
    {
      v79 = v78 + 1;
      v12 = (v10 - 1) & (v78 + v12);
      v13 = (__int64 *)(v11 + 16LL * v12);
      v14 = *v13;
      if ( a2 == *v13 )
        goto LABEL_12;
      v78 = v79;
    }
    return 0;
  }
LABEL_12:
  if ( v13 == (__int64 *)(v11 + 16 * v10) || !v13[1] )
    return 0;
  sub_1CC6C70(a1[4]);
  a1[4] = 0;
  a1[5] = a1 + 3;
  a1[6] = a1 + 3;
  a1[7] = 0;
  v90 = a1 + 3;
  v15 = sub_157EBA0(a2);
  v106 = sub_15F4D60(v15);
  if ( v106 )
  {
    for ( i = 0; i != v106; ++i )
    {
      v17 = sub_157EBA0(a2);
      v116 = sub_15F4DF0(v17, i);
      v18 = (_QWORD *)a1[10];
      if ( !v18 )
        return 0;
      v19 = v4;
      do
      {
        while ( 1 )
        {
          v20 = v18[2];
          v21 = v18[3];
          if ( v18[4] >= v116 )
            break;
          v18 = (_QWORD *)v18[3];
          if ( !v21 )
            goto LABEL_21;
        }
        v19 = v18;
        v18 = (_QWORD *)v18[2];
      }
      while ( v20 );
LABEL_21:
      if ( v4 == v19 || v19[4] > v116 )
        return 0;
      sub_1444990(a1 + 2, &v116);
    }
  }
  v22 = *(_QWORD *)(a2 + 40);
  v116 = 0;
  v117 = (__int64 *)v121;
  v118 = (__int64 *)v121;
  v119 = 8;
  v120 = 0;
  v97 = 0;
  v104 = 0;
  v103 = a2;
  v96 = a1;
  for ( j = (_QWORD *)(v22 & 0xFFFFFFFFFFFFFFF8LL); ; j = (_QWORD *)v107 )
  {
    v24 = (__int64)(j - 3);
    v107 = (unsigned __int64)j;
    if ( !j )
      v24 = 0;
    v25 = *(_QWORD **)(v103 + 48);
    if ( j != v25 )
      v107 = *j & 0xFFFFFFFFFFFFFFF8LL;
    v26 = *(_BYTE *)(v24 + 16);
    if ( v26 == 78 )
    {
      v44 = *(_QWORD *)(v24 - 24);
      if ( !*(_BYTE *)(v44 + 16)
        && (*(_BYTE *)(v44 + 33) & 0x20) != 0
        && (unsigned int)(*(_DWORD *)(v44 + 36) - 35) <= 3 )
      {
        goto LABEL_55;
      }
    }
    else if ( v26 == 54 )
    {
      v27 = **(_QWORD **)(v24 - 24);
      if ( *(_BYTE *)(v27 + 8) == 16 )
        v27 = **(_QWORD **)(v27 + 16);
      v28 = *(_DWORD *)(v27 + 8) >> 8;
      if ( v28 == 1 || v28 == 5 || !v28 )
        goto LABEL_55;
    }
    v91 = (_QWORD *)v96[1];
    if ( (unsigned __int8)sub_15F3040(v24) )
    {
      v66 = v117;
      if ( v118 != v117 )
        goto LABEL_123;
      v67 = &v117[HIDWORD(v119)];
      if ( v117 != v67 )
      {
        v68 = 0;
        do
        {
          if ( v24 == *v66 )
            goto LABEL_55;
          if ( *v66 == -2 )
            v68 = v66;
          ++v66;
        }
        while ( v67 != v66 );
        if ( v68 )
        {
          *v68 = v24;
          --v120;
          ++v116;
          goto LABEL_55;
        }
      }
      if ( HIDWORD(v119) >= (unsigned int)v119 )
      {
LABEL_123:
        sub_16CCBA0((__int64)&v116, v24);
        goto LABEL_55;
      }
      ++HIDWORD(v119);
      *v67 = v24;
      ++v116;
      goto LABEL_55;
    }
    v29 = *(_BYTE *)(v24 + 16);
    if ( v29 == 54 )
    {
      sub_141EB40(&v109, (__int64 *)v24);
      v31 = v118;
      if ( v118 == v117 )
        v32 = &v118[HIDWORD(v119)];
      else
        v32 = &v118[(unsigned int)v119];
      if ( v118 != v32 )
      {
        while ( (unsigned __int64)*v31 >= 0xFFFFFFFFFFFFFFFELL )
        {
          if ( v32 == ++v31 )
            goto LABEL_43;
        }
        if ( v32 != v31 )
        {
          v102 = j;
          v52 = v31;
          do
          {
            v53 = _mm_loadu_si128(&v109);
            v54 = _mm_loadu_si128(&v110);
            LOBYTE(v115) = 1;
            v112 = v53;
            v114 = v111;
            v113 = v54;
            if ( (sub_13575E0(v91, *v52, &v112, v30) & 2) != 0 )
            {
              j = v102;
              goto LABEL_55;
            }
            v55 = v52 + 1;
            if ( v52 + 1 == v32 )
              break;
            while ( 1 )
            {
              v52 = v55;
              if ( (unsigned __int64)*v55 < 0xFFFFFFFFFFFFFFFELL )
                break;
              if ( v32 == ++v55 )
                goto LABEL_98;
            }
          }
          while ( v32 != v55 );
LABEL_98:
          j = v102;
        }
      }
LABEL_43:
      v29 = *(_BYTE *)(v24 + 16);
    }
    if ( (unsigned int)v29 - 25 <= 9 || v29 == 77 || sub_1C30980(v24) )
      goto LABEL_55;
    v112.m128i_i32[2] = 0;
    v113.m128i_i64[0] = 0;
    v113.m128i_i64[1] = (__int64)&v112.m128i_i64[1];
    v114 = &v112.m128i_i64[1];
    v115 = 0;
    v109 = 0u;
    v110.m128i_i64[0] = 0;
    v33 = *(_QWORD *)(v24 + 8);
    if ( v33 )
    {
      v101 = v24;
      do
      {
        v34 = sub_1648700(v33);
        v35 = *((_BYTE *)v34 + 16);
        if ( v35 > 0x17u )
        {
          v108 = v34;
          if ( v35 == 77 )
            goto LABEL_51;
          v45 = v34[5];
          v46 = (_QWORD *)v96[4];
          if ( !v46 )
            goto LABEL_51;
          v47 = v90;
          do
          {
            while ( 1 )
            {
              v48 = v46[2];
              v49 = v46[3];
              if ( v46[4] >= v45 )
                break;
              v46 = (_QWORD *)v46[3];
              if ( !v49 )
                goto LABEL_84;
            }
            v47 = v46;
            v46 = (_QWORD *)v46[2];
          }
          while ( v48 );
LABEL_84:
          if ( v47 == v90 || v47[4] > v45 )
          {
LABEL_51:
            v36 = v109.m128i_i64[0];
LABEL_52:
            if ( v36 )
              j_j___libc_free_0(v36, v110.m128i_i64[0] - v36);
LABEL_54:
            sub_1CC6AA0(v113.m128i_i64[0]);
            goto LABEL_55;
          }
          v50 = (__m128i *)v113.m128i_i64[0];
          v51 = (__m128i *)&v112.m128i_u64[1];
          if ( !v113.m128i_i64[0] )
            goto LABEL_101;
          do
          {
            if ( v50[2].m128i_i64[0] < v45 )
            {
              v50 = (__m128i *)v50[1].m128i_i64[1];
            }
            else
            {
              v51 = v50;
              v50 = (__m128i *)v50[1].m128i_i64[0];
            }
          }
          while ( v50 );
          if ( v51 == (__m128i *)&v112.m128i_u64[1] || v51[2].m128i_i64[0] > v45 )
          {
LABEL_101:
            v99 = (__int64 *)v51;
            v56 = sub_22077B0(48);
            *(_QWORD *)(v56 + 32) = v45;
            v51 = (__m128i *)v56;
            *(_QWORD *)(v56 + 40) = 0;
            v57 = (__m128i *)sub_1CC7410(&v112, v99, (unsigned __int64 *)(v56 + 32));
            if ( v58 )
            {
              v59 = &v112.m128i_u64[1] == (unsigned __int64 *)v58 || v57 || v58[2].m128i_i64[0] > v45;
              sub_220F040(v59, v51, v58, &v112.m128i_u64[1]);
              ++v115;
            }
            else
            {
              v100 = v57;
              j_j___libc_free_0(v51, 48);
              v51 = v100;
            }
          }
          v51[2].m128i_i64[1] = 0;
          v38 = v109.m128i_i64[1];
          if ( v109.m128i_i64[1] == v110.m128i_i64[0] )
          {
            sub_170B610((__int64)&v109, (_BYTE *)v109.m128i_i64[1], &v108);
          }
          else
          {
            if ( v109.m128i_i64[1] )
            {
              *(_QWORD *)v109.m128i_i64[1] = v108;
              v38 = v109.m128i_i64[1];
            }
            v109.m128i_i64[1] = v38 + 8;
          }
        }
        v33 = *(_QWORD *)(v33 + 8);
      }
      while ( v33 );
      v24 = v101;
      v36 = v109.m128i_i64[0];
      if ( v96[7] != v115 )
        goto LABEL_52;
      v98 = v109.m128i_i64[1];
      if ( v109.m128i_i64[0] == v109.m128i_i64[1] )
        goto LABEL_117;
      v105 = v109.m128i_i64[0];
      v87 = j;
      v86 = v25;
      while ( 2 )
      {
        v39 = *(_QWORD *)v105;
        v40 = (__m128i *)v113.m128i_i64[0];
        v41 = *(_QWORD *)(*(_QWORD *)v105 + 40LL);
        if ( v113.m128i_i64[0] )
        {
          v42 = (__m128i *)v113.m128i_i64[0];
          v43 = (__m128i *)&v112.m128i_u64[1];
          do
          {
            if ( v42[2].m128i_i64[0] < v41 )
            {
              v42 = (__m128i *)v42[1].m128i_i64[1];
            }
            else
            {
              v43 = v42;
              v42 = (__m128i *)v42[1].m128i_i64[0];
            }
          }
          while ( v42 );
          if ( v43 != (__m128i *)&v112.m128i_u64[1] && v43[2].m128i_i64[0] <= v41 )
          {
            v69 = v43[2].m128i_i64[1];
            goto LABEL_136;
          }
        }
        else
        {
          v43 = (__m128i *)&v112.m128i_u64[1];
        }
        v70 = sub_22077B0(48);
        *(_QWORD *)(v70 + 32) = v41;
        v71 = v70;
        *(_QWORD *)(v70 + 40) = 0;
        v72 = sub_1CC7410(&v112, v43, (unsigned __int64 *)(v70 + 32));
        if ( v73 )
        {
          v74 = &v112.m128i_u64[1] == (unsigned __int64 *)v73 || v72 || v41 < v73[2].m128i_i64[0];
          sub_220F040(v74, v71, v73, &v112.m128i_u64[1]);
          ++v115;
        }
        else
        {
          v93 = v72;
          j_j___libc_free_0(v71, 48);
          v71 = (__int64)v93;
        }
        v40 = (__m128i *)v113.m128i_i64[0];
        v69 = *(_QWORD *)(v71 + 40);
        if ( !v113.m128i_i64[0] )
        {
          v65 = (__m128i *)&v112.m128i_u64[1];
          goto LABEL_109;
        }
LABEL_136:
        v65 = (__m128i *)&v112.m128i_u64[1];
        do
        {
          if ( v40[2].m128i_i64[0] < v41 )
          {
            v40 = (__m128i *)v40[1].m128i_i64[1];
          }
          else
          {
            v65 = v40;
            v40 = (__m128i *)v40[1].m128i_i64[0];
          }
        }
        while ( v40 );
        if ( v65 == (__m128i *)&v112.m128i_u64[1] || v65[2].m128i_i64[0] > v41 )
        {
LABEL_109:
          v88 = (__int64 *)v65;
          v60 = sub_22077B0(48);
          *(_QWORD *)(v60 + 32) = v41;
          *(_QWORD *)(v60 + 40) = 0;
          v92 = (__m128i *)v60;
          v61 = (__m128i *)sub_1CC7410(&v112, v88, (unsigned __int64 *)(v60 + 32));
          v63 = v61;
          if ( v62 )
          {
            v64 = v61 || &v112.m128i_u64[1] == (unsigned __int64 *)v62 || v41 < v62[2].m128i_i64[0];
            sub_220F040(v64, v92, v62, &v112.m128i_u64[1]);
            ++v115;
            v65 = v92;
          }
          else
          {
            j_j___libc_free_0(v92, 48);
            v65 = v63;
          }
        }
        if ( !v65[2].m128i_i64[1] )
        {
          v75 = sub_15F4880(v101);
          v76 = (__m128i *)&v112.m128i_u64[1];
          v69 = v75;
          v77 = (__m128i *)v113.m128i_i64[0];
          if ( !v113.m128i_i64[0] )
            goto LABEL_166;
          do
          {
            if ( v77[2].m128i_i64[0] < v41 )
            {
              v77 = (__m128i *)v77[1].m128i_i64[1];
            }
            else
            {
              v76 = v77;
              v77 = (__m128i *)v77[1].m128i_i64[0];
            }
          }
          while ( v77 );
          if ( v76 == (__m128i *)&v112.m128i_u64[1] || v76[2].m128i_i64[0] > v41 )
          {
LABEL_166:
            v89 = (__int64 *)v76;
            v80 = sub_22077B0(48);
            *(_QWORD *)(v80 + 32) = v41;
            *(_QWORD *)(v80 + 40) = 0;
            v94 = (__m128i *)v80;
            v81 = (__m128i *)sub_1CC7410(&v112, v89, (unsigned __int64 *)(v80 + 32));
            if ( v82 )
            {
              v83 = &v112.m128i_u64[1] == (unsigned __int64 *)v82 || v81 || v41 < v82[2].m128i_i64[0];
              sub_220F040(v83, v94, v82, &v112.m128i_u64[1]);
              ++v115;
              v76 = v94;
            }
            else
            {
              v85 = v94;
              v95 = v81;
              j_j___libc_free_0(v85, 48);
              v76 = v95;
            }
          }
          v76[2].m128i_i64[1] = v69;
          v84 = sub_157ED20(v41);
          sub_15F2120(v69, v84);
        }
        sub_1648780(v39, v101, v69);
        v105 += 8;
        if ( v98 == v105 )
        {
          j = v87;
          v25 = v86;
          v24 = v101;
          goto LABEL_117;
        }
        continue;
      }
    }
    if ( v96[7] )
      goto LABEL_54;
LABEL_117:
    sub_15F20C0((_QWORD *)v24);
    if ( v109.m128i_i64[0] )
      j_j___libc_free_0(v109.m128i_i64[0], v110.m128i_i64[0] - v109.m128i_i64[0]);
    sub_1CC6AA0(v113.m128i_i64[0]);
    ++v97;
    v104 = 1;
    if ( v97 >= SLODWORD(qword_4FBF1E0[20]) )
      break;
LABEL_55:
    if ( j == v25 )
      break;
  }
  if ( v118 != v117 )
    _libc_free((unsigned __int64)v118);
  return v104;
}
