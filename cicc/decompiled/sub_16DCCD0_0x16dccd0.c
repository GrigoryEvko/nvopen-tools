// Function: sub_16DCCD0
// Address: 0x16dccd0
//
int __fastcall sub_16DCCD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  unsigned int v6; // edi
  __int64 v7; // rax
  __int64 *v8; // r13
  __int64 *v9; // rbx
  __int64 *v10; // rsi
  __int64 v11; // r13
  __int64 v12; // r15
  __int64 *v13; // rbx
  __int64 *v14; // r14
  __int64 *v15; // rsi
  int v16; // edi
  _QWORD *v17; // rsi
  __int64 *v18; // rax
  __int64 v19; // rdx
  __int64 *v20; // r15
  size_t v21; // rbx
  unsigned __int8 *v22; // r12
  unsigned int v23; // r13d
  __int64 v24; // rax
  __int64 *v25; // rdx
  __int64 v26; // rax
  __int64 *v27; // rax
  __int64 v28; // rdx
  int v29; // edx
  size_t **v30; // rsi
  size_t *v31; // rax
  size_t **v32; // r14
  size_t v33; // rbx
  unsigned __int8 *v34; // r12
  size_t v35; // r13
  __int64 v36; // rax
  size_t *v37; // rdx
  size_t **v38; // rax
  __int64 v39; // rax
  unsigned int v40; // r8d
  _QWORD *v41; // r9
  _QWORD *v42; // rcx
  __int64 v43; // rdi
  int v44; // eax
  __int64 *v45; // rdx
  __int64 v46; // rax
  void *v47; // rax
  __int64 v48; // r13
  __int64 v49; // rax
  const __m128i *v50; // r15
  const __m128i *v51; // rdi
  const __m128i *v52; // r12
  const __m128i *v53; // rbx
  __m128i *v54; // r14
  __m128i v55; // xmm1
  const __m128i *v56; // rdi
  const __m128i *v57; // rax
  __int64 *v58; // rax
  __int64 v59; // rdx
  __int64 **v60; // r13
  __int64 **v61; // rbx
  __int64 *v62; // r14
  __int64 v63; // rdx
  _BYTE *v64; // rsi
  __m128i *v65; // rsi
  __m128i *v66; // rdi
  __int64 *v67; // rdx
  __int64 **v68; // rax
  __m128i *v69; // r13
  __m128i *v70; // r14
  __int64 v71; // rbx
  unsigned __int64 v72; // rax
  __m128i *v73; // rbx
  __m128i *v74; // rdi
  __int64 v75; // r14
  __m128i *v76; // r14
  const __m128i *v77; // rbx
  const __m128i *v78; // r12
  unsigned __int64 v79; // r8
  __int64 v80; // r12
  __int64 v81; // rbx
  unsigned __int64 v82; // rdi
  __int64 v83; // rbx
  __int64 v84; // r12
  __int64 v85; // rdi
  int result; // eax
  __int64 v87; // rax
  _QWORD *v88; // r10
  _QWORD *v89; // r8
  __int64 v90; // rdi
  int v91; // eax
  __int64 *v92; // rdx
  __int64 *v93; // rdx
  __int64 v94; // rax
  void *v95; // rax
  char v96; // [rsp+0h] [rbp-160h]
  __int64 v98; // [rsp+10h] [rbp-150h]
  __int64 v99; // [rsp+18h] [rbp-148h]
  __int64 *v100; // [rsp+20h] [rbp-140h]
  size_t **v101; // [rsp+20h] [rbp-140h]
  _QWORD *v102; // [rsp+28h] [rbp-138h]
  _QWORD *v103; // [rsp+28h] [rbp-138h]
  _QWORD *v104; // [rsp+28h] [rbp-138h]
  _QWORD *v105; // [rsp+28h] [rbp-138h]
  unsigned int v106; // [rsp+30h] [rbp-130h]
  _QWORD *v107; // [rsp+30h] [rbp-130h]
  _QWORD *v108; // [rsp+30h] [rbp-130h]
  _QWORD *v109; // [rsp+30h] [rbp-130h]
  unsigned int v110; // [rsp+30h] [rbp-130h]
  unsigned int v111; // [rsp+38h] [rbp-128h]
  _QWORD *v112; // [rsp+38h] [rbp-128h]
  _QWORD *v113; // [rsp+38h] [rbp-128h]
  __int64 v115; // [rsp+40h] [rbp-120h]
  _QWORD *v116; // [rsp+40h] [rbp-120h]
  __int64 i; // [rsp+48h] [rbp-118h]
  __int64 v118; // [rsp+48h] [rbp-118h]
  size_t v119; // [rsp+48h] [rbp-118h]
  __int64 v120[2]; // [rsp+50h] [rbp-110h] BYREF
  __int64 v121; // [rsp+60h] [rbp-100h] BYREF
  __m128i v122; // [rsp+68h] [rbp-F8h]
  const __m128i *v123; // [rsp+80h] [rbp-E0h] BYREF
  __m128i *v124; // [rsp+88h] [rbp-D8h]
  const __m128i *v125; // [rsp+90h] [rbp-D0h]
  __m128i *v126[4]; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v127; // [rsp+C0h] [rbp-A0h] BYREF
  __m128i v128; // [rsp+C8h] [rbp-98h]
  int v129; // [rsp+D8h] [rbp-88h]
  _QWORD *v130; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v131; // [rsp+E8h] [rbp-78h]
  __int64 v132; // [rsp+F0h] [rbp-70h]
  __m128i v133; // [rsp+100h] [rbp-60h] BYREF
  __m128i v134; // [rsp+110h] [rbp-50h] BYREF
  int v135; // [rsp+120h] [rbp-40h]

  if ( &_pthread_key_create )
  {
    v6 = pthread_mutex_lock(&stru_4FA16A0);
    if ( v6 )
      sub_4264C5(v6);
  }
  v127 = 0;
  v120[1] = (__int64)&v121;
  v7 = *(unsigned int *)(a1 + 1304);
  v8 = *(__int64 **)(a1 + 1296);
  v128 = 0u;
  v129 = 0;
  v121 = 0;
  v9 = &v8[10 * v7];
  v120[0] = a1;
  v122 = 0u;
  while ( v9 != v8 )
  {
    v10 = v8;
    v8 += 10;
    sub_16DB9E0(v120, v10, *(_QWORD *)(a1 + 11656));
  }
  v11 = qword_4FA1670;
  for ( i = qword_4FA1678; i != v11; v11 += 8 )
  {
    v12 = *(_QWORD *)v11;
    v13 = *(__int64 **)(*(_QWORD *)v11 + 1296LL);
    v14 = &v13[10 * *(unsigned int *)(*(_QWORD *)v11 + 1304LL)];
    while ( v14 != v13 )
    {
      v15 = v13;
      v13 += 10;
      sub_16DB9E0(v120, v15, *(_QWORD *)(v12 + 11656));
    }
  }
  v130 = 0;
  v132 = 0x1800000000LL;
  v131 = 0;
  v16 = *(_DWORD *)(a1 + 11560);
  if ( v16 )
  {
    v17 = *(_QWORD **)(a1 + 11552);
    if ( *v17 && *v17 != -8 )
    {
      v20 = *(__int64 **)(a1 + 11552);
    }
    else
    {
      v18 = v17 + 1;
      do
      {
        do
        {
          v19 = *v18;
          v20 = v18++;
        }
        while ( v19 == -8 );
      }
      while ( !v19 );
    }
    v100 = &v17[v16];
    if ( v100 != v20 )
    {
      while ( 1 )
      {
        v21 = *(_QWORD *)*v20;
        v22 = (unsigned __int8 *)(*v20 + 24);
        v118 = *(_QWORD *)(*v20 + 8);
        v115 = *(_QWORD *)(*v20 + 16);
        v23 = sub_16D19C0((__int64)&v130, v22, v21);
        v24 = v130[v23];
        if ( !v24 )
          goto LABEL_117;
        if ( v24 == -8 )
          break;
LABEL_18:
        *(_QWORD *)(v24 + 8) += v118;
        v25 = v20 + 1;
        *(_QWORD *)(v24 + 16) += v115;
        v26 = v20[1];
        if ( v26 && v26 != -8 )
        {
          ++v20;
          if ( v25 == v100 )
            goto LABEL_24;
        }
        else
        {
          v27 = v20 + 2;
          do
          {
            do
            {
              v28 = *v27;
              v20 = v27++;
            }
            while ( v28 == -8 );
          }
          while ( !v28 );
          if ( v20 == v100 )
            goto LABEL_24;
        }
      }
      LODWORD(v132) = v132 - 1;
LABEL_117:
      v103 = &v130[v23];
      v87 = malloc(v21 + 25);
      v88 = v103;
      v89 = (_QWORD *)v87;
      if ( v87 )
      {
        v90 = v87 + 24;
      }
      else
      {
        if ( v21 == -25 )
        {
          v94 = malloc(1u);
          v89 = 0;
          v88 = v103;
          if ( v94 )
          {
            v90 = v94 + 24;
            v89 = (_QWORD *)v94;
            goto LABEL_128;
          }
        }
        v104 = v88;
        v109 = v89;
        sub_16BD1C0("Allocation failed", 1u);
        v89 = v109;
        v90 = 24;
        v88 = v104;
      }
      if ( v21 + 1 <= 1 )
      {
LABEL_120:
        *(_BYTE *)(v90 + v21) = 0;
        *v89 = v21;
        v89[1] = 0;
        v89[2] = 0;
        *v88 = v89;
        ++HIDWORD(v131);
        v91 = sub_16D1CD0((__int64)&v130, v23);
        v92 = &v130[v91];
        v24 = *v92;
        if ( *v92 == -8 || !v24 )
        {
          v93 = v92 + 1;
          do
          {
            do
              v24 = *v93++;
            while ( v24 == -8 );
          }
          while ( !v24 );
        }
        goto LABEL_18;
      }
LABEL_128:
      v108 = v88;
      v112 = v89;
      v95 = memcpy((void *)v90, v22, v21);
      v88 = v108;
      v89 = v112;
      v90 = (__int64)v95;
      goto LABEL_120;
    }
  }
LABEL_24:
  v98 = qword_4FA1678;
  v99 = qword_4FA1670;
  if ( qword_4FA1678 != qword_4FA1670 )
  {
    while ( 1 )
    {
      v29 = *(_DWORD *)(*(_QWORD *)v99 + 11560LL);
      if ( !v29 )
        goto LABEL_48;
      v30 = *(size_t ***)(*(_QWORD *)v99 + 11552LL);
      v31 = *v30;
      v32 = v30;
      if ( *v30 == (size_t *)-8LL )
        break;
LABEL_28:
      if ( !v31 )
        goto LABEL_27;
      v101 = &v30[v29];
      if ( v101 != v32 )
      {
        while ( 1 )
        {
          v33 = **v32;
          v34 = (unsigned __int8 *)(*v32 + 3);
          v35 = (*v32)[2];
          v119 = (*v32)[1];
          a4 = (unsigned int)sub_16D19C0((__int64)&v130, v34, v33);
          a5 = a4;
          a6 = &v130[a4];
          v36 = *a6;
          if ( *a6 )
          {
            if ( v36 != -8 )
              goto LABEL_32;
            LODWORD(v132) = v132 - 1;
          }
          v102 = &v130[a4];
          v106 = a4;
          v39 = malloc(v33 + 25);
          v40 = v106;
          v41 = v102;
          v42 = (_QWORD *)v39;
          if ( v39 )
            break;
          if ( v33 != -25 || (v46 = malloc(1u), v42 = 0, v40 = v106, v41 = v102, !v46) )
          {
            v105 = v41;
            v110 = v40;
            v113 = v42;
            sub_16BD1C0("Allocation failed", 1u);
            v42 = v113;
            v43 = 24;
            v40 = v110;
            v41 = v105;
LABEL_40:
            if ( v33 + 1 <= 1 )
              goto LABEL_41;
            goto LABEL_47;
          }
          v43 = v46 + 24;
          v42 = (_QWORD *)v46;
LABEL_47:
          v107 = v41;
          v111 = v40;
          v116 = v42;
          v47 = memcpy((void *)v43, v34, v33);
          v41 = v107;
          v40 = v111;
          v42 = v116;
          v43 = (__int64)v47;
LABEL_41:
          *(_BYTE *)(v43 + v33) = 0;
          *v42 = v33;
          v42[1] = 0;
          v42[2] = 0;
          *v41 = v42;
          ++HIDWORD(v131);
          v44 = sub_16D1CD0((__int64)&v130, v40);
          v45 = &v130[v44];
          v36 = *v45;
          if ( *v45 != -8 )
            goto LABEL_43;
          do
          {
            do
            {
              v36 = v45[1];
              ++v45;
            }
            while ( v36 == -8 );
LABEL_43:
            ;
          }
          while ( !v36 );
LABEL_32:
          *(_QWORD *)(v36 + 16) += v35;
          *(_QWORD *)(v36 + 8) += v119;
          v37 = v32[1];
          v38 = v32 + 1;
          if ( v37 != (size_t *)-8LL )
            goto LABEL_34;
          do
          {
            do
            {
              v37 = v38[1];
              ++v38;
            }
            while ( v37 == (size_t *)-8LL );
LABEL_34:
            ;
          }
          while ( !v37 );
          if ( v38 == v101 )
            goto LABEL_48;
          v32 = v38;
        }
        v43 = v39 + 24;
        goto LABEL_40;
      }
LABEL_48:
      v99 += 8;
      if ( v98 == v99 )
        goto LABEL_49;
    }
    do
    {
LABEL_27:
      v31 = v32[1];
      ++v32;
    }
    while ( v31 == (size_t *)-8LL );
    goto LABEL_28;
  }
LABEL_49:
  v123 = 0;
  v124 = 0;
  v125 = 0;
  if ( HIDWORD(v131) )
  {
    v48 = 3LL * HIDWORD(v131);
    v49 = sub_22077B0(v48 * 16);
    v50 = v124;
    v51 = v123;
    v52 = (const __m128i *)v49;
    if ( v124 != v123 )
    {
      v53 = v123 + 1;
      v54 = (__m128i *)v49;
      while ( 1 )
      {
        if ( v54 )
        {
          v54->m128i_i64[0] = (__int64)v54[1].m128i_i64;
          v57 = (const __m128i *)v53[-1].m128i_i64[0];
          if ( v57 == v53 )
          {
            v54[1] = _mm_loadu_si128(v53);
          }
          else
          {
            v54->m128i_i64[0] = (__int64)v57;
            v54[1].m128i_i64[0] = v53->m128i_i64[0];
          }
          v54->m128i_i64[1] = v53[-1].m128i_i64[1];
          v55 = _mm_loadu_si128(v53 + 1);
          v53[-1].m128i_i64[0] = (__int64)v53;
          v53[-1].m128i_i64[1] = 0;
          v53->m128i_i8[0] = 0;
          v54[2] = v55;
        }
        v56 = (const __m128i *)v53[-1].m128i_i64[0];
        if ( v56 != v53 )
          j_j___libc_free_0(v56, v53->m128i_i64[0] + 1);
        v54 += 3;
        if ( v50 == &v53[2] )
          break;
        v53 += 3;
      }
      v51 = v123;
    }
    if ( v51 )
      j_j___libc_free_0(v51, (char *)v125 - (char *)v51);
    v123 = v52;
    v124 = (__m128i *)v52;
    v125 = &v52[v48];
  }
  if ( (_DWORD)v131 )
  {
    a4 = (__int64)v130;
    if ( *v130 && *v130 != -8 )
    {
      v60 = (__int64 **)v130;
    }
    else
    {
      v58 = v130 + 1;
      do
      {
        do
        {
          v59 = *v58;
          v60 = (__int64 **)v58++;
        }
        while ( !v59 );
      }
      while ( v59 == -8 );
    }
    v61 = (__int64 **)&v130[(unsigned int)v131];
    if ( v61 != v60 )
    {
      while ( 1 )
      {
        v62 = *v60;
        v63 = **v60;
        v64 = *v60 + 3;
        v133.m128i_i64[0] = (__int64)&v134;
        sub_16D9940(v133.m128i_i64, v64, (__int64)&v64[v63]);
        v65 = v124;
        if ( v124 == v125 )
        {
          sub_C99F30((__int64 *)&v123, v124, &v133, (const __m128i *)(v62 + 1));
          v66 = (__m128i *)v133.m128i_i64[0];
        }
        else
        {
          v66 = (__m128i *)v133.m128i_i64[0];
          if ( v124 )
          {
            v124->m128i_i64[0] = (__int64)v124[1].m128i_i64;
            if ( (__m128i *)v133.m128i_i64[0] == &v134 )
            {
              v65[1] = _mm_load_si128(&v134);
            }
            else
            {
              v65->m128i_i64[0] = v133.m128i_i64[0];
              v65[1].m128i_i64[0] = v134.m128i_i64[0];
            }
            v66 = &v134;
            v65->m128i_i64[1] = v133.m128i_i64[1];
            v133.m128i_i64[0] = (__int64)&v134;
            v133.m128i_i64[1] = 0;
            v134.m128i_i8[0] = 0;
            v65[2] = _mm_loadu_si128((const __m128i *)(v62 + 1));
            v65 = v124;
          }
          v124 = v65 + 3;
        }
        if ( v66 != &v134 )
          j_j___libc_free_0(v66, v134.m128i_i64[0] + 1);
        v67 = v60[1];
        v68 = v60 + 1;
        if ( v67 == (__int64 *)-8LL )
          break;
LABEL_82:
        if ( !v67 )
          goto LABEL_81;
        if ( v68 == v61 )
          goto LABEL_85;
        v60 = v68;
      }
      do
      {
LABEL_81:
        v67 = v68[1];
        ++v68;
      }
      while ( v67 == (__int64 *)-8LL );
      goto LABEL_82;
    }
  }
LABEL_85:
  v69 = v124;
  v70 = (__m128i *)v123;
  if ( v123 != v124 )
  {
    v71 = (char *)v124 - (char *)v123;
    _BitScanReverse64(&v72, 0xAAAAAAAAAAAAAAABLL * (v124 - v123));
    sub_16DC8B0((__int64)v123, (__int64)v124, (char *)(2LL * (int)(63 - (v72 ^ 0x3F))), a4, a5, (__int64)a6, v96);
    if ( v71 <= 768 )
    {
      sub_16D9DE0((__int64)v70, v69);
    }
    else
    {
      v73 = v70 + 48;
      sub_16D9DE0((__int64)v70, v70 + 48);
      if ( v69 != &v70[48] )
      {
        do
        {
          v74 = v73;
          v73 += 3;
          sub_16D9B90(v74);
        }
        while ( v69 != v73 );
      }
    }
  }
  v133.m128i_i8[0] = 7;
  v133.m128i_i64[1] = v121;
  v121 = 0;
  v134 = v122;
  v122 = 0u;
  sub_16DA620(v126, (__m128i *)"traceEvents", (__m128i *)0xB);
  v75 = sub_16F4840(&v127, v126);
  sub_16F2AA0(v75);
  sub_16F2270(v75, &v133);
  v76 = v126[0];
  if ( v126[0] )
  {
    if ( *(__m128i **)v126[0] != &v126[0][1] )
      j_j___libc_free_0(*(_QWORD *)v126[0], v126[0][1].m128i_i64[0] + 1);
    j_j___libc_free_0(v76, 32);
  }
  sub_16F2AA0(&v133);
  ++v127;
  v134 = v128;
  v133.m128i_i8[0] = 6;
  v133.m128i_i64[1] = 1;
  v135 = v129;
  v128 = 0u;
  v129 = 0;
  sub_16F34F0(a2, &v133);
  sub_16F2AA0(&v133);
  v77 = v124;
  v78 = v123;
  if ( v124 != v123 )
  {
    do
    {
      if ( (const __m128i *)v78->m128i_i64[0] != &v78[1] )
        j_j___libc_free_0(v78->m128i_i64[0], v78[1].m128i_i64[0] + 1);
      v78 += 3;
    }
    while ( v77 != v78 );
    v78 = v123;
  }
  if ( v78 )
    j_j___libc_free_0(v78, (char *)v125 - (char *)v78);
  v79 = (unsigned __int64)v130;
  if ( HIDWORD(v131) && (_DWORD)v131 )
  {
    v80 = 8LL * (unsigned int)v131;
    v81 = 0;
    do
    {
      v82 = *(_QWORD *)(v79 + v81);
      if ( v82 != -8 && v82 )
      {
        _libc_free(v82);
        v79 = (unsigned __int64)v130;
      }
      v81 += 8;
    }
    while ( v81 != v80 );
  }
  _libc_free(v79);
  v83 = v122.m128i_i64[0];
  v84 = v121;
  if ( v122.m128i_i64[0] != v121 )
  {
    do
    {
      v85 = v84;
      v84 += 40;
      sub_16F2AA0(v85);
    }
    while ( v83 != v84 );
    v84 = v121;
  }
  if ( v84 )
    j_j___libc_free_0(v84, v122.m128i_i64[1] - v84);
  sub_16DB620((__int64)&v127);
  result = j___libc_free_0(v128.m128i_i64[0]);
  if ( &_pthread_key_create )
    return pthread_mutex_unlock(&stru_4FA16A0);
  return result;
}
