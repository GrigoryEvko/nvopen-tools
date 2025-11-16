// Function: sub_356AF60
// Address: 0x356af60
//
unsigned __int64 __fastcall sub_356AF60(__int64 *a1, __int64 a2, char a3, unsigned int a4, unsigned int a5)
{
  __int64 v7; // rdi
  _BYTE *v8; // rax
  __int64 v9; // rax
  _WORD *v10; // rdx
  __int64 v11; // r8
  unsigned __int64 result; // rax
  __int64 *v13; // r15
  __int64 *v14; // r13
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  _WORD *v19; // rdx
  _BYTE *v20; // rax
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // r8
  __int64 v25; // r9
  const __m128i *v26; // rsi
  const __m128i *v27; // rdi
  const __m128i *v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rcx
  __m128i *v31; // rdx
  const __m128i *v32; // rax
  const __m128i *v33; // rcx
  __int64 v34; // rax
  __int64 v35; // rdx
  unsigned __int64 v36; // rdi
  __m128i *v37; // rdx
  const __m128i *v38; // rax
  unsigned __int64 v39; // rcx
  unsigned __int64 v40; // rax
  const char *v41; // rax
  size_t v42; // rdx
  __int64 v43; // r8
  __int64 v44; // r9
  _WORD *v45; // rdi
  unsigned __int8 *v46; // rsi
  unsigned __int64 v47; // rax
  __int64 v48; // r14
  unsigned __int64 v49; // r14
  __int64 v50; // rbx
  __int64 *v51; // rax
  __int64 v52; // rcx
  __int64 *v53; // rdx
  __int64 v54; // r15
  __int64 *v55; // rax
  __int64 v56; // rcx
  __int64 v57; // r8
  __int64 v58; // r9
  __int64 v59; // r8
  __int64 v60; // r9
  const __m128i *v61; // rcx
  unsigned __int64 v62; // r15
  __m128i *v63; // rax
  __int64 v64; // rcx
  const __m128i *v65; // rax
  const __m128i *v66; // rcx
  unsigned __int64 v67; // r15
  __int64 v68; // rax
  unsigned __int64 v69; // rdi
  __m128i *v70; // rdx
  __m128i *v71; // rax
  __int64 v72; // r15
  _WORD *v73; // rdi
  unsigned __int64 v74; // rax
  unsigned __int64 v75; // rdx
  __int64 *v76; // rsi
  const char *v77; // rax
  size_t v78; // rdx
  unsigned __int8 *v79; // rsi
  __int64 v80; // rax
  char v81; // dl
  unsigned __int64 v82; // rdx
  char v83; // si
  __int64 v84; // rax
  unsigned __int64 v85; // rax
  char v86; // si
  __int64 v87; // rax
  const __m128i *v88; // [rsp+0h] [rbp-440h]
  signed __int64 v89; // [rsp+0h] [rbp-440h]
  size_t v90; // [rsp+0h] [rbp-440h]
  size_t v91; // [rsp+8h] [rbp-438h]
  __int64 v93; // [rsp+18h] [rbp-428h]
  unsigned int v94; // [rsp+18h] [rbp-428h]
  unsigned int v96; // [rsp+28h] [rbp-418h]
  __int64 v98; // [rsp+30h] [rbp-410h] BYREF
  __int64 *v99; // [rsp+38h] [rbp-408h]
  unsigned int v100; // [rsp+40h] [rbp-400h]
  unsigned int v101; // [rsp+44h] [rbp-3FCh]
  char v102; // [rsp+4Ch] [rbp-3F4h]
  char v103[64]; // [rsp+50h] [rbp-3F0h] BYREF
  unsigned __int64 v104; // [rsp+90h] [rbp-3B0h] BYREF
  unsigned __int64 v105; // [rsp+98h] [rbp-3A8h]
  __int8 *v106; // [rsp+A0h] [rbp-3A0h]
  char v107[8]; // [rsp+B0h] [rbp-390h] BYREF
  unsigned __int64 v108; // [rsp+B8h] [rbp-388h]
  char v109; // [rsp+CCh] [rbp-374h]
  char v110[64]; // [rsp+D0h] [rbp-370h] BYREF
  unsigned __int64 v111; // [rsp+110h] [rbp-330h]
  unsigned __int64 v112; // [rsp+118h] [rbp-328h]
  __int64 v113; // [rsp+120h] [rbp-320h]
  char v114[8]; // [rsp+130h] [rbp-310h] BYREF
  unsigned __int64 v115; // [rsp+138h] [rbp-308h]
  char v116; // [rsp+14Ch] [rbp-2F4h]
  _BYTE v117[64]; // [rsp+150h] [rbp-2F0h] BYREF
  __m128i *v118; // [rsp+190h] [rbp-2B0h]
  __int64 v119; // [rsp+198h] [rbp-2A8h]
  __int8 *v120; // [rsp+1A0h] [rbp-2A0h]
  char v121[8]; // [rsp+1B0h] [rbp-290h] BYREF
  unsigned __int64 v122; // [rsp+1B8h] [rbp-288h]
  char v123; // [rsp+1CCh] [rbp-274h]
  _BYTE v124[64]; // [rsp+1D0h] [rbp-270h] BYREF
  unsigned __int64 v125; // [rsp+210h] [rbp-230h]
  unsigned __int64 i; // [rsp+218h] [rbp-228h]
  unsigned __int64 v127; // [rsp+220h] [rbp-220h]
  __m128i v128; // [rsp+230h] [rbp-210h] BYREF
  __int64 v129; // [rsp+240h] [rbp-200h] BYREF
  char v130; // [rsp+24Ch] [rbp-1F4h]
  const __m128i *v131; // [rsp+290h] [rbp-1B0h]
  const __m128i *v132; // [rsp+298h] [rbp-1A8h]
  char v133[8]; // [rsp+2A8h] [rbp-198h] BYREF
  unsigned __int64 v134; // [rsp+2B0h] [rbp-190h]
  char v135; // [rsp+2C4h] [rbp-17Ch]
  const __m128i *v136; // [rsp+308h] [rbp-138h]
  const __m128i *v137; // [rsp+310h] [rbp-130h]
  __m128i v138; // [rsp+320h] [rbp-120h] BYREF
  __int64 v139; // [rsp+330h] [rbp-110h] BYREF
  char v140; // [rsp+33Ch] [rbp-104h]
  const __m128i *v141; // [rsp+380h] [rbp-C0h]
  const __m128i *v142; // [rsp+388h] [rbp-B8h]
  char v143[8]; // [rsp+398h] [rbp-A8h] BYREF
  unsigned __int64 v144; // [rsp+3A0h] [rbp-A0h]
  char v145; // [rsp+3B4h] [rbp-8Ch]
  const __m128i *v146; // [rsp+3F8h] [rbp-48h]
  const __m128i *v147; // [rsp+400h] [rbp-40h]

  v96 = 2 * a4;
  if ( a3 )
  {
    v7 = sub_CB69B0(a2, 2 * a4);
    v8 = *(_BYTE **)(v7 + 32);
    if ( (unsigned __int64)v8 >= *(_QWORD *)(v7 + 24) )
    {
      v7 = sub_CB5D20(v7, 91);
    }
    else
    {
      *(_QWORD *)(v7 + 32) = v8 + 1;
      *v8 = 91;
    }
    v9 = sub_CB59D0(v7, a4);
    v10 = *(_WORD **)(v9 + 32);
    v11 = v9;
    if ( *(_QWORD *)(v9 + 24) - (_QWORD)v10 <= 1u )
    {
      v11 = sub_CB6200(v9, (unsigned __int8 *)"] ", 2u);
    }
    else
    {
      *v10 = 8285;
      *(_QWORD *)(v9 + 32) += 2LL;
    }
  }
  else
  {
    v11 = sub_CB69B0(a2, v96);
  }
  v93 = v11;
  sub_35677C0(&v138, a1);
  sub_CB6200(v93, (unsigned __int8 *)v138.m128i_i64[0], v138.m128i_u64[1]);
  if ( (__int64 *)v138.m128i_i64[0] != &v139 )
    j_j___libc_free_0(v138.m128i_u64[0]);
  result = *(_QWORD *)(a2 + 32);
  if ( result >= *(_QWORD *)(a2 + 24) )
  {
    result = sub_CB5D20(a2, 10);
  }
  else
  {
    *(_QWORD *)(a2 + 32) = result + 1;
    *(_BYTE *)result = 10;
  }
  if ( !a5 )
  {
    if ( a3 )
    {
      result = a1[5];
      v13 = (__int64 *)a1[6];
      if ( v13 != (__int64 *)result )
      {
        v94 = a4 + 1;
        goto LABEL_16;
      }
    }
    return result;
  }
  v18 = sub_CB69B0(a2, v96);
  v19 = *(_WORD **)(v18 + 32);
  if ( *(_QWORD *)(v18 + 24) - (_QWORD)v19 <= 1u )
  {
    sub_CB6200(v18, (unsigned __int8 *)"{\n", 2u);
  }
  else
  {
    *v19 = 2683;
    *(_QWORD *)(v18 + 32) += 2LL;
  }
  v94 = a4 + 1;
  sub_CB69B0(a2, v96 + 2);
  if ( a5 != 1 )
  {
    if ( a5 != 2 )
      goto LABEL_25;
    sub_356A570(&v138, a1);
    v26 = (const __m128i *)v117;
    v27 = (const __m128i *)v114;
    sub_C8CD80((__int64)v114, (__int64)v117, (__int64)&v138, v56, v57, v58);
    v61 = v142;
    v28 = v141;
    v118 = 0;
    v119 = 0;
    v120 = 0;
    v62 = (char *)v142 - (char *)v141;
    if ( v142 == v141 )
    {
      v63 = 0;
    }
    else
    {
      if ( v62 > 0x7FFFFFFFFFFFFFE0LL )
        goto LABEL_161;
      v63 = (__m128i *)sub_22077B0((char *)v142 - (char *)v141);
      v61 = v142;
      v28 = v141;
    }
    v118 = v63;
    v119 = (__int64)v63;
    v120 = &v63->m128i_i8[v62];
    if ( v28 == v61 )
    {
      v64 = (__int64)v63;
    }
    else
    {
      v64 = (__int64)v63->m128i_i64 + (char *)v61 - (char *)v28;
      do
      {
        if ( v63 )
        {
          *v63 = _mm_loadu_si128(v28);
          v63[1] = _mm_loadu_si128(v28 + 1);
        }
        v63 += 2;
        v28 += 2;
      }
      while ( (__m128i *)v64 != v63 );
    }
    v27 = (const __m128i *)v121;
    v119 = v64;
    v26 = (const __m128i *)v124;
    sub_C8CD80((__int64)v121, (__int64)v124, (__int64)v143, v64, v59, v60);
    v65 = v147;
    v66 = v146;
    v125 = 0;
    i = 0;
    v127 = 0;
    v67 = (char *)v147 - (char *)v146;
    if ( v147 == v146 )
    {
      v69 = 0;
      goto LABEL_82;
    }
    if ( v67 <= 0x7FFFFFFFFFFFFFE0LL )
    {
      v68 = sub_22077B0((char *)v147 - (char *)v146);
      v66 = v146;
      v69 = v68;
      v65 = v147;
LABEL_82:
      v125 = v69;
      i = v69;
      v127 = v69 + v67;
      if ( v66 == v65 )
      {
        v71 = (__m128i *)v69;
      }
      else
      {
        v70 = (__m128i *)v69;
        v71 = (__m128i *)(v69 + (char *)v65 - (char *)v66);
        do
        {
          if ( v70 )
          {
            *v70 = _mm_loadu_si128(v66);
            v70[1] = _mm_loadu_si128(v66 + 1);
          }
          v70 += 2;
          v66 += 2;
        }
        while ( v71 != v70 );
      }
      for ( i = (unsigned __int64)v71; ; v71 = (__m128i *)i )
      {
        v75 = (unsigned __int64)v118;
        if ( (__m128i *)(v119 - (_QWORD)v118) == (__m128i *)((char *)v71 - v69) )
        {
          if ( v118 == (__m128i *)v119 )
          {
LABEL_139:
            if ( v69 )
              j_j___libc_free_0(v69);
            if ( !v123 )
              _libc_free(v122);
            if ( v118 )
              j_j___libc_free_0((unsigned __int64)v118);
            if ( !v116 )
              _libc_free(v115);
            if ( v146 )
              j_j___libc_free_0((unsigned __int64)v146);
            if ( !v145 )
              _libc_free(v144);
            if ( v141 )
              j_j___libc_free_0((unsigned __int64)v141);
            if ( !v140 )
              _libc_free(v138.m128i_u64[1]);
            goto LABEL_25;
          }
          v85 = v69;
          while ( *(_QWORD *)v75 == *(_QWORD *)v85 )
          {
            v86 = *(_BYTE *)(v75 + 24);
            if ( v86 != *(_BYTE *)(v85 + 24) )
              break;
            if ( v86 )
            {
              if ( ((*(__int64 *)(v75 + 8) >> 1) & 3) != 0 )
              {
                if ( ((*(__int64 *)(v75 + 8) >> 1) & 3) != ((*(__int64 *)(v85 + 8) >> 1) & 3) )
                  break;
              }
              else if ( *(_QWORD *)(v75 + 16) != *(_QWORD *)(v85 + 16) )
              {
                break;
              }
            }
            v75 += 32LL;
            v85 += 32LL;
            if ( v119 == v75 )
              goto LABEL_139;
          }
        }
        v76 = *(__int64 **)(v119 - 32);
        if ( (*v76 & 4) != 0 )
          break;
        v77 = sub_2E31BC0(*v76 & 0xFFFFFFFFFFFFFFF8LL);
        v73 = *(_WORD **)(a2 + 32);
        v79 = (unsigned __int8 *)v77;
        v74 = *(_QWORD *)(a2 + 24) - (_QWORD)v73;
        if ( v78 > v74 )
        {
          v72 = sub_CB6200(a2, v79, v78);
LABEL_90:
          v73 = *(_WORD **)(v72 + 32);
          v74 = *(_QWORD *)(v72 + 24) - (_QWORD)v73;
          goto LABEL_91;
        }
        v72 = a2;
        if ( v78 )
        {
          v91 = v78;
          memcpy(v73, v79, v78);
          v80 = *(_QWORD *)(a2 + 24);
          v73 = (_WORD *)(v91 + *(_QWORD *)(a2 + 32));
          *(_QWORD *)(a2 + 32) = v73;
          v74 = v80 - (_QWORD)v73;
        }
LABEL_91:
        if ( v74 <= 1 )
        {
          sub_CB6200(v72, (unsigned __int8 *)", ", 2u);
        }
        else
        {
          *v73 = 8236;
          *(_QWORD *)(v72 + 32) += 2LL;
        }
        sub_3569CB0((__int64)v114);
        v69 = v125;
      }
      sub_35677C0(&v128, v76);
      v72 = sub_CB6200(a2, (unsigned __int8 *)v128.m128i_i64[0], v128.m128i_u64[1]);
      if ( (__int64 *)v128.m128i_i64[0] != &v129 )
        j_j___libc_free_0(v128.m128i_u64[0]);
      goto LABEL_90;
    }
LABEL_161:
    sub_4261EA(v27, v26, v28);
  }
  sub_356ACC0(&v128, a1);
  sub_C8CD80((__int64)&v98, (__int64)v103, (__int64)&v128, v21, v22, v23);
  v26 = v132;
  v27 = v131;
  v104 = 0;
  v105 = 0;
  v106 = 0;
  v28 = (const __m128i *)((char *)v132 - (char *)v131);
  if ( v132 == v131 )
  {
    v30 = 0;
  }
  else
  {
    if ( (unsigned __int64)v28 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_161;
    v88 = (const __m128i *)((char *)v132 - (char *)v131);
    v29 = sub_22077B0((char *)v132 - (char *)v131);
    v26 = v132;
    v27 = v131;
    v28 = v88;
    v30 = v29;
  }
  v104 = v30;
  v105 = v30;
  v106 = &v28->m128i_i8[v30];
  if ( v26 != v27 )
  {
    v31 = (__m128i *)v30;
    v32 = v27;
    do
    {
      if ( v31 )
      {
        *v31 = _mm_loadu_si128(v32);
        v24 = v32[1].m128i_i64[0];
        v31[1].m128i_i64[0] = v24;
      }
      v32 = (const __m128i *)((char *)v32 + 24);
      v31 = (__m128i *)((char *)v31 + 24);
    }
    while ( v32 != v26 );
    v30 += 8 * ((unsigned __int64)((char *)&v32[-2].m128i_u64[1] - (char *)v27) >> 3) + 24;
  }
  v105 = v30;
  v27 = (const __m128i *)v107;
  sub_C8CD80((__int64)v107, (__int64)v110, (__int64)v133, v30, v24, v25);
  v33 = v137;
  v26 = v136;
  v111 = 0;
  v112 = 0;
  v113 = 0;
  v28 = (const __m128i *)((char *)v137 - (char *)v136);
  if ( v137 == v136 )
  {
    v35 = 0;
    v36 = 0;
  }
  else
  {
    if ( (unsigned __int64)v28 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_161;
    v89 = (char *)v137 - (char *)v136;
    v34 = sub_22077B0((char *)v137 - (char *)v136);
    v33 = v137;
    v26 = v136;
    v35 = v89;
    v36 = v34;
  }
  v111 = v36;
  v113 = v36 + v35;
  v37 = (__m128i *)v36;
  v112 = v36;
  if ( v33 != v26 )
  {
    v38 = v26;
    do
    {
      if ( v37 )
      {
        *v37 = _mm_loadu_si128(v38);
        v37[1].m128i_i64[0] = v38[1].m128i_i64[0];
      }
      v38 = (const __m128i *)((char *)v38 + 24);
      v37 = (__m128i *)((char *)v37 + 24);
    }
    while ( v38 != v33 );
    v37 = (__m128i *)(v36 + 8 * ((unsigned __int64)((char *)&v38[-2].m128i_u64[1] - (char *)v26) >> 3) + 24);
  }
  v39 = v105;
  v40 = v104;
  v112 = (unsigned __int64)v37;
  if ( (__m128i *)(v105 - v104) == (__m128i *)((char *)v37 - v36) )
    goto LABEL_104;
  do
  {
LABEL_55:
    v41 = sub_2E31BC0(*(_QWORD *)(v39 - 24));
    v45 = *(_WORD **)(a2 + 32);
    v46 = (unsigned __int8 *)v41;
    v47 = *(_QWORD *)(a2 + 24) - (_QWORD)v45;
    if ( v47 < v42 )
    {
      v84 = sub_CB6200(a2, v46, v42);
      v45 = *(_WORD **)(v84 + 32);
      v48 = v84;
      if ( *(_QWORD *)(v84 + 24) - (_QWORD)v45 > 1u )
        goto LABEL_59;
    }
    else
    {
      v48 = a2;
      if ( v42 )
      {
        v90 = v42;
        memcpy(v45, v46, v42);
        v87 = *(_QWORD *)(a2 + 24);
        v45 = (_WORD *)(v90 + *(_QWORD *)(a2 + 32));
        *(_QWORD *)(a2 + 32) = v45;
        v47 = v87 - (_QWORD)v45;
      }
      if ( v47 > 1 )
      {
LABEL_59:
        *v45 = 8236;
        *(_QWORD *)(v48 + 32) += 2LL;
        goto LABEL_60;
      }
    }
    sub_CB6200(v48, (unsigned __int8 *)", ", 2u);
LABEL_60:
    v49 = v105;
    while ( 1 )
    {
      v50 = *(_QWORD *)(v49 - 24);
      if ( *(_BYTE *)(v49 - 8) )
        break;
      v51 = *(__int64 **)(v50 + 112);
      *(_BYTE *)(v49 - 8) = 1;
      *(_QWORD *)(v49 - 16) = v51;
      v52 = *(unsigned int *)(v50 + 120);
      if ( v51 != (__int64 *)(*(_QWORD *)(v50 + 112) + 8 * v52) )
        goto LABEL_63;
LABEL_69:
      v105 -= 24LL;
      v40 = v104;
      v49 = v105;
      if ( v105 == v104 )
      {
        v39 = v104;
        goto LABEL_103;
      }
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v52 = *(unsigned int *)(v50 + 120);
        v51 = *(__int64 **)(v49 - 16);
        if ( v51 == (__int64 *)(*(_QWORD *)(v50 + 112) + 8 * v52) )
          goto LABEL_69;
LABEL_63:
        v53 = v51 + 1;
        *(_QWORD *)(v49 - 16) = v51 + 1;
        v54 = *v51;
        if ( v102 )
          break;
LABEL_101:
        sub_C8CC70((__int64)&v98, v54, (__int64)v53, v52, v43, v44);
        if ( v81 )
          goto LABEL_102;
      }
      v55 = v99;
      v53 = &v99[v101];
      if ( v99 == v53 )
        break;
      while ( v54 != *v55 )
      {
        if ( v53 == ++v55 )
          goto LABEL_127;
      }
    }
LABEL_127:
    if ( v101 >= v100 )
      goto LABEL_101;
    ++v101;
    *v53 = v54;
    ++v98;
LABEL_102:
    v138.m128i_i64[0] = v54;
    LOBYTE(v139) = 0;
    sub_356AC80(&v104, &v138);
    v40 = v104;
    v39 = v105;
LABEL_103:
    v36 = v111;
  }
  while ( v39 - v40 != v112 - v111 );
LABEL_104:
  if ( v39 != v40 )
  {
    v82 = v36;
    while ( *(_QWORD *)v40 == *(_QWORD *)v82 )
    {
      v83 = *(_BYTE *)(v40 + 16);
      if ( v83 != *(_BYTE *)(v82 + 16) || v83 && *(_QWORD *)(v40 + 8) != *(_QWORD *)(v82 + 8) )
        break;
      v40 += 24LL;
      v82 += 24LL;
      if ( v40 == v39 )
        goto LABEL_111;
    }
    goto LABEL_55;
  }
LABEL_111:
  if ( v36 )
    j_j___libc_free_0(v36);
  if ( !v109 )
    _libc_free(v108);
  if ( v104 )
    j_j___libc_free_0(v104);
  if ( !v102 )
    _libc_free((unsigned __int64)v99);
  if ( v136 )
    j_j___libc_free_0((unsigned __int64)v136);
  if ( !v135 )
    _libc_free(v134);
  if ( v131 )
    j_j___libc_free_0((unsigned __int64)v131);
  if ( !v130 )
    _libc_free(v128.m128i_u64[1]);
LABEL_25:
  v20 = *(_BYTE **)(a2 + 32);
  if ( (unsigned __int64)v20 >= *(_QWORD *)(a2 + 24) )
  {
    sub_CB5D20(a2, 10);
  }
  else
  {
    *(_QWORD *)(a2 + 32) = v20 + 1;
    *v20 = 10;
  }
  if ( !a3 || (result = a1[5], v13 = (__int64 *)a1[6], v13 == (__int64 *)result) )
  {
LABEL_19:
    v16 = sub_CB69B0(a2, v96);
    v17 = *(_QWORD *)(v16 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(v16 + 24) - v17) <= 2 )
      return sub_CB6200(v16, "} \n", 3u);
    *(_BYTE *)(v17 + 2) = 10;
    *(_WORD *)v17 = 8317;
    *(_QWORD *)(v16 + 32) += 3LL;
    return 8317;
  }
LABEL_16:
  v14 = (__int64 *)result;
  do
  {
    v15 = *v14++;
    result = sub_356AF60(v15, a2, 1, v94, a5);
  }
  while ( v13 != v14 );
  if ( a5 )
    goto LABEL_19;
  return result;
}
