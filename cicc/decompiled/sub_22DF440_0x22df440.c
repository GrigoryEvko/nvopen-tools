// Function: sub_22DF440
// Address: 0x22df440
//
unsigned __int64 __fastcall sub_22DF440(__int64 *a1, __int64 a2, char a3, unsigned int a4, unsigned int a5)
{
  unsigned int v5; // r15d
  unsigned int v9; // esi
  __int64 v11; // rdi
  _BYTE *v12; // rax
  __int64 v13; // rax
  _WORD *v14; // rdx
  __int64 v15; // r8
  unsigned __int64 result; // rax
  unsigned int v17; // r13d
  __int64 *v18; // r14
  __int64 v19; // rdi
  _BYTE *v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  _WORD *v24; // rdx
  char *v25; // rdi
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // r8
  __int64 v30; // r9
  const __m128i *v31; // rcx
  const __m128i *v32; // rdx
  _BYTE *v33; // rsi
  __m128i *v34; // rax
  __int64 v35; // rcx
  const __m128i *v36; // rax
  const __m128i *v37; // rcx
  __int64 v38; // rax
  __int64 v39; // rdx
  unsigned __int64 v40; // rdi
  __m128i *v41; // rdx
  __m128i *v42; // rax
  const char *v43; // rax
  size_t v44; // rdx
  _WORD *v45; // rdi
  unsigned __int8 *v46; // rsi
  unsigned __int64 v47; // rax
  __int64 v48; // r8
  unsigned __int64 v49; // rcx
  unsigned __int64 v50; // rax
  char v51; // si
  __int64 v52; // rcx
  __int64 v53; // r8
  __int64 v54; // r9
  __int64 v55; // r9
  __int64 v56; // rcx
  __int64 v57; // r8
  __int64 v58; // rax
  unsigned __int64 v59; // rdi
  __m128i *v60; // rdx
  const __m128i *v61; // rax
  const __m128i *v62; // rcx
  unsigned __int64 v63; // r8
  __int64 v64; // rax
  unsigned __int64 v65; // rdi
  __m128i *v66; // rdx
  const __m128i *v67; // rax
  unsigned __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // r8
  _WORD *v71; // rdi
  unsigned __int64 v72; // rax
  unsigned __int64 v73; // rcx
  __int64 *v74; // rsi
  const char *v75; // rax
  size_t v76; // rdx
  unsigned __int8 *v77; // rsi
  __int64 v78; // rax
  __int64 v79; // rax
  unsigned __int64 v80; // rax
  char v81; // si
  __int64 v82; // rax
  _WORD *v83; // rdx
  __int64 v84; // [rsp+8h] [rbp-428h]
  _BYTE *v85; // [rsp+8h] [rbp-428h]
  signed __int64 v86; // [rsp+8h] [rbp-428h]
  size_t v87; // [rsp+8h] [rbp-428h]
  const __m128i *v88; // [rsp+10h] [rbp-420h]
  const __m128i *v89; // [rsp+10h] [rbp-420h]
  __int64 v90; // [rsp+10h] [rbp-420h]
  size_t v91; // [rsp+10h] [rbp-420h]
  __int64 *v93; // [rsp+18h] [rbp-418h]
  char v94[8]; // [rsp+20h] [rbp-410h] BYREF
  unsigned __int64 v95; // [rsp+28h] [rbp-408h]
  char v96; // [rsp+3Ch] [rbp-3F4h]
  char v97[64]; // [rsp+40h] [rbp-3F0h] BYREF
  __m128i *v98; // [rsp+80h] [rbp-3B0h]
  __int64 v99; // [rsp+88h] [rbp-3A8h]
  _BYTE *v100; // [rsp+90h] [rbp-3A0h]
  char v101[8]; // [rsp+A0h] [rbp-390h] BYREF
  unsigned __int64 v102; // [rsp+A8h] [rbp-388h]
  char v103; // [rsp+BCh] [rbp-374h]
  _BYTE v104[64]; // [rsp+C0h] [rbp-370h] BYREF
  unsigned __int64 v105; // [rsp+100h] [rbp-330h]
  unsigned __int64 j; // [rsp+108h] [rbp-328h]
  __int64 v107; // [rsp+110h] [rbp-320h]
  char v108[8]; // [rsp+120h] [rbp-310h] BYREF
  unsigned __int64 v109; // [rsp+128h] [rbp-308h]
  char v110; // [rsp+13Ch] [rbp-2F4h]
  _BYTE v111[64]; // [rsp+140h] [rbp-2F0h] BYREF
  unsigned __int64 v112; // [rsp+180h] [rbp-2B0h]
  unsigned __int64 v113; // [rsp+188h] [rbp-2A8h]
  __int8 *v114; // [rsp+190h] [rbp-2A0h]
  char v115[8]; // [rsp+1A0h] [rbp-290h] BYREF
  unsigned __int64 v116; // [rsp+1A8h] [rbp-288h]
  char v117; // [rsp+1BCh] [rbp-274h]
  _BYTE v118[64]; // [rsp+1C0h] [rbp-270h] BYREF
  unsigned __int64 v119; // [rsp+200h] [rbp-230h]
  unsigned __int64 i; // [rsp+208h] [rbp-228h]
  __int8 *v121; // [rsp+210h] [rbp-220h]
  __m128i v122; // [rsp+220h] [rbp-210h] BYREF
  __int64 v123; // [rsp+230h] [rbp-200h] BYREF
  char v124; // [rsp+23Ch] [rbp-1F4h]
  const __m128i *v125; // [rsp+280h] [rbp-1B0h]
  const __m128i *v126; // [rsp+288h] [rbp-1A8h]
  char v127[8]; // [rsp+298h] [rbp-198h] BYREF
  unsigned __int64 v128; // [rsp+2A0h] [rbp-190h]
  char v129; // [rsp+2B4h] [rbp-17Ch]
  const __m128i *v130; // [rsp+2F8h] [rbp-138h]
  const __m128i *v131; // [rsp+300h] [rbp-130h]
  __m128i v132; // [rsp+310h] [rbp-120h] BYREF
  __int64 v133; // [rsp+320h] [rbp-110h] BYREF
  char v134; // [rsp+32Ch] [rbp-104h]
  unsigned __int64 v135; // [rsp+370h] [rbp-C0h]
  __int64 v136; // [rsp+378h] [rbp-B8h]
  char v137[8]; // [rsp+388h] [rbp-A8h] BYREF
  unsigned __int64 v138; // [rsp+390h] [rbp-A0h]
  char v139; // [rsp+3A4h] [rbp-8Ch]
  const __m128i *v140; // [rsp+3E8h] [rbp-48h]
  const __m128i *v141; // [rsp+3F0h] [rbp-40h]

  v5 = 2 * a4;
  v9 = 2 * a4;
  if ( a3 )
  {
    v11 = sub_CB69B0(a2, v9);
    v12 = *(_BYTE **)(v11 + 32);
    if ( (unsigned __int64)v12 >= *(_QWORD *)(v11 + 24) )
    {
      v11 = sub_CB5D20(v11, 91);
    }
    else
    {
      *(_QWORD *)(v11 + 32) = v12 + 1;
      *v12 = 91;
    }
    v13 = sub_CB59D0(v11, a4);
    v14 = *(_WORD **)(v13 + 32);
    v15 = v13;
    if ( *(_QWORD *)(v13 + 24) - (_QWORD)v14 <= 1u )
    {
      v15 = sub_CB6200(v13, (unsigned __int8 *)"] ", 2u);
    }
    else
    {
      *v14 = 8285;
      *(_QWORD *)(v13 + 32) += 2LL;
    }
  }
  else
  {
    v15 = sub_CB69B0(a2, v9);
  }
  v84 = v15;
  sub_22DAE20(&v132, a1);
  sub_CB6200(v84, (unsigned __int8 *)v132.m128i_i64[0], v132.m128i_u64[1]);
  if ( (__int64 *)v132.m128i_i64[0] != &v133 )
    j_j___libc_free_0(v132.m128i_u64[0]);
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
      v17 = a4 + 1;
      v93 = (__int64 *)a1[6];
      if ( v93 != (__int64 *)result )
        goto LABEL_14;
    }
    return result;
  }
  v23 = sub_CB69B0(a2, v5);
  v24 = *(_WORD **)(v23 + 32);
  if ( *(_QWORD *)(v23 + 24) - (_QWORD)v24 <= 1u )
  {
    sub_CB6200(v23, (unsigned __int8 *)"{\n", 2u);
  }
  else
  {
    *v24 = 2683;
    *(_QWORD *)(v23 + 32) += 2LL;
  }
  v17 = a4 + 1;
  sub_CB69B0(a2, v5 + 2);
  if ( a5 != 1 )
  {
    if ( a5 != 2 )
      goto LABEL_21;
    sub_22DEC60(&v132, a1);
    v33 = v111;
    v25 = v108;
    sub_C8CD80((__int64)v108, (__int64)v111, (__int64)&v132, v52, v53, v54);
    v56 = v136;
    v57 = v135;
    v112 = 0;
    v113 = 0;
    v114 = 0;
    v32 = (const __m128i *)(v136 - v135);
    if ( v136 == v135 )
    {
      v59 = 0;
    }
    else
    {
      if ( (unsigned __int64)v32 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_149;
      v88 = (const __m128i *)(v136 - v135);
      v58 = sub_22077B0(v136 - v135);
      v56 = v136;
      v57 = v135;
      v32 = v88;
      v59 = v58;
    }
    v112 = v59;
    v113 = v59;
    v114 = &v32->m128i_i8[v59];
    if ( v57 != v56 )
    {
      v60 = (__m128i *)v59;
      v61 = (const __m128i *)v57;
      do
      {
        if ( v60 )
        {
          *v60 = _mm_loadu_si128(v61);
          v60[1] = _mm_loadu_si128(v61 + 1);
          v60[2].m128i_i64[0] = v61[2].m128i_i64[0];
        }
        v61 = (const __m128i *)((char *)v61 + 40);
        v60 = (__m128i *)((char *)v60 + 40);
      }
      while ( v61 != (const __m128i *)v56 );
      v59 += 8 * (((unsigned __int64)&v61[-3].m128i_u64[1] - v57) >> 3) + 40;
    }
    v113 = v59;
    v25 = v115;
    v33 = v118;
    sub_C8CD80((__int64)v115, (__int64)v118, (__int64)v137, v56, v57, v55);
    v62 = v141;
    v63 = (unsigned __int64)v140;
    v119 = 0;
    i = 0;
    v121 = 0;
    v32 = (const __m128i *)((char *)v141 - (char *)v140);
    if ( v141 == v140 )
    {
      v65 = 0;
      goto LABEL_94;
    }
    if ( (unsigned __int64)v32 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v89 = (const __m128i *)((char *)v141 - (char *)v140);
      v64 = sub_22077B0((char *)v141 - (char *)v140);
      v62 = v141;
      v63 = (unsigned __int64)v140;
      v32 = v89;
      v65 = v64;
LABEL_94:
      v119 = v65;
      i = v65;
      v121 = &v32->m128i_i8[v65];
      if ( v62 == (const __m128i *)v63 )
      {
        v68 = v65;
      }
      else
      {
        v66 = (__m128i *)v65;
        v67 = (const __m128i *)v63;
        do
        {
          if ( v66 )
          {
            *v66 = _mm_loadu_si128(v67);
            v66[1] = _mm_loadu_si128(v67 + 1);
            v66[2].m128i_i64[0] = v67[2].m128i_i64[0];
          }
          v67 = (const __m128i *)((char *)v67 + 40);
          v66 = (__m128i *)((char *)v66 + 40);
        }
        while ( v67 != v62 );
        v68 = v65 + 8 * (((unsigned __int64)&v67[-3].m128i_u64[1] - v63) >> 3) + 40;
      }
      for ( i = v68; ; v68 = i )
      {
        v73 = v112;
        if ( v113 - v112 == v68 - v65 )
        {
          if ( v112 == v113 )
          {
LABEL_124:
            if ( v65 )
              j_j___libc_free_0(v65);
            if ( !v117 )
              _libc_free(v116);
            if ( v112 )
              j_j___libc_free_0(v112);
            if ( !v110 )
              _libc_free(v109);
            if ( v140 )
              j_j___libc_free_0((unsigned __int64)v140);
            if ( !v139 )
              _libc_free(v138);
            if ( v135 )
              j_j___libc_free_0(v135);
            if ( !v134 )
              _libc_free(v132.m128i_u64[1]);
            goto LABEL_21;
          }
          v80 = v65;
          while ( *(_QWORD *)v73 == *(_QWORD *)v80 )
          {
            v81 = *(_BYTE *)(v73 + 32);
            if ( v81 != *(_BYTE *)(v80 + 32) )
              break;
            if ( v81 )
            {
              if ( ((*(__int64 *)(v73 + 8) >> 1) & 3) != 0 )
              {
                if ( ((*(__int64 *)(v73 + 8) >> 1) & 3) != ((*(__int64 *)(v80 + 8) >> 1) & 3) )
                  break;
              }
              else if ( *(_DWORD *)(v73 + 24) != *(_DWORD *)(v80 + 24) )
              {
                break;
              }
            }
            v73 += 40LL;
            v80 += 40LL;
            if ( v113 == v73 )
              goto LABEL_124;
          }
        }
        v74 = *(__int64 **)(v113 - 40);
        if ( (*v74 & 4) != 0 )
          break;
        v75 = sub_BD5D20(*v74 & 0xFFFFFFFFFFFFFFF8LL);
        v71 = *(_WORD **)(a2 + 32);
        v77 = (unsigned __int8 *)v75;
        v72 = *(_QWORD *)(a2 + 24) - (_QWORD)v71;
        if ( v76 > v72 )
        {
          v70 = sub_CB6200(a2, v77, v76);
LABEL_103:
          v71 = *(_WORD **)(v70 + 32);
          v72 = *(_QWORD *)(v70 + 24) - (_QWORD)v71;
          goto LABEL_104;
        }
        v70 = a2;
        if ( v76 )
        {
          v91 = v76;
          memcpy(v71, v77, v76);
          v78 = *(_QWORD *)(a2 + 24);
          v70 = a2;
          v71 = (_WORD *)(v91 + *(_QWORD *)(a2 + 32));
          *(_QWORD *)(a2 + 32) = v71;
          v72 = v78 - (_QWORD)v71;
        }
LABEL_104:
        if ( v72 <= 1 )
        {
          sub_CB6200(v70, (unsigned __int8 *)", ", 2u);
        }
        else
        {
          *v71 = 8236;
          *(_QWORD *)(v70 + 32) += 2LL;
        }
        sub_22DE060((__int64)v108);
        v65 = v119;
      }
      sub_22DAE20(&v122, v74);
      v69 = sub_CB6200(a2, (unsigned __int8 *)v122.m128i_i64[0], v122.m128i_u64[1]);
      v70 = v69;
      if ( (__int64 *)v122.m128i_i64[0] != &v123 )
      {
        v90 = v69;
        j_j___libc_free_0(v122.m128i_u64[0]);
        v70 = v90;
      }
      goto LABEL_103;
    }
LABEL_149:
    sub_4261EA(v25, v33, v32);
  }
  sub_22DD930(&v122, a1);
  v25 = v94;
  sub_C8CD80((__int64)v94, (__int64)v97, (__int64)&v122, v26, v27, v28);
  v31 = v126;
  v32 = v125;
  v98 = 0;
  v99 = 0;
  v100 = 0;
  v33 = (_BYTE *)((char *)v126 - (char *)v125);
  if ( v126 == v125 )
  {
    v34 = 0;
  }
  else
  {
    if ( (unsigned __int64)v33 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_149;
    v85 = (_BYTE *)((char *)v126 - (char *)v125);
    v34 = (__m128i *)sub_22077B0((char *)v126 - (char *)v125);
    v31 = v126;
    v32 = v125;
    v33 = v85;
  }
  v98 = v34;
  v99 = (__int64)v34;
  v100 = &v33[(_QWORD)v34];
  if ( v31 == v32 )
  {
    v35 = (__int64)v34;
  }
  else
  {
    v35 = (__int64)v34->m128i_i64 + (char *)v31 - (char *)v32;
    do
    {
      if ( v34 )
      {
        *v34 = _mm_loadu_si128(v32);
        v34[1] = _mm_loadu_si128(v32 + 1);
      }
      v34 += 2;
      v32 += 2;
    }
    while ( v34 != (__m128i *)v35 );
  }
  v25 = v101;
  v99 = v35;
  v33 = v104;
  sub_C8CD80((__int64)v101, (__int64)v104, (__int64)v127, v35, v29, v30);
  v36 = v131;
  v37 = v130;
  v105 = 0;
  j = 0;
  v107 = 0;
  v32 = (const __m128i *)((char *)v131 - (char *)v130);
  if ( v131 == v130 )
  {
    v39 = 0;
    v40 = 0;
  }
  else
  {
    if ( (unsigned __int64)v32 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_149;
    v86 = (char *)v131 - (char *)v130;
    v38 = sub_22077B0((char *)v131 - (char *)v130);
    v37 = v130;
    v39 = v86;
    v40 = v38;
    v36 = v131;
  }
  v105 = v40;
  j = v40;
  v107 = v40 + v39;
  if ( v37 == v36 )
  {
    v42 = (__m128i *)v40;
  }
  else
  {
    v41 = (__m128i *)v40;
    v42 = (__m128i *)(v40 + (char *)v36 - (char *)v37);
    do
    {
      if ( v41 )
      {
        *v41 = _mm_loadu_si128(v37);
        v41[1] = _mm_loadu_si128(v37 + 1);
      }
      v41 += 2;
      v37 += 2;
    }
    while ( v41 != v42 );
  }
  for ( j = (unsigned __int64)v42; ; v42 = (__m128i *)j )
  {
    v49 = (unsigned __int64)v98;
    if ( (__m128i *)(v99 - (_QWORD)v98) != (__m128i *)((char *)v42 - v40) )
      goto LABEL_52;
    if ( v98 == (__m128i *)v99 )
      break;
    v50 = v40;
    while ( *(_QWORD *)v49 == *(_QWORD *)v50 )
    {
      v51 = *(_BYTE *)(v49 + 24);
      if ( v51 != *(_BYTE *)(v50 + 24) || v51 && *(_DWORD *)(v49 + 16) != *(_DWORD *)(v50 + 16) )
        break;
      v49 += 32LL;
      v50 += 32LL;
      if ( v99 == v49 )
        goto LABEL_66;
    }
LABEL_52:
    v43 = sub_BD5D20(*(_QWORD *)(v99 - 32));
    v45 = *(_WORD **)(a2 + 32);
    v46 = (unsigned __int8 *)v43;
    v47 = *(_QWORD *)(a2 + 24) - (_QWORD)v45;
    if ( v47 < v44 )
    {
      v79 = sub_CB6200(a2, v46, v44);
      v45 = *(_WORD **)(v79 + 32);
      v48 = v79;
      v47 = *(_QWORD *)(v79 + 24) - (_QWORD)v45;
    }
    else
    {
      v48 = a2;
      if ( v44 )
      {
        v87 = v44;
        memcpy(v45, v46, v44);
        v82 = *(_QWORD *)(a2 + 24);
        v83 = (_WORD *)(*(_QWORD *)(a2 + 32) + v87);
        v48 = a2;
        *(_QWORD *)(a2 + 32) = v83;
        v45 = v83;
        v47 = v82 - (_QWORD)v83;
      }
    }
    if ( v47 <= 1 )
    {
      sub_CB6200(v48, (unsigned __int8 *)", ", 2u);
    }
    else
    {
      *v45 = 8236;
      *(_QWORD *)(v48 + 32) += 2LL;
    }
    sub_22DDBD0((__int64)v94);
    v40 = v105;
  }
LABEL_66:
  if ( v40 )
    j_j___libc_free_0(v40);
  if ( !v103 )
    _libc_free(v102);
  if ( v98 )
    j_j___libc_free_0((unsigned __int64)v98);
  if ( !v96 )
    _libc_free(v95);
  if ( v130 )
    j_j___libc_free_0((unsigned __int64)v130);
  if ( !v129 )
    _libc_free(v128);
  if ( v125 )
    j_j___libc_free_0((unsigned __int64)v125);
  if ( !v124 )
    _libc_free(v122.m128i_u64[1]);
LABEL_21:
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
  if ( !a3 || (result = a1[5], v93 = (__int64 *)a1[6], (__int64 *)result == v93) )
  {
LABEL_24:
    v21 = sub_CB69B0(a2, v5);
    v22 = *(_QWORD *)(v21 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(v21 + 24) - v22) <= 2 )
      return sub_CB6200(v21, "} \n", 3u);
    *(_BYTE *)(v22 + 2) = 10;
    *(_WORD *)v22 = 8317;
    *(_QWORD *)(v21 + 32) += 3LL;
    return 8317;
  }
LABEL_14:
  v18 = (__int64 *)result;
  do
  {
    v19 = *v18++;
    result = sub_22DF440(v19, a2, 1, v17, a5);
  }
  while ( v93 != v18 );
  if ( a5 )
    goto LABEL_24;
  return result;
}
