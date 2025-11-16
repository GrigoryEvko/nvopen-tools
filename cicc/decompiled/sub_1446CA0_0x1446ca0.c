// Function: sub_1446CA0
// Address: 0x1446ca0
//
unsigned __int64 __fastcall sub_1446CA0(__int64 *a1, __int64 a2, char a3, unsigned int a4, unsigned int a5)
{
  unsigned int v5; // r15d
  __int64 v9; // rsi
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
  unsigned __int64 v27; // rdx
  _BYTE *v28; // rsi
  __int64 v29; // rax
  __int64 v30; // rcx
  char v31; // si
  __int64 v32; // rax
  __int64 v33; // rcx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rdi
  __int64 v37; // rdx
  __int64 v38; // rax
  char v39; // si
  __int64 v40; // rax
  size_t v41; // rdx
  _WORD *v42; // rdi
  const char *v43; // rsi
  unsigned __int64 v44; // rax
  __int64 v45; // r8
  __int64 v46; // rcx
  __int64 v47; // rax
  char v48; // si
  unsigned __int64 v49; // rdi
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // rax
  __int64 v53; // rdi
  __int64 v54; // rdx
  __int64 v55; // rax
  char v56; // si
  _BYTE *v57; // r8
  __int64 v58; // rax
  unsigned __int64 v59; // rdi
  unsigned __int64 v60; // rdx
  _BYTE *v61; // rax
  char v62; // cl
  unsigned __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // r8
  _WORD *v66; // rdi
  unsigned __int64 v67; // rax
  __int64 v68; // rcx
  __int64 *v69; // rsi
  __int64 v70; // rax
  size_t v71; // rdx
  const char *v72; // rsi
  __int64 v73; // rax
  _WORD *v74; // rdx
  __int64 v75; // rax
  unsigned __int64 v76; // rsi
  char v77; // al
  char v78; // r8
  bool v79; // al
  __int64 v80; // rax
  _WORD *v81; // rdx
  __int64 v82; // [rsp+8h] [rbp-448h]
  _BYTE *v83; // [rsp+8h] [rbp-448h]
  __int64 v84; // [rsp+8h] [rbp-448h]
  size_t v85; // [rsp+8h] [rbp-448h]
  unsigned __int64 v86; // [rsp+10h] [rbp-440h]
  unsigned __int64 v87; // [rsp+10h] [rbp-440h]
  __int64 v88; // [rsp+10h] [rbp-440h]
  size_t v89; // [rsp+10h] [rbp-440h]
  __int64 *v91; // [rsp+18h] [rbp-438h]
  char v92[8]; // [rsp+20h] [rbp-430h] BYREF
  __int64 v93; // [rsp+28h] [rbp-428h]
  unsigned __int64 v94; // [rsp+30h] [rbp-420h]
  char v95[64]; // [rsp+48h] [rbp-408h] BYREF
  __int64 v96; // [rsp+88h] [rbp-3C8h]
  __int64 v97; // [rsp+90h] [rbp-3C0h]
  _BYTE *v98; // [rsp+98h] [rbp-3B8h]
  char v99[8]; // [rsp+A0h] [rbp-3B0h] BYREF
  __int64 v100; // [rsp+A8h] [rbp-3A8h]
  unsigned __int64 v101; // [rsp+B0h] [rbp-3A0h]
  _BYTE v102[64]; // [rsp+C8h] [rbp-388h] BYREF
  __int64 v103; // [rsp+108h] [rbp-348h]
  __int64 j; // [rsp+110h] [rbp-340h]
  __int64 v105; // [rsp+118h] [rbp-338h]
  char v106[8]; // [rsp+120h] [rbp-330h] BYREF
  __int64 v107; // [rsp+128h] [rbp-328h]
  unsigned __int64 v108; // [rsp+130h] [rbp-320h]
  _BYTE v109[64]; // [rsp+148h] [rbp-308h] BYREF
  __int64 v110; // [rsp+188h] [rbp-2C8h]
  __int64 v111; // [rsp+190h] [rbp-2C0h]
  unsigned __int64 v112; // [rsp+198h] [rbp-2B8h]
  char v113[8]; // [rsp+1A0h] [rbp-2B0h] BYREF
  __int64 v114; // [rsp+1A8h] [rbp-2A8h]
  unsigned __int64 v115; // [rsp+1B0h] [rbp-2A0h]
  char v116[64]; // [rsp+1C8h] [rbp-288h] BYREF
  unsigned __int64 v117; // [rsp+208h] [rbp-248h]
  unsigned __int64 i; // [rsp+210h] [rbp-240h]
  unsigned __int64 v119; // [rsp+218h] [rbp-238h]
  __m128i v120; // [rsp+220h] [rbp-230h] BYREF
  unsigned __int64 v121; // [rsp+230h] [rbp-220h] BYREF
  unsigned __int64 v122; // [rsp+288h] [rbp-1C8h]
  __int64 v123; // [rsp+290h] [rbp-1C0h]
  __int64 v124; // [rsp+298h] [rbp-1B8h]
  char v125[8]; // [rsp+2A0h] [rbp-1B0h] BYREF
  __int64 v126; // [rsp+2A8h] [rbp-1A8h]
  unsigned __int64 v127; // [rsp+2B0h] [rbp-1A0h]
  __int64 v128; // [rsp+308h] [rbp-148h]
  __int64 v129; // [rsp+310h] [rbp-140h]
  __int64 v130; // [rsp+318h] [rbp-138h]
  __m128i v131; // [rsp+320h] [rbp-130h] BYREF
  unsigned __int64 v132; // [rsp+330h] [rbp-120h] BYREF
  __int64 v133; // [rsp+388h] [rbp-C8h]
  __int64 v134; // [rsp+390h] [rbp-C0h]
  __int64 v135; // [rsp+398h] [rbp-B8h]
  char v136[8]; // [rsp+3A0h] [rbp-B0h] BYREF
  __int64 v137; // [rsp+3A8h] [rbp-A8h]
  unsigned __int64 v138; // [rsp+3B0h] [rbp-A0h]
  _BYTE *v139; // [rsp+408h] [rbp-48h]
  _BYTE *v140; // [rsp+410h] [rbp-40h]
  __int64 v141; // [rsp+418h] [rbp-38h]

  v5 = 2 * a4;
  v9 = 2 * a4;
  if ( a3 )
  {
    v11 = sub_16E8750(a2, v9);
    v12 = *(_BYTE **)(v11 + 24);
    if ( (unsigned __int64)v12 >= *(_QWORD *)(v11 + 16) )
    {
      v11 = sub_16E7DE0(v11, 91);
    }
    else
    {
      *(_QWORD *)(v11 + 24) = v12 + 1;
      *v12 = 91;
    }
    v13 = sub_16E7A90(v11, a4);
    v14 = *(_WORD **)(v13 + 24);
    v15 = v13;
    if ( *(_QWORD *)(v13 + 16) - (_QWORD)v14 <= 1u )
    {
      v15 = sub_16E7EE0(v13, "] ", 2);
    }
    else
    {
      *v14 = 8285;
      *(_QWORD *)(v13 + 24) += 2LL;
    }
  }
  else
  {
    v15 = sub_16E8750(a2, v9);
  }
  v82 = v15;
  sub_1442FC0(&v131, a1);
  sub_16E7EE0(v82, (const char *)v131.m128i_i64[0], v131.m128i_i64[1]);
  if ( (unsigned __int64 *)v131.m128i_i64[0] != &v132 )
    j_j___libc_free_0(v131.m128i_i64[0], v132 + 1);
  result = *(_QWORD *)(a2 + 24);
  if ( result >= *(_QWORD *)(a2 + 16) )
  {
    result = sub_16E7DE0(a2, 10);
  }
  else
  {
    *(_QWORD *)(a2 + 24) = result + 1;
    *(_BYTE *)result = 10;
  }
  if ( !a5 )
  {
    if ( a3 )
    {
      result = a1[5];
      v17 = a4 + 1;
      v91 = (__int64 *)a1[6];
      if ( (__int64 *)result != v91 )
        goto LABEL_14;
    }
    return result;
  }
  v23 = sub_16E8750(a2, v5);
  v24 = *(_WORD **)(v23 + 24);
  if ( *(_QWORD *)(v23 + 16) - (_QWORD)v24 <= 1u )
  {
    sub_16E7EE0(v23, "{\n", 2);
  }
  else
  {
    *v24 = 2683;
    *(_QWORD *)(v23 + 24) += 2LL;
  }
  v17 = a4 + 1;
  sub_16E8750(a2, v5 + 2);
  if ( a5 != 1 )
  {
    if ( a5 != 2 )
      goto LABEL_21;
    sub_1445F60(&v131, a1);
    v28 = v109;
    v25 = v106;
    sub_16CCCB0(v106, v109, &v131);
    v50 = v134;
    v51 = v133;
    v110 = 0;
    v111 = 0;
    v112 = 0;
    v27 = v134 - v133;
    if ( v134 == v133 )
    {
      v53 = 0;
    }
    else
    {
      if ( v27 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_157;
      v86 = v134 - v133;
      v52 = sub_22077B0(v134 - v133);
      v50 = v134;
      v51 = v133;
      v27 = v86;
      v53 = v52;
    }
    v110 = v53;
    v111 = v53;
    v112 = v53 + v27;
    if ( v50 != v51 )
    {
      v54 = v53;
      v55 = v51;
      do
      {
        if ( v54 )
        {
          *(_QWORD *)v54 = *(_QWORD *)v55;
          v56 = *(_BYTE *)(v55 + 32);
          *(_BYTE *)(v54 + 32) = v56;
          if ( v56 )
          {
            *(__m128i *)(v54 + 8) = _mm_loadu_si128((const __m128i *)(v55 + 8));
            *(_QWORD *)(v54 + 24) = *(_QWORD *)(v55 + 24);
          }
        }
        v55 += 40;
        v54 += 40;
      }
      while ( v50 != v55 );
      v53 += 8 * ((unsigned __int64)(v50 - 40 - v51) >> 3) + 40;
    }
    v111 = v53;
    v25 = v113;
    sub_16CCCB0(v113, v116, v136);
    v28 = v140;
    v57 = v139;
    v117 = 0;
    i = 0;
    v119 = 0;
    v27 = v140 - v139;
    if ( v140 == v139 )
    {
      v59 = 0;
      goto LABEL_97;
    }
    if ( v27 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v87 = v140 - v139;
      v58 = sub_22077B0(v140 - v139);
      v28 = v140;
      v57 = v139;
      v27 = v87;
      v59 = v58;
LABEL_97:
      v117 = v59;
      i = v59;
      v119 = v59 + v27;
      if ( v57 == v28 )
      {
        v63 = v59;
      }
      else
      {
        v60 = v59;
        v61 = v57;
        do
        {
          if ( v60 )
          {
            *(_QWORD *)v60 = *(_QWORD *)v61;
            v62 = v61[32];
            *(_BYTE *)(v60 + 32) = v62;
            if ( v62 )
            {
              *(__m128i *)(v60 + 8) = _mm_loadu_si128((const __m128i *)(v61 + 8));
              *(_QWORD *)(v60 + 24) = *((_QWORD *)v61 + 3);
            }
          }
          v61 += 40;
          v60 += 40LL;
        }
        while ( v61 != v28 );
        v63 = v59 + 8 * ((unsigned __int64)(v61 - 40 - v57) >> 3) + 40;
      }
      for ( i = v63; ; v63 = i )
      {
        v68 = v110;
        if ( v111 - v110 == v63 - v59 )
        {
          if ( v110 == v111 )
          {
LABEL_129:
            if ( v59 )
              j_j___libc_free_0(v59, v119 - v59);
            if ( v115 != v114 )
              _libc_free(v115);
            if ( v110 )
              j_j___libc_free_0(v110, v112 - v110);
            if ( v108 != v107 )
              _libc_free(v108);
            if ( v139 )
              j_j___libc_free_0(v139, v141 - (_QWORD)v139);
            if ( v138 != v137 )
              _libc_free(v138);
            if ( v133 )
              j_j___libc_free_0(v133, v135 - v133);
            v49 = v132;
            if ( v132 == v131.m128i_i64[1] )
              goto LABEL_21;
LABEL_83:
            _libc_free(v49);
            goto LABEL_21;
          }
          v76 = v59;
          while ( *(_QWORD *)v68 == *(_QWORD *)v76 )
          {
            v77 = *(_BYTE *)(v68 + 32);
            v78 = *(_BYTE *)(v76 + 32);
            if ( v77 && v78 )
            {
              if ( ((*(__int64 *)(v68 + 8) >> 1) & 3) != 0 )
                v79 = ((*(__int64 *)(v76 + 8) >> 1) & 3) == ((*(__int64 *)(v68 + 8) >> 1) & 3);
              else
                v79 = *(_DWORD *)(v68 + 24) == *(_DWORD *)(v76 + 24);
              if ( !v79 )
                break;
            }
            else if ( v77 != v78 )
            {
              break;
            }
            v68 += 40;
            v76 += 40LL;
            if ( v111 == v68 )
              goto LABEL_129;
          }
        }
        v69 = *(__int64 **)(v111 - 40);
        if ( (*v69 & 4) != 0 )
          break;
        v70 = sub_1649960(*v69 & 0xFFFFFFFFFFFFFFF8LL);
        v66 = *(_WORD **)(a2 + 24);
        v72 = (const char *)v70;
        v67 = *(_QWORD *)(a2 + 16) - (_QWORD)v66;
        if ( v71 > v67 )
        {
          v65 = sub_16E7EE0(a2, v72);
LABEL_107:
          v66 = *(_WORD **)(v65 + 24);
          v67 = *(_QWORD *)(v65 + 16) - (_QWORD)v66;
          goto LABEL_108;
        }
        v65 = a2;
        if ( v71 )
        {
          v89 = v71;
          memcpy(v66, v72, v71);
          v73 = *(_QWORD *)(a2 + 16);
          v74 = (_WORD *)(*(_QWORD *)(a2 + 24) + v89);
          v65 = a2;
          *(_QWORD *)(a2 + 24) = v74;
          v66 = v74;
          v67 = v73 - (_QWORD)v74;
        }
LABEL_108:
        if ( v67 <= 1 )
        {
          sub_16E7EE0(v65, ", ", 2);
        }
        else
        {
          *v66 = 8236;
          *(_QWORD *)(v65 + 24) += 2LL;
        }
        sub_1445BC0((__int64)v106);
        v59 = v117;
      }
      sub_1442FC0(&v120, v69);
      v64 = sub_16E7EE0(a2, (const char *)v120.m128i_i64[0], v120.m128i_i64[1]);
      v65 = v64;
      if ( (unsigned __int64 *)v120.m128i_i64[0] != &v121 )
      {
        v88 = v64;
        j_j___libc_free_0(v120.m128i_i64[0], v121 + 1);
        v65 = v88;
      }
      goto LABEL_107;
    }
LABEL_157:
    sub_4261EA(v25, v28, v27);
  }
  sub_14469E0(&v120, a1);
  v25 = v92;
  sub_16CCCB0(v92, v95, &v120);
  v26 = v123;
  v27 = v122;
  v96 = 0;
  v97 = 0;
  v98 = 0;
  v28 = (_BYTE *)(v123 - v122);
  if ( v123 == v122 )
  {
    v29 = 0;
  }
  else
  {
    if ( (unsigned __int64)v28 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_157;
    v83 = (_BYTE *)(v123 - v122);
    v29 = sub_22077B0(v123 - v122);
    v26 = v123;
    v27 = v122;
    v28 = v83;
  }
  v96 = v29;
  v97 = v29;
  v98 = &v28[v29];
  if ( v27 == v26 )
  {
    v30 = v29;
  }
  else
  {
    v30 = v29 + v26 - v27;
    do
    {
      if ( v29 )
      {
        *(_QWORD *)v29 = *(_QWORD *)v27;
        v31 = *(_BYTE *)(v27 + 24);
        *(_BYTE *)(v29 + 24) = v31;
        if ( v31 )
          *(__m128i *)(v29 + 8) = _mm_loadu_si128((const __m128i *)(v27 + 8));
      }
      v29 += 32;
      v27 += 32LL;
    }
    while ( v29 != v30 );
  }
  v25 = v99;
  v97 = v30;
  v28 = v102;
  sub_16CCCB0(v99, v102, v125);
  v32 = v129;
  v33 = v128;
  v103 = 0;
  j = 0;
  v105 = 0;
  v27 = v129 - v128;
  if ( v129 == v128 )
  {
    v35 = 0;
    v36 = 0;
  }
  else
  {
    if ( v27 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_157;
    v84 = v129 - v128;
    v34 = sub_22077B0(v129 - v128);
    v33 = v128;
    v35 = v84;
    v36 = v34;
    v32 = v129;
  }
  v103 = v36;
  j = v36;
  v105 = v36 + v35;
  if ( v33 == v32 )
  {
    v38 = v36;
  }
  else
  {
    v37 = v36;
    v38 = v36 + v32 - v33;
    do
    {
      if ( v37 )
      {
        *(_QWORD *)v37 = *(_QWORD *)v33;
        v39 = *(_BYTE *)(v33 + 24);
        *(_BYTE *)(v37 + 24) = v39;
        if ( v39 )
          *(__m128i *)(v37 + 8) = _mm_loadu_si128((const __m128i *)(v33 + 8));
      }
      v37 += 32;
      v33 += 32;
    }
    while ( v37 != v38 );
  }
  for ( j = v38; ; v38 = j )
  {
    v46 = v96;
    if ( v97 - v96 != v38 - v36 )
      goto LABEL_54;
    if ( v96 == v97 )
      break;
    v47 = v36;
    while ( *(_QWORD *)v46 == *(_QWORD *)v47 )
    {
      v48 = *(_BYTE *)(v47 + 24);
      if ( *(_BYTE *)(v46 + 24) )
      {
        if ( !v48 || *(_DWORD *)(v46 + 16) != *(_DWORD *)(v47 + 16) )
          break;
      }
      else if ( v48 )
      {
        break;
      }
      v46 += 32;
      v47 += 32;
      if ( v97 == v46 )
        goto LABEL_68;
    }
LABEL_54:
    v40 = sub_1649960(*(_QWORD *)(v97 - 32));
    v42 = *(_WORD **)(a2 + 24);
    v43 = (const char *)v40;
    v44 = *(_QWORD *)(a2 + 16) - (_QWORD)v42;
    if ( v44 < v41 )
    {
      v75 = sub_16E7EE0(a2, v43);
      v42 = *(_WORD **)(v75 + 24);
      v45 = v75;
      v44 = *(_QWORD *)(v75 + 16) - (_QWORD)v42;
    }
    else
    {
      v45 = a2;
      if ( v41 )
      {
        v85 = v41;
        memcpy(v42, v43, v41);
        v80 = *(_QWORD *)(a2 + 16);
        v81 = (_WORD *)(*(_QWORD *)(a2 + 24) + v85);
        v45 = a2;
        *(_QWORD *)(a2 + 24) = v81;
        v42 = v81;
        v44 = v80 - (_QWORD)v81;
      }
    }
    if ( v44 <= 1 )
    {
      sub_16E7EE0(v45, ", ", 2);
    }
    else
    {
      *v42 = 8236;
      *(_QWORD *)(v45 + 24) += 2LL;
    }
    sub_1446890((__int64)v92);
    v36 = v103;
  }
LABEL_68:
  if ( v36 )
    j_j___libc_free_0(v36, v105 - v36);
  if ( v101 != v100 )
    _libc_free(v101);
  if ( v96 )
    j_j___libc_free_0(v96, &v98[-v96]);
  if ( v94 != v93 )
    _libc_free(v94);
  if ( v128 )
    j_j___libc_free_0(v128, v130 - v128);
  if ( v127 != v126 )
    _libc_free(v127);
  if ( v122 )
    j_j___libc_free_0(v122, v124 - v122);
  v49 = v121;
  if ( v121 != v120.m128i_i64[1] )
    goto LABEL_83;
LABEL_21:
  v20 = *(_BYTE **)(a2 + 24);
  if ( (unsigned __int64)v20 >= *(_QWORD *)(a2 + 16) )
  {
    sub_16E7DE0(a2, 10);
  }
  else
  {
    *(_QWORD *)(a2 + 24) = v20 + 1;
    *v20 = 10;
  }
  if ( !a3 || (result = a1[5], v91 = (__int64 *)a1[6], v91 == (__int64 *)result) )
  {
LABEL_24:
    v21 = sub_16E8750(a2, v5);
    v22 = *(_QWORD *)(v21 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(v21 + 16) - v22) <= 2 )
      return sub_16E7EE0(v21, "} \n", 3);
    *(_BYTE *)(v22 + 2) = 10;
    *(_WORD *)v22 = 8317;
    *(_QWORD *)(v21 + 24) += 3LL;
    return 8317;
  }
LABEL_14:
  v18 = (__int64 *)result;
  do
  {
    v19 = *v18++;
    result = sub_1446CA0(v19, a2, 1, v17, a5);
  }
  while ( v91 != v18 );
  if ( a5 )
    goto LABEL_24;
  return result;
}
