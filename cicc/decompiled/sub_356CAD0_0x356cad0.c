// Function: sub_356CAD0
// Address: 0x356cad0
//
void __fastcall sub_356CAD0(_QWORD *a1, unsigned __int64 *a2, char a3)
{
  unsigned __int64 *v6; // rsi
  _BYTE *v7; // rsi
  char *v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r8
  __int64 v13; // r9
  const __m128i *v14; // rcx
  const __m128i *v15; // rdx
  unsigned __int64 v16; // r15
  __m128i *v17; // rax
  __int64 v18; // rcx
  const __m128i *v19; // rax
  const __m128i *v20; // rcx
  unsigned __int64 v21; // r15
  __int64 v22; // rax
  unsigned __int64 v23; // rdi
  __m128i *v24; // rdx
  __m128i *v25; // rax
  __int64 v26; // rsi
  unsigned __int64 v27; // rcx
  unsigned __int64 v28; // rax
  char v29; // si
  __int64 *v31; // r15
  __int64 *v32; // r13
  unsigned __int64 *v33; // rsi
  unsigned __int64 *v34; // r14
  unsigned __int64 *v35; // rsi
  __int64 *v36; // r14
  __int64 *v37; // rbx
  unsigned __int64 v38; // r13
  unsigned __int64 *v39; // r14
  unsigned __int64 *v40; // rbx
  __int64 *v41; // rdi
  __int64 v42; // r8
  __int64 v43; // rsi
  unsigned __int64 v44; // r10
  unsigned __int64 v45; // r9
  unsigned __int64 v46; // r13
  _QWORD *v47; // rax
  char *v48; // rsi
  _QWORD *v49; // rcx
  __int64 v50; // r12
  char *v51; // rax
  __int64 *v52; // r14
  __int64 v53; // rcx
  unsigned __int64 v54; // r13
  unsigned __int64 v55; // rax
  unsigned __int64 v56; // r12
  unsigned __int64 *v57; // rbx
  unsigned __int64 v58; // r12
  unsigned __int64 v59; // rsi
  unsigned __int64 v60; // r14
  bool v61; // cf
  unsigned __int64 v62; // r9
  unsigned __int64 v63; // r13
  __int64 *v64; // rax
  __int64 *v65; // rsi
  __int64 *v66; // rcx
  unsigned __int64 *v67; // rdx
  __int64 *v68; // r14
  __int64 *v69; // rbx
  __int64 *v70; // rdx
  __int64 *v71; // r15
  unsigned __int64 *v72; // r14
  unsigned __int64 v73; // r13
  char *v74; // rcx
  char *v75; // rax
  char *v76; // r11
  __int64 v77; // rax
  __int64 *v78; // rax
  __int64 *v79; // r8
  __int64 *v80; // rcx
  unsigned __int64 v81; // rax
  unsigned __int64 v82; // r12
  char *v83; // [rsp+8h] [rbp-238h]
  __int64 v84; // [rsp+10h] [rbp-230h]
  __int64 *v85; // [rsp+10h] [rbp-230h]
  unsigned __int64 v86; // [rsp+18h] [rbp-228h]
  __int64 *v87; // [rsp+18h] [rbp-228h]
  char v88[8]; // [rsp+20h] [rbp-220h] BYREF
  unsigned __int64 v89; // [rsp+28h] [rbp-218h]
  char v90; // [rsp+3Ch] [rbp-204h]
  _BYTE v91[64]; // [rsp+40h] [rbp-200h] BYREF
  __m128i *v92; // [rsp+80h] [rbp-1C0h]
  __int64 v93; // [rsp+88h] [rbp-1B8h]
  __int8 *v94; // [rsp+90h] [rbp-1B0h]
  char v95[8]; // [rsp+A0h] [rbp-1A0h] BYREF
  unsigned __int64 v96; // [rsp+A8h] [rbp-198h]
  char v97; // [rsp+BCh] [rbp-184h]
  _BYTE v98[64]; // [rsp+C0h] [rbp-180h] BYREF
  unsigned __int64 v99; // [rsp+100h] [rbp-140h]
  unsigned __int64 i; // [rsp+108h] [rbp-138h]
  unsigned __int64 v101; // [rsp+110h] [rbp-130h]
  unsigned __int64 *v102; // [rsp+120h] [rbp-120h] BYREF
  unsigned __int64 *v103; // [rsp+128h] [rbp-118h]
  unsigned __int64 *v104; // [rsp+130h] [rbp-110h]
  char v105; // [rsp+13Ch] [rbp-104h]
  const __m128i *v106; // [rsp+180h] [rbp-C0h]
  const __m128i *v107; // [rsp+188h] [rbp-B8h]
  char v108[8]; // [rsp+198h] [rbp-A8h] BYREF
  unsigned __int64 v109; // [rsp+1A0h] [rbp-A0h]
  char v110; // [rsp+1B4h] [rbp-8Ch]
  const __m128i *v111; // [rsp+1F8h] [rbp-48h]
  const __m128i *v112; // [rsp+200h] [rbp-40h]

  a2[1] = (unsigned __int64)a1;
  v102 = a2;
  v6 = (unsigned __int64 *)a1[6];
  if ( v6 != (unsigned __int64 *)a1[7] )
  {
    if ( v6 )
    {
      *v6 = (unsigned __int64)a2;
      a1[6] += 8LL;
      goto LABEL_4;
    }
    a1[6] = 8;
    v60 = (unsigned __int64)a2;
    goto LABEL_97;
  }
  sub_35691F0(a1 + 5, v6, (__int64 *)&v102);
  v60 = (unsigned __int64)v102;
  if ( v102 )
  {
LABEL_97:
    sub_3568740(v60);
    j_j___libc_free_0(v60);
  }
LABEL_4:
  if ( !a3 )
    return;
  sub_356A1B0(&v102, a1);
  v7 = v91;
  v8 = v88;
  sub_C8CD80((__int64)v88, (__int64)v91, (__int64)&v102, v9, v10, v11);
  v14 = v107;
  v15 = v106;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v16 = (char *)v107 - (char *)v106;
  if ( v107 == v106 )
  {
    v17 = 0;
  }
  else
  {
    if ( v16 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_162;
    v17 = (__m128i *)sub_22077B0((char *)v107 - (char *)v106);
    v14 = v107;
    v15 = v106;
  }
  v92 = v17;
  v93 = (__int64)v17;
  v94 = &v17->m128i_i8[v16];
  if ( v14 == v15 )
  {
    v18 = (__int64)v17;
  }
  else
  {
    v18 = (__int64)v17->m128i_i64 + (char *)v14 - (char *)v15;
    do
    {
      if ( v17 )
      {
        *v17 = _mm_loadu_si128(v15);
        v17[1] = _mm_loadu_si128(v15 + 1);
      }
      v17 += 2;
      v15 += 2;
    }
    while ( v17 != (__m128i *)v18 );
  }
  v8 = v95;
  v93 = v18;
  v7 = v98;
  sub_C8CD80((__int64)v95, (__int64)v98, (__int64)v108, v18, v12, v13);
  v19 = v112;
  v20 = v111;
  v99 = 0;
  i = 0;
  v101 = 0;
  v21 = (char *)v112 - (char *)v111;
  if ( v112 == v111 )
  {
    v23 = 0;
    goto LABEL_17;
  }
  if ( v21 > 0x7FFFFFFFFFFFFFE0LL )
LABEL_162:
    sub_4261EA(v8, v7, v15);
  v22 = sub_22077B0((char *)v112 - (char *)v111);
  v20 = v111;
  v23 = v22;
  v19 = v112;
LABEL_17:
  v99 = v23;
  i = v23;
  v101 = v23 + v21;
  if ( v20 == v19 )
  {
    v25 = (__m128i *)v23;
  }
  else
  {
    v24 = (__m128i *)v23;
    v25 = (__m128i *)(v23 + (char *)v19 - (char *)v20);
    do
    {
      if ( v24 )
      {
        *v24 = _mm_loadu_si128(v20);
        v24[1] = _mm_loadu_si128(v20 + 1);
      }
      v24 += 2;
      v20 += 2;
    }
    while ( v25 != v24 );
  }
  for ( i = (unsigned __int64)v25; ; v25 = (__m128i *)i )
  {
    v27 = (unsigned __int64)v92;
    if ( (__m128i *)(v93 - (_QWORD)v92) != (__m128i *)((char *)v25 - v23) )
      goto LABEL_23;
    if ( v92 == (__m128i *)v93 )
      break;
    v28 = v23;
    while ( *(_QWORD *)v27 == *(_QWORD *)v28 )
    {
      v29 = *(_BYTE *)(v27 + 24);
      if ( v29 != *(_BYTE *)(v28 + 24) )
        break;
      if ( v29 )
      {
        if ( !(((*(__int64 *)(v27 + 8) >> 1) & 3) != 0
             ? ((*(__int64 *)(v28 + 8) >> 1) & 3) == ((*(__int64 *)(v27 + 8) >> 1) & 3)
             : *(_QWORD *)(v27 + 16) == *(_QWORD *)(v28 + 16)) )
          break;
      }
      v27 += 32LL;
      v28 += 32LL;
      if ( v93 == v27 )
        goto LABEL_35;
    }
LABEL_23:
    v26 = **(_QWORD **)(v93 - 32);
    if ( (v26 & 4) == 0 )
    {
      v59 = v26 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (unsigned __int8)sub_3567D90(a2, v59) )
        sub_356C870(a1[2], v59, (__int64)a2);
    }
    sub_3569EC0((__int64)v88);
    v23 = v99;
  }
LABEL_35:
  if ( v23 )
    j_j___libc_free_0(v23);
  if ( !v97 )
    _libc_free(v96);
  if ( v92 )
    j_j___libc_free_0((unsigned __int64)v92);
  if ( !v90 )
    _libc_free(v89);
  if ( v111 )
    j_j___libc_free_0((unsigned __int64)v111);
  if ( !v110 )
    _libc_free(v109);
  if ( v106 )
    j_j___libc_free_0((unsigned __int64)v106);
  if ( !v105 )
    _libc_free((unsigned __int64)v103);
  v31 = (__int64 *)a1[5];
  v32 = (__int64 *)a1[6];
  v102 = 0;
  v103 = 0;
  v104 = 0;
  if ( v32 != v31 )
  {
    while ( 1 )
    {
      v34 = (unsigned __int64 *)*v31;
      if ( a2[4] )
      {
        if ( !(unsigned __int8)sub_3567D90(a2, *v34 & 0xFFFFFFFFFFFFFFF8LL)
          || !(unsigned __int8)sub_3567D90(a2, v34[4]) && v34[4] != a2[4] )
        {
LABEL_54:
          v33 = v103;
          if ( v103 == v104 )
          {
            sub_35691F0((unsigned __int64 *)&v102, v103, v31);
          }
          else
          {
            if ( v103 )
            {
              *v103 = *v31;
              *v31 = 0;
              v33 = v103;
            }
            v103 = v33 + 1;
          }
          goto LABEL_58;
        }
        v34 = (unsigned __int64 *)*v31;
      }
      if ( a2 == v34 )
        goto LABEL_54;
      v34[1] = (unsigned __int64)a2;
      v35 = (unsigned __int64 *)a2[6];
      if ( v35 == (unsigned __int64 *)a2[7] )
      {
        sub_35691F0(a2 + 5, v35, v31);
LABEL_58:
        if ( v32 == ++v31 )
          goto LABEL_65;
      }
      else
      {
        if ( v35 )
        {
          *v35 = *v31;
          *v31 = 0;
          v35 = (unsigned __int64 *)a2[6];
        }
        ++v31;
        a2[6] = (unsigned __int64)(v35 + 1);
        if ( v32 == v31 )
        {
LABEL_65:
          v36 = (__int64 *)a1[5];
          v31 = (__int64 *)a1[6];
          if ( v31 != v36 )
          {
            v37 = (__int64 *)a1[5];
            do
            {
              v38 = *v37;
              if ( *v37 )
              {
                sub_3568740(*v37);
                j_j___libc_free_0(v38);
              }
              ++v37;
            }
            while ( v31 != v37 );
            a1[6] = v36;
            v31 = (__int64 *)a1[5];
          }
          break;
        }
      }
    }
  }
  v39 = v103;
  v40 = v102;
  if ( v102 != v103 )
  {
    v41 = (__int64 *)a1[6];
    v42 = (char *)v103 - (char *)v102;
    v43 = (char *)v41 - (char *)v31;
    v44 = v103 - v102;
    v86 = v44;
    v45 = v41 - v31;
    v46 = v45;
    if ( a1[7] - (_QWORD)v41 >= (unsigned __int64)((char *)v103 - (char *)v102) )
    {
      if ( v42 >= (unsigned __int64)v43 )
      {
        v74 = (char *)v102 + v43;
        if ( v103 == (unsigned __int64 *)((char *)v102 + v43) )
        {
          v77 = a1[6];
        }
        else
        {
          v75 = (char *)a1[6];
          v76 = (char *)v41 + (char *)v103 - v74;
          do
          {
            if ( v75 )
            {
              *(_QWORD *)v75 = *(_QWORD *)v74;
              *(_QWORD *)v74 = 0;
            }
            v75 += 8;
            v74 += 8;
          }
          while ( v75 != v76 );
          v77 = a1[6];
        }
        v78 = (__int64 *)(v77 + 8 * (v44 - v45));
        a1[6] = v78;
        if ( v31 == v41 )
        {
          a1[6] = (char *)v78 + v43;
        }
        else
        {
          v79 = v31;
          v80 = (__int64 *)((char *)v78 + (char *)v41 - (char *)v31);
          do
          {
            if ( v78 )
            {
              *v78 = *v79;
              *v79 = 0;
            }
            ++v78;
            ++v79;
          }
          while ( v78 != v80 );
          a1[6] += v43;
          if ( v43 > 0 )
          {
            do
            {
              v81 = *v40;
              *v40 = 0;
              v82 = *v31;
              *v31 = v81;
              if ( v82 )
              {
                sub_3568740(v82);
                j_j___libc_free_0(v82);
              }
              ++v40;
              ++v31;
              --v46;
            }
            while ( v46 );
          }
        }
      }
      else
      {
        v47 = (_QWORD *)a1[6];
        v48 = (char *)v41 - v42;
        v49 = (__int64 *)((char *)v41 - v42);
        do
        {
          if ( v47 )
          {
            *v47 = *v49;
            *v49 = 0;
          }
          ++v47;
          ++v49;
        }
        while ( v47 != (__int64 *)((char *)v41 + v42) );
        a1[6] += v42;
        v50 = (v48 - (char *)v31) >> 3;
        if ( v48 - (char *)v31 > 0 )
        {
          v51 = &v48[-8 * v50];
          v52 = &v41[-v50];
          do
          {
            v53 = *(_QWORD *)&v51[8 * v50 - 8];
            *(_QWORD *)&v51[8 * v50 - 8] = 0;
            v54 = v52[v50 - 1];
            v52[v50 - 1] = v53;
            if ( v54 )
            {
              v83 = v51;
              v84 = v42;
              sub_3568740(v54);
              j_j___libc_free_0(v54);
              v51 = v83;
              v42 = v84;
            }
            --v50;
          }
          while ( v50 );
        }
        if ( v42 > 0 )
        {
          do
          {
            v55 = *v40;
            *v40 = 0;
            v56 = *v31;
            *v31 = v55;
            if ( v56 )
            {
              sub_3568740(v56);
              j_j___libc_free_0(v56);
            }
            ++v40;
            ++v31;
            --v86;
          }
          while ( v86 );
        }
      }
LABEL_87:
      v39 = v103;
      v57 = v102;
      if ( v102 != v103 )
      {
        do
        {
          v58 = *v57;
          if ( *v57 )
          {
            sub_3568740(*v57);
            j_j___libc_free_0(v58);
          }
          ++v57;
        }
        while ( v39 != v57 );
        v39 = v102;
      }
      goto LABEL_92;
    }
    if ( v44 > 0xFFFFFFFFFFFFFFFLL - v45 )
      sub_4262D8((__int64)"vector::_M_range_insert");
    if ( v45 >= v44 )
      v44 = v41 - v31;
    v61 = __CFADD__(v44, v45);
    v62 = v44 + v45;
    if ( v61 )
    {
      v63 = 0xFFFFFFFFFFFFFFFLL;
    }
    else
    {
      if ( !v62 )
      {
        v85 = 0;
        v64 = 0;
        v87 = 0;
LABEL_121:
        v67 = v40;
        v68 = (__int64 *)((char *)v64 + (char *)v39 - (char *)v40);
        do
        {
          if ( v64 )
          {
            *v64 = *v67;
            *v67 = 0;
          }
          ++v64;
          ++v67;
        }
        while ( v68 != v64 );
        v69 = (__int64 *)a1[6];
        if ( v31 == v69 )
        {
          v71 = v68;
        }
        else
        {
          v70 = v31;
          v71 = (__int64 *)((char *)v68 + (char *)v69 - (char *)v31);
          do
          {
            if ( v68 )
            {
              *v68 = *v70;
              *v70 = 0;
            }
            ++v68;
            ++v70;
          }
          while ( v71 != v68 );
          v69 = (__int64 *)a1[6];
        }
        v72 = (unsigned __int64 *)a1[5];
        if ( v69 != (__int64 *)v72 )
        {
          do
          {
            v73 = *v72;
            if ( *v72 )
            {
              sub_3568740(*v72);
              j_j___libc_free_0(v73);
            }
            ++v72;
          }
          while ( v69 != (__int64 *)v72 );
          v72 = (unsigned __int64 *)a1[5];
        }
        if ( v72 )
          j_j___libc_free_0((unsigned __int64)v72);
        a1[6] = v71;
        a1[5] = v87;
        a1[7] = v85;
        goto LABEL_87;
      }
      if ( v62 > 0xFFFFFFFFFFFFFFFLL )
        v62 = 0xFFFFFFFFFFFFFFFLL;
      v63 = v62;
    }
    v64 = (__int64 *)sub_22077B0(v63 * 8);
    v65 = (__int64 *)a1[5];
    v87 = v64;
    if ( v31 == v65 )
    {
      v85 = &v64[v63];
    }
    else
    {
      v66 = v64;
      v64 = (__int64 *)((char *)v64 + (char *)v31 - (char *)v65);
      do
      {
        if ( v66 )
        {
          *v66 = *v65;
          *v65 = 0;
        }
        ++v66;
        ++v65;
      }
      while ( v66 != v64 );
      v85 = &v87[v63];
    }
    goto LABEL_121;
  }
LABEL_92:
  if ( v39 )
    j_j___libc_free_0((unsigned __int64)v39);
}
