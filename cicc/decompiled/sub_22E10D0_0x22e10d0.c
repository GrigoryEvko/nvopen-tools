// Function: sub_22E10D0
// Address: 0x22e10d0
//
void __fastcall sub_22E10D0(_QWORD *a1, unsigned __int64 *a2, char a3)
{
  unsigned __int64 *v6; // rsi
  _BYTE *v7; // rsi
  char *v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 v13; // r9
  __int64 v14; // rcx
  __int64 v15; // r8
  unsigned __int64 v16; // r15
  __int64 v17; // rax
  unsigned __int64 v18; // rdi
  __m128i *v19; // rdx
  const __m128i *v20; // rax
  const __m128i *v21; // rcx
  unsigned __int64 v22; // r8
  unsigned __int64 v23; // r15
  __int64 v24; // rax
  unsigned __int64 v25; // rdi
  __m128i *v26; // rdx
  const __m128i *v27; // rax
  unsigned __int64 v28; // rax
  __int64 v29; // rsi
  unsigned __int64 v30; // rcx
  unsigned __int64 v31; // rax
  char v32; // si
  __int64 *v34; // r15
  __int64 *v35; // r13
  unsigned __int64 *v36; // rsi
  unsigned __int64 *v37; // r14
  unsigned __int64 *v38; // rsi
  __int64 *v39; // r14
  __int64 *v40; // rbx
  unsigned __int64 v41; // r13
  unsigned __int64 *v42; // r14
  unsigned __int64 *v43; // rbx
  __int64 *v44; // rdi
  __int64 v45; // r8
  __int64 v46; // rsi
  unsigned __int64 v47; // r10
  unsigned __int64 v48; // r9
  unsigned __int64 v49; // r13
  _QWORD *v50; // rax
  char *v51; // rsi
  _QWORD *v52; // rcx
  __int64 v53; // r12
  char *v54; // rax
  __int64 *v55; // r14
  __int64 v56; // rcx
  unsigned __int64 v57; // r13
  unsigned __int64 v58; // rax
  unsigned __int64 v59; // r12
  unsigned __int64 *v60; // rbx
  unsigned __int64 v61; // r12
  unsigned __int64 v62; // rsi
  unsigned __int64 v63; // r14
  bool v64; // cf
  unsigned __int64 v65; // r9
  unsigned __int64 v66; // r13
  __int64 *v67; // rax
  __int64 *v68; // rsi
  __int64 *v69; // rcx
  unsigned __int64 *v70; // rdx
  __int64 *v71; // r14
  __int64 *v72; // rbx
  __int64 *v73; // rdx
  __int64 *v74; // r15
  unsigned __int64 *v75; // r14
  unsigned __int64 v76; // r13
  char *v77; // rcx
  char *v78; // rax
  char *v79; // r11
  __int64 v80; // rax
  __int64 *v81; // rax
  __int64 *v82; // r8
  __int64 *v83; // rcx
  unsigned __int64 v84; // rax
  unsigned __int64 v85; // r12
  char *v86; // [rsp+8h] [rbp-238h]
  __int64 v87; // [rsp+10h] [rbp-230h]
  __int64 *v88; // [rsp+10h] [rbp-230h]
  unsigned __int64 v89; // [rsp+18h] [rbp-228h]
  __int64 *v90; // [rsp+18h] [rbp-228h]
  char v91[8]; // [rsp+20h] [rbp-220h] BYREF
  unsigned __int64 v92; // [rsp+28h] [rbp-218h]
  char v93; // [rsp+3Ch] [rbp-204h]
  _BYTE v94[64]; // [rsp+40h] [rbp-200h] BYREF
  unsigned __int64 v95; // [rsp+80h] [rbp-1C0h]
  unsigned __int64 v96; // [rsp+88h] [rbp-1B8h]
  unsigned __int64 v97; // [rsp+90h] [rbp-1B0h]
  char v98[8]; // [rsp+A0h] [rbp-1A0h] BYREF
  unsigned __int64 v99; // [rsp+A8h] [rbp-198h]
  char v100; // [rsp+BCh] [rbp-184h]
  _BYTE v101[64]; // [rsp+C0h] [rbp-180h] BYREF
  unsigned __int64 v102; // [rsp+100h] [rbp-140h]
  unsigned __int64 i; // [rsp+108h] [rbp-138h]
  unsigned __int64 v104; // [rsp+110h] [rbp-130h]
  unsigned __int64 *v105; // [rsp+120h] [rbp-120h] BYREF
  unsigned __int64 *v106; // [rsp+128h] [rbp-118h]
  unsigned __int64 *v107; // [rsp+130h] [rbp-110h]
  char v108; // [rsp+13Ch] [rbp-104h]
  unsigned __int64 v109; // [rsp+180h] [rbp-C0h]
  __int64 v110; // [rsp+188h] [rbp-B8h]
  char v111[8]; // [rsp+198h] [rbp-A8h] BYREF
  unsigned __int64 v112; // [rsp+1A0h] [rbp-A0h]
  char v113; // [rsp+1B4h] [rbp-8Ch]
  const __m128i *v114; // [rsp+1F8h] [rbp-48h]
  const __m128i *v115; // [rsp+200h] [rbp-40h]

  a2[1] = (unsigned __int64)a1;
  v105 = a2;
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
    v63 = (unsigned __int64)a2;
    goto LABEL_99;
  }
  sub_22DD010(a1 + 5, v6, (__int64 *)&v105);
  v63 = (unsigned __int64)v105;
  if ( v105 )
  {
LABEL_99:
    sub_22DBFB0(v63);
    j_j___libc_free_0(v63);
  }
LABEL_4:
  if ( !a3 )
    return;
  sub_22DE8A0(&v105, a1);
  v7 = v94;
  v8 = v91;
  sub_C8CD80((__int64)v91, (__int64)v94, (__int64)&v105, v9, v10, v11);
  v14 = v110;
  v15 = v109;
  v95 = 0;
  v96 = 0;
  v97 = 0;
  v16 = v110 - v109;
  if ( v110 == v109 )
  {
    v18 = 0;
  }
  else
  {
    if ( v16 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_166;
    v17 = sub_22077B0(v110 - v109);
    v14 = v110;
    v15 = v109;
    v18 = v17;
  }
  v95 = v18;
  v96 = v18;
  v97 = v18 + v16;
  if ( v15 != v14 )
  {
    v19 = (__m128i *)v18;
    v20 = (const __m128i *)v15;
    do
    {
      if ( v19 )
      {
        *v19 = _mm_loadu_si128(v20);
        v19[1] = _mm_loadu_si128(v20 + 1);
        v19[2].m128i_i64[0] = v20[2].m128i_i64[0];
      }
      v20 = (const __m128i *)((char *)v20 + 40);
      v19 = (__m128i *)((char *)v19 + 40);
    }
    while ( v20 != (const __m128i *)v14 );
    v18 += 8 * (((unsigned __int64)&v20[-3].m128i_u64[1] - v15) >> 3) + 40;
  }
  v96 = v18;
  v8 = v98;
  v7 = v101;
  sub_C8CD80((__int64)v98, (__int64)v101, (__int64)v111, v14, v15, v13);
  v21 = v115;
  v22 = (unsigned __int64)v114;
  v102 = 0;
  i = 0;
  v104 = 0;
  v23 = (char *)v115 - (char *)v114;
  if ( v115 == v114 )
  {
    v25 = 0;
    goto LABEL_18;
  }
  if ( v23 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_166:
    sub_4261EA(v8, v7, v12);
  v24 = sub_22077B0((char *)v115 - (char *)v114);
  v21 = v115;
  v22 = (unsigned __int64)v114;
  v25 = v24;
LABEL_18:
  v102 = v25;
  i = v25;
  v104 = v25 + v23;
  if ( (const __m128i *)v22 == v21 )
  {
    v28 = v25;
  }
  else
  {
    v26 = (__m128i *)v25;
    v27 = (const __m128i *)v22;
    do
    {
      if ( v26 )
      {
        *v26 = _mm_loadu_si128(v27);
        v26[1] = _mm_loadu_si128(v27 + 1);
        v26[2].m128i_i64[0] = v27[2].m128i_i64[0];
      }
      v27 = (const __m128i *)((char *)v27 + 40);
      v26 = (__m128i *)((char *)v26 + 40);
    }
    while ( v27 != v21 );
    v28 = v25 + 8 * (((unsigned __int64)&v27[-3].m128i_u64[1] - v22) >> 3) + 40;
  }
  for ( i = v28; ; v28 = i )
  {
    v30 = v95;
    if ( v96 - v95 != v28 - v25 )
      goto LABEL_25;
    if ( v95 == v96 )
      break;
    v31 = v25;
    while ( *(_QWORD *)v30 == *(_QWORD *)v31 )
    {
      v32 = *(_BYTE *)(v30 + 32);
      if ( v32 != *(_BYTE *)(v31 + 32) )
        break;
      if ( v32 )
      {
        if ( !(((*(__int64 *)(v30 + 8) >> 1) & 3) != 0
             ? ((*(__int64 *)(v31 + 8) >> 1) & 3) == ((*(__int64 *)(v30 + 8) >> 1) & 3)
             : *(_DWORD *)(v30 + 24) == *(_DWORD *)(v31 + 24)) )
          break;
      }
      v30 += 40LL;
      v31 += 40LL;
      if ( v96 == v30 )
        goto LABEL_37;
    }
LABEL_25:
    v29 = **(_QWORD **)(v96 - 40);
    if ( (v29 & 4) == 0 )
    {
      v62 = v29 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (unsigned __int8)sub_22DB400(a2, v62) )
        sub_22E0E70(a1[2], v62, (__int64)a2);
    }
    sub_22DE410((__int64)v91);
    v25 = v102;
  }
LABEL_37:
  if ( v25 )
    j_j___libc_free_0(v25);
  if ( !v100 )
    _libc_free(v99);
  if ( v95 )
    j_j___libc_free_0(v95);
  if ( !v93 )
    _libc_free(v92);
  if ( v114 )
    j_j___libc_free_0((unsigned __int64)v114);
  if ( !v113 )
    _libc_free(v112);
  if ( v109 )
    j_j___libc_free_0(v109);
  if ( !v108 )
    _libc_free((unsigned __int64)v106);
  v34 = (__int64 *)a1[5];
  v35 = (__int64 *)a1[6];
  v105 = 0;
  v106 = 0;
  v107 = 0;
  if ( v35 != v34 )
  {
    while ( 1 )
    {
      v37 = (unsigned __int64 *)*v34;
      if ( a2[4] )
      {
        if ( !(unsigned __int8)sub_22DB400(a2, *v37 & 0xFFFFFFFFFFFFFFF8LL)
          || !(unsigned __int8)sub_22DB400(a2, v37[4]) && v37[4] != a2[4] )
        {
LABEL_56:
          v36 = v106;
          if ( v106 == v107 )
          {
            sub_22DD010((unsigned __int64 *)&v105, v106, v34);
          }
          else
          {
            if ( v106 )
            {
              *v106 = *v34;
              *v34 = 0;
              v36 = v106;
            }
            v106 = v36 + 1;
          }
          goto LABEL_60;
        }
        v37 = (unsigned __int64 *)*v34;
      }
      if ( a2 == v37 )
        goto LABEL_56;
      v37[1] = (unsigned __int64)a2;
      v38 = (unsigned __int64 *)a2[6];
      if ( v38 == (unsigned __int64 *)a2[7] )
      {
        sub_22DD010(a2 + 5, v38, v34);
LABEL_60:
        if ( v35 == ++v34 )
          goto LABEL_67;
      }
      else
      {
        if ( v38 )
        {
          *v38 = *v34;
          *v34 = 0;
          v38 = (unsigned __int64 *)a2[6];
        }
        ++v34;
        a2[6] = (unsigned __int64)(v38 + 1);
        if ( v35 == v34 )
        {
LABEL_67:
          v39 = (__int64 *)a1[5];
          v34 = (__int64 *)a1[6];
          if ( v34 != v39 )
          {
            v40 = (__int64 *)a1[5];
            do
            {
              v41 = *v40;
              if ( *v40 )
              {
                sub_22DBFB0(*v40);
                j_j___libc_free_0(v41);
              }
              ++v40;
            }
            while ( v34 != v40 );
            a1[6] = v39;
            v34 = (__int64 *)a1[5];
          }
          break;
        }
      }
    }
  }
  v42 = v106;
  v43 = v105;
  if ( v105 != v106 )
  {
    v44 = (__int64 *)a1[6];
    v45 = (char *)v106 - (char *)v105;
    v46 = (char *)v44 - (char *)v34;
    v47 = v106 - v105;
    v89 = v47;
    v48 = v44 - v34;
    v49 = v48;
    if ( a1[7] - (_QWORD)v44 >= (unsigned __int64)((char *)v106 - (char *)v105) )
    {
      if ( v45 >= (unsigned __int64)v46 )
      {
        v77 = (char *)v105 + v46;
        if ( v106 == (unsigned __int64 *)((char *)v105 + v46) )
        {
          v80 = a1[6];
        }
        else
        {
          v78 = (char *)a1[6];
          v79 = (char *)v44 + (char *)v106 - v77;
          do
          {
            if ( v78 )
            {
              *(_QWORD *)v78 = *(_QWORD *)v77;
              *(_QWORD *)v77 = 0;
            }
            v78 += 8;
            v77 += 8;
          }
          while ( v79 != v78 );
          v80 = a1[6];
        }
        v81 = (__int64 *)(v80 + 8 * (v47 - v48));
        a1[6] = v81;
        if ( v34 == v44 )
        {
          a1[6] = (char *)v81 + v46;
        }
        else
        {
          v82 = v34;
          v83 = (__int64 *)((char *)v81 + (char *)v44 - (char *)v34);
          do
          {
            if ( v81 )
            {
              *v81 = *v82;
              *v82 = 0;
            }
            ++v81;
            ++v82;
          }
          while ( v83 != v81 );
          a1[6] += v46;
          if ( v46 > 0 )
          {
            do
            {
              v84 = *v43;
              *v43 = 0;
              v85 = *v34;
              *v34 = v84;
              if ( v85 )
              {
                sub_22DBFB0(v85);
                j_j___libc_free_0(v85);
              }
              ++v43;
              ++v34;
              --v49;
            }
            while ( v49 );
          }
        }
      }
      else
      {
        v50 = (_QWORD *)a1[6];
        v51 = (char *)v44 - v45;
        v52 = (__int64 *)((char *)v44 - v45);
        do
        {
          if ( v50 )
          {
            *v50 = *v52;
            *v52 = 0;
          }
          ++v50;
          ++v52;
        }
        while ( v50 != (__int64 *)((char *)v44 + v45) );
        a1[6] += v45;
        v53 = (v51 - (char *)v34) >> 3;
        if ( v51 - (char *)v34 > 0 )
        {
          v54 = &v51[-8 * v53];
          v55 = &v44[-v53];
          do
          {
            v56 = *(_QWORD *)&v54[8 * v53 - 8];
            *(_QWORD *)&v54[8 * v53 - 8] = 0;
            v57 = v55[v53 - 1];
            v55[v53 - 1] = v56;
            if ( v57 )
            {
              v86 = v54;
              v87 = v45;
              sub_22DBFB0(v57);
              j_j___libc_free_0(v57);
              v54 = v86;
              v45 = v87;
            }
            --v53;
          }
          while ( v53 );
        }
        if ( v45 > 0 )
        {
          do
          {
            v58 = *v43;
            *v43 = 0;
            v59 = *v34;
            *v34 = v58;
            if ( v59 )
            {
              sub_22DBFB0(v59);
              j_j___libc_free_0(v59);
            }
            ++v43;
            ++v34;
            --v89;
          }
          while ( v89 );
        }
      }
LABEL_89:
      v42 = v106;
      v60 = v105;
      if ( v105 != v106 )
      {
        do
        {
          v61 = *v60;
          if ( *v60 )
          {
            sub_22DBFB0(*v60);
            j_j___libc_free_0(v61);
          }
          ++v60;
        }
        while ( v60 != v42 );
        v42 = v105;
      }
      goto LABEL_94;
    }
    if ( v47 > 0xFFFFFFFFFFFFFFFLL - v48 )
      sub_4262D8((__int64)"vector::_M_range_insert");
    if ( v48 >= v47 )
      v47 = v44 - v34;
    v64 = __CFADD__(v47, v48);
    v65 = v47 + v48;
    if ( v64 )
    {
      v66 = 0xFFFFFFFFFFFFFFFLL;
    }
    else
    {
      if ( !v65 )
      {
        v88 = 0;
        v67 = 0;
        v90 = 0;
LABEL_123:
        v70 = v43;
        v71 = (__int64 *)((char *)v67 + (char *)v42 - (char *)v43);
        do
        {
          if ( v67 )
          {
            *v67 = *v70;
            *v70 = 0;
          }
          ++v67;
          ++v70;
        }
        while ( v71 != v67 );
        v72 = (__int64 *)a1[6];
        if ( v34 == v72 )
        {
          v74 = v71;
        }
        else
        {
          v73 = v34;
          v74 = (__int64 *)((char *)v71 + (char *)v72 - (char *)v34);
          do
          {
            if ( v71 )
            {
              *v71 = *v73;
              *v73 = 0;
            }
            ++v71;
            ++v73;
          }
          while ( v74 != v71 );
          v72 = (__int64 *)a1[6];
        }
        v75 = (unsigned __int64 *)a1[5];
        if ( v72 != (__int64 *)v75 )
        {
          do
          {
            v76 = *v75;
            if ( *v75 )
            {
              sub_22DBFB0(*v75);
              j_j___libc_free_0(v76);
            }
            ++v75;
          }
          while ( v72 != (__int64 *)v75 );
          v75 = (unsigned __int64 *)a1[5];
        }
        if ( v75 )
          j_j___libc_free_0((unsigned __int64)v75);
        a1[6] = v74;
        a1[5] = v90;
        a1[7] = v88;
        goto LABEL_89;
      }
      if ( v65 > 0xFFFFFFFFFFFFFFFLL )
        v65 = 0xFFFFFFFFFFFFFFFLL;
      v66 = v65;
    }
    v67 = (__int64 *)sub_22077B0(v66 * 8);
    v68 = (__int64 *)a1[5];
    v90 = v67;
    if ( v34 == v68 )
    {
      v88 = &v67[v66];
    }
    else
    {
      v69 = v67;
      v67 = (__int64 *)((char *)v67 + (char *)v34 - (char *)v68);
      do
      {
        if ( v69 )
        {
          *v69 = *v68;
          *v68 = 0;
        }
        ++v69;
        ++v68;
      }
      while ( v69 != v67 );
      v88 = &v90[v66];
    }
    goto LABEL_123;
  }
LABEL_94:
  if ( v42 )
    j_j___libc_free_0((unsigned __int64)v42);
}
