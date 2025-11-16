// Function: sub_3966880
// Address: 0x3966880
//
__int64 __fastcall sub_3966880(__int64 *a1)
{
  __int64 v1; // rsi
  int v2; // r8d
  int v3; // r9d
  __int64 v4; // r14
  unsigned __int64 i; // r15
  __int64 v6; // r13
  __int64 v7; // r12
  __int64 j; // r13
  char v9; // al
  __int64 v10; // rax
  unsigned __int64 v11; // r13
  _BYTE *v12; // r15
  __int64 *v13; // rdx
  __int64 v14; // r14
  unsigned __int8 v15; // al
  __int64 v16; // rbx
  __int64 v17; // r12
  __int64 v18; // rsi
  __int64 v19; // rax
  unsigned __int64 v20; // rax
  _BYTE *v21; // rsi
  const char **v22; // rdi
  __int64 v23; // rdx
  const __m128i *v24; // rcx
  unsigned __int64 v25; // r8
  unsigned __int64 v26; // r12
  __int64 v27; // rax
  const __m128i *v28; // rdi
  __m128i *v29; // rdx
  const __m128i *v30; // rax
  int v31; // r9d
  const __m128i *v32; // rcx
  unsigned __int64 v33; // r8
  unsigned __int64 v34; // r12
  __int64 v35; // rax
  unsigned __int64 v36; // rdi
  __m128i *v37; // rdx
  const __m128i *v38; // rax
  __m128i *v39; // r12
  const __m128i *v40; // rax
  _QWORD *v41; // r15
  _QWORD *v42; // rax
  char v43; // dl
  __int64 v44; // rdx
  unsigned __int64 v45; // rdi
  int v46; // eax
  unsigned int v47; // esi
  __int64 v48; // rdi
  __int64 v49; // r15
  __int64 *v50; // rax
  char v51; // dl
  unsigned __int64 v52; // rax
  __int64 *v53; // rsi
  __int64 *v54; // rdi
  __int64 v55; // rdx
  __int64 *v56; // rdx
  unsigned __int64 v57; // rdx
  _QWORD *v58; // r12
  __int64 v59; // r15
  char v60; // bl
  __int64 v61; // rsi
  __int64 v62; // r13
  __int64 v63; // rax
  __int64 v64; // rbx
  __int64 v65; // rdx
  __int64 v66; // rdi
  unsigned __int8 v67; // al
  unsigned int v68; // eax
  _BYTE *v70; // [rsp+10h] [rbp-330h]
  unsigned __int8 v71; // [rsp+1Fh] [rbp-321h]
  _BYTE *v72; // [rsp+28h] [rbp-318h]
  _QWORD *v73; // [rsp+28h] [rbp-318h]
  unsigned __int64 v74; // [rsp+30h] [rbp-310h] BYREF
  __int64 v75; // [rsp+38h] [rbp-308h]
  __m128i v76; // [rsp+50h] [rbp-2F0h] BYREF
  __int64 v77; // [rsp+60h] [rbp-2E0h]
  _BYTE *v78; // [rsp+70h] [rbp-2D0h] BYREF
  __int64 v79; // [rsp+78h] [rbp-2C8h]
  _BYTE v80[64]; // [rsp+80h] [rbp-2C0h] BYREF
  _BYTE *v81; // [rsp+C0h] [rbp-280h] BYREF
  __int64 v82; // [rsp+C8h] [rbp-278h]
  _BYTE v83[64]; // [rsp+D0h] [rbp-270h] BYREF
  __int64 v84; // [rsp+110h] [rbp-230h] BYREF
  __int64 *v85; // [rsp+118h] [rbp-228h]
  __int64 *v86; // [rsp+120h] [rbp-220h]
  unsigned int v87; // [rsp+128h] [rbp-218h]
  unsigned int v88; // [rsp+12Ch] [rbp-214h]
  int v89; // [rsp+130h] [rbp-210h]
  _BYTE v90[64]; // [rsp+138h] [rbp-208h] BYREF
  const __m128i *v91; // [rsp+178h] [rbp-1C8h] BYREF
  __m128i *v92; // [rsp+180h] [rbp-1C0h]
  __m128i *v93; // [rsp+188h] [rbp-1B8h]
  const char *v94; // [rsp+190h] [rbp-1B0h] BYREF
  __int64 v95; // [rsp+198h] [rbp-1A8h]
  unsigned __int64 v96; // [rsp+1A0h] [rbp-1A0h]
  _BYTE v97[64]; // [rsp+1B8h] [rbp-188h] BYREF
  unsigned __int64 v98; // [rsp+1F8h] [rbp-148h]
  __m128i *v99; // [rsp+200h] [rbp-140h]
  unsigned __int64 v100; // [rsp+208h] [rbp-138h]
  const char **v101; // [rsp+210h] [rbp-130h] BYREF
  const char *v102; // [rsp+218h] [rbp-128h]
  const char *v103; // [rsp+220h] [rbp-120h]
  const __m128i *v104; // [rsp+278h] [rbp-C8h]
  const __m128i *v105; // [rsp+280h] [rbp-C0h]
  char v106[8]; // [rsp+290h] [rbp-B0h] BYREF
  __int64 v107; // [rsp+298h] [rbp-A8h]
  unsigned __int64 v108; // [rsp+2A0h] [rbp-A0h]
  const __m128i *v109; // [rsp+2F8h] [rbp-48h]
  const __m128i *v110; // [rsp+300h] [rbp-40h]

  v1 = *a1;
  v78 = v80;
  v79 = 0x800000000LL;
  sub_39659E0(&v74, v1);
  v4 = v75;
  for ( i = v74; i != v4; v4 -= 8 )
  {
    v6 = *(_QWORD *)(v4 - 8);
    v7 = *(_QWORD *)(v6 + 48);
    for ( j = v6 + 40; j != v7; v7 = *(_QWORD *)(v7 + 8) )
    {
      while ( 1 )
      {
        if ( !v7 )
LABEL_139:
          BUG();
        v9 = *(_BYTE *)(v7 - 8);
        if ( v9 == 86 || v9 == 83 )
          break;
        v7 = *(_QWORD *)(v7 + 8);
        if ( j == v7 )
          goto LABEL_11;
      }
      v10 = (unsigned int)v79;
      if ( (unsigned int)v79 >= HIDWORD(v79) )
      {
        sub_16CD150((__int64)&v78, v80, 0, 8, v2, v3);
        v10 = (unsigned int)v79;
      }
      *(_QWORD *)&v78[8 * v10] = v7 - 24;
      LODWORD(v79) = v79 + 1;
    }
LABEL_11:
    ;
  }
  v11 = (unsigned __int64)v78;
  v12 = &v78[8 * (unsigned int)v79];
  if ( v78 != v12 )
  {
    v71 = 0;
    while ( 1 )
    {
      v17 = *(_QWORD *)v11;
      if ( (*(_BYTE *)(*(_QWORD *)v11 + 23LL) & 0x40) != 0 )
        v13 = *(__int64 **)(v17 - 8);
      else
        v13 = (__int64 *)(v17 - 24LL * (*(_DWORD *)(v17 + 20) & 0xFFFFFFF));
      v14 = *v13;
      v15 = *(_BYTE *)(*v13 + 16);
      if ( v15 > 0x17u )
      {
        if ( *(_BYTE *)(v17 + 16) != 83 || (v18 = v13[3], *(_BYTE *)(v18 + 16) <= 0x17u) )
        {
          v16 = *(_QWORD *)(v14 + 40);
          if ( v15 != 77 )
            goto LABEL_18;
          goto LABEL_26;
        }
        if ( sub_15CCEE0(a1[3], v18, *v13) )
        {
          v16 = *(_QWORD *)(v14 + 40);
          if ( *(_BYTE *)(v14 + 16) != 77 )
          {
LABEL_18:
            if ( v16 != *(_QWORD *)(v17 + 40) )
            {
              sub_15F2300((_QWORD *)v17, v14);
              v71 = 1;
            }
            goto LABEL_20;
          }
LABEL_26:
          v19 = sub_157ED20(v16);
          v16 = *(_QWORD *)(v19 + 40);
          v14 = v19;
          if ( v19 == sub_157EBA0(v16) )
          {
            if ( *(_QWORD *)(v16 + 48) == v14 + 24 || (v20 = *(_QWORD *)(v14 + 24) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
              BUG();
            v16 = *(_QWORD *)(v20 + 16);
            v14 = v20 - 24;
          }
          goto LABEL_18;
        }
      }
LABEL_20:
      v11 += 8LL;
      if ( v12 == (_BYTE *)v11 )
        goto LABEL_31;
    }
  }
  v71 = 0;
LABEL_31:
  v81 = v83;
  v82 = 0x800000000LL;
  sub_39638F0(&v101, *a1);
  v21 = v90;
  v22 = (const char **)&v84;
  sub_16CCCB0(&v84, (__int64)v90, (__int64)&v101);
  v24 = v105;
  v25 = (unsigned __int64)v104;
  v91 = 0;
  v92 = 0;
  v93 = 0;
  v26 = (char *)v105 - (char *)v104;
  if ( v105 == v104 )
  {
    v26 = 0;
    v28 = 0;
  }
  else
  {
    if ( v26 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_137;
    v27 = sub_22077B0((char *)v105 - (char *)v104);
    v24 = v105;
    v25 = (unsigned __int64)v104;
    v28 = (const __m128i *)v27;
  }
  v91 = v28;
  v92 = (__m128i *)v28;
  v93 = (__m128i *)((char *)v28 + v26);
  if ( v24 != (const __m128i *)v25 )
  {
    v29 = (__m128i *)v28;
    v30 = (const __m128i *)v25;
    do
    {
      if ( v29 )
      {
        *v29 = _mm_loadu_si128(v30);
        v29[1].m128i_i64[0] = v30[1].m128i_i64[0];
      }
      v30 = (const __m128i *)((char *)v30 + 24);
      v29 = (__m128i *)((char *)v29 + 24);
    }
    while ( v30 != v24 );
    v28 = (const __m128i *)((char *)v28 + 8 * (((unsigned __int64)&v30[-2].m128i_u64[1] - v25) >> 3) + 24);
  }
  v92 = (__m128i *)v28;
  v21 = v97;
  v22 = &v94;
  sub_16CCCB0(&v94, (__int64)v97, (__int64)v106);
  v32 = v110;
  v33 = (unsigned __int64)v109;
  v98 = 0;
  v99 = 0;
  v100 = 0;
  v34 = (char *)v110 - (char *)v109;
  if ( v110 == v109 )
  {
    v36 = 0;
    goto LABEL_43;
  }
  if ( v34 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_137:
    sub_4261EA(v22, v21, v23);
  v35 = sub_22077B0((char *)v110 - (char *)v109);
  v32 = v110;
  v33 = (unsigned __int64)v109;
  v36 = v35;
LABEL_43:
  v98 = v36;
  v37 = (__m128i *)v36;
  v99 = (__m128i *)v36;
  v100 = v36 + v34;
  if ( v32 != (const __m128i *)v33 )
  {
    v38 = (const __m128i *)v33;
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
    while ( v38 != v32 );
    v37 = (__m128i *)(v36 + 8 * (((unsigned __int64)&v38[-2].m128i_u64[1] - v33) >> 3) + 24);
  }
  v99 = v37;
  v39 = v92;
LABEL_50:
  v40 = v91;
  if ( (__m128i *)((char *)v39 - (char *)v91) == (__m128i *)((char *)v37 - v36) )
    goto LABEL_85;
  while ( 1 )
  {
    do
    {
      v41 = (_QWORD *)(v39[-2].m128i_i64[1] + 40);
      v42 = (_QWORD *)(*v41 & 0xFFFFFFFFFFFFFFF8LL);
      if ( v41 != v42 )
      {
        do
        {
          while ( 1 )
          {
            if ( !v42 )
              goto LABEL_139;
            v43 = *((_BYTE *)v42 - 8);
            if ( v43 == 87 || v43 == 84 )
              break;
            v42 = (_QWORD *)(*v42 & 0xFFFFFFFFFFFFFFF8LL);
            if ( v41 == v42 )
              goto LABEL_60;
          }
          v44 = (unsigned int)v82;
          if ( (unsigned int)v82 >= HIDWORD(v82) )
          {
            v73 = v42;
            sub_16CD150((__int64)&v81, v83, 0, 8, v33, v31);
            v44 = (unsigned int)v82;
            v42 = v73;
          }
          *(_QWORD *)&v81[8 * v44] = v42 - 3;
          LODWORD(v82) = v82 + 1;
          v42 = (_QWORD *)(*v42 & 0xFFFFFFFFFFFFFFF8LL);
        }
        while ( v41 != v42 );
LABEL_60:
        v39 = v92;
      }
      v40 = v91;
      v39 = (__m128i *)((char *)v39 - 24);
      v92 = v39;
      if ( v39 != v91 )
      {
        while ( 1 )
        {
          while ( 1 )
          {
            v45 = sub_157EBA0(v39[-2].m128i_i64[1]);
            v46 = 0;
            if ( v45 )
            {
              v46 = sub_15F4D60(v45);
              v39 = v92;
            }
            v47 = v39[-1].m128i_u32[2];
            if ( v47 == v46 )
            {
              v36 = v98;
              v37 = v99;
              goto LABEL_50;
            }
            v48 = v39[-1].m128i_i64[0];
            v39[-1].m128i_i32[2] = v47 + 1;
            v49 = sub_15F4DF0(v48, v47);
            v50 = v85;
            if ( v86 != v85 )
              goto LABEL_66;
            v53 = &v85[v88];
            if ( v85 != v53 )
              break;
LABEL_81:
            if ( v88 < v87 )
            {
              ++v88;
              *v53 = v49;
              v39 = v92;
              ++v84;
              goto LABEL_67;
            }
LABEL_66:
            sub_16CCBA0((__int64)&v84, v49);
            v39 = v92;
            if ( v51 )
            {
LABEL_67:
              v52 = sub_157EBA0(v49);
              v76.m128i_i64[0] = v49;
              v76.m128i_i64[1] = v52;
              LODWORD(v77) = 0;
              if ( v39 == v93 )
              {
                sub_13FDF40(&v91, v39, &v76);
                v39 = v92;
              }
              else
              {
                if ( v39 )
                {
                  *v39 = _mm_loadu_si128(&v76);
                  v39[1].m128i_i64[0] = v77;
                  v39 = v92;
                }
                v39 = (__m128i *)((char *)v39 + 24);
                v92 = v39;
              }
            }
          }
          v54 = 0;
          while ( 2 )
          {
            v55 = *v50;
            if ( v49 != *v50 )
            {
              while ( v55 == -2 )
              {
                v56 = v50 + 1;
                v54 = v50;
                if ( v53 == v50 + 1 )
                  goto LABEL_77;
                ++v50;
                v55 = *v56;
                if ( v49 == v55 )
                  goto LABEL_80;
              }
              if ( v53 != ++v50 )
                continue;
              if ( v54 )
              {
LABEL_77:
                *v54 = v49;
                v39 = v92;
                --v89;
                ++v84;
                goto LABEL_67;
              }
              goto LABEL_81;
            }
            break;
          }
LABEL_80:
          v39 = v92;
        }
      }
      v36 = v98;
    }
    while ( (__m128i *)((char *)v39 - (char *)v91) != (__m128i *)((char *)v99 - v98) );
LABEL_85:
    if ( v40 == v39 )
      break;
    v57 = v36;
    while ( v40->m128i_i64[0] == *(_QWORD *)v57 && v40[1].m128i_i32[0] == *(_DWORD *)(v57 + 16) )
    {
      v40 = (const __m128i *)((char *)v40 + 24);
      v57 += 24LL;
      if ( v40 == v39 )
        goto LABEL_90;
    }
  }
LABEL_90:
  if ( v36 )
    j_j___libc_free_0(v36);
  if ( v96 != v95 )
    _libc_free(v96);
  if ( v91 )
    j_j___libc_free_0((unsigned __int64)v91);
  if ( v86 != v85 )
    _libc_free((unsigned __int64)v86);
  if ( v109 )
    j_j___libc_free_0((unsigned __int64)v109);
  if ( v108 != v107 )
    _libc_free(v108);
  if ( v104 )
    j_j___libc_free_0((unsigned __int64)v104);
  if ( v103 != v102 )
    _libc_free((unsigned __int64)v103);
  v70 = &v81[8 * (unsigned int)v82];
  if ( v81 != v70 )
  {
    v72 = v81;
    while ( 1 )
    {
      v58 = *(_QWORD **)v72;
      v59 = *(_QWORD *)(*(_QWORD *)v72 + 8LL);
      if ( !v59 )
        goto LABEL_136;
      v60 = 0;
      do
      {
        while ( 1 )
        {
          v62 = (__int64)sub_1648700(v59);
          v67 = *(_BYTE *)(v62 + 16);
          if ( v67 <= 0x17u || *(_QWORD *)(v62 + 40) == v58[5] )
            goto LABEL_115;
          if ( v67 == 77 )
          {
            v68 = sub_1648720(v59);
            if ( (*(_BYTE *)(v62 + 23) & 0x40) != 0 )
              v61 = *(_QWORD *)(v62 - 8);
            else
              v61 = v62 - 24LL * (*(_DWORD *)(v62 + 20) & 0xFFFFFFF);
            v62 = sub_157EBA0(*(_QWORD *)(v61 + 8LL * v68 + 24LL * *(unsigned int *)(v62 + 56) + 8));
          }
          v63 = v58[1];
          if ( v63 )
          {
            if ( !*(_QWORD *)(v63 + 8) )
              break;
          }
          v64 = sub_15F4880((__int64)v58);
          v94 = sub_1649960((__int64)v58);
          LOWORD(v103) = 773;
          v95 = v65;
          v101 = &v94;
          v102 = ".pre-remat";
          sub_164B780(v64, (__int64 *)&v101);
          v66 = v64;
          v60 = 1;
          sub_15F2120(v66, v62);
LABEL_115:
          v59 = *(_QWORD *)(v59 + 8);
          if ( !v59 )
            goto LABEL_125;
        }
        v60 = 1;
        sub_15F22F0(v58, v62);
        v59 = *(_QWORD *)(v59 + 8);
      }
      while ( v59 );
LABEL_125:
      if ( v58[1] )
      {
        v71 |= v60;
        goto LABEL_127;
      }
LABEL_136:
      sub_15F20C0(v58);
      v71 = 1;
LABEL_127:
      v72 += 8;
      if ( v70 == v72 )
      {
        v70 = v81;
        break;
      }
    }
  }
  if ( v70 != v83 )
    _libc_free((unsigned __int64)v70);
  if ( v74 )
    j_j___libc_free_0(v74);
  if ( v78 != v80 )
    _libc_free((unsigned __int64)v78);
  return v71;
}
