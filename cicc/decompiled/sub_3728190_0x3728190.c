// Function: sub_3728190
// Address: 0x3728190
//
void __fastcall sub_3728190(_QWORD *a1, __int64 a2, __int64 a3, __int64 *a4, _BYTE *a5, __int64 a6)
{
  unsigned __int64 v7; // r12
  unsigned int v9; // r14d
  char *v10; // rax
  __int64 v11; // r14
  char *v12; // rdx
  unsigned __int64 v13; // rdi
  __int64 v14; // rdi
  _DWORD *v15; // rax
  _DWORD *i; // rdx
  __int64 *v17; // r15
  unsigned __int64 v18; // rbx
  _BYTE *v19; // r14
  int v20; // r12d
  __int64 v21; // rax
  const __m128i *v22; // rsi
  int v23; // edx
  _DWORD *v24; // rdi
  int v25; // r11d
  unsigned int v26; // ecx
  _DWORD *v27; // rax
  int v28; // r8d
  _DWORD *v29; // rax
  __int64 v30; // rax
  __int64 v31; // r12
  int v32; // ecx
  const __m128i *v33; // rax
  const __m128i *v34; // rsi
  int v35; // edx
  int v36; // r14d
  __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // r12
  void (__fastcall *v40)(__int64, _QWORD, _QWORD); // rbx
  __int64 v41; // rax
  __int64 v42; // r8
  __int64 v43; // r9
  __int16 v44; // ax
  __int64 v45; // r8
  __int64 v46; // rdx
  __int16 v47; // ax
  __int64 v48; // rdx
  char v49; // r9
  char *v50; // r12
  char *v51; // rbx
  unsigned __int64 v52; // rdi
  __int64 v53; // rsi
  __int64 *v54; // r13
  __int64 *v55; // rbx
  unsigned int v56; // ecx
  __int64 v57; // rdi
  __int64 *v58; // rbx
  __int64 *v59; // r12
  __int64 v60; // rdi
  int v61; // eax
  int v62; // eax
  int v63; // ecx
  unsigned int v64; // esi
  int v65; // r8d
  int v66; // r11d
  _DWORD *v67; // r10
  int v68; // ecx
  int v69; // r11d
  unsigned int v70; // esi
  int v71; // r8d
  size_t v72; // rdx
  __int128 v73; // [rsp-10h] [rbp-2B0h]
  const __m128i *v74; // [rsp+0h] [rbp-2A0h]
  __int16 v77; // [rsp+3Ch] [rbp-264h] BYREF
  __int16 v78; // [rsp+3Eh] [rbp-262h] BYREF
  const __m128i *v79; // [rsp+40h] [rbp-260h] BYREF
  __m128i *v80; // [rsp+48h] [rbp-258h]
  const __m128i *v81; // [rsp+50h] [rbp-250h]
  unsigned __int64 v82; // [rsp+60h] [rbp-240h] BYREF
  __m128i *v83; // [rsp+68h] [rbp-238h]
  __m128i *v84; // [rsp+70h] [rbp-230h]
  _BYTE *v85; // [rsp+80h] [rbp-220h] BYREF
  __int64 v86; // [rsp+88h] [rbp-218h]
  _BYTE v87[4]; // [rsp+90h] [rbp-210h] BYREF
  char v88; // [rsp+94h] [rbp-20Ch] BYREF
  __int64 v89; // [rsp+A0h] [rbp-200h] BYREF
  _DWORD *v90; // [rsp+A8h] [rbp-1F8h]
  __int64 v91; // [rsp+B0h] [rbp-1F0h]
  unsigned int v92; // [rsp+B8h] [rbp-1E8h]
  _QWORD v93[4]; // [rsp+C0h] [rbp-1E0h] BYREF
  _BYTE *v94; // [rsp+E0h] [rbp-1C0h] BYREF
  __int64 v95; // [rsp+E8h] [rbp-1B8h]
  _BYTE v96[32]; // [rsp+F0h] [rbp-1B0h] BYREF
  __m128i v97; // [rsp+110h] [rbp-190h] BYREF
  __int64 v98; // [rsp+150h] [rbp-150h] BYREF
  char *v99; // [rsp+160h] [rbp-140h]
  int v100; // [rsp+168h] [rbp-138h]
  char v101; // [rsp+170h] [rbp-130h] BYREF
  __int64 *v102; // [rsp+1A8h] [rbp-F8h]
  int v103; // [rsp+1B0h] [rbp-F0h]
  char v104; // [rsp+1B8h] [rbp-E8h] BYREF
  __int64 *v105; // [rsp+1D8h] [rbp-C8h]
  unsigned int v106; // [rsp+1E0h] [rbp-C0h]
  char v107; // [rsp+1E8h] [rbp-B8h] BYREF
  __int64 v108; // [rsp+258h] [rbp-48h]
  unsigned int v109; // [rsp+268h] [rbp-38h]

  v7 = (unsigned __int64)a5;
  v9 = *(_DWORD *)(a2 + 216);
  v94 = v96;
  v95 = 0x100000000LL;
  if ( v9 )
  {
    a5 = v96;
    v72 = 24;
    if ( v9 == 1
      || (sub_C8D5F0((__int64)&v94, v96, v9, 0x18u, (__int64)v96, v9),
          a5 = v94,
          (v72 = 24LL * *(unsigned int *)(a2 + 216)) != 0) )
    {
      memcpy(a5, *(const void **)(a2 + 208), v72);
    }
    LODWORD(v95) = v9;
  }
  v79 = 0;
  v10 = v87;
  v80 = 0;
  v81 = 0;
  v82 = 0;
  v83 = 0;
  v84 = 0;
  v85 = v87;
  v86 = 0x100000000LL;
  if ( v7 )
  {
    v11 = 4 * v7;
    v12 = &v88;
    if ( v7 == 1
      || (sub_C8D5F0((__int64)&v85, v87, v7, 4u, (__int64)a5, a6),
          v10 = &v85[4 * (unsigned int)v86],
          v12 = &v85[v11],
          v10 != &v85[v11]) )
    {
      do
      {
        if ( v10 )
          *(_DWORD *)v10 = 0;
        v10 += 4;
      }
      while ( v12 != v10 );
    }
    LODWORD(v86) = v7;
  }
  v89 = 0;
  if ( !(_DWORD)v95 )
  {
    v90 = 0;
    v17 = &a4[v7];
    v91 = 0;
    v92 = 0;
    if ( a4 == v17 )
      goto LABEL_43;
    goto LABEL_31;
  }
  v13 = (((((((4 * (int)v95 / 3u + 1) | ((unsigned __int64)(4 * (int)v95 / 3u + 1) >> 1)) >> 2)
          | (4 * (int)v95 / 3u + 1)
          | ((unsigned __int64)(4 * (int)v95 / 3u + 1) >> 1)) >> 4)
        | (((4 * (int)v95 / 3u + 1) | ((unsigned __int64)(4 * (int)v95 / 3u + 1) >> 1)) >> 2)
        | (4 * (int)v95 / 3u + 1)
        | ((unsigned __int64)(4 * (int)v95 / 3u + 1) >> 1)) >> 8)
      | (((((4 * (int)v95 / 3u + 1) | ((unsigned __int64)(4 * (int)v95 / 3u + 1) >> 1)) >> 2)
        | (4 * (int)v95 / 3u + 1)
        | ((unsigned __int64)(4 * (int)v95 / 3u + 1) >> 1)) >> 4)
      | (((4 * (int)v95 / 3u + 1) | ((unsigned __int64)(4 * (int)v95 / 3u + 1) >> 1)) >> 2)
      | (4 * (int)v95 / 3u + 1)
      | ((unsigned __int64)(4 * (int)v95 / 3u + 1) >> 1);
  v14 = ((v13 >> 16) | v13) + 1;
  v92 = v14;
  v15 = (_DWORD *)sub_C7D670(8 * v14, 4);
  v91 = 0;
  v90 = v15;
  for ( i = &v15[2 * v92]; i != v15; v15 += 2 )
  {
    if ( v15 )
      *v15 = -1;
  }
  v17 = &a4[v7];
  if ( a4 != v17 )
  {
LABEL_31:
    v31 = 0;
    v32 = 0;
    v33 = &v97;
    do
    {
      v35 = *(_DWORD *)(*(_QWORD *)(*a4 + 80) + 36LL);
      if ( !v35 || v35 == 3 )
      {
        v36 = v32 + 1;
        *(_DWORD *)&v85[v31] = v32;
        v37 = *a4;
        if ( *(_BYTE *)(a3 + 3769) )
          v37 = *(_QWORD *)(v37 + 408);
        v38 = *(_QWORD *)(v37 + 192);
        v34 = v80;
        v97.m128i_i8[8] = 0;
        v97.m128i_i64[0] = v38;
        if ( v80 == v81 )
        {
          v74 = v33;
          sub_3723EF0((unsigned __int64 *)&v79, v80, v33);
          v33 = v74;
        }
        else
        {
          if ( v80 )
          {
            *v80 = _mm_loadu_si128(&v97);
            v34 = v80;
          }
          v80 = (__m128i *)&v34[1];
        }
        v32 = v36;
      }
      ++a4;
      v31 += 4;
    }
    while ( v17 != a4 );
  }
  v18 = (unsigned __int64)v94;
  v19 = &v94[24 * (unsigned int)v95];
  if ( v19 != v94 )
  {
    v20 = 0;
    do
    {
      if ( v92 )
      {
        v23 = *(_DWORD *)(v18 + 16);
        v24 = 0;
        v25 = 1;
        v26 = (v92 - 1) & (37 * v23);
        v27 = &v90[2 * v26];
        v28 = *v27;
        if ( v23 == *v27 )
        {
LABEL_24:
          v29 = v27 + 1;
          goto LABEL_25;
        }
        while ( v28 != -1 )
        {
          if ( !v24 && v28 == -2 )
            v24 = v27;
          v26 = (v92 - 1) & (v25 + v26);
          v27 = &v90[2 * v26];
          v28 = *v27;
          if ( v23 == *v27 )
            goto LABEL_24;
          ++v25;
        }
        if ( !v24 )
          v24 = v27;
        ++v89;
        v61 = v91 + 1;
        if ( 4 * ((int)v91 + 1) < 3 * v92 )
        {
          if ( v92 - HIDWORD(v91) - v61 > v92 >> 3 )
            goto LABEL_88;
          sub_A09770((__int64)&v89, v92);
          if ( !v92 )
          {
LABEL_121:
            LODWORD(v91) = v91 + 1;
            BUG();
          }
          v68 = *(_DWORD *)(v18 + 16);
          v67 = 0;
          v69 = 1;
          v70 = (v92 - 1) & (37 * v68);
          v24 = &v90[2 * v70];
          v71 = *v24;
          v61 = v91 + 1;
          if ( *v24 == v68 )
            goto LABEL_88;
          while ( v71 != -1 )
          {
            if ( v71 == -2 && !v67 )
              v67 = v24;
            v70 = (v92 - 1) & (v69 + v70);
            v24 = &v90[2 * v70];
            v71 = *v24;
            if ( v68 == *v24 )
              goto LABEL_88;
            ++v69;
          }
          goto LABEL_97;
        }
      }
      else
      {
        ++v89;
      }
      sub_A09770((__int64)&v89, 2 * v92);
      if ( !v92 )
        goto LABEL_121;
      v63 = *(_DWORD *)(v18 + 16);
      v64 = (v92 - 1) & (37 * v63);
      v24 = &v90[2 * v64];
      v65 = *v24;
      v61 = v91 + 1;
      if ( v63 == *v24 )
        goto LABEL_88;
      v66 = 1;
      v67 = 0;
      while ( v65 != -1 )
      {
        if ( !v67 && v65 == -2 )
          v67 = v24;
        v64 = (v92 - 1) & (v66 + v64);
        v24 = &v90[2 * v64];
        v65 = *v24;
        if ( v63 == *v24 )
          goto LABEL_88;
        ++v66;
      }
LABEL_97:
      if ( v67 )
        v24 = v67;
LABEL_88:
      LODWORD(v91) = v61;
      if ( *v24 != -1 )
        --HIDWORD(v91);
      v62 = *(_DWORD *)(v18 + 16);
      v24[1] = 0;
      *v24 = v62;
      v29 = v24 + 1;
LABEL_25:
      *v29 = v20;
      if ( *(_BYTE *)(a3 + 3769) )
      {
        if ( *(_BYTE *)(v18 + 8) != 1 )
          goto LABEL_120;
        v21 = *(_QWORD *)v18;
        v22 = v83;
        v97.m128i_i8[8] = 1;
        v97.m128i_i64[0] = v21;
        if ( v83 == v84 )
          goto LABEL_91;
        if ( v83 )
          goto LABEL_19;
      }
      else
      {
        if ( *(_BYTE *)(v18 + 8) )
LABEL_120:
          abort();
        v30 = *(_QWORD *)v18;
        v22 = v83;
        v97.m128i_i8[8] = 0;
        v97.m128i_i64[0] = v30;
        if ( v83 == v84 )
        {
LABEL_91:
          sub_3723EF0(&v82, v22, &v97);
          goto LABEL_21;
        }
        if ( v83 )
        {
LABEL_19:
          *v83 = _mm_loadu_si128(&v97);
          v22 = v83;
        }
      }
      v83 = (__m128i *)&v22[1];
LABEL_21:
      v18 += 24LL;
      ++v20;
    }
    while ( v19 != (_BYTE *)v18 );
  }
LABEL_43:
  if ( v80 != v79 )
  {
    v39 = a1[28];
    v40 = *(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v39 + 176LL);
    v41 = sub_31DA6B0((__int64)a1);
    v40(v39, *(_QWORD *)(v41 + 192), 0);
    sub_3724C70(a2, a1, "names", (char *)5, v42, v43);
    v44 = 11;
    v45 = v80 - v79;
    v46 = v45 - 1;
    if ( ((v45 - 1) & 0xFFFFFFFFFFFFFF00LL) != 0 )
    {
      v44 = 5;
      if ( (v46 & 0xFFFFFFFFFFFF0000LL) != 0 )
        v44 = (v46 != (unsigned int)v46) + 6;
    }
    v77 = v44;
    v47 = 11;
    v48 = ((__int64)((__int64)v83->m128i_i64 - v82) >> 4) - 1;
    if ( (v48 & 0xFFFFFFFFFFFFFF00LL) != 0 )
    {
      v47 = 5;
      if ( (v48 & 0xFFFFFFFFFFFF0000LL) != 0 )
        v47 = (v48 != (unsigned int)v48) + 6;
    }
    v78 = v47;
    v93[0] = &v89;
    *((_QWORD *)&v73 + 1) = v93;
    *(_QWORD *)&v73 = sub_3724070;
    v49 = *(_BYTE *)(a3 + 3769);
    v93[1] = &v78;
    v93[2] = &v85;
    v93[3] = &v77;
    sub_37272A0(
      (__int64)&v97,
      (__int64)a1,
      a2,
      (__int64)v79,
      v45,
      v49,
      v82,
      (__int64)((__int64)v83->m128i_i64 - v82) >> 4,
      v73);
    sub_3726A90((unsigned __int16 *)&v97);
    v50 = v99;
    v51 = &v99[8 * v100];
    if ( v99 != v51 )
    {
      do
      {
        v52 = *(_QWORD *)(*(_QWORD *)v50 + 16LL);
        if ( v52 != *(_QWORD *)v50 + 32LL )
          _libc_free(v52);
        v50 += 8;
      }
      while ( v51 != v50 );
    }
    v53 = 16LL * v109;
    sub_C7D6A0(v108, v53, 8);
    v54 = v102;
    v55 = &v102[v103];
    if ( v102 != v55 )
    {
      LOBYTE(v56) = 0;
      while ( 1 )
      {
        v57 = *v54++;
        v53 = 4096LL << v56;
        sub_C7D6A0(v57, 4096LL << v56, 16);
        if ( v55 == v54 )
          break;
        v56 = (unsigned int)(v54 - v102) >> 7;
        if ( (unsigned int)(v54 - v102) > 0xEFF )
          LOBYTE(v56) = 30;
      }
    }
    v58 = v105;
    v59 = &v105[2 * v106];
    if ( v105 != v59 )
    {
      do
      {
        v53 = v58[1];
        v60 = *v58;
        v58 += 2;
        sub_C7D6A0(v60, v53, 16);
      }
      while ( v59 != v58 );
      v59 = v105;
    }
    if ( v59 != (__int64 *)&v107 )
      _libc_free((unsigned __int64)v59);
    if ( v102 != (__int64 *)&v104 )
      _libc_free((unsigned __int64)v102);
    if ( v99 != &v101 )
      _libc_free((unsigned __int64)v99);
    sub_C65770(&v98, v53);
  }
  sub_C7D6A0((__int64)v90, 8LL * v92, 4);
  if ( v85 != v87 )
    _libc_free((unsigned __int64)v85);
  if ( v82 )
    j_j___libc_free_0(v82);
  if ( v79 )
    j_j___libc_free_0((unsigned __int64)v79);
  if ( v94 != v96 )
    _libc_free((unsigned __int64)v94);
}
