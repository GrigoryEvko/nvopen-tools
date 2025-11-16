// Function: sub_1A0B880
// Address: 0x1a0b880
//
void __fastcall sub_1A0B880(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128 a4,
        __m128 a5,
        __m128i a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r12
  char v11; // al
  int v12; // r8d
  int v13; // r9d
  unsigned __int64 v14; // rdx
  int v15; // r13d
  int v16; // ebx
  _BYTE *v17; // rsi
  __int64 v18; // r15
  int v19; // eax
  int v20; // r8d
  unsigned __int64 v21; // r14
  int v22; // r9d
  __int64 v23; // rdx
  char *v24; // rax
  unsigned __int64 v25; // rdx
  __m128i *v26; // r13
  __int64 v27; // rax
  __m128i *v28; // r8
  __int64 v29; // r14
  __m128i *v30; // r12
  __int64 v31; // rbx
  __int64 v32; // rax
  __m128i *v33; // r9
  __int64 v34; // rdx
  __m128i *v35; // r8
  __int64 v36; // rax
  const __m128i *v37; // rax
  __m128i *v38; // r9
  __int64 v39; // rcx
  int v40; // r8d
  double v41; // xmm4_8
  double v42; // xmm5_8
  __int64 v43; // rax
  double v44; // xmm4_8
  double v45; // xmm5_8
  __m128i *v46; // rbx
  __int64 v47; // rsi
  __m128i *v48; // r15
  __int64 v49; // rsi
  unsigned __int8 *v50; // rsi
  _BYTE *v51; // rdi
  _BYTE *v52; // rbx
  unsigned __int64 v53; // r12
  __int64 v54; // rdi
  unsigned int v55; // ebx
  __int64 v56; // rdi
  char v57; // al
  char *v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rcx
  char v61; // bl
  __int64 v62; // r8
  int v63; // r9d
  __int64 v64; // rsi
  __int64 v65; // rsi
  _BYTE *v66; // rbx
  __int64 v67; // rdi
  int v68; // r9d
  __m128i *v69; // rsi
  __int64 v70; // rax
  char *v71; // r8
  __int64 v72; // r13
  unsigned int v73; // edx
  int v74; // eax
  __int64 v75; // rcx
  __int32 v76; // eax
  unsigned int *v77; // rdx
  __int64 v78; // rcx
  __int64 v79; // rdx
  __int64 v81; // [rsp+10h] [rbp-1F0h]
  int v82; // [rsp+10h] [rbp-1F0h]
  char *v83; // [rsp+18h] [rbp-1E8h]
  __m128i *v84; // [rsp+28h] [rbp-1D8h]
  int v85; // [rsp+28h] [rbp-1D8h]
  __int16 *v86; // [rsp+28h] [rbp-1D8h]
  __m128i *v87; // [rsp+28h] [rbp-1D8h]
  __int64 v88[4]; // [rsp+30h] [rbp-1D0h] BYREF
  __m128i v89; // [rsp+50h] [rbp-1B0h] BYREF
  unsigned int v90; // [rsp+60h] [rbp-1A0h]
  void *src; // [rsp+70h] [rbp-190h] BYREF
  __int64 v92; // [rsp+78h] [rbp-188h]
  _BYTE v93[128]; // [rsp+80h] [rbp-180h] BYREF
  _BYTE *v94; // [rsp+100h] [rbp-100h] BYREF
  __int64 v95; // [rsp+108h] [rbp-F8h]
  _BYTE v96[240]; // [rsp+110h] [rbp-F0h] BYREF

  v10 = a1;
  v94 = v96;
  v95 = 0x800000000LL;
  v11 = sub_1A04A80(
          (__int64 *)a2,
          (__int64)&v94,
          a3,
          *(double *)a4.m128_u64,
          *(double *)a5.m128_u64,
          *(double *)a6.m128i_i64,
          a7,
          a8,
          a9,
          a10);
  v14 = (unsigned int)v95;
  *(_BYTE *)(a1 + 752) |= v11;
  src = v93;
  v92 = 0x800000000LL;
  if ( (unsigned int)v14 > 8 )
  {
    sub_16CD150((__int64)&src, v93, v14, 16, v12, v13);
    v15 = v95;
    if ( !(_DWORD)v95 )
      goto LABEL_20;
  }
  else
  {
    v15 = v14;
    if ( !(_DWORD)v14 )
    {
      v28 = (__m128i *)v93;
      v26 = (__m128i *)v93;
      goto LABEL_69;
    }
  }
  v16 = 0;
  do
  {
    v17 = &v94[24 * v16];
    v18 = *(_QWORD *)v17;
    v89.m128i_i64[0] = *(_QWORD *)v17;
    v90 = *((_DWORD *)v17 + 4);
    if ( v90 > 0x40 )
    {
      sub_16A4FD0((__int64)&v89.m128i_i64[1], (const void **)v17 + 1);
      v18 = v89.m128i_i64[0];
    }
    else
    {
      v89.m128i_i64[1] = *((_QWORD *)v17 + 1);
    }
    v19 = sub_1A03A70(a1, v18);
    v21 = v89.m128i_u64[1];
    v22 = v19;
    if ( v90 > 0x40 )
      v21 = *(_QWORD *)v89.m128i_i64[1];
    v23 = (unsigned int)v92;
    if ( v21 > HIDWORD(v92) - (unsigned __int64)(unsigned int)v92 )
    {
      v85 = v19;
      sub_16CD150((__int64)&src, v93, v21 + (unsigned int)v92, 16, v20, v19);
      v23 = (unsigned int)v92;
      v22 = v85;
    }
    v24 = (char *)src + 16 * v23;
    if ( v21 )
    {
      v25 = v21;
      do
      {
        if ( v24 )
        {
          *(_DWORD *)v24 = v22;
          *((_QWORD *)v24 + 1) = v18;
        }
        v24 += 16;
        --v25;
      }
      while ( v25 );
      LODWORD(v23) = v92;
    }
    LODWORD(v92) = v21 + v23;
    if ( v90 > 0x40 && v89.m128i_i64[1] )
      j_j___libc_free_0_0(v89.m128i_i64[1]);
    ++v16;
  }
  while ( v16 != v15 );
LABEL_20:
  v26 = (__m128i *)src;
  v27 = 16LL * (unsigned int)v92;
  v28 = (__m128i *)((char *)src + v27);
  if ( !v27 )
  {
LABEL_69:
    v31 = 0;
    sub_1A00210(v26, v28);
    v38 = 0;
    goto LABEL_27;
  }
  v29 = v27 >> 4;
  v30 = (__m128i *)((char *)src + v27);
  while ( 1 )
  {
    v31 = v29;
    v32 = sub_2207800(16 * v29, &unk_435FF63);
    v33 = (__m128i *)v32;
    if ( v32 )
      break;
    v29 >>= 1;
    if ( !v29 )
    {
      v28 = v30;
      v10 = a1;
      goto LABEL_69;
    }
  }
  a5 = (__m128)_mm_loadu_si128(v26);
  v34 = v32 + v31 * 16;
  v35 = v30;
  v36 = v32 + 16;
  v10 = a1;
  *(__m128 *)(v36 - 16) = a5;
  if ( v34 == v36 )
  {
    v37 = v33;
  }
  else
  {
    do
    {
      a4 = (__m128)_mm_loadu_si128((const __m128i *)(v36 - 16));
      v36 += 16;
      *(__m128 *)(v36 - 16) = a4;
    }
    while ( v34 != v36 );
    v37 = &v33[v31 - 1];
  }
  a6 = _mm_loadu_si128(v37);
  v84 = v33;
  *v26 = a6;
  sub_1A008E0(v26, v35, v33, v29);
  v38 = v84;
LABEL_27:
  j_j___libc_free_0(v38, v31 * 16);
  v43 = sub_1A0B650(v10, a2, &src, a3, a4, *(double *)a5.m128_u64, *(double *)a6.m128i_i64, v41, v42, a9, a10, v39, v40);
  v46 = (__m128i *)v43;
  if ( v43 )
  {
    if ( a2 != v43 )
    {
      sub_164D160(
        a2,
        v43,
        a3,
        *(double *)a4.m128_u64,
        *(double *)a5.m128_u64,
        *(double *)a6.m128i_i64,
        v44,
        v45,
        a9,
        a10);
      if ( v46[1].m128i_i8[0] > 0x17u )
      {
        v47 = *(_QWORD *)(a2 + 48);
        if ( v47 )
        {
          v48 = v46 + 3;
          v89.m128i_i64[0] = *(_QWORD *)(a2 + 48);
          sub_1623A60((__int64)&v89, v47, 2);
          if ( &v46[3] != &v89 )
            goto LABEL_32;
          goto LABEL_66;
        }
      }
      goto LABEL_36;
    }
    v51 = src;
    goto LABEL_37;
  }
  v55 = v92;
  v56 = *(_QWORD *)(a2 + 8);
  if ( v56 && !*(_QWORD *)(v56 + 8) )
  {
    v57 = *(_BYTE *)(a2 + 16);
    if ( v57 == 39 )
    {
      if ( *((_BYTE *)sub_1648700(v56) + 16) == 35 )
      {
        v69 = (__m128i *)src;
        v70 = 16LL * v55;
        v71 = (char *)src + v70 - 16;
        v72 = *((_QWORD *)v71 + 1);
        if ( *(_BYTE *)(v72 + 16) == 13 )
        {
          v73 = *(_DWORD *)(v72 + 32);
          if ( v73 <= 0x40 )
          {
            v75 = 64 - v73;
            if ( *(_QWORD *)(v72 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v73) )
              goto LABEL_61;
          }
          else
          {
            v82 = *(_DWORD *)(v72 + 32);
            v83 = (char *)src + v70 - 16;
            v87 = (__m128i *)src;
            v74 = sub_16A58F0(v72 + 24);
            v69 = v87;
            v71 = v83;
            if ( v82 != v74 )
              goto LABEL_61;
          }
          v76 = *(_DWORD *)v71;
          v89.m128i_i64[1] = v72;
          LODWORD(v92) = v55 - 1;
          v89.m128i_i32[0] = v76;
          sub_19FFA50((__int64)&src, v69, &v89, v75, (__int64)v71, v68);
          v55 = v92;
        }
      }
    }
    else if ( v57 == 40 && *((_BYTE *)sub_1648700(v56) + 16) == 36 )
    {
      v58 = (char *)src + 16 * v55;
      if ( *(_BYTE *)(*((_QWORD *)v58 - 1) + 16LL) == 14 )
      {
        v81 = *((_QWORD *)v58 - 1);
        a3 = (__m128)0xBFF0000000000000LL;
        v86 = (__int16 *)sub_1698280();
        sub_169D3F0((__int64)v88, -1.0);
        sub_169E320(&v89.m128i_i64[1], v88, v86);
        sub_1698460((__int64)v88);
        sub_16A3360((__int64)&v89, *(__int16 **)(v81 + 32), 0, (bool *)v88);
        v61 = sub_1594120(v81, (__int64)&v89, v59, v60);
        sub_127D120(&v89.m128i_i64[1]);
        if ( v61 )
        {
          v77 = (unsigned int *)((char *)src + 16 * (unsigned int)v92 - 16);
          v78 = *v77;
          v79 = *((_QWORD *)v77 + 1);
          LODWORD(v92) = v92 - 1;
          v89.m128i_i64[1] = v79;
          v89.m128i_i32[0] = v78;
          sub_19FFA50((__int64)&src, (__m128i *)src, &v89, v78, v62, v63);
        }
        v55 = v92;
      }
    }
  }
LABEL_61:
  if ( v55 == 1 )
  {
    v51 = src;
    v64 = *((_QWORD *)src + 1);
    if ( v64 != a2 )
    {
      sub_164D160(
        a2,
        v64,
        a3,
        *(double *)a4.m128_u64,
        *(double *)a5.m128_u64,
        *(double *)a6.m128i_i64,
        v44,
        v45,
        a9,
        a10);
      v46 = (__m128i *)*((_QWORD *)src + 1);
      if ( v46[1].m128i_i8[0] > 0x17u )
      {
        v65 = *(_QWORD *)(a2 + 48);
        v48 = v46 + 3;
        v89.m128i_i64[0] = v65;
        if ( v65 )
        {
          sub_1623A60((__int64)&v89, v65, 2);
          if ( v48 != &v89 )
          {
LABEL_32:
            v49 = v46[3].m128i_i64[0];
            if ( !v49 )
              goto LABEL_34;
            goto LABEL_33;
          }
LABEL_66:
          if ( v89.m128i_i64[0] )
            sub_161E7C0((__int64)&v89, v89.m128i_i64[0]);
          goto LABEL_36;
        }
        if ( v48 != &v89 )
        {
          v49 = v46[3].m128i_i64[0];
          if ( v49 )
          {
LABEL_33:
            sub_161E7C0((__int64)v48, v49);
LABEL_34:
            v50 = (unsigned __int8 *)v89.m128i_i64[0];
            v46[3].m128i_i64[0] = v89.m128i_i64[0];
            if ( v50 )
              sub_1623210((__int64)&v89, v50, (__int64)v48);
          }
        }
      }
LABEL_36:
      v89.m128i_i64[0] = a2;
      sub_1A062A0(v10 + 64, &v89);
      v51 = src;
    }
LABEL_37:
    if ( v51 != v93 )
      _libc_free((unsigned __int64)v51);
    v52 = v94;
    v53 = (unsigned __int64)&v94[24 * (unsigned int)v95];
    if ( v94 != (_BYTE *)v53 )
    {
      do
      {
        v53 -= 24LL;
        if ( *(_DWORD *)(v53 + 16) > 0x40u )
        {
          v54 = *(_QWORD *)(v53 + 8);
          if ( v54 )
            j_j___libc_free_0_0(v54);
        }
      }
      while ( v52 != (_BYTE *)v53 );
LABEL_44:
      v53 = (unsigned __int64)v94;
      goto LABEL_45;
    }
    goto LABEL_45;
  }
  sub_1A08FD0(v10, a2, (__int64 *)&src);
  if ( src != v93 )
    _libc_free((unsigned __int64)src);
  v66 = v94;
  v53 = (unsigned __int64)&v94[24 * (unsigned int)v95];
  if ( v94 != (_BYTE *)v53 )
  {
    do
    {
      v53 -= 24LL;
      if ( *(_DWORD *)(v53 + 16) > 0x40u )
      {
        v67 = *(_QWORD *)(v53 + 8);
        if ( v67 )
          j_j___libc_free_0_0(v67);
      }
    }
    while ( v66 != (_BYTE *)v53 );
    goto LABEL_44;
  }
LABEL_45:
  if ( (_BYTE *)v53 != v96 )
    _libc_free(v53);
}
