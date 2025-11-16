// Function: sub_1CF6830
// Address: 0x1cf6830
//
void __fastcall sub_1CF6830(
        _QWORD *a1,
        __m128 a2,
        __m128i a3,
        __m128i a4,
        __m128i a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 v10; // r13
  int v11; // r8d
  unsigned __int64 v12; // r12
  __int64 v13; // rax
  unsigned int v14; // r10d
  __int64 v15; // r14
  unsigned __int64 v16; // rbx
  __int64 v17; // rcx
  __int64 i; // rax
  __int64 v19; // rdx
  unsigned int v20; // r13d
  unsigned __int64 v21; // r14
  char *v23; // rax
  char *v24; // rsi
  char *v25; // rbx
  const __m128i *v26; // r13
  const __m128i *v27; // r14
  char *v28; // rsi
  __int64 v29; // r13
  __int64 v30; // r14
  char *v31; // rsi
  char *v32; // r14
  __int64 v33; // r13
  unsigned __int64 v34; // rax
  double v35; // xmm4_8
  double v36; // xmm5_8
  char *v37; // r13
  unsigned __int32 v38; // eax
  unsigned __int64 v39; // r8
  __int64 v40; // r10
  unsigned int v41; // edi
  __int64 v42; // rdx
  __int64 v43; // rsi
  __int64 v44; // rcx
  int v45; // ecx
  unsigned __int64 v46; // r11
  unsigned int v47; // r9d
  unsigned int v49; // eax
  __m128i *v50; // rcx
  __m128i *v51; // rsi
  int v52; // ebx
  const __m128i *v53; // r9
  __int64 v54; // rdx
  unsigned int v55; // eax
  unsigned int v56; // r8d
  __int64 v57; // rdx
  int v58; // ecx
  unsigned __int64 v59; // rdi
  unsigned __int64 v60; // rdi
  __int64 v63; // r8
  char *v64; // rax
  unsigned __int64 v65; // r8
  __int64 v66; // rsi
  unsigned __int64 v67; // rax
  __int64 *v68; // r13
  __int64 *v69; // rdi
  __int64 v70; // r9
  __int64 v71; // rax
  int v72; // [rsp+4h] [rbp-1BCh]
  unsigned __int64 v73; // [rsp+8h] [rbp-1B8h]
  int v74; // [rsp+8h] [rbp-1B8h]
  unsigned int v75; // [rsp+10h] [rbp-1B0h]
  int v76; // [rsp+10h] [rbp-1B0h]
  unsigned int v77; // [rsp+10h] [rbp-1B0h]
  unsigned int v78; // [rsp+18h] [rbp-1A8h]
  int v79; // [rsp+18h] [rbp-1A8h]
  unsigned __int64 v80; // [rsp+18h] [rbp-1A8h]
  unsigned __int64 v81; // [rsp+18h] [rbp-1A8h]
  __int64 v82; // [rsp+20h] [rbp-1A0h]
  unsigned int v83; // [rsp+20h] [rbp-1A0h]
  unsigned int v84; // [rsp+20h] [rbp-1A0h]
  unsigned int v85; // [rsp+20h] [rbp-1A0h]
  unsigned __int64 v86; // [rsp+28h] [rbp-198h]
  unsigned __int64 v87; // [rsp+28h] [rbp-198h]
  __int64 v88; // [rsp+30h] [rbp-190h]
  unsigned int v89; // [rsp+30h] [rbp-190h]
  unsigned __int64 v90; // [rsp+30h] [rbp-190h]
  unsigned int v91; // [rsp+38h] [rbp-188h]
  const __m128i *v92; // [rsp+38h] [rbp-188h]
  __int64 v93; // [rsp+38h] [rbp-188h]
  unsigned int v94; // [rsp+38h] [rbp-188h]
  unsigned int v95; // [rsp+38h] [rbp-188h]
  unsigned __int64 v96; // [rsp+38h] [rbp-188h]
  unsigned __int64 v97; // [rsp+38h] [rbp-188h]
  __int64 v98; // [rsp+38h] [rbp-188h]
  const __m128i *v99; // [rsp+40h] [rbp-180h] BYREF
  const __m128i *v100; // [rsp+48h] [rbp-178h]
  __m128i *v101; // [rsp+50h] [rbp-170h]
  void *src; // [rsp+60h] [rbp-160h] BYREF
  char *v103; // [rsp+68h] [rbp-158h]
  char *v104; // [rsp+70h] [rbp-150h]
  __m128i v105; // [rsp+80h] [rbp-140h] BYREF
  __m128i v106; // [rsp+90h] [rbp-130h] BYREF
  __m128i v107; // [rsp+A0h] [rbp-120h] BYREF
  __m128i v108; // [rsp+B0h] [rbp-110h] BYREF

  v10 = a1[13];
  v88 = a1[14];
  if ( v10 == v88 )
  {
    v99 = 0;
    v12 = 0;
    v100 = 0;
    v101 = 0;
    goto LABEL_16;
  }
  v86 = 0;
  v11 = 0;
  v12 = 0;
  do
  {
    v91 = v11;
    v13 = sub_3950BA0(a1[4], *(_QWORD *)(v10 + 48));
    v11 = v91;
    v14 = *(_DWORD *)(v13 + 24);
    v15 = v13;
    if ( v14 <= v91 )
    {
      v16 = (v14 + 63) >> 6;
      goto LABEL_5;
    }
    v16 = (v14 + 63) >> 6;
    v45 = v91 & 0x3F;
    v46 = (v91 + 63) >> 6;
    v47 = (v91 + 63) >> 6;
    if ( v14 <= v86 << 6 )
    {
      if ( v86 > v46 )
        goto LABEL_100;
      goto LABEL_70;
    }
    v72 = v91 & 0x3F;
    v73 = (v91 + 63) >> 6;
    v63 = 2 * v86;
    v75 = (v91 + 63) >> 6;
    v78 = *(_DWORD *)(v13 + 24);
    if ( 2 * v86 < v16 )
      v63 = (v14 + 63) >> 6;
    v82 = v63;
    v93 = 8 * v63;
    v64 = realloc(v12, 8 * v63, 8 * (int)v63, v45, v63, v47);
    v65 = v82;
    v45 = v72;
    v12 = (unsigned __int64)v64;
    v14 = v78;
    v47 = v75;
    v46 = v73;
    if ( !v64 )
    {
      if ( v93 )
      {
        sub_16BD1C0("Allocation failed", 1u);
        v45 = v72;
        v46 = v73;
        v47 = v75;
        v65 = v82;
        v14 = v78;
        v16 = (unsigned int)(*(_DWORD *)(v15 + 24) + 63) >> 6;
      }
      else
      {
        v71 = malloc(1u);
        v14 = v78;
        v65 = v82;
        v47 = v75;
        v46 = v73;
        v12 = v71;
        v45 = v72;
        if ( !v71 )
        {
          sub_16BD1C0("Allocation failed", 1u);
          v14 = v78;
          v65 = v82;
          v47 = v75;
          v46 = v73;
          v45 = v72;
          v16 = (unsigned int)(*(_DWORD *)(v15 + 24) + 63) >> 6;
        }
      }
    }
    if ( v65 > v46 && v65 != v46 )
    {
      v74 = v45;
      v77 = v47;
      v81 = v65;
      v85 = v14;
      v97 = v46;
      memset((void *)(v12 + 8 * v46), 0, 8 * (v65 - v46));
      v45 = v74;
      v47 = v77;
      v65 = v81;
      v14 = v85;
      v46 = v97;
    }
    if ( v45 )
    {
      v66 = (unsigned int)v86;
      *(_QWORD *)(v12 + 8LL * (v47 - 1)) &= ~(-1LL << v45);
      v67 = v65 - (unsigned int)v86;
      if ( v65 == (unsigned int)v86 )
        goto LABEL_99;
    }
    else
    {
      v66 = (unsigned int)v86;
      v67 = v65 - (unsigned int)v86;
      if ( v65 == (unsigned int)v86 )
        goto LABEL_99;
    }
    v76 = v45;
    v80 = v46;
    v84 = v47;
    v87 = v65;
    v95 = v14;
    memset((void *)(v12 + 8 * v66), 0, 8 * v67);
    v45 = v76;
    v46 = v80;
    v47 = v84;
    v65 = v87;
    v14 = v95;
LABEL_99:
    v86 = v65;
    if ( v65 > v46 )
    {
LABEL_100:
      if ( v86 != v46 )
      {
        v79 = v45;
        v83 = v47;
        v94 = v14;
        memset((void *)(v12 + 8 * v46), 0, 8 * (v86 - v46));
        v45 = v79;
        v47 = v83;
        v14 = v94;
      }
    }
LABEL_70:
    if ( v45 )
      *(_QWORD *)(v12 + 8LL * (v47 - 1)) &= ~(-1LL << v45);
    v11 = v14;
LABEL_5:
    if ( v16 )
    {
      v17 = *(_QWORD *)(v15 + 8);
      for ( i = 0; i != v16; ++i )
        *(_QWORD *)(v12 + 8 * i) |= *(_QWORD *)(v17 + 8 * i);
    }
    v10 += 64;
  }
  while ( v10 != v88 );
  v99 = 0;
  v100 = 0;
  v101 = 0;
  if ( v11 )
  {
    v19 = 0;
    v20 = (unsigned int)(v11 - 1) >> 6;
    v21 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v11;
    while ( 1 )
    {
      _RCX = *(_QWORD *)(v12 + 8 * v19);
      if ( v20 == (_DWORD)v19 )
        _RCX = v21 & *(_QWORD *)(v12 + 8 * v19);
      if ( _RCX )
        break;
      if ( v20 + 1 == ++v19 )
        goto LABEL_15;
    }
    __asm { tzcnt   rcx, rcx }
    if ( ((_DWORD)v19 << 6) + (_DWORD)_RCX != -1 )
    {
      v49 = ((_DWORD)v19 << 6) + _RCX;
      v50 = 0;
      v51 = 0;
      v52 = v11;
      v53 = &v105;
      while ( 1 )
      {
        v54 = *(_QWORD *)(*(_QWORD *)(a1[4] + 40LL) + 8LL * v49);
        v105 = 0u;
        v106 = (__m128i)0xFFFFFFFFFFFFFFFFLL;
        v107 = 0u;
        v108.m128i_i64[0] = v54;
        v108.m128i_i32[2] = 0;
        if ( v51 == v50 )
        {
          v89 = v49;
          v92 = v53;
          sub_1CF64E0(&v99, v51, v53);
          v49 = v89;
          v53 = v92;
        }
        else
        {
          if ( v51 )
          {
            a2 = (__m128)_mm_loadu_si128(&v105);
            *v51 = (__m128i)a2;
            a3 = _mm_loadu_si128(&v106);
            v51[1] = a3;
            a4 = _mm_loadu_si128(&v107);
            v51[2] = a4;
            a5 = _mm_loadu_si128(&v108);
            v51[3] = a5;
            v51 = (__m128i *)v100;
          }
          v100 = v51 + 4;
        }
        v55 = v49 + 1;
        if ( v52 == v55 )
          break;
        v56 = v55 >> 6;
        if ( v20 < v55 >> 6 )
          break;
        v57 = v56;
        v58 = 64 - (v55 & 0x3F);
        v59 = 0xFFFFFFFFFFFFFFFFLL >> v58;
        if ( v58 == 64 )
          v59 = 0;
        v60 = ~v59;
        while ( 1 )
        {
          _RAX = *(_QWORD *)(v12 + 8 * v57);
          if ( v56 == (_DWORD)v57 )
            _RAX = v60 & *(_QWORD *)(v12 + 8 * v57);
          if ( (_DWORD)v57 == v20 )
            _RAX &= v21;
          if ( _RAX )
            break;
          if ( v20 < (unsigned int)++v57 )
            goto LABEL_15;
        }
        __asm { tzcnt   rax, rax }
        v49 = ((_DWORD)v57 << 6) + _RAX;
        if ( v49 == -1 )
          break;
        v51 = (__m128i *)v100;
        v50 = v101;
      }
    }
  }
LABEL_15:
  v10 = a1[13];
  v88 = a1[14];
LABEL_16:
  src = 0;
  v103 = 0;
  v104 = 0;
  if ( v88 != v10 )
  {
    v23 = 0;
    v24 = 0;
    while ( 1 )
    {
      v105.m128i_i64[0] = v10;
      if ( v24 == v23 )
      {
        v10 += 64;
        sub_1CF66A0((__int64)&src, v24, &v105);
        v24 = v103;
        if ( v88 == v10 )
          goto LABEL_24;
      }
      else
      {
        if ( v24 )
        {
          *(_QWORD *)v24 = v10;
          v24 = v103;
        }
        v24 += 8;
        v10 += 64;
        v103 = v24;
        if ( v88 == v10 )
        {
LABEL_24:
          v25 = v24;
          goto LABEL_25;
        }
      }
      v23 = v104;
    }
  }
  v25 = 0;
LABEL_25:
  v26 = v99;
  v27 = v100;
  if ( v100 != v99 )
  {
    v28 = v25;
    do
    {
      while ( 1 )
      {
        v105.m128i_i64[0] = (__int64)v26;
        if ( v28 != v104 )
          break;
        v26 += 4;
        sub_1CF66A0((__int64)&src, v28, &v105);
        v28 = v103;
        if ( v27 == v26 )
          goto LABEL_32;
      }
      if ( v28 )
      {
        *(_QWORD *)v28 = v26;
        v28 = v103;
      }
      v28 += 8;
      v26 += 4;
      v103 = v28;
    }
    while ( v27 != v26 );
LABEL_32:
    v25 = v28;
  }
  v29 = a1[16];
  v30 = a1[17];
  if ( v30 != v29 )
  {
    v31 = v25;
    do
    {
      while ( 1 )
      {
        v105.m128i_i64[0] = v29;
        if ( v104 != v31 )
          break;
        v29 += 64;
        sub_1CF66A0((__int64)&src, v31, &v105);
        v31 = v103;
        if ( v30 == v29 )
          goto LABEL_40;
      }
      if ( v31 )
      {
        *(_QWORD *)v31 = v29;
        v31 = v103;
      }
      v31 += 8;
      v29 += 64;
      v103 = v31;
    }
    while ( v30 != v29 );
LABEL_40:
    v25 = v31;
  }
  v32 = (char *)src;
  if ( src == v25 )
    goto LABEL_63;
  v33 = v25 - (_BYTE *)src;
  _BitScanReverse64(&v34, (v25 - (_BYTE *)src) >> 3);
  sub_1CF51D0((char *)src, v25, 2LL * (int)(63 - (v34 ^ 0x3F)));
  if ( v33 > 128 )
  {
    v68 = (__int64 *)(v32 + 128);
    sub_1CF4F70(v32, v32 + 128);
    if ( v32 + 128 != v25 )
    {
      do
      {
        v69 = v68++;
        sub_1CF4F20(v69);
      }
      while ( v25 != (char *)v68 );
    }
  }
  else
  {
    sub_1CF4F70(v32, v25);
  }
  v37 = v103;
  v25 = (char *)src;
  v105.m128i_i64[0] = (__int64)&v106;
  v105.m128i_i64[1] = 0x2000000000LL;
  if ( v103 == src )
    goto LABEL_63;
  v38 = 0;
  v39 = v12;
  while ( 1 )
  {
    v70 = *(_QWORD *)v25;
    if ( v38 )
      break;
LABEL_58:
    v38 = 0;
    if ( (*(_BYTE *)v70 & 4) == 0 )
    {
LABEL_48:
      if ( v105.m128i_i32[3] <= v38 )
      {
        v90 = v39;
        v98 = v70;
        sub_16CD150((__int64)&v105, &v106, 0, 8, v39, v70);
        v39 = v90;
        v70 = v98;
      }
      *(_QWORD *)(v105.m128i_i64[0] + 8LL * v105.m128i_u32[2]) = v70;
      ++v105.m128i_i32[2];
      goto LABEL_51;
    }
    v25 += 8;
    if ( v37 == v25 )
      goto LABEL_60;
LABEL_52:
    v38 = v105.m128i_u32[2];
  }
  v40 = *(_QWORD *)(v70 + 48);
  v41 = *(_DWORD *)(v40 + 48);
  v42 = v105.m128i_i64[0] + 8LL * v38;
  while ( 2 )
  {
    v43 = *(_QWORD *)(v42 - 8);
    v44 = *(_QWORD *)(v43 + 48);
    if ( *(_DWORD *)(v44 + 48) == v41 )
    {
      if ( *(_DWORD *)(v43 + 56) <= *(_DWORD *)(v70 + 56) )
        break;
      goto LABEL_57;
    }
    if ( *(_DWORD *)(v44 + 48) > v41 || *(_DWORD *)(v40 + 52) > *(_DWORD *)(v44 + 52) )
    {
LABEL_57:
      --v38;
      v42 -= 8;
      v105.m128i_i32[2] = v38;
      if ( !v38 )
        goto LABEL_58;
      continue;
    }
    break;
  }
  if ( (*(_BYTE *)v70 & 4) == 0 )
    goto LABEL_48;
  if ( (*(_QWORD *)v43 & 0xFFFFFFFFFFFFFFF8LL) != 0 && a1[6] == (*(_QWORD *)v43 & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v96 = v39;
    sub_1CF5980(
      (__int64)a1,
      (__int64 *)v70,
      a2,
      *(double *)a3.m128i_i64,
      *(double *)a4.m128i_i64,
      *(double *)a5.m128i_i64,
      v35,
      v36,
      a8,
      a9);
    v39 = v96;
  }
LABEL_51:
  v25 += 8;
  if ( v37 != v25 )
    goto LABEL_52;
LABEL_60:
  v12 = v39;
  if ( (__m128i *)v105.m128i_i64[0] != &v106 )
    _libc_free(v105.m128i_u64[0]);
  v25 = (char *)src;
LABEL_63:
  if ( v25 )
    j_j___libc_free_0(v25, v104 - v25);
  if ( v99 )
    j_j___libc_free_0(v99, (char *)v101 - (char *)v99);
  _libc_free(v12);
}
