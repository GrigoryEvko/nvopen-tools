// Function: sub_3773D20
// Address: 0x3773d20
//
__int64 __fastcall sub_3773D20(const __m128i *a1, __m128i a2)
{
  const __m128i *v2; // rax
  __int64 *v3; // rax
  __int64 v4; // r13
  const __m128i *v5; // r15
  __int64 v6; // rbx
  const __m128i *v7; // r14
  __int64 v8; // rax
  __int64 v9; // rbx
  const __m128i *v10; // rbx
  __int16 v11; // ax
  __int16 v12; // ax
  __int16 v13; // ax
  __int16 v14; // ax
  unsigned int v15; // r12d
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  const __m128i *v20; // r12
  __int64 v21; // r14
  unsigned __int64 v22; // rbx
  unsigned __int64 v23; // rsi
  unsigned __int64 v24; // rbx
  __int32 v25; // r13d
  __int64 *v26; // rdi
  int v27; // esi
  int v28; // r9d
  __int64 *v29; // r8
  unsigned int v30; // edx
  __int64 *v31; // rax
  unsigned int v32; // edx
  __int16 v33; // ax
  bool v34; // al
  __int16 v35; // ax
  bool v36; // al
  __int16 v37; // ax
  bool v38; // al
  unsigned int v39; // esi
  __int64 v40; // rbx
  __int64 v41; // r13
  unsigned __int32 v42; // eax
  unsigned int v43; // ecx
  __int64 *v44; // rax
  __int64 *v45; // rdx
  unsigned __int32 v46; // edx
  unsigned __int32 v47; // edi
  unsigned int v48; // r8d
  unsigned int v49; // eax
  unsigned int v50; // ebx
  __int64 v51; // rax
  __int64 *v52; // rdx
  __int64 *v53; // rax
  __int64 *v54; // r8
  int v55; // ecx
  int v56; // edi
  __int64 *v57; // rsi
  unsigned int j; // edx
  unsigned int v59; // edx
  __int64 *v60; // r8
  int v61; // esi
  int v62; // edi
  __int64 *v63; // rcx
  unsigned int i; // edx
  unsigned int v65; // edx
  int v66; // r10d
  __int64 *v67; // rax
  __int64 *v68; // rdx
  int v69; // r9d
  int v70; // r9d
  const __m128i *v71; // [rsp+0h] [rbp-8B0h]
  __int64 v72; // [rsp+10h] [rbp-8A0h]
  __m128i v73; // [rsp+40h] [rbp-870h] BYREF
  const __m128i *v74[2]; // [rsp+50h] [rbp-860h] BYREF
  unsigned __int8 v75; // [rsp+60h] [rbp-850h]
  __m128i v76; // [rsp+68h] [rbp-848h] BYREF
  __int64 *v77; // [rsp+78h] [rbp-838h] BYREF
  unsigned int v78; // [rsp+80h] [rbp-830h]
  _BYTE v79[56]; // [rsp+878h] [rbp-38h] BYREF

  v2 = (const __m128i *)a1[1].m128i_i64[0];
  v74[0] = a1;
  v75 = 0;
  v74[1] = v2;
  v3 = (__int64 *)&v77;
  v76.m128i_i64[0] = 0;
  v76.m128i_i64[1] = 1;
  do
  {
    *v3 = 0;
    v3 += 4;
    *((_DWORD *)v3 - 6) = -1;
  }
  while ( v3 != (__int64 *)v79 );
  v71 = v74[0];
  v4 = v74[0][25].m128i_i64[1];
  v72 = *(_QWORD *)((v74[0][25].m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) + 8);
  if ( v4 == v72 )
  {
LABEL_19:
    v15 = 0;
    goto LABEL_20;
  }
  while ( 1 )
  {
    if ( !v4 )
      BUG();
    v5 = *(const __m128i **)(v4 + 40);
    v6 = 16LL * *(unsigned int *)(v4 + 60);
    v7 = &v5[(unsigned __int64)v6 / 0x10];
    v8 = v6 >> 4;
    v9 = v6 >> 6;
    if ( v9 )
      break;
LABEL_41:
    if ( v8 != 2 )
    {
      if ( v8 != 3 )
      {
        if ( v8 != 1 )
          goto LABEL_18;
        goto LABEL_44;
      }
      v35 = v5->m128i_i16[0];
      v73 = _mm_loadu_si128(v5);
      if ( v35 )
        v36 = (unsigned __int16)(v35 - 17) <= 0xD3u;
      else
        v36 = sub_30070B0((__int64)&v73);
      if ( v36 )
      {
LABEL_17:
        if ( v7 != v5 )
          goto LABEL_25;
        goto LABEL_18;
      }
      ++v5;
    }
    v37 = v5->m128i_i16[0];
    v73 = _mm_loadu_si128(v5);
    if ( v37 )
      v38 = (unsigned __int16)(v37 - 17) <= 0xD3u;
    else
      v38 = sub_30070B0((__int64)&v73);
    if ( v38 )
      goto LABEL_17;
    ++v5;
LABEL_44:
    v33 = v5->m128i_i16[0];
    v73 = _mm_loadu_si128(v5);
    if ( v33 )
      v34 = (unsigned __int16)(v33 - 17) <= 0xD3u;
    else
      v34 = sub_30070B0((__int64)&v73);
    if ( v34 )
      goto LABEL_17;
LABEL_18:
    v4 = *(_QWORD *)(v4 + 8);
    if ( v4 == v72 )
      goto LABEL_19;
  }
  v10 = &v5[4 * v9];
  while ( 1 )
  {
    a2 = _mm_loadu_si128(v5);
    v14 = v5->m128i_i16[0];
    v73 = a2;
    if ( v14 )
    {
      if ( (unsigned __int16)(v14 - 17) <= 0xD3u )
        goto LABEL_17;
    }
    else if ( sub_30070B0((__int64)&v73) )
    {
      goto LABEL_17;
    }
    v11 = v5[1].m128i_i16[0];
    v73 = _mm_loadu_si128(v5 + 1);
    if ( v11 )
    {
      if ( (unsigned __int16)(v11 - 17) <= 0xD3u )
        break;
      goto LABEL_10;
    }
    if ( sub_30070B0((__int64)&v73) )
      break;
LABEL_10:
    v12 = v5[2].m128i_i16[0];
    v73 = _mm_loadu_si128(v5 + 2);
    if ( v12 )
    {
      if ( (unsigned __int16)(v12 - 17) <= 0xD3u )
        goto LABEL_37;
    }
    else if ( sub_30070B0((__int64)&v73) )
    {
LABEL_37:
      v5 += 2;
      goto LABEL_17;
    }
    v13 = v5[3].m128i_i16[0];
    v73 = _mm_loadu_si128(v5 + 3);
    if ( v13 )
    {
      if ( (unsigned __int16)(v13 - 17) <= 0xD3u )
        goto LABEL_39;
    }
    else if ( sub_30070B0((__int64)&v73) )
    {
LABEL_39:
      v5 += 3;
      goto LABEL_17;
    }
    v5 += 4;
    if ( v10 == v5 )
    {
      v8 = v7 - v5;
      goto LABEL_41;
    }
  }
  if ( v7 == &v5[1] )
    goto LABEL_18;
LABEL_25:
  sub_33E2990((__int64)v71);
  v20 = v74[0];
  v21 = v74[0][25].m128i_i64[1];
  v22 = v74[0][25].m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
  if ( v21 != *(_QWORD *)(v22 + 8) )
  {
    do
    {
      v23 = v21 - 8;
      if ( !v21 )
        v23 = 0;
      sub_376DE90((__int64)v74, v23, 0, a2, v17, v18, v19);
      v21 = *(_QWORD *)(v21 + 8);
    }
    while ( v21 != *(_QWORD *)(v22 + 8) );
    v20 = v74[0];
  }
  v24 = v20[24].m128i_u64[0];
  v25 = v20[24].m128i_i32[2];
  if ( (v76.m128i_i8[8] & 1) != 0 )
  {
    v26 = (__int64 *)&v77;
    v27 = 63;
  }
  else
  {
    v39 = v78;
    v26 = v77;
    if ( !v78 )
    {
      v46 = v76.m128i_u32[2];
      ++v76.m128i_i64[0];
      v31 = 0;
      v47 = ((unsigned __int32)v76.m128i_i32[2] >> 1) + 1;
      goto LABEL_76;
    }
    v27 = v78 - 1;
  }
  v28 = 1;
  v29 = 0;
  v30 = v27 & (v25 + ((v24 >> 9) ^ (v24 >> 4)));
  while ( 2 )
  {
    v31 = &v26[4 * v30];
    if ( *v31 == v24 && *((_DWORD *)v31 + 2) == v25 )
    {
      v40 = v31[2];
      v41 = v31[3];
      if ( !v40 )
        goto LABEL_82;
      nullsub_1875();
      v20[24].m128i_i64[0] = v40;
      v20[24].m128i_i32[2] = v41;
      sub_33E2B60();
      goto LABEL_64;
    }
    if ( *v31 )
    {
LABEL_35:
      v32 = v28 + v30;
      ++v28;
      v30 = v27 & v32;
      continue;
    }
    break;
  }
  v66 = *((_DWORD *)v31 + 2);
  if ( v66 != -1 )
  {
    if ( v66 == -2 && !v29 )
      v29 = &v26[4 * v30];
    goto LABEL_35;
  }
  v46 = v76.m128i_u32[2];
  v39 = 64;
  if ( v29 )
    v31 = v29;
  ++v76.m128i_i64[0];
  v48 = 192;
  v47 = ((unsigned __int32)v76.m128i_i32[2] >> 1) + 1;
  if ( (v76.m128i_i8[8] & 1) == 0 )
  {
    v39 = v78;
LABEL_76:
    v48 = 3 * v39;
  }
  if ( v48 <= 4 * v47 )
  {
    sub_376D3B0(&v76, 2 * v39);
    if ( (v76.m128i_i8[8] & 1) != 0 )
    {
      v60 = (__int64 *)&v77;
      v61 = 63;
    }
    else
    {
      v60 = v77;
      if ( !v78 )
        goto LABEL_152;
      v61 = v78 - 1;
    }
    v62 = 1;
    v63 = 0;
    for ( i = v61 & (v25 + ((v24 >> 9) ^ (v24 >> 4))); ; i = v61 & v65 )
    {
      v31 = &v60[4 * i];
      if ( *v31 == v24 && *((_DWORD *)v31 + 2) == v25 )
        break;
      if ( !*v31 )
      {
        v70 = *((_DWORD *)v31 + 2);
        if ( v70 == -1 )
        {
          if ( !v63 )
            goto LABEL_114;
          v46 = v76.m128i_u32[2];
          v31 = v63;
          goto LABEL_79;
        }
        if ( v70 == -2 && !v63 )
          v63 = &v60[4 * i];
      }
      v65 = v62 + i;
      ++v62;
    }
    goto LABEL_114;
  }
  if ( v39 - v76.m128i_i32[3] - v47 <= v39 >> 3 )
  {
    sub_376D3B0(&v76, v39);
    if ( (v76.m128i_i8[8] & 1) != 0 )
    {
      v54 = (__int64 *)&v77;
      v55 = 63;
      goto LABEL_98;
    }
    v54 = v77;
    if ( v78 )
    {
      v55 = v78 - 1;
LABEL_98:
      v56 = 1;
      v57 = 0;
      for ( j = v55 & (v25 + ((v24 >> 9) ^ (v24 >> 4))); ; j = v55 & v59 )
      {
        v31 = &v54[4 * j];
        if ( *v31 == v24 && *((_DWORD *)v31 + 2) == v25 )
          break;
        if ( !*v31 )
        {
          v69 = *((_DWORD *)v31 + 2);
          if ( v69 == -1 )
          {
            if ( v57 )
              v31 = v57;
            break;
          }
          if ( !v57 && v69 == -2 )
            v57 = &v54[4 * j];
        }
        v59 = v56 + j;
        ++v56;
      }
LABEL_114:
      v46 = v76.m128i_u32[2];
      goto LABEL_79;
    }
LABEL_152:
    v76.m128i_i32[2] = (2 * ((unsigned __int32)v76.m128i_i32[2] >> 1) + 2) | v76.m128i_i8[8] & 1;
    BUG();
  }
LABEL_79:
  v76.m128i_i32[2] = (2 * (v46 >> 1) + 2) | v46 & 1;
  if ( *v31 || *((_DWORD *)v31 + 2) != -1 )
    --v76.m128i_i32[3];
  *v31 = v24;
  v31[2] = 0;
  *((_DWORD *)v31 + 2) = v25;
  *((_DWORD *)v31 + 6) = 0;
  v41 = v31[3];
LABEL_82:
  v20[24].m128i_i64[0] = 0;
  v20[24].m128i_i32[2] = v41;
LABEL_64:
  ++v76.m128i_i64[0];
  v42 = (unsigned __int32)v76.m128i_i32[2] >> 1;
  if ( (unsigned __int32)v76.m128i_i32[2] >> 1 )
  {
    v43 = 4 * v42;
    if ( (v76.m128i_i8[8] & 1) == 0 )
      goto LABEL_67;
    goto LABEL_74;
  }
  if ( !v76.m128i_i32[3] )
    goto LABEL_72;
  v43 = 0;
  if ( (v76.m128i_i8[8] & 1) != 0 )
  {
LABEL_74:
    v45 = (__int64 *)v79;
    v44 = (__int64 *)&v77;
    goto LABEL_70;
  }
LABEL_67:
  if ( v78 <= v43 || v78 <= 0x40 )
  {
    v44 = v77;
    v45 = &v77[4 * v78];
    if ( v77 != v45 )
    {
      do
      {
LABEL_70:
        *v44 = 0;
        v44 += 4;
        *((_DWORD *)v44 - 6) = -1;
      }
      while ( v44 != v45 );
    }
    v76.m128i_i64[1] = v76.m128i_i8[8] & 1;
    goto LABEL_72;
  }
  if ( !v42 )
  {
    sub_C7D6A0((__int64)v77, 32LL * v78, 8);
    v76.m128i_i8[8] |= 1u;
    goto LABEL_90;
  }
  v49 = v42 - 1;
  if ( !v49 )
  {
    sub_C7D6A0((__int64)v77, 32LL * v78, 8);
LABEL_132:
    v76.m128i_i8[8] |= 1u;
    goto LABEL_90;
  }
  _BitScanReverse(&v49, v49);
  v50 = 1 << (33 - (v49 ^ 0x1F));
  if ( v50 == v78 )
  {
    v76.m128i_i64[1] = v76.m128i_i8[8] & 1;
    if ( v76.m128i_i64[1] )
    {
      v68 = (__int64 *)v79;
      v67 = (__int64 *)&v77;
    }
    else
    {
      v67 = v77;
      v68 = &v77[4 * v50];
    }
    do
    {
      if ( v67 )
      {
        *v67 = 0;
        *((_DWORD *)v67 + 2) = -1;
      }
      v67 += 4;
    }
    while ( v68 != v67 );
    goto LABEL_72;
  }
  sub_C7D6A0((__int64)v77, 32LL * v78, 8);
  if ( v50 <= 0x40 )
    goto LABEL_132;
  v76.m128i_i8[8] &= ~1u;
  v51 = sub_C7D670(32LL * v50, 8);
  v78 = v50;
  v77 = (__int64 *)v51;
LABEL_90:
  v76.m128i_i64[1] = v76.m128i_i8[8] & 1;
  if ( v76.m128i_i64[1] )
  {
    v52 = (__int64 *)v79;
    v53 = (__int64 *)&v77;
  }
  else
  {
    v53 = v77;
    v52 = &v77[4 * v78];
    if ( v77 == v52 )
      goto LABEL_72;
  }
  do
  {
    if ( v53 )
    {
      *v53 = 0;
      *((_DWORD *)v53 + 2) = -1;
    }
    v53 += 4;
  }
  while ( v53 != v52 );
LABEL_72:
  sub_33F7860(v74[0]);
  v15 = v75;
LABEL_20:
  if ( (v76.m128i_i8[8] & 1) == 0 )
    sub_C7D6A0((__int64)v77, 32LL * v78, 8);
  return v15;
}
