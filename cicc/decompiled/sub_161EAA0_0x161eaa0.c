// Function: sub_161EAA0
// Address: 0x161eaa0
//
void __fastcall sub_161EAA0(const __m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int32 v6; // edx
  unsigned __int32 v7; // eax
  const __m128i *v8; // r12
  const __m128i *v9; // r14
  unsigned __int32 v10; // edx
  unsigned int v11; // eax
  __int64 v12; // rsi
  __m128i *v13; // rax
  __int64 v14; // rdx
  __m128i *i; // rdx
  char *v16; // r12
  char *v17; // r13
  __int64 v18; // rax
  unsigned __int64 v19; // rdi
  const __m128i *v20; // rax
  __int64 v21; // r15
  const __m128i *v22; // rdx
  __m128i *v23; // rcx
  const __m128i *v24; // rax
  __m128i *v25; // r14
  __int64 v26; // r12
  __m128i *v27; // r15
  unsigned __int64 v28; // rax
  __m128i *v29; // r12
  __m128i *v30; // rdi
  unsigned __int32 v31; // edx
  bool v32; // zf
  __m128i *v33; // rax
  __int64 v34; // rdx
  __m128i *j; // rdx
  unsigned int v36; // edx
  unsigned int v37; // eax
  unsigned int v38; // r12d
  __int8 v39; // al
  __int64 v40; // rdi
  __int64 v41; // rax
  __m128i *v42; // r13
  __int64 v43; // rax
  __m128i *v44; // rax
  __m128i *v45; // [rsp+0h] [rbp-100h] BYREF
  __int64 v46; // [rsp+8h] [rbp-F8h]
  _BYTE v47[240]; // [rsp+10h] [rbp-F0h] BYREF

  v6 = a1[1].m128i_u32[2];
  v7 = v6 >> 1;
  if ( (a1[1].m128i_i8[8] & 1) != 0 )
  {
    v8 = a1 + 8;
    v9 = a1 + 2;
    if ( !v7 )
    {
LABEL_7:
      HIDWORD(v46) = 8;
      v45 = (__m128i *)v47;
      goto LABEL_8;
    }
  }
  else
  {
    v9 = (const __m128i *)a1[2].m128i_i64[0];
    v8 = (const __m128i *)((char *)v9 + 24 * a1[2].m128i_u32[2]);
    if ( !v7 )
      goto LABEL_7;
  }
  if ( v9 == v8 )
    goto LABEL_7;
  while ( v9->m128i_i64[0] == -4 || v9->m128i_i64[0] == -8 )
  {
    v9 = (const __m128i *)((char *)v9 + 24);
    if ( v9 == v8 )
      goto LABEL_7;
  }
  v45 = (__m128i *)v47;
  v46 = 0x800000000LL;
  if ( v9 == v8 )
  {
LABEL_8:
    ++a1[1].m128i_i64[0];
    v10 = v6 >> 1;
    LODWORD(v46) = 0;
    if ( !v10 )
      goto LABEL_49;
    goto LABEL_9;
  }
  v20 = v9;
  v21 = 0;
  while ( 1 )
  {
    v22 = (const __m128i *)((char *)v20 + 24);
    if ( &v20[1].m128i_u64[1] == (unsigned __int64 *)v8 )
      break;
    while ( 1 )
    {
      v20 = v22;
      if ( v22->m128i_i64[0] != -4 && v22->m128i_i64[0] != -8 )
        break;
      v22 = (const __m128i *)((char *)v22 + 24);
      if ( v8 == v22 )
        goto LABEL_35;
    }
    ++v21;
    if ( v22 == v8 )
      goto LABEL_36;
  }
LABEL_35:
  ++v21;
LABEL_36:
  v23 = (__m128i *)v47;
  if ( v21 > 8 )
  {
    sub_16CD150(&v45, v47, v21, 24);
    v23 = (__m128i *)((char *)v45 + 24 * (unsigned int)v46);
  }
  do
  {
    if ( v23 )
    {
      *v23 = _mm_loadu_si128(v9);
      v23[1].m128i_i64[0] = v9[1].m128i_i64[0];
    }
    v24 = (const __m128i *)((char *)v9 + 24);
    if ( &v9[1].m128i_u64[1] == (unsigned __int64 *)v8 )
      break;
    while ( 1 )
    {
      v9 = v24;
      if ( v24->m128i_i64[0] != -8 && v24->m128i_i64[0] != -4 )
        break;
      v24 = (const __m128i *)((char *)v24 + 24);
      if ( v8 == v24 )
        goto LABEL_44;
    }
    v23 = (__m128i *)((char *)v23 + 24);
  }
  while ( v24 != v8 );
LABEL_44:
  v25 = v45;
  LODWORD(v46) = v46 + v21;
  v26 = 24LL * (unsigned int)v46;
  v27 = (__m128i *)((char *)v45 + v26);
  if ( &v45->m128i_i8[v26] != (__int8 *)v45 )
  {
    _BitScanReverse64(&v28, 0xAAAAAAAAAAAAAAABLL * (v26 >> 3));
    sub_161D610(v45->m128i_i8, (__m128i *)((char *)v45 + v26), 2LL * (int)(63 - (v28 ^ 0x3F)), (__int64)v23, a5);
    if ( (unsigned __int64)v26 <= 0x180 )
    {
      sub_161D900(v25, v27);
    }
    else
    {
      v29 = v25 + 24;
      sub_161D900(v25, v25 + 24);
      if ( v27 != &v25[24] )
      {
        do
        {
          v30 = v29;
          v29 = (__m128i *)((char *)v29 + 24);
          sub_161CBB0(v30);
        }
        while ( v27 != v29 );
      }
    }
  }
  v31 = a1[1].m128i_u32[2];
  ++a1[1].m128i_i64[0];
  v10 = v31 >> 1;
  if ( !v10 )
  {
LABEL_49:
    if ( !a1[1].m128i_i32[3] )
      goto LABEL_17;
    v11 = 0;
    if ( (a1[1].m128i_i8[8] & 1) != 0 )
      goto LABEL_51;
    goto LABEL_11;
  }
LABEL_9:
  if ( (a1[1].m128i_i8[8] & 1) != 0 )
  {
LABEL_51:
    v13 = (__m128i *)&a1[2];
    v14 = 96;
    goto LABEL_14;
  }
  v11 = 4 * v10;
LABEL_11:
  v12 = a1[2].m128i_u32[2];
  if ( v11 >= (unsigned int)v12 || (unsigned int)v12 <= 0x40 )
  {
    v13 = (__m128i *)a1[2].m128i_i64[0];
    v14 = 24 * v12;
LABEL_14:
    for ( i = (__m128i *)((char *)v13 + v14); i != v13; v13 = (__m128i *)((char *)v13 + 24) )
      v13->m128i_i64[0] = -4;
    a1[1].m128i_i64[1] &= 1uLL;
    goto LABEL_17;
  }
  if ( !v10 || (v36 = v10 - 1) == 0 )
  {
    j___libc_free_0(a1[2].m128i_i64[0]);
    a1[1].m128i_i8[8] |= 1u;
    goto LABEL_61;
  }
  _BitScanReverse(&v37, v36);
  v38 = 1 << (33 - (v37 ^ 0x1F));
  if ( v38 - 5 <= 0x3A )
  {
    v38 = 64;
    j___libc_free_0(a1[2].m128i_i64[0]);
    v39 = a1[1].m128i_i8[8];
    v40 = 1536;
    goto LABEL_73;
  }
  if ( (_DWORD)v12 != v38 )
  {
    j___libc_free_0(a1[2].m128i_i64[0]);
    v39 = a1[1].m128i_i8[8] | 1;
    a1[1].m128i_i8[8] = v39;
    if ( v38 <= 4 )
      goto LABEL_61;
    v40 = 24LL * v38;
LABEL_73:
    a1[1].m128i_i8[8] = v39 & 0xFE;
    v41 = sub_22077B0(v40);
    a1[2].m128i_i32[2] = v38;
    a1[2].m128i_i64[0] = v41;
LABEL_61:
    v32 = (a1[1].m128i_i64[1] & 1) == 0;
    a1[1].m128i_i64[1] &= 1uLL;
    if ( v32 )
    {
      v33 = (__m128i *)a1[2].m128i_i64[0];
      v34 = 24LL * a1[2].m128i_u32[2];
    }
    else
    {
      v33 = (__m128i *)&a1[2];
      v34 = 96;
    }
    for ( j = (__m128i *)((char *)v33 + v34); j != v33; v33 = (__m128i *)((char *)v33 + 24) )
    {
      if ( v33 )
        v33->m128i_i64[0] = -4;
    }
    goto LABEL_17;
  }
  v32 = (a1[1].m128i_i64[1] & 1) == 0;
  a1[1].m128i_i64[1] &= 1uLL;
  if ( v32 )
  {
    v42 = (__m128i *)a1[2].m128i_i64[0];
    v43 = 24 * v12;
  }
  else
  {
    v42 = (__m128i *)&a1[2];
    v43 = 96;
  }
  v44 = (__m128i *)((char *)v42 + v43);
  do
  {
    if ( v42 )
      v42->m128i_i64[0] = -4;
    v42 = (__m128i *)((char *)v42 + 24);
  }
  while ( v44 != v42 );
LABEL_17:
  v16 = (char *)v45;
  v17 = &v45->m128i_i8[24 * (unsigned int)v46];
  if ( v45 != (__m128i *)v17 )
  {
    do
    {
      v18 = *((_QWORD *)v16 + 1);
      v19 = v18 & 0xFFFFFFFFFFFFFFFCLL;
      if ( (v18 & 0xFFFFFFFFFFFFFFFCLL) != 0
        && (v18 & 2) != 0
        && (unsigned __int8)(*(_BYTE *)v19 - 4) <= 0x1Eu
        && (*(_BYTE *)(v19 + 1) == 2 || *(_DWORD *)(v19 + 12)) )
      {
        sub_161EA80(v19);
      }
      v16 += 24;
    }
    while ( v17 != v16 );
    v17 = (char *)v45;
  }
  if ( v17 != v47 )
    _libc_free((unsigned __int64)v17);
}
