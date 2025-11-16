// Function: sub_B967C0
// Address: 0xb967c0
//
__m128i *__fastcall sub_B967C0(__m128i *a1, __m128i *a2)
{
  __m128i *v2; // r14
  unsigned __int32 v4; // edx
  const __m128i *v5; // rbx
  const __m128i *v6; // r12
  __m128i v7; // xmm0
  char *v8; // r14
  __int64 v9; // rbx
  char *v10; // r12
  unsigned __int64 v11; // rax
  char *v12; // rbx
  __int64 v13; // rdx
  __int64 v14; // rcx
  __m128i *v15; // rax
  __int64 v16; // rax
  char *v17; // rbx
  __int64 v18; // r15
  __int64 v19; // rax
  __m128i *v21; // r8
  int v22; // edx
  unsigned int v23; // ecx
  __int64 *v24; // rax
  __int64 v25; // r9
  __int64 v26; // rdx
  _QWORD *v27; // rax
  unsigned int v28; // edx
  unsigned int v29; // eax
  int v30; // edi
  unsigned int v31; // r8d
  unsigned int v32; // esi
  __int64 v33; // rdx
  int v34; // r11d
  __int64 *v35; // r10
  _QWORD *v36; // [rsp+8h] [rbp-B8h]
  __int64 m128i_i64; // [rsp+10h] [rbp-B0h]
  __int64 *v38; // [rsp+28h] [rbp-98h] BYREF
  __m128i v39; // [rsp+30h] [rbp-90h] BYREF
  __int64 v40; // [rsp+40h] [rbp-80h]
  void *src; // [rsp+50h] [rbp-70h] BYREF
  __int64 v42; // [rsp+58h] [rbp-68h]
  _BYTE v43[96]; // [rsp+60h] [rbp-60h] BYREF

  v2 = a2;
  src = v43;
  v4 = a2[1].m128i_u32[2];
  v42 = 0x600000000LL;
  m128i_i64 = (__int64)a2[1].m128i_i64;
  if ( !(v4 >> 1) )
    goto LABEL_46;
  if ( (a2[1].m128i_i8[8] & 1) != 0 )
  {
    v5 = a2 + 2;
    v6 = a2 + 8;
  }
  else
  {
    v5 = (const __m128i *)a2[2].m128i_i64[0];
    v6 = (const __m128i *)((char *)v5 + 24 * a2[2].m128i_u32[2]);
    if ( v5 == v6 )
      goto LABEL_7;
  }
  do
  {
    if ( v5->m128i_i64[0] != -4096 && v5->m128i_i64[0] != -8192 )
      break;
    v5 = (const __m128i *)((char *)v5 + 24);
  }
  while ( v5 != v6 );
LABEL_7:
  if ( v6 == v5 )
  {
LABEL_46:
    v10 = v43;
    goto LABEL_47;
  }
  do
  {
    v7 = _mm_loadu_si128(v5);
    v39 = v7;
    v40 = v5[1].m128i_i64[0];
    if ( (v7.m128i_i64[1] & 0xFFFFFFFFFFFFFFFCLL) != 0 && (v7.m128i_i8[8] & 3) == 2 )
    {
      a2 = (__m128i *)(v2[1].m128i_i8[8] & 1);
      if ( (v2[1].m128i_i8[8] & 1) != 0 )
      {
        v21 = v2 + 2;
        v22 = 3;
      }
      else
      {
        v28 = v2[2].m128i_u32[2];
        v21 = (__m128i *)v2[2].m128i_i64[0];
        if ( !v28 )
        {
          v29 = v2[1].m128i_u32[2];
          ++v2[1].m128i_i64[0];
          v38 = 0;
          v30 = (v29 >> 1) + 1;
          goto LABEL_40;
        }
        v22 = v28 - 1;
      }
      v23 = v22 & (((unsigned __int32)v39.m128i_i32[0] >> 9) ^ ((unsigned __int32)v39.m128i_i32[0] >> 4));
      v24 = &v21->m128i_i64[3 * v23];
      v25 = *v24;
      if ( v39.m128i_i64[0] == *v24 )
      {
LABEL_34:
        v26 = (unsigned int)v42;
        v27 = v24 + 1;
        if ( (unsigned __int64)(unsigned int)v42 + 1 > HIDWORD(v42) )
        {
          a2 = (__m128i *)v43;
          v36 = v27;
          sub_C8D5F0(&src, v43, (unsigned int)v42 + 1LL, 8);
          v26 = (unsigned int)v42;
          v27 = v36;
        }
        *((_QWORD *)src + v26) = v27;
        LODWORD(v42) = v42 + 1;
        goto LABEL_10;
      }
      v34 = 1;
      v35 = 0;
      while ( v25 != -4096 )
      {
        if ( v25 == -8192 && !v35 )
          v35 = v24;
        v23 = v22 & (v34 + v23);
        v24 = &v21->m128i_i64[3 * v23];
        v25 = *v24;
        if ( v39.m128i_i64[0] == *v24 )
          goto LABEL_34;
        ++v34;
      }
      v31 = 12;
      v28 = 4;
      if ( !v35 )
        v35 = v24;
      v29 = v2[1].m128i_u32[2];
      ++v2[1].m128i_i64[0];
      v38 = v35;
      v30 = (v29 >> 1) + 1;
      if ( (_BYTE)a2 )
      {
LABEL_41:
        v32 = 2 * v28;
        if ( 4 * v30 < v31 )
        {
          a2 = (__m128i *)(v28 >> 3);
          if ( v28 - v2[1].m128i_i32[3] - v30 > (unsigned int)a2 )
          {
LABEL_43:
            v2[1].m128i_i32[2] = (2 * (v29 >> 1) + 2) | v29 & 1;
            v24 = v38;
            if ( *v38 != -4096 )
              --v2[1].m128i_i32[3];
            v33 = v39.m128i_i64[0];
            v24[1] = 0;
            v24[2] = 0;
            *v24 = v33;
            goto LABEL_34;
          }
          v32 = v28;
        }
        sub_B95E60(m128i_i64, v32);
        a2 = &v39;
        sub_B926F0(m128i_i64, v39.m128i_i64, &v38);
        v29 = v2[1].m128i_u32[2];
        goto LABEL_43;
      }
      v28 = v2[2].m128i_u32[2];
LABEL_40:
      v31 = 3 * v28;
      goto LABEL_41;
    }
LABEL_10:
    v5 = (const __m128i *)((char *)v5 + 24);
    if ( v5 == v6 )
      break;
    while ( v5->m128i_i64[0] == -8192 || v5->m128i_i64[0] == -4096 )
    {
      v5 = (const __m128i *)((char *)v5 + 24);
      if ( v6 == v5 )
        goto LABEL_14;
    }
  }
  while ( v5 != v6 );
LABEL_14:
  v8 = (char *)src;
  v9 = 8LL * (unsigned int)v42;
  v10 = (char *)src + v9;
  if ( src == (char *)src + v9 )
  {
LABEL_47:
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    a1->m128i_i64[1] = 0x600000000LL;
    goto LABEL_25;
  }
  _BitScanReverse64(&v11, v9 >> 3);
  sub_B8EEB0((char *)src, (__int64 *)((char *)src + v9), 2LL * (int)(63 - (v11 ^ 0x3F)));
  if ( (unsigned __int64)v9 <= 0x80 )
  {
    a2 = (__m128i *)v10;
    sub_B8F660(v8, v10);
  }
  else
  {
    v12 = v8 + 128;
    a2 = (__m128i *)(v8 + 128);
    sub_B8F660(v8, v8 + 128);
    if ( v8 + 128 != v10 )
    {
      do
      {
        while ( 1 )
        {
          v13 = *((_QWORD *)v12 - 1);
          v14 = *(_QWORD *)v12;
          v15 = (__m128i *)(v12 - 8);
          if ( *(_QWORD *)(*(_QWORD *)v12 + 8LL) > *(_QWORD *)(v13 + 8) )
            break;
          a2 = (__m128i *)v12;
          v12 += 8;
          a2->m128i_i64[0] = v14;
          if ( v12 == v10 )
            goto LABEL_20;
        }
        do
        {
          v15->m128i_i64[1] = v13;
          a2 = v15;
          v13 = v15[-1].m128i_i64[1];
          v15 = (__m128i *)((char *)v15 - 8);
        }
        while ( *(_QWORD *)(v14 + 8) > *(_QWORD *)(v13 + 8) );
        v12 += 8;
        a2->m128i_i64[0] = v14;
      }
      while ( v12 != v10 );
    }
  }
LABEL_20:
  v16 = (unsigned int)v42;
  v17 = (char *)src;
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  v10 = &v17[8 * v16];
  a1->m128i_i64[1] = 0x600000000LL;
  if ( v17 != v10 )
  {
    do
    {
      v18 = sub_B90FC0(**(_QWORD **)v17 & 0xFFFFFFFFFFFFFFFCLL);
      v19 = a1->m128i_u32[2];
      if ( v19 + 1 > (unsigned __int64)a1->m128i_u32[3] )
      {
        a2 = a1 + 1;
        sub_C8D5F0(a1, &a1[1], v19 + 1, 8);
        v19 = a1->m128i_u32[2];
      }
      v17 += 8;
      *(_QWORD *)(a1->m128i_i64[0] + 8 * v19) = v18;
      ++a1->m128i_i32[2];
    }
    while ( v10 != v17 );
    v10 = (char *)src;
  }
LABEL_25:
  if ( v10 != v43 )
    _libc_free(v10, a2);
  return a1;
}
