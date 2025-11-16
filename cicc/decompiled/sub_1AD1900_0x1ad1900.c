// Function: sub_1AD1900
// Address: 0x1ad1900
//
const void **__fastcall sub_1AD1900(__int64 a1)
{
  __m128i *v2; // rcx
  __int64 v3; // r13
  const void **v4; // r12
  __int64 v5; // r12
  __int64 v6; // r15
  unsigned __int64 v7; // rax
  __m128i *v8; // r12
  const void *v9; // r13
  size_t v10; // r15
  int v11; // eax
  __m128i v12; // xmm0
  size_t v13; // r14
  const void *v14; // rsi
  const void **v15; // rax
  const void **v16; // rdx
  const void **result; // rax
  __int64 v18; // r12
  unsigned __int8 *v19; // r14
  size_t v20; // r13
  __int64 v21; // rdx
  unsigned int v22; // r15d
  _QWORD *v23; // r8
  _QWORD *v24; // rdx
  __int64 v25; // rsi
  __int64 v26; // rax
  _QWORD *v27; // r8
  _QWORD *v28; // rcx
  __int64 v29; // rdi
  __int64 v30; // rax
  void *v31; // rax
  _QWORD *v32; // [rsp+0h] [rbp-50h]
  _QWORD *v33; // [rsp+0h] [rbp-50h]
  _QWORD *v34; // [rsp+8h] [rbp-48h]
  _QWORD *v35; // [rsp+8h] [rbp-48h]
  const void **v36; // [rsp+10h] [rbp-40h]
  _QWORD *v37; // [rsp+10h] [rbp-40h]
  const void **dest; // [rsp+18h] [rbp-38h]
  const void **desta; // [rsp+18h] [rbp-38h]

  v2 = *(__m128i **)(a1 + 40);
  v3 = *(_QWORD *)(a1 + 32);
  v36 = (const void **)v2;
  v4 = (const void **)v2;
  if ( (__m128i *)v3 == v2 )
  {
LABEL_16:
    v15 = sub_1AD17F0(v36, v4);
  }
  else
  {
    v5 = (__int64)v2->m128i_i64 - v3;
    v6 = *(_QWORD *)(a1 + 40);
    _BitScanReverse64(&v7, ((__int64)v2->m128i_i64 - v3) >> 4);
    sub_1AD04B0((__m128i *)v3, v2, 2LL * (int)(63 - (v7 ^ 0x3F)));
    if ( v5 <= 256 )
    {
      sub_1ACFE20((const void **)v3, v36);
    }
    else
    {
      sub_1ACFE20((const void **)v3, (const void **)(v3 + 256));
      dest = (const void **)(v3 + 256);
      if ( v6 != v3 + 256 )
      {
        while ( 1 )
        {
          v8 = (__m128i *)dest;
          v9 = *dest;
          v10 = (size_t)dest[1];
          while ( 1 )
          {
            v13 = v8[-1].m128i_u64[1];
            v14 = (const void *)v8[-1].m128i_i64[0];
            if ( v13 < v10 )
              break;
            if ( v10 )
            {
              v11 = memcmp(v9, v14, v10);
              if ( v11 )
                goto LABEL_13;
            }
            if ( v13 == v10 )
              goto LABEL_14;
LABEL_8:
            if ( v13 <= v10 )
              goto LABEL_14;
LABEL_9:
            v12 = _mm_loadu_si128(--v8);
            v8[1] = v12;
          }
          if ( !v13 )
            goto LABEL_14;
          v11 = memcmp(v9, v14, v8[-1].m128i_u64[1]);
          if ( !v11 )
            goto LABEL_8;
LABEL_13:
          if ( v11 < 0 )
            goto LABEL_9;
LABEL_14:
          dest += 2;
          v8->m128i_i64[0] = (__int64)v9;
          v8->m128i_i64[1] = v10;
          if ( v36 == dest )
          {
            v4 = *(const void ***)(a1 + 40);
            v36 = *(const void ***)(a1 + 32);
            goto LABEL_16;
          }
        }
      }
    }
    v4 = *(const void ***)(a1 + 40);
    v15 = sub_1AD17F0(*(const void ***)(a1 + 32), v4);
  }
  desta = v15;
  if ( v15 == v4 )
  {
    result = *(const void ***)(a1 + 40);
    desta = result;
  }
  else
  {
    v16 = *(const void ***)(a1 + 40);
    if ( v16 == v4
      || (memmove(v15, v4, (char *)v16 - (char *)v4),
          result = *(const void ***)(a1 + 40),
          desta = (const void **)((char *)desta + (char *)result - (char *)v4),
          desta != result) )
    {
      result = desta;
      *(_QWORD *)(a1 + 40) = desta;
    }
  }
  v18 = *(_QWORD *)(a1 + 32);
  if ( (const void **)v18 != desta )
  {
    while ( 1 )
    {
      v19 = *(unsigned __int8 **)v18;
      v20 = *(_QWORD *)(v18 + 8);
      v21 = (unsigned int)sub_16D19C0(a1, *(unsigned __int8 **)v18, v20);
      result = *(const void ***)a1;
      v22 = v21;
      v23 = (_QWORD *)(*(_QWORD *)a1 + 8 * v21);
      v24 = (_QWORD *)*v23;
      if ( !*v23 )
        goto LABEL_32;
      if ( v24 == (_QWORD *)-8LL )
        break;
LABEL_25:
      v25 = v24[1];
      if ( *(_BYTE *)(v25 + 89) )
      {
        v18 += 16;
        if ( desta == (const void **)v18 )
          return result;
      }
      else
      {
        v18 += 16;
        result = (const void **)sub_1AD1560(a1, v25);
        if ( desta == (const void **)v18 )
          return result;
      }
    }
    --*(_DWORD *)(a1 + 16);
LABEL_32:
    v32 = v23;
    v26 = malloc(v20 + 17);
    v27 = v32;
    v28 = (_QWORD *)v26;
    if ( v26 )
    {
      v29 = v26 + 16;
    }
    else
    {
      if ( v20 == -17 )
      {
        v30 = malloc(1u);
        v27 = v32;
        v28 = 0;
        if ( v30 )
        {
          v29 = v30 + 16;
          v28 = (_QWORD *)v30;
          goto LABEL_43;
        }
      }
      v33 = v28;
      v35 = v27;
      sub_16BD1C0("Allocation failed", 1u);
      v27 = v35;
      v28 = v33;
      v29 = 16;
    }
    if ( v20 + 1 <= 1 )
    {
LABEL_35:
      *(_BYTE *)(v29 + v20) = 0;
      *v28 = v20;
      v28[1] = 0;
      *v27 = v28;
      ++*(_DWORD *)(a1 + 12);
      result = (const void **)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_16D1CD0(a1, v22));
      v24 = *result;
      if ( *result == (const void *)-8LL || !v24 )
      {
        ++result;
        do
        {
          do
            v24 = *result++;
          while ( v24 == (_QWORD *)-8LL );
        }
        while ( !v24 );
      }
      goto LABEL_25;
    }
LABEL_43:
    v34 = v28;
    v37 = v27;
    v31 = memcpy((void *)v29, v19, v20);
    v28 = v34;
    v27 = v37;
    v29 = (__int64)v31;
    goto LABEL_35;
  }
  return result;
}
