// Function: sub_3986080
// Address: 0x3986080
//
__int64 __fastcall sub_3986080(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8)
{
  __int64 v8; // r14
  __int64 v9; // rbx
  __int64 i; // r14
  __int64 v11; // rbx
  __int64 v12; // r13
  __int64 v13; // r9
  __m128i *v14; // r13
  unsigned __int64 v15; // r12
  __int64 result; // rax
  __m128i *v17; // rcx
  __int64 v18; // r12
  __m128i v19; // xmm7
  __int64 v20; // r13
  __int64 j; // rbx
  __m128i *v22; // r12
  const __m128i *v23; // r15
  __m128i v24; // xmm4
  __int64 v25; // r8
  __int64 v26; // rdx
  __int64 v28; // [rsp+8h] [rbp-C8h]
  __int64 v29; // [rsp+10h] [rbp-C0h]
  __int64 v30; // [rsp+20h] [rbp-B0h]
  __int64 v31; // [rsp+30h] [rbp-A0h]
  unsigned __int64 v32; // [rsp+30h] [rbp-A0h]
  char v33[8]; // [rsp+40h] [rbp-90h] BYREF
  unsigned __int64 v34; // [rsp+48h] [rbp-88h]
  char v35[8]; // [rsp+60h] [rbp-70h] BYREF
  unsigned __int64 v36; // [rsp+68h] [rbp-68h]
  __m128i v37; // [rsp+80h] [rbp-50h] BYREF
  __m128i v38[4]; // [rsp+90h] [rbp-40h] BYREF

  v8 = a1;
  v9 = a7;
  v29 = a3 & 1;
  v30 = (a3 - 1) / 2;
  if ( a2 >= v30 )
  {
    result = 32 * a2;
    v14 = (__m128i *)(a1 + 32 * a2);
    if ( (a3 & 1) != 0 )
    {
      v37 = _mm_loadu_si128((const __m128i *)&a7);
      v38[0] = _mm_loadu_si128((const __m128i *)&a8);
      goto LABEL_14;
    }
    v18 = a2;
    goto LABEL_17;
  }
  v28 = a7;
  for ( i = a2; ; i = v11 )
  {
    v11 = 2 * (i + 1);
    v12 = (i + 1) << 6;
    v13 = a1 + v12 - 32;
    v14 = (__m128i *)(a1 + v12);
    v31 = v13;
    sub_15B1350(
      (__int64)v35,
      *(unsigned __int64 **)(v14->m128i_i64[0] + 24),
      *(unsigned __int64 **)(v14->m128i_i64[0] + 32));
    v15 = v36;
    result = sub_15B1350(
               (__int64)&v37,
               *(unsigned __int64 **)(*(_QWORD *)v31 + 24LL),
               *(unsigned __int64 **)(*(_QWORD *)v31 + 32LL));
    if ( v15 < v37.m128i_i64[1] )
    {
      --v11;
      v14 = (__m128i *)(a1 + 32 * v11);
    }
    v17 = (__m128i *)(a1 + 32 * i);
    *v17 = _mm_loadu_si128(v14);
    v17[1] = _mm_loadu_si128(v14 + 1);
    if ( v11 >= v30 )
      break;
  }
  v18 = v11;
  v8 = a1;
  v9 = v28;
  if ( !v29 )
  {
LABEL_17:
    if ( (a3 - 2) / 2 == v18 )
    {
      v25 = v18 + 1;
      v18 = 2 * (v18 + 1) - 1;
      v26 = v8 + (v25 << 6);
      result = 32 * v18;
      *v14 = _mm_loadu_si128((const __m128i *)(v26 - 32));
      v14[1] = _mm_loadu_si128((const __m128i *)(v26 - 16));
      v14 = (__m128i *)(v8 + 32 * v18);
    }
  }
  v19 = _mm_loadu_si128((const __m128i *)&a8);
  v37 = _mm_loadu_si128((const __m128i *)&a7);
  v38[0] = v19;
  if ( v18 > a2 )
  {
    v20 = v9;
    for ( j = (v18 - 1) / 2; ; j = (j - 1) / 2 )
    {
      v22 = (__m128i *)(v8 + 32 * v18);
      v23 = (const __m128i *)(v8 + 32 * j);
      sub_15B1350(
        (__int64)v33,
        *(unsigned __int64 **)(v23->m128i_i64[0] + 24),
        *(unsigned __int64 **)(v23->m128i_i64[0] + 32));
      v32 = v34;
      sub_15B1350((__int64)v35, *(unsigned __int64 **)(v20 + 24), *(unsigned __int64 **)(v20 + 32));
      result = v32;
      if ( v32 >= v36 )
      {
        v9 = v20;
        v14 = v22;
        goto LABEL_14;
      }
      *v22 = _mm_loadu_si128(v23);
      v22[1] = _mm_loadu_si128(v23 + 1);
      result = (j - 1) / 2;
      v18 = j;
      if ( a2 >= j )
        break;
    }
    v9 = v20;
    v14 = (__m128i *)v23;
  }
LABEL_14:
  v37.m128i_i64[0] = v9;
  v24 = _mm_loadu_si128(&v37);
  v14[1] = _mm_loadu_si128(v38);
  *v14 = v24;
  return result;
}
