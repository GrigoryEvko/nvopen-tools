// Function: sub_3913930
// Address: 0x3913930
//
__int64 __fastcall sub_3913930(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8)
{
  __int64 i; // r15
  __int64 v9; // rbx
  __m128i *v10; // r12
  char v11; // al
  bool v12; // zf
  __m128i *v13; // rax
  __int64 v14; // r14
  __m128i *v15; // r15
  __int64 result; // rax
  const __m128i *v17; // rax
  __int64 v19; // [rsp+10h] [rbp-70h]
  __int64 v20; // [rsp+18h] [rbp-68h]
  __m128i v21; // [rsp+30h] [rbp-50h] BYREF
  __int64 v22; // [rsp+40h] [rbp-40h]

  v19 = a3 & 1;
  v20 = (a3 - 1) / 2;
  if ( a2 >= v20 )
  {
    v10 = (__m128i *)(a1 + 24 * a2);
    if ( (a3 & 1) != 0 )
    {
      v21 = _mm_loadu_si128((const __m128i *)&a7);
      v22 = a8;
      goto LABEL_15;
    }
    v9 = a2;
    goto LABEL_18;
  }
  for ( i = a2; ; i = v9 )
  {
    v9 = 2 * (i + 1);
    v10 = (__m128i *)(a1 + 48 * (i + 1));
    v11 = sub_3913890(v10, (_QWORD *)(a1 + 24 * (v9 - 1)));
    v12 = v11 == 0;
    if ( v11 )
      v10 = (__m128i *)(a1 + 24 * (v9 - 1));
    v13 = (__m128i *)(a1 + 24 * i);
    if ( !v12 )
      --v9;
    *v13 = _mm_loadu_si128(v10);
    v13[1].m128i_i64[0] = v10[1].m128i_i64[0];
    if ( v9 >= v20 )
      break;
  }
  if ( !v19 )
  {
LABEL_18:
    if ( (a3 - 2) / 2 == v9 )
    {
      v9 = 2 * v9 + 1;
      v17 = (const __m128i *)(a1 + 24 * v9);
      *v10 = _mm_loadu_si128(v17);
      v10[1].m128i_i64[0] = v17[1].m128i_i64[0];
      v10 = (__m128i *)v17;
    }
  }
  v22 = a8;
  v21 = _mm_loadu_si128((const __m128i *)&a7);
  v14 = (v9 - 1) / 2;
  if ( v9 > a2 )
  {
    while ( 1 )
    {
      v15 = (__m128i *)(a1 + 24 * v14);
      v10 = (__m128i *)(a1 + 24 * v9);
      if ( !sub_3913890(v15, &v21) )
        break;
      v9 = v14;
      *v10 = _mm_loadu_si128(v15);
      v10[1].m128i_i64[0] = v15[1].m128i_i64[0];
      if ( a2 >= v14 )
      {
        v10 = (__m128i *)(a1 + 24 * v14);
        break;
      }
      v14 = (v14 - 1) / 2;
    }
  }
LABEL_15:
  result = v22;
  *v10 = _mm_loadu_si128(&v21);
  v10[1].m128i_i64[0] = result;
  return result;
}
