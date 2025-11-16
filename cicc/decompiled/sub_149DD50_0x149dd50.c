// Function: sub_149DD50
// Address: 0x149dd50
//
__int64 __fastcall sub_149DD50(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 (__fastcall *a4)(__m128i *),
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int64 a9)
{
  __int64 i; // r15
  __int64 v10; // r13
  __m128i *v11; // r12
  char v12; // al
  bool v13; // zf
  __m128i *v14; // rax
  __int64 v15; // r14
  const __m128i *v16; // r15
  __m128i v17; // xmm5
  __int64 result; // rax
  const __m128i *v19; // rax
  __int64 v21; // [rsp+8h] [rbp-88h]
  __int64 v22; // [rsp+10h] [rbp-80h]
  __m128i v24; // [rsp+30h] [rbp-60h] BYREF
  __m128i v25; // [rsp+40h] [rbp-50h] BYREF
  __int64 v26; // [rsp+50h] [rbp-40h]

  v22 = (a3 - 1) / 2;
  v21 = a3 & 1;
  if ( a2 >= v22 )
  {
    v11 = (__m128i *)(a1 + 40 * a2);
    if ( (a3 & 1) != 0 )
    {
      v24 = _mm_loadu_si128((const __m128i *)&a7);
      v26 = a9;
      v25 = _mm_loadu_si128((const __m128i *)&a8);
      goto LABEL_15;
    }
    v10 = a2;
    goto LABEL_18;
  }
  for ( i = a2; ; i = v10 )
  {
    v10 = 2 * (i + 1);
    v11 = (__m128i *)(a1 + 80 * (i + 1));
    v12 = a4(v11);
    v13 = v12 == 0;
    if ( v12 )
      v11 = (__m128i *)(a1 + 40 * (v10 - 1));
    v14 = (__m128i *)(a1 + 40 * i);
    if ( !v13 )
      --v10;
    *v14 = _mm_loadu_si128(v11);
    v14[1] = _mm_loadu_si128(v11 + 1);
    v14[2].m128i_i32[0] = v11[2].m128i_i32[0];
    if ( v10 >= v22 )
      break;
  }
  if ( !v21 )
  {
LABEL_18:
    if ( (a3 - 2) / 2 == v10 )
    {
      v10 = 2 * v10 + 1;
      v19 = (const __m128i *)(a1 + 40 * v10);
      *v11 = _mm_loadu_si128(v19);
      v11[1] = _mm_loadu_si128(v19 + 1);
      v11[2].m128i_i32[0] = v19[2].m128i_i32[0];
      v11 = (__m128i *)v19;
    }
  }
  v26 = a9;
  v24 = _mm_loadu_si128((const __m128i *)&a7);
  v25 = _mm_loadu_si128((const __m128i *)&a8);
  v15 = (v10 - 1) / 2;
  if ( v10 > a2 )
  {
    while ( 1 )
    {
      v16 = (const __m128i *)(a1 + 40 * v15);
      v11 = (__m128i *)(a1 + 40 * v10);
      if ( !((unsigned __int8 (__fastcall *)(const __m128i *, __m128i *))a4)(v16, &v24) )
        break;
      v10 = v15;
      *v11 = _mm_loadu_si128(v16);
      v11[1] = _mm_loadu_si128(v16 + 1);
      v11[2].m128i_i32[0] = v16[2].m128i_i32[0];
      if ( a2 >= v15 )
      {
        v11 = (__m128i *)(a1 + 40 * v15);
        break;
      }
      v15 = (v15 - 1) / 2;
    }
  }
LABEL_15:
  v17 = _mm_loadu_si128(&v25);
  result = (unsigned int)v26;
  *v11 = _mm_loadu_si128(&v24);
  v11[2].m128i_i32[0] = result;
  v11[1] = v17;
  return result;
}
