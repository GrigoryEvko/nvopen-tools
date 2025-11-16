// Function: sub_33665B0
// Address: 0x33665b0
//
__int64 __fastcall sub_33665B0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        const __m128i a7,
        __int128 a8,
        __int64 a9)
{
  __int64 v9; // r11
  unsigned __int32 v10; // r13d
  __int64 i; // r14
  unsigned __int32 v12; // eax
  bool v13; // zf
  __m128i *v14; // rbx
  __m128i *v15; // rax
  __int64 v16; // r12
  __int64 v17; // r9
  __m128i *v18; // r10
  unsigned int v19; // eax
  __int64 v20; // rdx
  const __m128i *v21; // r14
  int v22; // eax
  __m128i v24; // xmm5
  __m128i v25; // xmm4
  const __m128i *v26; // rax
  __int64 v28; // [rsp+10h] [rbp-90h]
  __int64 v29; // [rsp+18h] [rbp-88h]
  __int64 v30; // [rsp+20h] [rbp-80h]
  __int64 v31; // [rsp+30h] [rbp-70h]
  __m128i v32; // [rsp+40h] [rbp-60h] BYREF
  __m128i v33; // [rsp+50h] [rbp-50h] BYREF
  __int64 v34; // [rsp+60h] [rbp-40h]

  v9 = (a3 - 1) / 2;
  v10 = a9;
  v28 = a3 & 1;
  v29 = a7.m128i_i64[1];
  if ( a2 >= v9 )
  {
    v14 = (__m128i *)(a1 + 40 * a2);
    if ( (a3 & 1) != 0 )
    {
      v32 = _mm_loadu_si128(&a7);
      v33 = _mm_loadu_si128((const __m128i *)&a8);
      goto LABEL_20;
    }
    v16 = a2;
    goto LABEL_23;
  }
  for ( i = a2; ; i = v16 )
  {
    v16 = 2 * (i + 1);
    v17 = v16 - 1;
    v14 = (__m128i *)(a1 + 80 * (i + 1));
    v18 = (__m128i *)(a1 + 40 * (v16 - 1));
    v12 = v14[2].m128i_u32[0];
    if ( v18[2].m128i_i32[0] == v12 )
    {
      v30 = v9;
      v19 = sub_C4C880(v14->m128i_i64[1] + 24, v18->m128i_i64[1] + 24);
      v9 = v30;
      v17 = v16 - 1;
      v18 = (__m128i *)(a1 + 40 * (v16 - 1));
      v12 = v19 >> 31;
    }
    else
    {
      LOBYTE(v12) = v18[2].m128i_i32[0] < v12;
    }
    v13 = (_BYTE)v12 == 0;
    if ( (_BYTE)v12 )
      v14 = v18;
    v15 = (__m128i *)(a1 + 40 * i);
    if ( !v13 )
      v16 = v17;
    *v15 = _mm_loadu_si128(v14);
    v15[1] = _mm_loadu_si128(v14 + 1);
    v15[2].m128i_i32[0] = v14[2].m128i_i32[0];
    if ( v16 >= v9 )
      break;
  }
  if ( !v28 )
  {
LABEL_23:
    if ( (a3 - 2) / 2 == v16 )
    {
      v16 = 2 * v16 + 1;
      v26 = (const __m128i *)(a1 + 40 * v16);
      *v14 = _mm_loadu_si128(v26);
      v14[1] = _mm_loadu_si128(v26 + 1);
      v14[2].m128i_i32[0] = v26[2].m128i_i32[0];
      v14 = (__m128i *)v26;
    }
  }
  v34 = a9;
  v32 = _mm_loadu_si128(&a7);
  v33 = _mm_loadu_si128((const __m128i *)&a8);
  v20 = (v16 - 1) / 2;
  if ( v16 > a2 )
  {
    while ( 1 )
    {
      v21 = (const __m128i *)(a1 + 40 * v20);
      if ( v21[2].m128i_i32[0] == v10 )
      {
        v31 = v20;
        v22 = sub_C4C880(v21->m128i_i64[1] + 24, v29 + 24);
        v20 = v31;
        v14 = (__m128i *)(a1 + 40 * v16);
        if ( v22 >= 0 )
          goto LABEL_20;
      }
      else
      {
        v14 = (__m128i *)(a1 + 40 * v16);
        if ( v21[2].m128i_i32[0] <= v10 )
          goto LABEL_20;
      }
      v16 = v20;
      *v14 = _mm_loadu_si128(v21);
      v14[1] = _mm_loadu_si128(v21 + 1);
      v14[2].m128i_i32[0] = v21[2].m128i_i32[0];
      if ( a2 >= v20 )
        break;
      v20 = (v20 - 1) / 2;
    }
    v14 = (__m128i *)v21;
  }
LABEL_20:
  v24 = _mm_loadu_si128(&v33);
  v14[2].m128i_i32[0] = v10;
  v32.m128i_i64[1] = v29;
  v25 = _mm_loadu_si128(&v32);
  v14[1] = v24;
  *v14 = v25;
  return v29;
}
