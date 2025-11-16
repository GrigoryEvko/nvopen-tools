// Function: sub_982780
// Address: 0x982780
//
__int64 __fastcall sub_982780(
        const __m128i *a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 (__fastcall *a4)(__m128i *, __m128i *))
{
  __int64 result; // rax
  __m128i *v7; // r15
  __m128i *v8; // r12
  __m128i *v9; // r14
  __m128i *v10; // rsi
  __int64 v11; // r14
  __int64 v12; // r11
  __int64 v13; // r10
  __int64 v14; // r9
  __int32 v15; // r8d
  __int8 v16; // di
  __int8 v17; // si
  __int64 v18; // rcx
  __int64 v19; // rax
  const __m128i *v20; // r14
  unsigned __int8 (__fastcall *v21)(__m128i *); // r15
  __m128i *v22; // r12
  __int128 v23; // rax
  __m128i *v24; // r13
  __int128 v25; // rcx
  __int64 v26; // r13
  __int64 v27; // rsi
  __int64 v28; // rdi
  __int64 v29; // r8
  __int64 v30; // r9
  __int128 v31; // [rsp-20h] [rbp-B0h]
  __int128 v32; // [rsp-10h] [rbp-A0h]
  __m128i *v33; // [rsp+0h] [rbp-90h]
  __m128i *v34; // [rsp+8h] [rbp-88h]
  __int64 v35; // [rsp+10h] [rbp-80h]
  unsigned __int64 v36; // [rsp+18h] [rbp-78h]

  result = a2 - (_QWORD)a1;
  v35 = a3;
  v34 = (__m128i *)a2;
  if ( (__int64)(a2 - (_QWORD)a1) <= 1024 )
    return result;
  if ( !a3 )
  {
    v36 = a2;
    goto LABEL_15;
  }
  v33 = (__m128i *)&a1[4];
  while ( 2 )
  {
    v7 = v34;
    v8 = v33;
    --v35;
    sub_981E60(
      (__int64)a1,
      v33,
      (__m128i *)&a1[4
                   * ((__int64)(((unsigned __int64)((char *)v34 - (char *)a1) >> 63) + (((char *)v34 - (char *)a1) >> 6)) >> 1)],
      v34 - 4,
      a4);
    while ( 1 )
    {
      v36 = (unsigned __int64)v8;
      if ( (unsigned __int8)a4(v8, (__m128i *)a1) )
        goto LABEL_10;
      v9 = v7 - 4;
      do
      {
        v10 = v9;
        v7 = v9;
        v9 -= 4;
      }
      while ( (unsigned __int8)a4((__m128i *)a1, v10) );
      if ( v8 >= v7 )
        break;
      v11 = v8->m128i_i64[0];
      v12 = v8->m128i_i64[1];
      v13 = v8[1].m128i_i64[0];
      *v8 = _mm_loadu_si128(v7);
      v14 = v8[1].m128i_i64[1];
      v15 = v8[2].m128i_i32[0];
      v16 = v8[2].m128i_i8[4];
      v8[1] = _mm_loadu_si128(v7 + 1);
      v17 = v8[2].m128i_i8[8];
      v18 = v8[3].m128i_i64[0];
      v19 = v8[3].m128i_i64[1];
      v8[2] = _mm_loadu_si128(v7 + 2);
      v8[3] = _mm_loadu_si128(v7 + 3);
      v7->m128i_i64[0] = v11;
      v7->m128i_i64[1] = v12;
      v7[1].m128i_i64[0] = v13;
      v7[1].m128i_i64[1] = v14;
      v7[2].m128i_i32[0] = v15;
      v7[2].m128i_i8[4] = v16;
      v7[2].m128i_i8[8] = v17;
      v7[3].m128i_i64[0] = v18;
      v7[3].m128i_i64[1] = v19;
LABEL_10:
      v8 += 4;
    }
    sub_982780(v8, v34, v35, a4);
    result = (char *)v8 - (char *)a1;
    if ( (char *)v8 - (char *)a1 > 1024 )
    {
      if ( v35 )
      {
        v34 = v8;
        continue;
      }
LABEL_15:
      v20 = a1;
      v21 = (unsigned __int8 (__fastcall *)(__m128i *))a4;
      v22 = (__m128i *)(v36 - 64);
      sub_9825D0(a1, v36, v36, (unsigned __int8 (__fastcall *)(__m128i *))a4);
      do
      {
        *((_QWORD *)&v23 + 1) = v22->m128i_i64[1];
        v24 = v22;
        v22 -= 4;
        *(_QWORD *)&v23 = v22[4].m128i_i64[0];
        *(_QWORD *)&v25 = v22[5].m128i_i64[0];
        v26 = (char *)v24 - (char *)v20;
        v22[4] = _mm_loadu_si128(v20);
        *((_QWORD *)&v25 + 1) = v22[5].m128i_i64[1];
        v27 = v22[6].m128i_i64[0];
        v28 = v22[6].m128i_i64[1];
        v22[5] = _mm_loadu_si128(v20 + 1);
        v29 = v22[7].m128i_i64[0];
        v30 = v22[7].m128i_i64[1];
        v22[6] = _mm_loadu_si128(v20 + 2);
        v22[7] = _mm_loadu_si128(v20 + 3);
        *((_QWORD *)&v32 + 1) = v30;
        *(_QWORD *)&v32 = v29;
        *((_QWORD *)&v31 + 1) = v28;
        *(_QWORD *)&v31 = v27;
        result = sub_982320((__int64)v20, 0, v26 >> 6, v21, v29, v30, v23, v25, v31, v32);
      }
      while ( v26 > 64 );
    }
    return result;
  }
}
