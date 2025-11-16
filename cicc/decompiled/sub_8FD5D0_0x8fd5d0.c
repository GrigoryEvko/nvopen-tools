// Function: sub_8FD5D0
// Address: 0x8fd5d0
//
__m128i *__fastcall sub_8FD5D0(__m128i *a1, __int64 a2, _QWORD *a3)
{
  __int64 v6; // rsi
  __int64 v7; // r8
  __int64 v8; // rdx
  __int64 v9; // rcx
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // r10
  _QWORD *v12; // rsi
  unsigned __int64 v13; // r10
  __m128i *v14; // rax
  __int64 v15; // rcx
  __m128i *v16; // rdx

  v6 = a2 + 16;
  v7 = *(_QWORD *)(v6 - 8);
  v8 = a3[1];
  v9 = *(_QWORD *)(v6 - 16);
  v10 = v7 + v8;
  if ( v9 == v6 )
    v11 = 15;
  else
    v11 = *(_QWORD *)(a2 + 16);
  v12 = (_QWORD *)*a3;
  if ( v10 > v11 )
  {
    v13 = v12 == a3 + 2 ? 15LL : a3[2];
    if ( v10 <= v13 )
    {
      v14 = (__m128i *)sub_2241130(a3, 0, 0, v9, v7);
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      v15 = v14->m128i_i64[0];
      v16 = v14 + 1;
      if ( (__m128i *)v14->m128i_i64[0] != &v14[1] )
        goto LABEL_8;
LABEL_11:
      a1[1] = _mm_loadu_si128(v14 + 1);
      goto LABEL_9;
    }
  }
  v14 = (__m128i *)sub_2241490(a2, v12, v8, v9);
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  v15 = v14->m128i_i64[0];
  v16 = v14 + 1;
  if ( (__m128i *)v14->m128i_i64[0] == &v14[1] )
    goto LABEL_11;
LABEL_8:
  a1->m128i_i64[0] = v15;
  a1[1].m128i_i64[0] = v14[1].m128i_i64[0];
LABEL_9:
  a1->m128i_i64[1] = v14->m128i_i64[1];
  v14->m128i_i64[0] = (__int64)v16;
  v14->m128i_i64[1] = 0;
  v14[1].m128i_i8[0] = 0;
  return a1;
}
