// Function: sub_1462F20
// Address: 0x1462f20
//
void __fastcall sub_1462F20(
        __int64 *src,
        __int64 **a2,
        char *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        const __m128i a7,
        const __m128i a8)
{
  __int64 v10; // rdx
  __int64 **v11; // r14
  __int64 **v12; // r15
  __int64 *v13; // rdi
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // r15
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // [rsp+0h] [rbp-60h]
  __int64 v21; // [rsp+8h] [rbp-58h]
  __m128i v22; // [rsp+10h] [rbp-50h]
  __m128i v23; // [rsp+20h] [rbp-40h]

  v10 = ((char *)a2 - (char *)src) >> 3;
  v11 = (__int64 **)&a3[(char *)a2 - (char *)src];
  v20 = (char *)a2 - (char *)src;
  v21 = v10;
  v22 = _mm_loadu_si128(&a7);
  v23 = _mm_loadu_si128(&a8);
  if ( (char *)a2 - (char *)src <= 48 )
  {
    sub_1462D40(
      src,
      a2,
      v10,
      a4,
      a5,
      a6,
      a7.m128i_i64[0],
      (_QWORD *)a7.m128i_i64[1],
      (__int64 *)a8.m128i_i64[0],
      a8.m128i_i64[1]);
  }
  else
  {
    v12 = (__int64 **)src;
    do
    {
      v13 = (__int64 *)v12;
      v12 += 7;
      sub_1462D40(
        v13,
        v12,
        v10,
        a4,
        a5,
        a6,
        v22.m128i_i64[0],
        (_QWORD *)v22.m128i_i64[1],
        (__int64 *)v23.m128i_i64[0],
        v23.m128i_i64[1]);
    }
    while ( (char *)a2 - (char *)v12 > 48 );
    sub_1462D40(
      (__int64 *)v12,
      a2,
      v10,
      a4,
      a5,
      a6,
      v22.m128i_i64[0],
      (_QWORD *)v22.m128i_i64[1],
      (__int64 *)v23.m128i_i64[0],
      v23.m128i_i64[1]);
    if ( v20 > 56 )
    {
      v16 = 7;
      do
      {
        sub_1462C90(
          (__int64 **)src,
          a2,
          a3,
          v16,
          v14,
          v15,
          a7.m128i_i64[0],
          (_QWORD *)a7.m128i_i64[1],
          (__int64 *)a8.m128i_i64[0],
          a8.m128i_i64[1]);
        v17 = 2 * v16;
        v16 *= 4;
        sub_1462C90(
          (__int64 **)a3,
          v11,
          (char *)src,
          v17,
          v18,
          v19,
          a7.m128i_i64[0],
          (_QWORD *)a7.m128i_i64[1],
          (__int64 *)a8.m128i_i64[0],
          a8.m128i_i64[1]);
      }
      while ( v21 > v16 );
    }
  }
}
