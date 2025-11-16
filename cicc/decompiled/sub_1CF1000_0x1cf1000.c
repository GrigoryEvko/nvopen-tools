// Function: sub_1CF1000
// Address: 0x1cf1000
//
__m128i *sub_1CF1000()
{
  __m128i *v0; // rax
  __m128i *v1; // r12
  __m128i v2; // xmm0
  __m128i v3; // xmm1
  __int8 v4; // dl
  __int64 v5; // rax
  __int64 v6; // rax
  __m128i v8; // [rsp+0h] [rbp-30h] BYREF
  void (__fastcall *v9)(__m128i *, __m128i *, __int64); // [rsp+10h] [rbp-20h]
  __int64 v10; // [rsp+18h] [rbp-18h]

  v0 = (__m128i *)sub_22077B0(192);
  v1 = v0;
  if ( v0 )
  {
    v2 = _mm_loadu_si128(&v8);
    v0->m128i_i64[1] = 0;
    v3 = _mm_loadu_si128(v0 + 10);
    v0[1].m128i_i32[2] = 3;
    v4 = qword_4FC08E0[20];
    v0[10] = v2;
    v8 = v3;
    v0[1].m128i_i64[0] = (__int64)&unk_4FC0718;
    v0[5].m128i_i64[0] = (__int64)v0[4].m128i_i64;
    v0[5].m128i_i64[1] = (__int64)v0[4].m128i_i64;
    v0[8].m128i_i64[0] = (__int64)v0[7].m128i_i64;
    v0[8].m128i_i64[1] = (__int64)v0[7].m128i_i64;
    v0[9].m128i_i8[9] = v4;
    v0[2].m128i_i64[0] = 0;
    v0->m128i_i64[0] = (__int64)&unk_49F91A8;
    v0[2].m128i_i64[1] = 0;
    v0[11].m128i_i64[0] = (__int64)sub_1CEFD30;
    v5 = v0[11].m128i_i64[1];
    v1[3].m128i_i64[0] = 0;
    v1[4].m128i_i32[0] = 0;
    v1[4].m128i_i64[1] = 0;
    v1[6].m128i_i64[0] = 0;
    v1[7].m128i_i32[0] = 0;
    v1[7].m128i_i64[1] = 0;
    v1[9].m128i_i64[0] = 0;
    v1[9].m128i_i8[8] = 0;
    v9 = 0;
    v10 = v5;
    v1[11].m128i_i64[1] = (__int64)nullsub_675;
    v6 = sub_163A1D0();
    sub_1CF0F10(v6);
    if ( v9 )
      v9(&v8, &v8, 3);
  }
  return v1;
}
