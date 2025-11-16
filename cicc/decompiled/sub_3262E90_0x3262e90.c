// Function: sub_3262E90
// Address: 0x3262e90
//
void __fastcall sub_3262E90(__m128i *a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 m128i_i64; // r14
  __int64 v4; // rbx

  if ( a2 - (__int64)a1 <= 224 )
  {
    sub_32613F0((unsigned int *)a1, a2);
  }
  else
  {
    v2 = (a2 - (__int64)a1) >> 5;
    m128i_i64 = (__int64)a1[v2].m128i_i64;
    v4 = (16 * v2) >> 4;
    sub_3262E90(a1, m128i_i64);
    sub_3262E90(m128i_i64, a2);
    sub_3262B80(a1, m128i_i64, a2, v4, (a2 - m128i_i64) >> 4);
  }
}
