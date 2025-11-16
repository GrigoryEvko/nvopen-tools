// Function: sub_9D5100
// Address: 0x9d5100
//
__m128i *__fastcall sub_9D5100(
        __m128i *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        const __m128i *a8,
        unsigned __int64 a9)
{
  char v10; // dl
  const __m128i *v11; // rdi
  __m128i v12; // xmm1
  __m128i v13; // xmm2
  char v14; // al
  __m128i v15; // xmm3
  __int64 v17; // rax
  unsigned __int64 v18; // rax
  __int64 v19; // [rsp+8h] [rbp-68h] BYREF
  const __m128i *v20[3]; // [rsp+10h] [rbp-60h] BYREF
  char v21; // [rsp+28h] [rbp-48h]
  const char *v22; // [rsp+30h] [rbp-40h] BYREF
  char v23; // [rsp+50h] [rbp-20h]
  char v24; // [rsp+51h] [rbp-1Fh]

  sub_9D5080((__int64)v20, a2, a3, a4, a5, a6, a7, a8, a9);
  v10 = v21 & 1;
  v21 = (2 * (v21 & 1)) | v21 & 0xFD;
  if ( v10 )
  {
    v18 = (unsigned __int64)v20[0];
    a1[4].m128i_i8[0] |= 3u;
    a1->m128i_i64[0] = v18 & 0xFFFFFFFFFFFFFFFELL;
    return a1;
  }
  else
  {
    v11 = v20[0];
    if ( (const __m128i *)((char *)v20[1] - (char *)v20[0]) == (const __m128i *)64 )
    {
      v12 = _mm_loadu_si128(v20[0] + 1);
      v13 = _mm_loadu_si128(v20[0] + 2);
      v14 = a1[4].m128i_i8[0] & 0xFC;
      v15 = _mm_loadu_si128(v20[0] + 3);
      *a1 = _mm_loadu_si128(v20[0]);
      a1[1] = v12;
      a1[4].m128i_i8[0] = v14 | 2;
      a1[2] = v13;
      a1[3] = v15;
LABEL_4:
      j_j___libc_free_0(v11, (char *)v20[2] - (char *)v11);
      return a1;
    }
    v24 = 1;
    v22 = "Expected a single module";
    v23 = 3;
    sub_9C8190(&v19, (__int64)&v22);
    v17 = v19;
    a1[4].m128i_i8[0] |= 3u;
    a1->m128i_i64[0] = v17 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v21 & 2) != 0 )
      sub_9D43B0(v20);
    v11 = v20[0];
    if ( (v21 & 1) == 0 )
    {
      if ( !v20[0] )
        return a1;
      goto LABEL_4;
    }
    if ( !v20[0] )
      return a1;
    (*(void (**)(void))(*(_QWORD *)v20[0] + 8LL))();
    return a1;
  }
}
