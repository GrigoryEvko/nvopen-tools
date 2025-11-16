// Function: sub_14F5920
// Address: 0x14f5920
//
__m128i *__fastcall sub_14F5920(
        __m128i *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __m128i *a8,
        unsigned __int64 a9)
{
  char v10; // dl
  const __m128i *v11; // rdi
  __m128i v12; // xmm1
  __m128i v13; // xmm2
  char v14; // al
  __m128i v15; // xmm3
  __int64 v17; // rdx
  __int64 v18; // rax
  unsigned __int64 v19; // rax
  __int64 v20; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v21[2]; // [rsp+10h] [rbp-50h] BYREF
  char v22; // [rsp+20h] [rbp-40h]
  char v23; // [rsp+21h] [rbp-3Fh]
  const __m128i *v24[3]; // [rsp+30h] [rbp-30h] BYREF
  char v25; // [rsp+48h] [rbp-18h]

  sub_14F58A0((__int64)v24, a2, a3, a4, a5, a6, a7, a8, a9);
  v10 = v25 & 1;
  v25 = (2 * (v25 & 1)) | v25 & 0xFD;
  if ( v10 )
  {
    v19 = (unsigned __int64)v24[0];
    a1[4].m128i_i8[0] |= 3u;
    a1->m128i_i64[0] = v19 & 0xFFFFFFFFFFFFFFFELL;
    return a1;
  }
  else
  {
    v11 = v24[0];
    if ( (const __m128i *)((char *)v24[1] - (char *)v24[0]) == (const __m128i *)64 )
    {
      v12 = _mm_loadu_si128(v24[0] + 1);
      v13 = _mm_loadu_si128(v24[0] + 2);
      v14 = a1[4].m128i_i8[0] & 0xFC;
      v15 = _mm_loadu_si128(v24[0] + 3);
      *a1 = _mm_loadu_si128(v24[0]);
      a1[1] = v12;
      a1[4].m128i_i8[0] = v14 | 2;
      a1[2] = v13;
      a1[3] = v15;
LABEL_4:
      j_j___libc_free_0(v11, (char *)v24[2] - (char *)v11);
      return a1;
    }
    v23 = 1;
    v21[0] = "Expected a single module";
    v22 = 3;
    sub_14EE0F0(&v20, (__int64)v21);
    v18 = v20;
    a1[4].m128i_i8[0] |= 3u;
    a1->m128i_i64[0] = v18 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v25 & 2) != 0 )
      sub_14F4EA0(v24, (__int64)v21, v17);
    v11 = v24[0];
    if ( (v25 & 1) == 0 )
    {
      if ( !v24[0] )
        return a1;
      goto LABEL_4;
    }
    if ( !v24[0] )
      return a1;
    (*(void (**)(void))(*(_QWORD *)v24[0] + 8LL))();
    return a1;
  }
}
