// Function: sub_11BE290
// Address: 0x11be290
//
__m128i *__fastcall sub_11BE290(
        __m128i *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        unsigned __int8 *a8)
{
  unsigned __int8 *v8; // rax
  __m128i v9; // xmm1
  unsigned __int8 *v11; // r14
  unsigned __int8 *v12; // r13
  __int64 v13; // rbx
  __int64 v14; // rax
  unsigned __int8 *v15; // rax
  __m128i v16; // xmm2
  __m128i v17; // xmm3
  __int128 *v18; // [rsp+0h] [rbp-40h] BYREF
  __int64 v19; // [rsp+8h] [rbp-38h]

  if ( (_DWORD)a7 == 86 )
  {
    v19 = a2;
    v18 = &a7;
    v15 = sub_BD4CB0(a8, (void (__fastcall *)(__int64, unsigned __int8 *))sub_11BE170, (__int64)&v18);
    v16 = _mm_loadu_si128((const __m128i *)&a7);
    a8 = v15;
    a1[1].m128i_i64[0] = (__int64)v15;
    *a1 = v16;
    return a1;
  }
  if ( (unsigned int)a7 > 0x56 )
  {
    if ( (unsigned int)(a7 - 90) > 1 )
      goto LABEL_13;
    v11 = a8;
    LODWORD(v19) = sub_AE43F0(a2, *((_QWORD *)a8 + 1));
    if ( (unsigned int)v19 > 0x40 )
      sub_C43690((__int64)&v18, 0, 0);
    else
      v18 = 0;
    v12 = sub_BD45C0(v11, a2, (__int64)&v18, 0, 0, 0, 0, 0);
    if ( (unsigned int)v19 > 0x40 )
    {
      v13 = *(_QWORD *)v18;
      j_j___libc_free_0_0(v18);
    }
    else
    {
      v13 = 0;
      if ( !(_DWORD)v19 )
        goto LABEL_17;
      v13 = (__int64)((_QWORD)v18 << (64 - (unsigned __int8)v19)) >> (64 - (unsigned __int8)v19);
    }
    if ( v13 < 0 )
    {
LABEL_13:
      v14 = (__int64)a8;
      *a1 = _mm_loadu_si128((const __m128i *)&a7);
      a1[1].m128i_i64[0] = v14;
      return a1;
    }
LABEL_17:
    *((_QWORD *)&a7 + 1) += v13;
    v17 = _mm_loadu_si128((const __m128i *)&a7);
    a8 = v12;
    a1[1].m128i_i64[0] = (__int64)v12;
    *a1 = v17;
    return a1;
  }
  if ( (_DWORD)a7 != 43 )
    goto LABEL_13;
  v8 = sub_98ACB0(a8, 6u);
  v9 = _mm_loadu_si128((const __m128i *)&a7);
  a8 = v8;
  a1[1].m128i_i64[0] = (__int64)v8;
  *a1 = v9;
  return a1;
}
