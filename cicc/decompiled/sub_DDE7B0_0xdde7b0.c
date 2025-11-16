// Function: sub_DDE7B0
// Address: 0xdde7b0
//
__m128i *__fastcall sub_DDE7B0(
        __m128i *a1,
        __int64 *a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v10; // r8
  __int64 v11; // r9
  __m128i v12; // xmm1
  __int64 *v14; // rax
  __int64 *v15; // r14
  __m128i v16; // xmm3
  __int64 *v17; // [rsp+0h] [rbp-70h]
  __int64 v19; // [rsp+10h] [rbp-60h]
  __int64 v21; // [rsp+18h] [rbp-58h]
  __m128i v22; // [rsp+20h] [rbp-50h] BYREF
  __m128i v23[4]; // [rsp+30h] [rbp-40h] BYREF

  sub_DDE580((__int64)&v22, a2, a3, a4, a5, a6, a7, a8);
  v10 = a5;
  v11 = a6;
  if ( v23[0].m128i_i8[8] )
  {
    v12 = _mm_loadu_si128(v23);
    *a1 = _mm_loadu_si128(&v22);
    a1[1] = v12;
  }
  else if ( *(_WORD *)(a8 + 24) != 11 || (v14 = *(__int64 **)(a8 + 32), v17 = &v14[*(_QWORD *)(a8 + 40)], v17 == v14) )
  {
LABEL_10:
    a1[1].m128i_i8[8] = 0;
  }
  else
  {
    v15 = *(__int64 **)(a8 + 32);
    while ( 1 )
    {
      v19 = v11;
      v21 = v10;
      sub_DDE580((__int64)&v22, a2, a3, a4, v10, v11, a7, *v15);
      v10 = v21;
      v11 = v19;
      if ( v23[0].m128i_i8[8] )
        break;
      if ( v17 == ++v15 )
        goto LABEL_10;
    }
    v16 = _mm_loadu_si128(v23);
    *a1 = _mm_loadu_si128(&v22);
    a1[1] = v16;
  }
  return a1;
}
