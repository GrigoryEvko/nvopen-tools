// Function: sub_234B580
// Address: 0x234b580
//
__int64 *__fastcall sub_234B580(__int64 *a1, __m128i *a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __m128i v4; // xmm1
  __m128i v5; // xmm0
  __int64 v6; // rax
  __int64 v7; // rax
  __m128i v8; // xmm2
  void (__fastcall *v10)(__m128i *, __m128i *, __int64); // rax
  __m128i v11; // [rsp+0h] [rbp-40h] BYREF
  void (__fastcall *v12)(__m128i *, __m128i *, __int64); // [rsp+10h] [rbp-30h]
  __int64 v13; // [rsp+18h] [rbp-28h]
  char v14; // [rsp+20h] [rbp-20h]

  v2 = a2[1].m128i_i64[0];
  v3 = v13;
  a2[1].m128i_i64[0] = 0;
  v4 = _mm_loadu_si128(&v11);
  v5 = _mm_loadu_si128(a2);
  v12 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v2;
  v6 = a2[1].m128i_i64[1];
  a2[1].m128i_i64[1] = v3;
  v13 = v6;
  LOBYTE(v6) = a2[2].m128i_i8[0];
  *a2 = v4;
  v14 = v6;
  v11 = v5;
  v7 = sub_22077B0(0x30u);
  if ( v7 )
  {
    v8 = _mm_loadu_si128(&v11);
    *a1 = v7;
    *(__m128i *)(v7 + 8) = v8;
    *(_QWORD *)v7 = &unk_4A0EC38;
    *(_QWORD *)(v7 + 24) = v12;
    *(_QWORD *)(v7 + 32) = v13;
    *(_BYTE *)(v7 + 40) = v14;
    return a1;
  }
  v10 = v12;
  *a1 = 0;
  if ( !v10 )
    return a1;
  v10(&v11, &v11, 3);
  return a1;
}
