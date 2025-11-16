// Function: sub_C2F500
// Address: 0xc2f500
//
__int64 *__fastcall sub_C2F500(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8)
{
  __int64 v9; // rax
  __m128i v10; // xmm1
  __m128i v12; // [rsp+20h] [rbp-30h] BYREF
  __int64 v13; // [rsp+30h] [rbp-20h]

  v12 = _mm_loadu_si128((const __m128i *)&a7);
  v13 = a8;
  v9 = sub_22077B0(40);
  if ( v9 )
  {
    v10 = _mm_loadu_si128(&v12);
    *(_QWORD *)(v9 + 8) = a3;
    *(__m128i *)(v9 + 16) = v10;
    *(_QWORD *)v9 = &unk_49DBEA0;
    *(_QWORD *)(v9 + 32) = v13;
  }
  *a1 = v9;
  return a1;
}
