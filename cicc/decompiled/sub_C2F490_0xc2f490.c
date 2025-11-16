// Function: sub_C2F490
// Address: 0xc2f490
//
__int64 *__fastcall sub_C2F490(
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
  __int64 v10; // rdx
  __m128i v11; // xmm1
  __m128i v13; // [rsp+40h] [rbp-40h] BYREF
  __int64 v14; // [rsp+50h] [rbp-30h]

  v13 = _mm_loadu_si128((const __m128i *)&a7);
  v14 = a8;
  v9 = sub_22077B0(48);
  if ( v9 )
  {
    v10 = v14;
    v11 = _mm_loadu_si128(&v13);
    *(_QWORD *)(v9 + 8) = a3;
    *(_QWORD *)(v9 + 40) = a2 + 32;
    *(_QWORD *)(v9 + 32) = v10;
    *(__m128i *)(v9 + 16) = v11;
    *(_QWORD *)v9 = &unk_49DBEC8;
  }
  *a1 = v9;
  return a1;
}
