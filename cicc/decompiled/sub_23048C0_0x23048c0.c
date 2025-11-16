// Function: sub_23048C0
// Address: 0x23048c0
//
__int64 *__fastcall sub_23048C0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __m128i v4; // rax
  __int64 v5; // rax
  __m128i v7; // [rsp+20h] [rbp-20h] BYREF

  v4.m128i_i64[0] = sub_30EBF20(a2 + 8, a3, a4);
  v7 = v4;
  v5 = sub_22077B0(0x18u);
  if ( v5 )
  {
    *(__m128i *)(v5 + 8) = _mm_loadu_si128(&v7);
    *(_QWORD *)v5 = &unk_4A0AFC0;
  }
  *a1 = v5;
  return a1;
}
