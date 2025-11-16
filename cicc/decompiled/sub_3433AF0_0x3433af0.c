// Function: sub_3433AF0
// Address: 0x3433af0
//
__m128i *__fastcall sub_3433AF0(__int64 a1, __int64 a2, __int64 a3, const __m128i *a4)
{
  bool v4; // r8
  _QWORD *v5; // r15
  __m128i *v8; // r12
  unsigned __int64 v10; // rax
  char v11; // [rsp+Ch] [rbp-34h]

  v4 = 1;
  v5 = (_QWORD *)(a1 + 8);
  if ( !a2 && (_QWORD *)a3 != v5 )
  {
    v10 = *(_QWORD *)(a3 + 32);
    v4 = a4->m128i_i64[0] < v10 || a4->m128i_i64[0] == v10 && a4->m128i_i32[2] < *(_DWORD *)(a3 + 40);
  }
  v11 = v4;
  v8 = (__m128i *)sub_22077B0(0x30u);
  v8[2] = _mm_loadu_si128(a4);
  sub_220F040(v11, (__int64)v8, (_QWORD *)a3, v5);
  ++*(_QWORD *)(a1 + 40);
  return v8;
}
