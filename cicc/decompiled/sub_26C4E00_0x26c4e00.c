// Function: sub_26C4E00
// Address: 0x26c4e00
//
__m128i *__fastcall sub_26C4E00(__int64 a1, const __m128i *a2)
{
  __m128i *result; // rax
  _QWORD *v3; // rdx
  _QWORD *v4; // r14
  _QWORD *v5; // rcx
  char v6; // r15
  __m128i *v7; // rbx
  int v8; // eax
  _QWORD *v9; // [rsp+8h] [rbp-38h]

  result = (__m128i *)sub_26C4C60(a1, (__int64)a2);
  if ( v3 )
  {
    v4 = v3;
    v5 = (_QWORD *)(a1 + 8);
    v6 = 1;
    if ( !result && v3 != v5 )
    {
      v8 = sub_C1F8C0(a2->m128i_i64[1], v3[5]);
      v5 = (_QWORD *)(a1 + 8);
      v6 = v8 < 0;
    }
    v9 = v5;
    v7 = (__m128i *)sub_22077B0(0x38u);
    v7[2] = _mm_loadu_si128(a2);
    v7[3].m128i_i64[0] = a2[1].m128i_i64[0];
    sub_220F040(v6, (__int64)v7, v4, v9);
    ++*(_QWORD *)(a1 + 40);
    return v7;
  }
  return result;
}
