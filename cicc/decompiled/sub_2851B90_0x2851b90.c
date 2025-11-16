// Function: sub_2851B90
// Address: 0x2851b90
//
__m128i *__fastcall sub_2851B90(__int64 a1, __int64 a2, __int64 a3, const __m128i *a4)
{
  bool v4; // r8
  _QWORD *v5; // r15
  __m128i *v8; // r12
  unsigned __int64 v10; // rax
  char v11; // al
  char v12; // [rsp+Ch] [rbp-34h]

  v4 = 1;
  v5 = (_QWORD *)(a1 + 8);
  if ( !a2 && (_QWORD *)a3 != v5 )
  {
    v10 = *(_QWORD *)(a3 + 32);
    if ( a4->m128i_i64[0] != v10 )
    {
      v4 = a4->m128i_i64[0] < v10;
      goto LABEL_2;
    }
    v11 = *(_BYTE *)(a3 + 48);
    if ( a4[1].m128i_i8[0] )
    {
      v4 = 0;
      if ( !v11 )
        goto LABEL_2;
      goto LABEL_7;
    }
    if ( !v11 )
LABEL_7:
      v4 = a4->m128i_i64[1] < *(_QWORD *)(a3 + 40);
  }
LABEL_2:
  v12 = v4;
  v8 = (__m128i *)sub_22077B0(0x38u);
  v8[2] = _mm_loadu_si128(a4);
  v8[3].m128i_i64[0] = a4[1].m128i_i64[0];
  sub_220F040(v12, (__int64)v8, (_QWORD *)a3, v5);
  ++*(_QWORD *)(a1 + 40);
  return v8;
}
