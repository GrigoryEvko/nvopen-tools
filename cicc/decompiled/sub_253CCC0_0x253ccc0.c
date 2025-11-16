// Function: sub_253CCC0
// Address: 0x253ccc0
//
__m128i *__fastcall sub_253CCC0(__int64 a1, __int64 a2, __int64 a3, const __m128i *a4)
{
  bool v4; // r8
  _QWORD *v5; // r15
  __m128i *v8; // r12
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rax
  char v13; // [rsp+Ch] [rbp-34h]

  v4 = 1;
  v5 = (_QWORD *)(a1 + 8);
  if ( !a2 && (_QWORD *)a3 != v5 )
  {
    v10 = *(_QWORD *)(a3 + 32);
    if ( a4->m128i_i64[0] == v10 )
    {
      v11 = a4->m128i_u64[1];
      v12 = *(_QWORD *)(a3 + 40);
      if ( v11 == v12 )
      {
        v4 = a4[1].m128i_i8[0] < *(_BYTE *)(a3 + 48);
        goto LABEL_2;
      }
      goto LABEL_8;
    }
    if ( a4->m128i_i64[0] >= v10 )
    {
      v4 = 0;
      if ( a4->m128i_i64[0] == v10 )
      {
        v11 = a4->m128i_u64[1];
        v12 = *(_QWORD *)(a3 + 40);
LABEL_8:
        v4 = v11 < v12;
      }
    }
  }
LABEL_2:
  v13 = v4;
  v8 = (__m128i *)sub_22077B0(0x38u);
  v8[2] = _mm_loadu_si128(a4);
  v8[3].m128i_i64[0] = a4[1].m128i_i64[0];
  sub_220F040(v13, (__int64)v8, (_QWORD *)a3, v5);
  ++*(_QWORD *)(a1 + 40);
  return v8;
}
