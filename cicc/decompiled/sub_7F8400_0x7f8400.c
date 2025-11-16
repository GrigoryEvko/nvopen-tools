// Function: sub_7F8400
// Address: 0x7f8400
//
_QWORD *__fastcall sub_7F8400(__m128i *a1, __int64 a2)
{
  _QWORD *v3; // rbx
  _QWORD *result; // rax
  __int64 v5; // r12
  void *v6; // rax
  __m128i *v7; // rax
  const __m128i *v8; // rdx
  __m128i *v9; // rdi
  const __m128i *v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rcx
  _UNKNOWN *__ptr32 *v13; // r8
  __int64 v14; // rax
  __m128i *v15; // [rsp+8h] [rbp-28h] BYREF

  v3 = sub_72BA30(byte_4D03F80[0]);
  result = (_QWORD *)sub_8D6740(v3);
  if ( v3 != result )
  {
    v5 = (__int64)result;
    if ( !v3 || !result || !dword_4F07588 || (result = (_QWORD *)v3[4], *(_QWORD **)(v5 + 32) != result) || !result )
    {
      if ( a1[1].m128i_i8[8] == 2 )
      {
        v7 = (__m128i *)sub_724DC0();
        v8 = (const __m128i *)a1[3].m128i_i64[1];
        v15 = v7;
        *v7 = _mm_loadu_si128(v8);
        v9 = v15;
        v7[1] = _mm_loadu_si128(v8 + 1);
        v7[2] = _mm_loadu_si128(v8 + 2);
        v7[3] = _mm_loadu_si128(v8 + 3);
        v7[4] = _mm_loadu_si128(v8 + 4);
        v7[5] = _mm_loadu_si128(v8 + 5);
        v7[6] = _mm_loadu_si128(v8 + 6);
        v7[7] = _mm_loadu_si128(v8 + 7);
        v7[8] = _mm_loadu_si128(v8 + 8);
        v7[9] = _mm_loadu_si128(v8 + 9);
        v7[10] = _mm_loadu_si128(v8 + 10);
        v7[11] = _mm_loadu_si128(v8 + 11);
        v7[12] = _mm_loadu_si128(v8 + 12);
        sub_7EAFC0(v9);
        v10 = v15;
        v15[8].m128i_i64[0] = v5;
        v14 = sub_73A460(v10, a2, v11, v12, v13);
        a1->m128i_i64[0] = v5;
        a1[3].m128i_i64[1] = v14;
        return sub_724E30((__int64)&v15);
      }
      else
      {
        v6 = sub_730FF0(a1);
        return (_QWORD *)sub_7E2300((__int64)a1, (__int64)v6, v5);
      }
    }
  }
  return result;
}
