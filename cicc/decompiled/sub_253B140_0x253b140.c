// Function: sub_253B140
// Address: 0x253b140
//
__m128i *__fastcall sub_253B140(__int64 a1, __int64 a2, __int64 a3, const __m128i *a4)
{
  bool v4; // r8
  _QWORD *v5; // r15
  __m128i *v8; // r12
  unsigned __int64 v10; // rax
  bool v11; // cf
  unsigned __int64 v12; // rax
  __int32 v13; // eax
  char v14; // [rsp+Ch] [rbp-34h]

  v4 = 1;
  v5 = (_QWORD *)(a1 + 24);
  if ( !a2 && (_QWORD *)a3 != v5 )
  {
    v10 = *(_QWORD *)(a3 + 32);
    v11 = a4->m128i_i64[0] < v10;
    if ( a4->m128i_i64[0] == v10 && (v12 = *(_QWORD *)(a3 + 40), v11 = a4->m128i_i64[1] < v12, a4->m128i_i64[1] == v12) )
    {
      v13 = *(_DWORD *)(a3 + 48);
      v4 = 0;
      if ( a4[1].m128i_i32[0] != v13 )
        v4 = a4[1].m128i_i32[0] < v13;
    }
    else
    {
      v4 = v11;
    }
  }
  v14 = v4;
  v8 = (__m128i *)sub_22077B0(0x38u);
  v8[2] = _mm_loadu_si128(a4);
  v8[3].m128i_i64[0] = a4[1].m128i_i64[0];
  sub_220F040(v14, (__int64)v8, (_QWORD *)a3, v5);
  ++*(_QWORD *)(a1 + 56);
  return v8;
}
