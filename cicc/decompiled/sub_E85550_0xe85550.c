// Function: sub_E85550
// Address: 0xe85550
//
__m128i *__fastcall sub_E85550(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  __m128i *result; // rax
  __int64 v7; // rdi
  __int64 v8; // rsi
  __m128i v9; // [rsp+0h] [rbp-40h] BYREF
  __m128i *v10; // [rsp+10h] [rbp-30h]

  v5 = sub_E6C430(*(_QWORD *)(a1 + 8), a2, a3, a4, a5);
  sub_E85210(a1, v5, 0);
  result = *(__m128i **)(a1 + 296);
  v9.m128i_i32[0] = a2;
  v9.m128i_i64[1] = v5;
  v7 = result[1].m128i_i64[1];
  v10 = 0;
  v8 = *(_QWORD *)(v7 + 208);
  if ( v8 == *(_QWORD *)(v7 + 216) )
    return sub_E835F0(v7 + 200, (_BYTE *)v8, &v9);
  if ( v8 )
  {
    *(__m128i *)v8 = _mm_loadu_si128(&v9);
    result = v10;
    *(_QWORD *)(v8 + 16) = v10;
    v8 = *(_QWORD *)(v7 + 208);
  }
  *(_QWORD *)(v7 + 208) = v8 + 24;
  return result;
}
