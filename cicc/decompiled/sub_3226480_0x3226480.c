// Function: sub_3226480
// Address: 0x3226480
//
__int64 __fastcall sub_3226480(__int64 a1, __int64 a2, __int32 a3)
{
  __int64 result; // rax
  _QWORD *v4; // rdx
  _QWORD *v5; // r12
  char v6; // r14
  __m128i *v7; // rax
  __m128i v8; // [rsp+0h] [rbp-40h] BYREF

  v8.m128i_i64[1] = a2;
  v8.m128i_i32[0] = a3;
  result = sub_32263E0(a1, (__int64)&v8);
  if ( v4 )
  {
    v5 = v4;
    v6 = 1;
    if ( !result && (_QWORD *)(a1 + 8) != v4 )
      v6 = sub_321DF40((__int64)&v8, (__int64)(v4 + 4));
    v7 = (__m128i *)sub_22077B0(0x30u);
    v7[2] = _mm_loadu_si128(&v8);
    result = (__int64)sub_220F040(v6, (__int64)v7, v5, (_QWORD *)(a1 + 8));
    ++*(_QWORD *)(a1 + 40);
  }
  return result;
}
