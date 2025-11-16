// Function: sub_266DE70
// Address: 0x266de70
//
__int64 __fastcall sub_266DE70(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __m128i *v4; // rdx
  __int64 result; // rax
  __m128i v6; // [rsp+0h] [rbp-20h] BYREF

  v2 = (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, _QWORD *))(*(_QWORD *)**a1 + 112LL))(
         **a1,
         *(unsigned int *)a1[1],
         a2,
         a1[2]);
  v6.m128i_i64[1] = v3;
  v4 = (__m128i *)a1[3];
  v6.m128i_i64[0] = v2;
  if ( !v4->m128i_i8[8]
    || (result = v6.m128i_u8[8], v6.m128i_i8[8]) && (result = 0, v4->m128i_i64[0] == v6.m128i_i64[0]) )
  {
    *v4 = _mm_loadu_si128(&v6);
    return 1;
  }
  return result;
}
