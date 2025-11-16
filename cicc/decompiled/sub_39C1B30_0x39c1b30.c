// Function: sub_39C1B30
// Address: 0x39c1b30
//
__int64 __fastcall sub_39C1B30(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 result; // rax
  __m128i v7; // [rsp+0h] [rbp-20h] BYREF

  v7.m128i_i64[0] = a2;
  v7.m128i_i64[1] = a3;
  v5 = sub_39C1660(a1, &v7);
  result = *(_QWORD *)v5 + 16LL * *(unsigned int *)(v5 + 8);
  *(_QWORD *)(result - 8) = a4;
  return result;
}
