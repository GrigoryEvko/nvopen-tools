// Function: sub_9471C0
// Address: 0x9471c0
//
__int64 __fastcall sub_9471C0(__int64 a1, __int64 a2)
{
  __m128i v2; // xmm1
  __m128i *v3; // rax
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 result; // rax
  char v7; // cl
  __int64 v8; // rdx
  __int64 i; // [rsp+8h] [rbp-58h]
  __m128i v10; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int8 v11; // [rsp+20h] [rbp-40h]

  sub_946FA0((__int64)&v10, a1, *(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL));
  v2 = _mm_loadu_si128(&v10);
  v3 = *(__m128i **)(a2 + 16);
  v3[1].m128i_i8[0] = v11;
  *v3 = v2;
  v4 = *(_QWORD *)(a2 + 16);
  v5 = v4 + 40;
  result = v4 + 8 * (5LL * *(unsigned int *)(a2 + 8) + 5);
  for ( i = result; v5 != i; *(_BYTE *)(v5 - 24) = result )
  {
    v7 = *(_BYTE *)(v5 + 33);
    v8 = *(_QWORD *)(v5 + 24);
    v5 += 40;
    sub_947060((__int64)&v10, a1, v8, v7);
    result = v11;
    *(__m128i *)(v5 - 40) = _mm_loadu_si128(&v10);
  }
  return result;
}
