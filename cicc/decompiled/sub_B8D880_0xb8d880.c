// Function: sub_B8D880
// Address: 0xb8d880
//
__m128i *__fastcall sub_B8D880(__m128i *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  int v5; // edi
  __int64 v6; // rax
  __m128i v7; // xmm1
  __m128i v9; // [rsp+0h] [rbp-30h] BYREF
  __m128i v10; // [rsp+10h] [rbp-20h] BYREF

  v3 = *(unsigned int *)(a2 + 24);
  v4 = *(_QWORD *)(a2 + 8);
  v9.m128i_i64[0] = a2;
  v5 = *(_DWORD *)(a2 + 16);
  v6 = v4 + 32 * v3;
  v9.m128i_i64[1] = *(_QWORD *)a2;
  if ( v5 )
  {
    v10.m128i_i64[0] = v4;
    v10.m128i_i64[1] = v6;
    sub_B8D830((__int64)&v9);
  }
  else
  {
    v10.m128i_i64[0] = v6;
    v10.m128i_i64[1] = v6;
  }
  v7 = _mm_loadu_si128(&v10);
  *a1 = _mm_loadu_si128(&v9);
  a1[1] = v7;
  return a1;
}
