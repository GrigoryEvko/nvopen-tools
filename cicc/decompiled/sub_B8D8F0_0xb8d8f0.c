// Function: sub_B8D8F0
// Address: 0xb8d8f0
//
__int64 __fastcall sub_B8D8F0(__int64 a1, const void *a2, size_t a3)
{
  __int64 v4; // r13
  __int64 v5; // rax
  int v6; // ecx
  __int64 v7; // rdx
  __int64 v8; // r13
  __int64 v9; // rbx
  __m128i v11; // [rsp+0h] [rbp-70h] BYREF
  __m128i v12; // [rsp+10h] [rbp-60h]
  __m128i v13; // [rsp+20h] [rbp-50h] BYREF
  __m128i v14; // [rsp+30h] [rbp-40h] BYREF

  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = *(_DWORD *)(a1 + 16);
  v7 = *(_QWORD *)a1;
  v13.m128i_i64[0] = a1;
  v8 = v5 + 32 * v4;
  v13.m128i_i64[1] = v7;
  if ( v6 )
  {
    v14.m128i_i64[1] = v8;
    v14.m128i_i64[0] = v5;
    sub_B8D830((__int64)&v13);
    v8 = *(_QWORD *)(a1 + 8) + 32LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v14.m128i_i64[0] = v8;
    v14.m128i_i64[1] = v8;
  }
  v9 = v14.m128i_i64[0];
  v11 = _mm_loadu_si128(&v13);
  v12 = _mm_loadu_si128(&v14);
  if ( v14.m128i_i64[0] == v8 )
    return 0;
  while ( a3 != *(_QWORD *)(v9 + 8) || a3 && memcmp(*(const void **)v9, a2, a3) )
  {
    v12.m128i_i64[0] = v9 + 32;
    sub_B8D830((__int64)&v11);
    v9 = v12.m128i_i64[0];
    if ( v12.m128i_i64[0] == v8 )
      return 0;
  }
  return 1;
}
