// Function: sub_25AFF40
// Address: 0x25aff40
//
__int64 __fastcall sub_25AFF40(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 result; // rax
  __int64 v8; // rax
  __m128i v9; // xmm0
  __m128i v10; // [rsp+0h] [rbp-30h] BYREF
  __m128i v11; // [rsp+10h] [rbp-20h] BYREF

  v11.m128i_i64[0] = a2;
  v11.m128i_i64[1] = a3;
  v5 = (unsigned int)sub_25AFE60(a1, v11.m128i_i64);
  result = 0;
  if ( !(_BYTE)v5 )
  {
    v8 = *(unsigned int *)(a4 + 8);
    v9 = _mm_loadu_si128(&v11);
    if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
    {
      v10 = v9;
      sub_C8D5F0(a4, (const void *)(a4 + 16), v8 + 1, 0x10u, v5, v6);
      v8 = *(unsigned int *)(a4 + 8);
      v9 = _mm_load_si128(&v10);
    }
    *(__m128i *)(*(_QWORD *)a4 + 16 * v8) = v9;
    ++*(_DWORD *)(a4 + 8);
    return 1;
  }
  return result;
}
