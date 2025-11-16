// Function: sub_18CA7B0
// Address: 0x18ca7b0
//
__int64 __fastcall sub_18CA7B0(__int64 a1, const char *a2, __int64 *a3)
{
  size_t v5; // rax
  size_t v6; // r12
  __m128i *v7; // rdx
  __int64 v8; // r14
  __int64 result; // rax
  __m128i *v10; // r12
  __int64 v11; // rax
  __m128i *v12; // rdi
  __int64 v13; // [rsp+18h] [rbp-58h] BYREF
  __m128i *v14; // [rsp+20h] [rbp-50h] BYREF
  size_t v15; // [rsp+28h] [rbp-48h]
  __m128i v16[4]; // [rsp+30h] [rbp-40h] BYREF

  if ( *(_DWORD *)(a1 + 8) >= *(_DWORD *)(a1 + 12) )
    sub_1740340(a1, 0);
  v14 = v16;
  v5 = strlen(a2);
  v13 = v5;
  v6 = v5;
  if ( v5 > 0xF )
  {
    v14 = (__m128i *)sub_22409D0(&v14, &v13, 0);
    v12 = v14;
    v16[0].m128i_i64[0] = v13;
  }
  else
  {
    if ( v5 == 1 )
    {
      v16[0].m128i_i8[0] = *a2;
      v7 = v16;
      goto LABEL_6;
    }
    if ( !v5 )
    {
      v7 = v16;
      goto LABEL_6;
    }
    v12 = v16;
  }
  memcpy(v12, a2, v6);
  v5 = v13;
  v7 = v14;
LABEL_6:
  v15 = v5;
  v7->m128i_i8[v5] = 0;
  v8 = *a3;
  result = 7LL * *(unsigned int *)(a1 + 8);
  v10 = (__m128i *)(*(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8));
  if ( v10 )
  {
    v10->m128i_i64[0] = (__int64)v10[1].m128i_i64;
    if ( v14 == v16 )
    {
      v10[1] = _mm_load_si128(v16);
    }
    else
    {
      v10->m128i_i64[0] = (__int64)v14;
      v10[1].m128i_i64[0] = v16[0].m128i_i64[0];
    }
    v11 = v15;
    v10[2].m128i_i64[0] = 0;
    v10[2].m128i_i64[1] = 0;
    v10->m128i_i64[1] = v11;
    v10[3].m128i_i64[0] = 0;
    v14 = v16;
    v15 = 0;
    v16[0].m128i_i8[0] = 0;
    result = sub_22077B0(8);
    v10[2].m128i_i64[0] = result;
    v10[3].m128i_i64[0] = result + 8;
    *(_QWORD *)result = v8;
    v10[2].m128i_i64[1] = result + 8;
  }
  if ( v14 != v16 )
    result = j_j___libc_free_0(v14, v16[0].m128i_i64[0] + 1);
  ++*(_DWORD *)(a1 + 8);
  return result;
}
