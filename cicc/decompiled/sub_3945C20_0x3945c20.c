// Function: sub_3945C20
// Address: 0x3945c20
//
void __fastcall sub_3945C20(__int64 a1, __int64 a2, __int32 a3, __int64 a4, int a5, int a6)
{
  __m128i *v7; // rax
  __m128i *v8; // rdx
  __m128i *v9; // r12
  __int64 v10; // rsi
  unsigned __int64 v11; // rax
  __m128i *v12; // rsi
  const __m128i *v13; // rdx
  __m128i *v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rax
  __m128i v18; // [rsp+0h] [rbp-20h] BYREF
  __m128i v19; // [rsp+10h] [rbp-10h] BYREF

  v7 = *(__m128i **)a1;
  v7->m128i_i32[3] = a4;
  v7->m128i_i64[0] = a2;
  v7->m128i_i32[2] = a3;
  v8 = *(__m128i **)a1;
  v18.m128i_i32[3] = HIDWORD(a4);
  v9 = v8 + 1;
  v10 = *(_QWORD *)(v8->m128i_i64[0] + 8LL * v8->m128i_u32[3]);
  v18.m128i_i32[2] = (v10 & 0x3F) + 1;
  v11 = *(unsigned int *)(a1 + 8);
  v18.m128i_i64[0] = v10 & 0xFFFFFFFFFFFFFFC0LL;
  v12 = &v8[v11];
  if ( &v8[1] == v12 )
  {
    if ( (unsigned int)v11 >= *(_DWORD *)(a1 + 12) )
    {
      sub_16CD150(a1, (const void *)(a1 + 16), 0, 16, a5, a6);
      v12 = (__m128i *)(*(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8));
    }
    *v12 = _mm_load_si128(&v18);
    ++*(_DWORD *)(a1 + 8);
  }
  else
  {
    v13 = &v8[v11 - 1];
    if ( v11 < *(unsigned int *)(a1 + 12)
      || (sub_16CD150(a1, (const void *)(a1 + 16), 0, 16, a5, a6),
          v14 = *(__m128i **)a1,
          v15 = *(unsigned int *)(a1 + 8),
          v16 = 16 * v15,
          v9 = (__m128i *)(*(_QWORD *)a1 + 16LL),
          v13 = (const __m128i *)(*(_QWORD *)a1 + 16 * v15 - 16),
          (v12 = (__m128i *)(16 * v15 + *(_QWORD *)a1)) != 0) )
    {
      *v12 = _mm_loadu_si128(v13);
      v14 = *(__m128i **)a1;
      v15 = *(unsigned int *)(a1 + 8);
      v16 = 16 * v15;
      v13 = (const __m128i *)(*(_QWORD *)a1 + 16 * v15 - 16);
    }
    if ( v9 != v13 )
    {
      memmove(&v14->m128i_i8[v16 - ((char *)v13 - (char *)v9)], v9, (char *)v13 - (char *)v9);
      LODWORD(v15) = *(_DWORD *)(a1 + 8);
    }
    v17 = (unsigned int)(v15 + 1);
    *(_DWORD *)(a1 + 8) = v17;
    if ( v9 <= &v18 && (unsigned __int64)&v18 < *(_QWORD *)a1 + 16 * v17 )
      *v9 = _mm_load_si128(&v19);
    else
      *v9 = _mm_load_si128(&v18);
  }
}
