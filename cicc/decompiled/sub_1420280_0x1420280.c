// Function: sub_1420280
// Address: 0x1420280
//
void __fastcall sub_1420280(__int64 a1)
{
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // rbx
  __int64 v4; // r13
  unsigned __int64 v5; // rdi
  __m128i *v6; // rax
  unsigned __int64 v7; // rcx
  const __m128i *v8; // rdx
  __m128i *v9; // rsi
  __int8 v10; // cl

  v2 = (((((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
         | (*(unsigned int *)(a1 + 12) + 2LL)
         | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4)
       | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 8)
     | (((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4)
     | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
     | (*(unsigned int *)(a1 + 12) + 2LL)
     | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v3 = (v2 | (v2 >> 16) | HIDWORD(v2)) + 1;
  if ( v3 > 0xFFFFFFFF )
    v3 = 0xFFFFFFFFLL;
  v4 = malloc(v3 << 6);
  if ( !v4 )
    sub_16BD1C0("Allocation failed");
  v5 = *(_QWORD *)a1;
  v6 = (__m128i *)v4;
  v7 = (unsigned __int64)*(unsigned int *)(a1 + 8) << 6;
  v8 = *(const __m128i **)a1;
  v9 = (__m128i *)(v4 + v7);
  if ( v7 )
  {
    do
    {
      if ( v6 )
      {
        *v6 = _mm_loadu_si128(v8);
        v6[1] = _mm_loadu_si128(v8 + 1);
        v6[2].m128i_i64[0] = v8[2].m128i_i64[0];
        v6[2].m128i_i64[1] = v8[2].m128i_i64[1];
        v6[3].m128i_i64[0] = v8[3].m128i_i64[0];
        v10 = v8[3].m128i_i8[12];
        v6[3].m128i_i8[12] = v10;
        if ( v10 )
          v6[3].m128i_i32[2] = v8[3].m128i_i32[2];
      }
      v6 += 4;
      v8 += 4;
    }
    while ( v6 != v9 );
  }
  if ( v5 != a1 + 16 )
    _libc_free(v5);
  *(_QWORD *)a1 = v4;
  *(_DWORD *)(a1 + 12) = v3;
}
