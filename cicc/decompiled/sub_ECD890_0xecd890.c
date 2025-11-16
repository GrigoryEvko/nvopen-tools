// Function: sub_ECD890
// Address: 0xecd890
//
__int64 __fastcall sub_ECD890(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r15
  const __m128i *v8; // rsi
  __int64 v10; // r14
  __int64 result; // rax
  __int64 v12; // r12
  const __m128i *v13; // rbx
  __m128i *v14; // r8
  __int64 v15; // rax
  __m128i *v16; // rdi
  unsigned __int64 v17; // r9
  const __m128i *v18; // rax
  size_t v19; // rdx
  const __m128i *v20; // rbx
  __int64 v21; // rdi
  int v22; // ebx
  unsigned __int64 v23; // [rsp+0h] [rbp-50h]
  __m128i *v24; // [rsp+0h] [rbp-50h]
  __m128i *v25; // [rsp+8h] [rbp-48h]
  unsigned __int64 v26; // [rsp+8h] [rbp-48h]
  unsigned __int64 v27[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = a1 + 16;
  v8 = (const __m128i *)(a1 + 16);
  v10 = sub_C8D7D0(a1, a1 + 16, a2, 0x70u, v27, a6);
  result = *(unsigned int *)(a1 + 8);
  v12 = *(_QWORD *)a1 + 112 * result;
  if ( *(_QWORD *)a1 != v12 )
  {
    v13 = (const __m128i *)(*(_QWORD *)a1 + 32LL);
    v14 = (__m128i *)v10;
    while ( 1 )
    {
      if ( v14 )
      {
        v15 = v13[-2].m128i_i64[0];
        v16 = v14 + 2;
        v14[1].m128i_i64[0] = 0;
        v14->m128i_i64[1] = (__int64)v14[2].m128i_i64;
        v14->m128i_i64[0] = v15;
        v14[1].m128i_i64[1] = 64;
        v17 = v13[-1].m128i_u64[0];
        if ( v17 && &v14->m128i_u64[1] != &v13[-2].m128i_u64[1] )
        {
          v18 = (const __m128i *)v13[-2].m128i_i64[1];
          if ( v18 == v13 )
          {
            v19 = v13[-1].m128i_u64[0];
            v8 = v13;
            if ( v17 <= 0x40 )
              goto LABEL_11;
            v24 = v14;
            v26 = v13[-1].m128i_u64[0];
            sub_C8D290((__int64)&v14->m128i_i64[1], &v14[2], v19, 1u, (__int64)v14, v17);
            v19 = v13[-1].m128i_u64[0];
            v14 = v24;
            v8 = (const __m128i *)v13[-2].m128i_i64[1];
            v17 = v26;
            v16 = (__m128i *)v24->m128i_i64[1];
            if ( v19 )
            {
LABEL_11:
              v23 = v17;
              v25 = v14;
              memcpy(v16, v8, v19);
              v17 = v23;
              v14 = v25;
            }
            v14[1].m128i_i64[0] = v17;
            v13[-1].m128i_i64[0] = 0;
          }
          else
          {
            v14->m128i_i64[1] = (__int64)v18;
            v14[1].m128i_i64[0] = v13[-1].m128i_i64[0];
            v14[1].m128i_i64[1] = v13[-1].m128i_i64[1];
            v13[-2].m128i_i64[1] = (__int64)v13;
            v13[-1].m128i_i64[1] = 0;
            v13[-1].m128i_i64[0] = 0;
          }
        }
        v14[6] = _mm_loadu_si128(v13 + 4);
      }
      v14 += 7;
      if ( (const __m128i *)v12 == &v13[5] )
        break;
      v13 += 7;
    }
    result = *(unsigned int *)(a1 + 8);
    v20 = *(const __m128i **)a1;
    v12 = *(_QWORD *)a1 + 112 * result;
    if ( v12 != *(_QWORD *)a1 )
    {
      do
      {
        v12 -= 112;
        v21 = *(_QWORD *)(v12 + 8);
        result = v12 + 32;
        if ( v21 != v12 + 32 )
          result = _libc_free(v21, v8);
      }
      while ( (const __m128i *)v12 != v20 );
      v12 = *(_QWORD *)a1;
    }
  }
  v22 = v27[0];
  if ( v7 != v12 )
    result = _libc_free(v12, v8);
  *(_QWORD *)a1 = v10;
  *(_DWORD *)(a1 + 12) = v22;
  return result;
}
