// Function: sub_2E0E3F0
// Address: 0x2e0e3f0
//
unsigned __int64 __fastcall sub_2E0E3F0(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int64 a6)
{
  __int64 v7; // rsi
  unsigned __int64 result; // rax
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 v12; // r9
  unsigned __int64 v13; // rbx
  __m128i *v14; // rsi
  __int64 v15; // rax
  __m128i *v16; // rcx
  unsigned __int64 v17; // rdx
  __int64 v18; // rcx
  __m128i *v19; // r14
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rax
  const void *v22; // rsi
  __m128i v23; // [rsp+0h] [rbp-40h] BYREF
  unsigned __int64 v24; // [rsp+10h] [rbp-30h]

  if ( *(_QWORD *)(a1 + 96) )
  {
    v7 = *(_QWORD *)(a2 + 8);
    v23.m128i_i64[0] = a1;
    return sub_2E0DC20(v23.m128i_i64, v7, 0, a2, a5, a6);
  }
  else
  {
    v9 = *(_QWORD *)(a2 + 8);
    v10 = sub_2E09D00((__int64 *)a1, v9);
    v13 = *(_QWORD *)a1;
    v14 = (__m128i *)v10;
    v15 = *(unsigned int *)(a1 + 8);
    v16 = (__m128i *)(*(_QWORD *)a1 + 24 * v15);
    v17 = v9 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v14 == v16 )
    {
      v23.m128i_i64[0] = v9;
      v19 = &v23;
      v23.m128i_i64[1] = v17 | 6;
      v20 = v15 + 1;
      v21 = *(unsigned int *)(a1 + 12);
      v24 = a2;
      if ( v20 > v21 )
      {
        v22 = (const void *)(a1 + 16);
        if ( v13 > (unsigned __int64)&v23 || v16 <= &v23 )
        {
          sub_C8D5F0(a1, v22, v20, 0x18u, v11, v12);
          v16 = (__m128i *)(*(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8));
        }
        else
        {
          sub_C8D5F0(a1, v22, v20, 0x18u, v11, v12);
          v19 = (__m128i *)((char *)&v23 + *(_QWORD *)a1 - v13);
          v16 = (__m128i *)(*(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8));
        }
      }
      *v16 = _mm_loadu_si128(v19);
      v16[1].m128i_i64[0] = v19[1].m128i_i64[0];
      ++*(_DWORD *)(a1 + 8);
      return a2;
    }
    else
    {
      v18 = v14->m128i_i64[0];
      if ( v17 == (v14->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) )
      {
        result = v14[1].m128i_u64[0];
        if ( (*(_DWORD *)(v17 + 24) | (unsigned int)(v18 >> 1) & 3) >= ((unsigned int)(v9 >> 1) & 3
                                                                      | *(_DWORD *)(v17 + 24))
          && v18 != v9 )
        {
          *(_QWORD *)(result + 8) = v9;
          result = v14[1].m128i_u64[0];
          v14->m128i_i64[0] = *(_QWORD *)(result + 8);
        }
      }
      else
      {
        v23.m128i_i64[0] = v9;
        v23.m128i_i64[1] = v17 | 6;
        v24 = a2;
        sub_2E0C1A0(a1, v14, &v23);
        return a2;
      }
    }
  }
  return result;
}
