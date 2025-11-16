// Function: sub_293A710
// Address: 0x293a710
//
unsigned __int64 __fastcall sub_293A710(
        __m128i *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        const __m128i *a6,
        __int64 a7)
{
  __m128i *v8; // rdi
  __m128i v9; // xmm0
  __m128i v10; // xmm1
  unsigned __int64 result; // rax
  unsigned int v12; // edx
  unsigned __int64 v13; // rbx
  unsigned __int64 v14; // r12

  v8 = a1 + 6;
  v9 = _mm_loadu_si128(a6);
  v8[-6].m128i_i64[0] = a2;
  v10 = _mm_loadu_si128(a6 + 1);
  v8[-6].m128i_i64[1] = a3;
  v8[-5].m128i_i64[0] = a4;
  v8[-5].m128i_i64[1] = a5;
  v8[-4] = v9;
  v8[-3] = v10;
  v8[-2].m128i_i64[1] = a7;
  a1[5].m128i_i64[0] = (__int64)v8;
  a1[5].m128i_i64[1] = 0x800000000LL;
  result = *(_QWORD *)(a5 + 8);
  a1[4].m128i_i8[0] = *(_BYTE *)(result + 8) == 14;
  if ( a7 )
  {
    v12 = a6->m128i_u32[3];
    result = *(unsigned int *)(a7 + 8);
    if ( v12 > (unsigned int)result && v12 != result )
    {
      if ( v12 >= result )
      {
        v13 = v12 - result;
        if ( v12 > (unsigned __int64)*(unsigned int *)(a7 + 12) )
        {
          sub_C8D5F0(a7, (const void *)(a7 + 16), v12, 8u, v12, (__int64)a6);
          result = *(unsigned int *)(a7 + 8);
        }
        if ( 8 * v13 )
        {
          memset((void *)(*(_QWORD *)a7 + 8 * result), 0, 8 * v13);
          result = *(unsigned int *)(a7 + 8);
        }
        result += v13;
        *(_DWORD *)(a7 + 8) = result;
      }
      else
      {
        *(_DWORD *)(a7 + 8) = v12;
      }
    }
  }
  else
  {
    v14 = a6->m128i_u32[3];
    if ( a6->m128i_i32[3] )
    {
      if ( v14 > 8 )
      {
        sub_C8D5F0((__int64)a1[5].m128i_i64, v8, a6->m128i_u32[3], 8u, (__int64)a1[5].m128i_i64, (__int64)a6);
        v8 = (__m128i *)(a1[5].m128i_i64[0] + 8LL * a1[5].m128i_u32[2]);
      }
      result = (unsigned __int64)memset(v8, 0, 8 * v14);
      a1[5].m128i_i32[2] += v14;
    }
  }
  return result;
}
