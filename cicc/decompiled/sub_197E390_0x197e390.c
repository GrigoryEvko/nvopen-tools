// Function: sub_197E390
// Address: 0x197e390
//
unsigned __int64 __fastcall sub_197E390(__m128i *a1, const __m128i *a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __m128i v7; // xmm0
  __int32 v8; // eax
  unsigned __int64 result; // rax
  int v10; // r8d
  int v11; // r9d
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r13
  unsigned __int64 v15; // r14
  _QWORD *v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rsi
  __int64 v19; // rdx

  v7 = _mm_loadu_si128(a2 + 1);
  a1->m128i_i64[1] = a2->m128i_i64[1];
  v8 = a2[2].m128i_i32[0];
  a1[1] = v7;
  a1[2].m128i_i32[0] = v8;
  a1->m128i_i64[0] = (__int64)&unk_49EC708;
  a1[2].m128i_i64[1] = (__int64)&a1[3].m128i_i64[1];
  a1[3].m128i_i64[0] = 0x1000000000LL;
  if ( a2[3].m128i_i32[0] )
    sub_197CFC0((__int64)&a1[2].m128i_i64[1], (__int64)&a2[2].m128i_i64[1], a3, a4, a5, a6);
  a1[11].m128i_i64[1] = 0;
  a1[12].m128i_i64[0] = 0;
  a1[12].m128i_i64[1] = 0;
  a1[13].m128i_i32[0] = 0;
  j___libc_free_0(0);
  result = a2[13].m128i_u32[0];
  a1[13].m128i_i32[0] = result;
  if ( (_DWORD)result )
  {
    result = sub_22077B0(56 * result);
    v12 = a2[12].m128i_i64[1];
    v13 = a1[13].m128i_u32[0];
    a1[12].m128i_i64[0] = result;
    a1[12].m128i_i64[1] = v12;
    if ( (_DWORD)v13 )
    {
      v14 = 0;
      v15 = 0;
      while ( 1 )
      {
        v16 = (_QWORD *)(v14 + result);
        if ( v16 )
        {
          *v16 = *(_QWORD *)(a2[12].m128i_i64[0] + v14);
          v16 = (_QWORD *)(v14 + a1[12].m128i_i64[0]);
        }
        if ( *v16 != -16 && *v16 != -8 )
        {
          v17 = a2[12].m128i_i64[0];
          v16[2] = 0x400000000LL;
          v16[1] = v16 + 3;
          v18 = v14 + v17;
          v19 = *(unsigned int *)(v18 + 16);
          if ( (_DWORD)v19 )
            sub_197CFC0((__int64)(v16 + 1), v18 + 8, v19, v13, v10, v11);
        }
        result = a1[13].m128i_u32[0];
        ++v15;
        v14 += 56;
        if ( result <= v15 )
          break;
        result = a1[12].m128i_u64[0];
      }
    }
  }
  else
  {
    a1[12].m128i_i64[0] = 0;
    a1[12].m128i_i64[1] = 0;
  }
  return result;
}
