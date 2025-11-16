// Function: sub_1B1DDA0
// Address: 0x1b1dda0
//
__int64 __fastcall sub_1B1DDA0(__int64 a1, const __m128i *a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v8; // rdi
  unsigned __int64 *v9; // rsi
  unsigned __int64 *v10; // r14
  unsigned __int64 v11; // rdi
  __int64 v12; // rax
  _QWORD *v13; // rbx
  _QWORD *v14; // r14
  unsigned __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 result; // rax
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rax
  __int32 v21; // ebx
  __int64 v22; // r15
  unsigned __int64 *v23; // rsi
  __int64 v24; // rdx

  v8 = a1 + 216;
  *(_QWORD *)(v8 - 32) = a2->m128i_i64[1];
  *(__m128i *)(v8 - 24) = _mm_loadu_si128(a2 + 1);
  *(_DWORD *)(v8 - 8) = a2[2].m128i_i32[0];
  if ( (unsigned __int64 *)v8 != &a2[2].m128i_u64[1] )
  {
    v9 = (unsigned __int64 *)a2[2].m128i_i64[1];
    v10 = &a2[3].m128i_u64[1];
    if ( v9 == &a2[3].m128i_u64[1] )
    {
      v19 = a2[3].m128i_u32[0];
      v20 = *(unsigned int *)(a1 + 224);
      v21 = a2[3].m128i_i32[0];
      if ( v19 <= v20 )
      {
        if ( a2[3].m128i_i32[0] )
          memmove(*(void **)(a1 + 216), v9, 8 * v19);
      }
      else
      {
        if ( v19 > *(unsigned int *)(a1 + 228) )
        {
          *(_DWORD *)(a1 + 224) = 0;
          sub_16CD150(v8, (const void *)(a1 + 232), v19, 8, a5, a6);
          v10 = (unsigned __int64 *)a2[2].m128i_i64[1];
          v19 = a2[3].m128i_u32[0];
          v20 = 0;
          v23 = v10;
        }
        else
        {
          v22 = v20;
          v23 = &a2[3].m128i_u64[1];
          if ( *(_DWORD *)(a1 + 224) )
          {
            memmove(*(void **)(a1 + 216), v23, 8 * v20);
            v10 = (unsigned __int64 *)a2[2].m128i_i64[1];
            v19 = a2[3].m128i_u32[0];
            v20 = v22 * 8;
            v23 = &v10[v22];
          }
        }
        v24 = v19;
        if ( v23 != &v10[v24] )
          memcpy((void *)(v20 + *(_QWORD *)(a1 + 216)), v23, v24 * 8 - v20);
      }
      *(_DWORD *)(a1 + 224) = v21;
      a2[3].m128i_i32[0] = 0;
    }
    else
    {
      v11 = *(_QWORD *)(a1 + 216);
      if ( v11 != a1 + 232 )
      {
        _libc_free(v11);
        v9 = (unsigned __int64 *)a2[2].m128i_i64[1];
      }
      *(_QWORD *)(a1 + 216) = v9;
      *(_DWORD *)(a1 + 224) = a2[3].m128i_i32[0];
      *(_DWORD *)(a1 + 228) = a2[3].m128i_i32[1];
      a2[2].m128i_i64[1] = (__int64)v10;
      a2[3].m128i_i64[0] = 0;
    }
  }
  v12 = *(unsigned int *)(a1 + 384);
  if ( (_DWORD)v12 )
  {
    v13 = *(_QWORD **)(a1 + 368);
    v14 = &v13[7 * v12];
    do
    {
      if ( *v13 != -16 && *v13 != -8 )
      {
        v15 = v13[1];
        if ( (_QWORD *)v15 != v13 + 3 )
          _libc_free(v15);
      }
      v13 += 7;
    }
    while ( v14 != v13 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 368));
  *(_QWORD *)(a1 + 376) = 0;
  *(_QWORD *)(a1 + 368) = 0;
  *(_DWORD *)(a1 + 384) = 0;
  ++*(_QWORD *)(a1 + 360);
  v16 = a2[12].m128i_i64[0];
  ++a2[11].m128i_i64[1];
  v17 = *(_QWORD *)(a1 + 368);
  *(_QWORD *)(a1 + 368) = v16;
  LODWORD(v16) = a2[12].m128i_i32[2];
  a2[12].m128i_i64[0] = v17;
  LODWORD(v17) = *(_DWORD *)(a1 + 376);
  *(_DWORD *)(a1 + 376) = v16;
  LODWORD(v16) = a2[12].m128i_i32[3];
  a2[12].m128i_i32[2] = v17;
  LODWORD(v17) = *(_DWORD *)(a1 + 380);
  *(_DWORD *)(a1 + 380) = v16;
  LODWORD(v16) = a2[13].m128i_i32[0];
  a2[12].m128i_i32[3] = v17;
  result = *(unsigned int *)(a1 + 384);
  *(_DWORD *)(a1 + 384) = v16;
  a2[13].m128i_i32[0] = result;
  return result;
}
