// Function: sub_1DF1C10
// Address: 0x1df1c10
//
__int64 __fastcall sub_1DF1C10(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, char a7)
{
  __int64 *v9; // rbx
  __int64 v10; // r12
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rbx
  _QWORD *i; // r13
  __int64 v15; // r14
  __m128i *v16; // rcx
  __m128i *v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rax
  const __m128i *v20; // rax
  __int64 *v21; // rbx
  __int64 result; // rax
  __int64 v23; // r13
  __int64 v24; // rdx
  __int64 v28; // [rsp+28h] [rbp-38h]

  v9 = *(__int64 **)a3;
  v28 = *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8);
  if ( *(_QWORD *)a3 != v28 )
  {
    do
    {
      v10 = *v9++;
      sub_1DD5BA0((__int64 *)(a1 + 16), v10);
      v11 = *a2;
      v12 = *(_QWORD *)v10;
      *(_QWORD *)(v10 + 8) = a2;
      v11 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v10 = v11 | v12 & 7;
      *(_QWORD *)(v11 + 8) = v10;
      *a2 = *a2 & 7 | v10;
    }
    while ( (__int64 *)v28 != v9 );
  }
  v13 = *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8);
  for ( i = *(_QWORD **)a4; (_QWORD *)v13 != i; ++i )
  {
    v15 = *i;
    sub_1E162E0(*i);
    v16 = *(__m128i **)a6;
    v17 = *(__m128i **)a6;
    v18 = 24LL * *(unsigned int *)(a6 + 8);
    if ( v18 )
    {
      do
      {
        if ( v17->m128i_i64[1] == v15 )
        {
          v20 = (__m128i *)((char *)v16 + v18 - 24);
          if ( v20 != v17 )
          {
            *v17 = _mm_loadu_si128(v20);
            v17[1].m128i_i32[0] = v20[1].m128i_i32[0];
            *(_BYTE *)(*(_QWORD *)(a6 + 208) + *(unsigned int *)(*(_QWORD *)a6 + 24LL * *(unsigned int *)(a6 + 8) - 24)) = -85 * (((__int64)v17->m128i_i64 - *(_QWORD *)a6) >> 3);
            v16 = *(__m128i **)a6;
          }
          v19 = (unsigned int)(*(_DWORD *)(a6 + 8) - 1);
          *(_DWORD *)(a6 + 8) = v19;
        }
        else
        {
          v19 = *(unsigned int *)(a6 + 8);
          v17 = (__m128i *)((char *)v17 + 24);
        }
        v18 = 24 * v19;
      }
      while ( v17 != (__m128i *)&v16->m128i_i8[v18] );
    }
  }
  if ( !a7 )
    return sub_1E807B0(a5, a1);
  v21 = *(__int64 **)a3;
  result = *(unsigned int *)(a3 + 8);
  v23 = *(_QWORD *)a3 + 8 * result;
  if ( v23 != *(_QWORD *)a3 )
  {
    do
    {
      v24 = *v21++;
      result = sub_1E82EC0(a5, a1, v24, a6);
    }
    while ( (__int64 *)v23 != v21 );
  }
  return result;
}
