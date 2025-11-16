// Function: sub_3909530
// Address: 0x3909530
//
__int64 __fastcall sub_3909530(unsigned int *a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // r12
  int v8; // r8d
  unsigned __int64 v9; // r14
  __int64 v10; // r15
  const __m128i *i; // rbx
  __int64 v12; // rax
  void *v13; // rdi
  unsigned int v14; // r9d
  const __m128i *v15; // rax
  __int64 v16; // rbx
  unsigned __int64 v17; // rdi
  size_t v19; // rdx
  const __m128i *v20; // rsi
  unsigned int v21; // [rsp+4h] [rbp-3Ch]
  unsigned int v22; // [rsp+4h] [rbp-3Ch]
  __int64 v23; // [rsp+8h] [rbp-38h]

  v3 = a2;
  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation", 1u);
  v4 = ((((a1[3] + 2LL) | (((unsigned __int64)a1[3] + 2) >> 1)) >> 2)
      | (a1[3] + 2LL)
      | (((unsigned __int64)a1[3] + 2) >> 1)) >> 4;
  v5 = ((v4
       | (((a1[3] + 2LL) | (((unsigned __int64)a1[3] + 2) >> 1)) >> 2)
       | (a1[3] + 2LL)
       | (((unsigned __int64)a1[3] + 2) >> 1)) >> 8)
     | v4
     | (((a1[3] + 2LL) | (((unsigned __int64)a1[3] + 2) >> 1)) >> 2)
     | (a1[3] + 2LL)
     | (((unsigned __int64)a1[3] + 2) >> 1);
  v6 = (v5 | (v5 >> 16) | HIDWORD(v5)) + 1;
  if ( v6 >= a2 )
    v3 = v6;
  v7 = v3;
  if ( v3 > 0xFFFFFFFF )
    v7 = 0xFFFFFFFFLL;
  v23 = malloc(104 * v7);
  if ( !v23 )
    sub_16BD1C0("Allocation failed", 1u);
  v9 = *(_QWORD *)a1 + 104LL * a1[2];
  if ( *(_QWORD *)a1 != v9 )
  {
    v10 = v23;
    for ( i = (const __m128i *)(*(_QWORD *)a1 + 24LL); ; i = (const __m128i *)((char *)i + 104) )
    {
      if ( v10 )
      {
        v12 = i[-2].m128i_i64[1];
        v13 = (void *)(v10 + 24);
        *(_DWORD *)(v10 + 16) = 0;
        *(_QWORD *)(v10 + 8) = v10 + 24;
        *(_QWORD *)v10 = v12;
        *(_DWORD *)(v10 + 20) = 64;
        v14 = i[-1].m128i_u32[2];
        if ( v14 && (const __m128i *)(v10 + 8) != &i[-1] )
        {
          v15 = (const __m128i *)i[-1].m128i_i64[0];
          if ( v15 == i )
          {
            v19 = v14;
            v20 = i;
            if ( v14 <= 0x40
              || (v22 = i[-1].m128i_u32[2],
                  sub_16CD150(v10 + 8, (const void *)(v10 + 24), v14, 1, v8, v14),
                  v19 = i[-1].m128i_u32[2],
                  v13 = *(void **)(v10 + 8),
                  v20 = (const __m128i *)i[-1].m128i_i64[0],
                  v14 = v22,
                  i[-1].m128i_i32[2]) )
            {
              v21 = v14;
              memcpy(v13, v20, v19);
              v14 = v21;
            }
            *(_DWORD *)(v10 + 16) = v14;
            i[-1].m128i_i32[2] = 0;
          }
          else
          {
            *(_QWORD *)(v10 + 8) = v15;
            *(_DWORD *)(v10 + 16) = i[-1].m128i_i32[2];
            *(_DWORD *)(v10 + 20) = i[-1].m128i_i32[3];
            i[-1].m128i_i64[0] = (__int64)i;
            i[-1].m128i_i32[3] = 0;
            i[-1].m128i_i32[2] = 0;
          }
        }
        *(__m128i *)(v10 + 88) = _mm_loadu_si128(i + 4);
      }
      v10 += 104;
      if ( (const __m128i *)v9 == &i[5] )
        break;
    }
    v16 = *(_QWORD *)a1;
    v9 = *(_QWORD *)a1 + 104LL * a1[2];
    if ( *(_QWORD *)a1 != v9 )
    {
      do
      {
        v9 -= 104LL;
        v17 = *(_QWORD *)(v9 + 8);
        if ( v17 != v9 + 24 )
          _libc_free(v17);
      }
      while ( v9 != v16 );
      v9 = *(_QWORD *)a1;
    }
  }
  if ( (unsigned int *)v9 != a1 + 4 )
    _libc_free(v9);
  a1[3] = v7;
  *(_QWORD *)a1 = v23;
  return v23;
}
