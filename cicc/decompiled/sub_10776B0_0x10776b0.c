// Function: sub_10776B0
// Address: 0x10776b0
//
void __fastcall sub_10776B0(char *src, char *a2)
{
  char *i; // rbx
  __int32 v4; // ecx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // r12
  __int64 v8; // r13
  __m128i *v9; // rdi
  unsigned __int64 v10; // rdx
  const __m128i *v11; // rax
  __m128i v12; // xmm0
  __m128i v13; // xmm1
  __int64 v14; // rdx
  int v15; // [rsp-4Ch] [rbp-4Ch]
  __int64 v16; // [rsp-48h] [rbp-48h]
  __int64 v17; // [rsp-40h] [rbp-40h]

  if ( src != a2 )
  {
    for ( i = src + 40; a2 != i; *((_QWORD *)src + 4) = v7 )
    {
      while ( 1 )
      {
        v7 = *((_QWORD *)i + 4);
        v8 = *(_QWORD *)i;
        v9 = (__m128i *)i;
        v6 = *((_QWORD *)i + 1);
        v5 = *((_QWORD *)i + 2);
        v10 = *(_QWORD *)i + *(_QWORD *)(v7 + 160);
        v4 = *((_DWORD *)i + 6);
        if ( v10 < *(_QWORD *)(*((_QWORD *)src + 4) + 160LL) + *(_QWORD *)src )
          break;
        v11 = (const __m128i *)(i - 40);
        if ( v10 < *(_QWORD *)(*((_QWORD *)i - 1) + 160LL) + *((_QWORD *)i - 5) )
        {
          do
          {
            v12 = _mm_loadu_si128(v11);
            v13 = _mm_loadu_si128(v11 + 1);
            v9 = (__m128i *)v11;
            v11 = (const __m128i *)((char *)v11 - 40);
            v14 = v11[4].m128i_i64[1];
            v11[5] = v12;
            v11[6] = v13;
            v11[7].m128i_i64[0] = v14;
          }
          while ( (unsigned __int64)(v8 + *(_QWORD *)(v7 + 160)) < *(_QWORD *)(v11[2].m128i_i64[0] + 160)
                                                                 + v11->m128i_i64[0] );
        }
        i += 40;
        v9->m128i_i64[0] = v8;
        v9->m128i_i64[1] = v6;
        v9[1].m128i_i64[0] = v5;
        v9[1].m128i_i32[2] = v4;
        v9[2].m128i_i64[0] = v7;
        if ( a2 == i )
          return;
      }
      if ( src != i )
      {
        v15 = *((_DWORD *)i + 6);
        v16 = *((_QWORD *)i + 2);
        v17 = *((_QWORD *)i + 1);
        memmove(src + 40, src, i - src);
        v4 = v15;
        v5 = v16;
        v6 = v17;
      }
      i += 40;
      *(_QWORD *)src = v8;
      *((_QWORD *)src + 1) = v6;
      *((_QWORD *)src + 2) = v5;
      *((_DWORD *)src + 6) = v4;
    }
  }
}
