// Function: sub_2852510
// Address: 0x2852510
//
void __fastcall sub_2852510(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  const __m128i *v5; // r15
  __m128i *v7; // r9
  unsigned __int64 v8; // rax
  __m128i *v9; // r12
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rdx
  const __m128i *v12; // rbx
  __m128i *v13; // rax
  const __m128i *i; // r12
  const __m128i *v15; // r12
  __int64 v16; // rbx
  __m128i *v17; // rbx
  __m128i *v18; // rdx
  __m128i *v19; // rbx
  const __m128i *v20; // r12
  __int64 v21; // rbx
  __int64 v22; // rbx
  __m128i *v23; // r12
  __int64 v24; // rdx
  __m128i *v25; // rbx
  __m128i *v26; // rbx
  __int64 v27; // rax
  unsigned __int64 v28; // [rsp-50h] [rbp-50h]
  int v29; // [rsp-44h] [rbp-44h]
  __m128i *v30; // [rsp-40h] [rbp-40h]
  __m128i *v31; // [rsp-40h] [rbp-40h]
  __m128i *v32; // [rsp-40h] [rbp-40h]
  __m128i *v33; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v5 = (const __m128i *)(a2 + 16);
    v7 = *(__m128i **)a1;
    v8 = *(unsigned int *)(a1 + 8);
    v9 = *(__m128i **)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v10 = *(unsigned int *)(a2 + 8);
      v29 = v10;
      if ( v10 <= v8 )
      {
        v18 = *(__m128i **)a1;
        if ( v10 )
        {
          v26 = &v7[5 * v10];
          do
          {
            v7->m128i_i64[0] = v5->m128i_i64[0];
            v7->m128i_i64[1] = v5->m128i_i64[1];
            if ( v7 != v5 )
            {
              v33 = v7;
              sub_C8CF80((__int64)v7[1].m128i_i64, &v7[3], 2, (__int64)v5[3].m128i_i64, (__int64)v5[1].m128i_i64);
              v7 = v33;
            }
            v27 = v5[4].m128i_i64[0];
            v7 += 5;
            v5 += 5;
            v7[-1].m128i_i64[0] = v27;
            v7[-1].m128i_i8[8] = v5[-1].m128i_i8[8];
          }
          while ( v7 != v26 );
          v18 = *(__m128i **)a1;
          v8 = *(unsigned int *)(a1 + 8);
        }
        v19 = &v18[5 * v8];
        if ( v19 != v7 )
        {
          do
          {
            while ( 1 )
            {
              v19 -= 5;
              if ( !v19[2].m128i_i8[12] )
                break;
              if ( v7 == v19 )
                goto LABEL_31;
            }
            v31 = v7;
            _libc_free(v19[1].m128i_u64[1]);
            v7 = v31;
          }
          while ( v31 != v19 );
        }
LABEL_31:
        *(_DWORD *)(a1 + 8) = v10;
        v20 = *(const __m128i **)a2;
        v21 = *(_QWORD *)a2 + 80LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v21 )
        {
          do
          {
            v21 -= 80;
            if ( !*(_BYTE *)(v21 + 44) )
              _libc_free(*(_QWORD *)(v21 + 24));
          }
          while ( v20 != (const __m128i *)v21 );
        }
      }
      else
      {
        v11 = *(unsigned int *)(a1 + 12);
        if ( v10 > v11 )
        {
          v25 = &v7[5 * v8];
          while ( v25 != v9 )
          {
            while ( 1 )
            {
              v25 -= 5;
              if ( v25[2].m128i_i8[12] )
                break;
              _libc_free(v25[1].m128i_u64[1]);
              if ( v25 == v9 )
                goto LABEL_46;
            }
          }
LABEL_46:
          *(_DWORD *)(a1 + 8) = 0;
          sub_2851200(a1, v10, v11, a4, a5, (__int64)v7);
          v5 = *(const __m128i **)a2;
          v10 = *(unsigned int *)(a2 + 8);
          v8 = 0;
          v9 = *(__m128i **)a1;
          v12 = *(const __m128i **)a2;
        }
        else
        {
          v12 = v5;
          if ( *(_DWORD *)(a1 + 8) )
          {
            v22 = 80 * v8;
            v8 = v22;
            v23 = &v7[(unsigned __int64)v22 / 0x10];
            do
            {
              v7->m128i_i64[0] = v5->m128i_i64[0];
              v7->m128i_i64[1] = v5->m128i_i64[1];
              if ( v7 != v5 )
              {
                v28 = v8;
                v32 = v7;
                sub_C8CF80((__int64)v7[1].m128i_i64, &v7[3], 2, (__int64)v5[3].m128i_i64, (__int64)v5[1].m128i_i64);
                v8 = v28;
                v7 = v32;
              }
              v24 = v5[4].m128i_i64[0];
              v7 += 5;
              v5 += 5;
              v7[-1].m128i_i64[0] = v24;
              v7[-1].m128i_i8[8] = v5[-1].m128i_i8[8];
            }
            while ( v23 != v7 );
            v5 = *(const __m128i **)a2;
            v10 = *(unsigned int *)(a2 + 8);
            v9 = *(__m128i **)a1;
            v12 = (const __m128i *)(*(_QWORD *)a2 + v22);
          }
        }
        v13 = (__m128i *)((char *)v9 + v8);
        for ( i = &v5[5 * v10]; i != v12; v13 += 5 )
        {
          if ( v13 )
          {
            v30 = v13;
            v13->m128i_i64[0] = v12->m128i_i64[0];
            v13->m128i_i64[1] = v12->m128i_i64[1];
            sub_C8CF70((__int64)v13[1].m128i_i64, &v13[3], 2, (__int64)v12[3].m128i_i64, (__int64)v12[1].m128i_i64);
            v13 = v30;
            v30[4] = _mm_loadu_si128(v12 + 4);
          }
          v12 += 5;
        }
        *(_DWORD *)(a1 + 8) = v29;
        v15 = *(const __m128i **)a2;
        v16 = *(_QWORD *)a2 + 80LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v16 )
        {
          do
          {
            while ( 1 )
            {
              v16 -= 80;
              if ( !*(_BYTE *)(v16 + 44) )
                break;
              if ( v15 == (const __m128i *)v16 )
                goto LABEL_15;
            }
            _libc_free(*(_QWORD *)(v16 + 24));
          }
          while ( v15 != (const __m128i *)v16 );
        }
      }
LABEL_15:
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v17 = &v7[5 * v8];
      if ( v17 != v7 )
      {
        do
        {
          while ( 1 )
          {
            v17 -= 5;
            if ( !v17[2].m128i_i8[12] )
              break;
            if ( v17 == v9 )
              goto LABEL_21;
          }
          _libc_free(v17[1].m128i_u64[1]);
        }
        while ( v17 != v9 );
LABEL_21:
        v7 = *(__m128i **)a1;
      }
      if ( v7 != (__m128i *)(a1 + 16) )
        _libc_free((unsigned __int64)v7);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v5;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
