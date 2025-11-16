// Function: sub_23507F0
// Address: 0x23507f0
//
void __fastcall sub_23507F0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  unsigned __int64 v5; // r8
  unsigned __int64 v6; // rbx
  unsigned __int64 *v7; // r15
  unsigned __int64 v8; // r9
  const __m128i *v9; // rdi
  __m128i *v10; // rax
  const __m128i *v11; // rdx
  const __m128i *v12; // rcx
  __m128i *v13; // rsi
  const __m128i *v14; // rcx
  unsigned __int64 *v15; // r13
  unsigned __int64 *v16; // rbx
  unsigned __int64 *v17; // rbx
  unsigned __int64 v18; // rax
  unsigned __int64 *v19; // rbx
  unsigned __int64 *v20; // r13
  unsigned __int64 *v21; // rbx
  __int64 v22; // r15
  __int64 v23; // rsi
  unsigned __int64 *v24; // rbx
  unsigned __int64 v25; // rbx
  __int64 v26; // rsi
  int v27; // [rsp-44h] [rbp-44h]
  unsigned __int64 v28; // [rsp-40h] [rbp-40h]
  unsigned __int64 v29; // [rsp-40h] [rbp-40h]
  unsigned __int64 v30; // [rsp-40h] [rbp-40h]
  unsigned __int64 v31; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v2 = a2 + 16;
    v5 = *(_QWORD *)a1;
    v6 = *(unsigned int *)(a1 + 8);
    v7 = *(unsigned __int64 **)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v8 = *(unsigned int *)(a2 + 8);
      v27 = *(_DWORD *)(a2 + 8);
      if ( v8 <= v6 )
      {
        v18 = *(_QWORD *)a1;
        if ( *(_DWORD *)(a2 + 8) )
        {
          v25 = v5 + 32 * v8;
          do
          {
            v26 = v2;
            v31 = v5;
            v2 += 32;
            sub_2305040(v5, v26);
            v5 = v31 + 32;
          }
          while ( v31 + 32 != v25 );
          v18 = *(_QWORD *)a1;
          v6 = *(unsigned int *)(a1 + 8);
        }
        v19 = (unsigned __int64 *)(v18 + 32 * v6);
        while ( (unsigned __int64 *)v5 != v19 )
        {
          v19 -= 4;
          if ( (unsigned __int64 *)*v19 != v19 + 2 )
          {
            v28 = v5;
            j_j___libc_free_0(*v19);
            v5 = v28;
          }
        }
        *(_DWORD *)(a1 + 8) = v27;
        v20 = *(unsigned __int64 **)a2;
        v21 = (unsigned __int64 *)(*(_QWORD *)a2 + 32LL * *(unsigned int *)(a2 + 8));
        if ( *(unsigned __int64 **)a2 != v21 )
        {
          do
          {
            v21 -= 4;
            if ( (unsigned __int64 *)*v21 != v21 + 2 )
              j_j___libc_free_0(*v21);
          }
          while ( v20 != v21 );
        }
      }
      else
      {
        if ( v8 > *(unsigned int *)(a1 + 12) )
        {
          v24 = (unsigned __int64 *)(v5 + 32 * v6);
          while ( v24 != v7 )
          {
            while ( 1 )
            {
              v24 -= 4;
              if ( (unsigned __int64 *)*v24 == v24 + 2 )
                break;
              v30 = v8;
              j_j___libc_free_0(*v24);
              v8 = v30;
              if ( v24 == v7 )
                goto LABEL_45;
            }
          }
LABEL_45:
          *(_DWORD *)(a1 + 8) = 0;
          v6 = 0;
          sub_95D880(a1, v8);
          v2 = *(_QWORD *)a2;
          v8 = *(unsigned int *)(a2 + 8);
          v7 = *(unsigned __int64 **)a1;
          v9 = *(const __m128i **)a2;
        }
        else
        {
          v9 = (const __m128i *)(a2 + 16);
          if ( v6 )
          {
            v6 *= 32LL;
            v22 = v5 + v6;
            do
            {
              v23 = v2;
              v29 = v5;
              v2 += 32;
              sub_2305040(v5, v23);
              v5 = v29 + 32;
            }
            while ( v22 != v29 + 32 );
            v2 = *(_QWORD *)a2;
            v8 = *(unsigned int *)(a2 + 8);
            v7 = *(unsigned __int64 **)a1;
            v9 = (const __m128i *)(*(_QWORD *)a2 + v6);
          }
        }
        v10 = (__m128i *)((char *)v7 + v6);
        v11 = v9 + 1;
        v12 = (const __m128i *)(v2 + 32 * v8);
        v13 = (__m128i *)((char *)v7 + v6 + (char *)v12 - (char *)v9);
        if ( v12 != v9 )
        {
          do
          {
            if ( v10 )
            {
              v10->m128i_i64[0] = (__int64)v10[1].m128i_i64;
              v14 = (const __m128i *)v11[-1].m128i_i64[0];
              if ( v11 == v14 )
              {
                v10[1] = _mm_loadu_si128(v11);
              }
              else
              {
                v10->m128i_i64[0] = (__int64)v14;
                v10[1].m128i_i64[0] = v11->m128i_i64[0];
              }
              v10->m128i_i64[1] = v11[-1].m128i_i64[1];
              v11[-1].m128i_i64[0] = (__int64)v11;
              v11[-1].m128i_i64[1] = 0;
              v11->m128i_i8[0] = 0;
            }
            v10 += 2;
            v11 += 2;
          }
          while ( v10 != v13 );
        }
        *(_DWORD *)(a1 + 8) = v27;
        v15 = *(unsigned __int64 **)a2;
        v16 = (unsigned __int64 *)(*(_QWORD *)a2 + 32LL * *(unsigned int *)(a2 + 8));
        if ( *(unsigned __int64 **)a2 != v16 )
        {
          do
          {
            v16 -= 4;
            if ( (unsigned __int64 *)*v16 != v16 + 2 )
              j_j___libc_free_0(*v16);
          }
          while ( v15 != v16 );
        }
      }
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v17 = (unsigned __int64 *)(v5 + 32 * v6);
      if ( v17 != (unsigned __int64 *)v5 )
      {
        do
        {
          v17 -= 4;
          if ( (unsigned __int64 *)*v17 != v17 + 2 )
            j_j___libc_free_0(*v17);
        }
        while ( v17 != v7 );
        v5 = *(_QWORD *)a1;
      }
      if ( v5 != a1 + 16 )
        _libc_free(v5);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v2;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
