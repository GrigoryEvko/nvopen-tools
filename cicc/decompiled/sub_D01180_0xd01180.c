// Function: sub_D01180
// Address: 0xd01180
//
void __fastcall sub_D01180(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rcx
  const __m128i *v8; // rbx
  __m128i *v9; // r12
  unsigned __int64 v10; // rax
  __m128i *v11; // r14
  unsigned __int64 v12; // rsi
  int v13; // r13d
  const __m128i *v14; // rdx
  __m128i *v15; // rax
  const __m128i *i; // rdi
  __int64 v17; // rsi
  const __m128i *v18; // r12
  __int64 v19; // rbx
  __int64 v20; // rdi
  __m128i *v21; // r13
  __int64 v22; // rdi
  __m128i *v23; // rdi
  __m128i *v24; // rbx
  __int64 v25; // rdi
  const __m128i *v26; // r12
  __int64 v27; // rbx
  __int64 v28; // rdi
  __m128i *v29; // r14
  bool v30; // cc
  __int64 v31; // rdi
  __int64 v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // rdx
  __m128i *v35; // rbx
  __int64 v36; // rdi
  __m128i *v37; // r14
  __int64 v38; // rdi
  __int64 v39; // rax
  __int64 v40; // rax
  unsigned __int64 v41; // [rsp-50h] [rbp-50h]
  __int64 v42; // [rsp-48h] [rbp-48h]
  __int64 v43; // [rsp-48h] [rbp-48h]
  __int64 v44; // [rsp-40h] [rbp-40h]
  __int64 v45; // [rsp-40h] [rbp-40h]
  __int64 v46; // [rsp-40h] [rbp-40h]
  unsigned __int64 v47; // [rsp-40h] [rbp-40h]
  __m128i **v48; // [rsp-40h] [rbp-40h]
  __int64 v49; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v6 = a1;
    v8 = (const __m128i *)(a2 + 16);
    v9 = *(__m128i **)a1;
    v10 = *(unsigned int *)(a1 + 8);
    v11 = *(__m128i **)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v12 = *(unsigned int *)(a2 + 8);
      v13 = v12;
      if ( v12 <= v10 )
      {
        v23 = *(__m128i **)a1;
        if ( v12 )
        {
          v37 = (__m128i *)((char *)v9 + 56 * v12);
          do
          {
            v30 = v9[2].m128i_i32[0] <= 0x40u;
            *v9 = _mm_loadu_si128(v8);
            v9[1].m128i_i32[0] = v8[1].m128i_i32[0];
            v9[1].m128i_i8[4] = v8[1].m128i_i8[4];
            if ( !v30 )
            {
              v38 = v9[1].m128i_i64[1];
              if ( v38 )
              {
                v49 = v6;
                j_j___libc_free_0_0(v38);
                v6 = v49;
              }
            }
            v39 = v8[1].m128i_i64[1];
            v9 = (__m128i *)((char *)v9 + 56);
            v8 = (const __m128i *)((char *)v8 + 56);
            v9[-2].m128i_i64[0] = v39;
            v9[-2].m128i_i32[2] = v8[-2].m128i_i32[2];
            v40 = v8[-1].m128i_i64[0];
            v8[-2].m128i_i32[2] = 0;
            v9[-1].m128i_i64[0] = v40;
            v9[-1].m128i_i8[8] = v8[-1].m128i_i8[8];
            v9[-1].m128i_i8[9] = v8[-1].m128i_i8[9];
          }
          while ( v9 != v37 );
          v23 = *(__m128i **)v6;
          v10 = *(unsigned int *)(v6 + 8);
        }
        v24 = (__m128i *)((char *)v23 + 56 * v10);
        while ( v9 != v24 )
        {
          v24 = (__m128i *)((char *)v24 - 56);
          if ( v24[2].m128i_i32[0] > 0x40u )
          {
            v25 = v24[1].m128i_i64[1];
            if ( v25 )
            {
              v46 = v6;
              j_j___libc_free_0_0(v25);
              v6 = v46;
            }
          }
        }
        *(_DWORD *)(v6 + 8) = v12;
        v26 = *(const __m128i **)a2;
        v27 = *(_QWORD *)a2 + 56LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v27 )
        {
          do
          {
            v27 -= 56;
            if ( *(_DWORD *)(v27 + 32) > 0x40u )
            {
              v28 = *(_QWORD *)(v27 + 24);
              if ( v28 )
                j_j___libc_free_0_0(v28);
            }
          }
          while ( v26 != (const __m128i *)v27 );
        }
      }
      else
      {
        if ( v12 > *(unsigned int *)(a1 + 12) )
        {
          v34 = 7 * v10;
          v35 = (__m128i *)((char *)v9 + 56 * v10);
          while ( v35 != v11 )
          {
            while ( 1 )
            {
              v35 = (__m128i *)((char *)v35 - 56);
              if ( v35[2].m128i_i32[0] <= 0x40u )
                break;
              v36 = v35[1].m128i_i64[1];
              if ( !v36 )
                break;
              v43 = v6;
              j_j___libc_free_0_0(v36);
              v6 = v43;
              if ( v35 == v11 )
                goto LABEL_49;
            }
          }
LABEL_49:
          *(_DWORD *)(v6 + 8) = 0;
          v48 = (__m128i **)v6;
          sub_D00C80(v6, v12, v34, v6, a5, a6);
          v8 = *(const __m128i **)a2;
          v6 = (__int64)v48;
          v10 = 0;
          v12 = *(unsigned int *)(a2 + 8);
          v11 = *v48;
          v14 = *(const __m128i **)a2;
        }
        else
        {
          v14 = v8;
          if ( *(_DWORD *)(a1 + 8) )
          {
            v10 *= 56LL;
            v41 = v10;
            v29 = (__m128i *)((char *)v9 + v10);
            do
            {
              v30 = v9[2].m128i_i32[0] <= 0x40u;
              *v9 = _mm_loadu_si128(v8);
              v9[1].m128i_i32[0] = v8[1].m128i_i32[0];
              v9[1].m128i_i8[4] = v8[1].m128i_i8[4];
              if ( !v30 )
              {
                v31 = v9[1].m128i_i64[1];
                if ( v31 )
                {
                  v42 = v6;
                  v47 = v10;
                  j_j___libc_free_0_0(v31);
                  v6 = v42;
                  v10 = v47;
                }
              }
              v32 = v8[1].m128i_i64[1];
              v9 = (__m128i *)((char *)v9 + 56);
              v8 = (const __m128i *)((char *)v8 + 56);
              v9[-2].m128i_i64[0] = v32;
              v9[-2].m128i_i32[2] = v8[-2].m128i_i32[2];
              v33 = v8[-1].m128i_i64[0];
              v8[-2].m128i_i32[2] = 0;
              v9[-1].m128i_i64[0] = v33;
              v9[-1].m128i_i8[8] = v8[-1].m128i_i8[8];
              v9[-1].m128i_i8[9] = v8[-1].m128i_i8[9];
            }
            while ( v9 != v29 );
            v8 = *(const __m128i **)a2;
            v12 = *(unsigned int *)(a2 + 8);
            v11 = *(__m128i **)v6;
            v14 = (const __m128i *)(*(_QWORD *)a2 + v41);
          }
        }
        v15 = (__m128i *)((char *)v11 + v10);
        for ( i = (const __m128i *)((char *)v8 + 56 * v12); i != v14; v15 = (__m128i *)((char *)v15 + 56) )
        {
          if ( v15 )
          {
            *v15 = _mm_loadu_si128(v14);
            v15[1].m128i_i64[0] = v14[1].m128i_i64[0];
            v15[2].m128i_i32[0] = v14[2].m128i_i32[0];
            v15[1].m128i_i64[1] = v14[1].m128i_i64[1];
            v17 = v14[2].m128i_i64[1];
            v14[2].m128i_i32[0] = 0;
            v15[2].m128i_i64[1] = v17;
            v15[3].m128i_i8[0] = v14[3].m128i_i8[0];
            v15[3].m128i_i8[1] = v14[3].m128i_i8[1];
          }
          v14 = (const __m128i *)((char *)v14 + 56);
        }
        *(_DWORD *)(v6 + 8) = v13;
        v18 = *(const __m128i **)a2;
        v19 = *(_QWORD *)a2 + 56LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v19 )
        {
          do
          {
            v19 -= 56;
            if ( *(_DWORD *)(v19 + 32) > 0x40u )
            {
              v20 = *(_QWORD *)(v19 + 24);
              if ( v20 )
                j_j___libc_free_0_0(v20);
            }
          }
          while ( v18 != (const __m128i *)v19 );
        }
      }
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v21 = (__m128i *)((char *)v9 + 56 * v10);
      if ( v21 != v9 )
      {
        do
        {
          v21 = (__m128i *)((char *)v21 - 56);
          if ( v21[2].m128i_i32[0] > 0x40u )
          {
            v22 = v21[1].m128i_i64[1];
            if ( v22 )
            {
              v44 = v6;
              j_j___libc_free_0_0(v22);
              v6 = v44;
            }
          }
        }
        while ( v21 != v11 );
        v9 = *(__m128i **)v6;
      }
      if ( v9 != (__m128i *)(v6 + 16) )
      {
        v45 = v6;
        _libc_free(v9, a2);
        v6 = v45;
      }
      *(_QWORD *)v6 = *(_QWORD *)a2;
      *(_DWORD *)(v6 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(v6 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v8;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
