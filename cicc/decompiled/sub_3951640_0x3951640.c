// Function: sub_3951640
// Address: 0x3951640
//
void __fastcall sub_3951640(__int64 a1, int a2)
{
  __int64 v3; // rbx
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rax
  _QWORD *v6; // rax
  const __m128i *v7; // r14
  _QWORD *i; // rdx
  const __m128i *v9; // rbx
  __int64 v10; // rax
  int v11; // edx
  int v12; // esi
  __int64 v13; // r8
  int v14; // r10d
  __m128i *v15; // r9
  unsigned int v16; // ecx
  __m128i *v17; // rdx
  __int64 v18; // rdi
  __int64 v19; // rdx
  _QWORD *j; // rdx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
            | (unsigned int)(a2 - 1)
            | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
          | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
        | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 16)
      | (((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
      | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
      | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
      | (unsigned int)(a2 - 1)
      | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1))
     + 1;
  if ( (unsigned int)v5 < 0x40 )
    LODWORD(v5) = 64;
  *(_DWORD *)(a1 + 24) = v5;
  v6 = (_QWORD *)sub_22077B0(40LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = v6;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = (const __m128i *)(v4 + 40 * v3);
    for ( i = &v6[5 * *(unsigned int *)(a1 + 24)]; i != v6; v6 += 5 )
    {
      if ( v6 )
        *v6 = -8;
    }
    if ( v7 != (const __m128i *)v4 )
    {
      v9 = (const __m128i *)v4;
      do
      {
        v10 = v9->m128i_i64[0];
        if ( v9->m128i_i64[0] != -16 && v10 != -8 )
        {
          v11 = *(_DWORD *)(a1 + 24);
          if ( !v11 )
          {
            MEMORY[0] = v9->m128i_i64[0];
            BUG();
          }
          v12 = v11 - 1;
          v13 = *(_QWORD *)(a1 + 8);
          v14 = 1;
          v15 = 0;
          v16 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
          v17 = (__m128i *)(v13 + 40LL * v16);
          v18 = v17->m128i_i64[0];
          if ( v10 != v17->m128i_i64[0] )
          {
            while ( v18 != -8 )
            {
              if ( v18 == -16 && !v15 )
                v15 = v17;
              v16 = v12 & (v14 + v16);
              v17 = (__m128i *)(v13 + 40LL * v16);
              v18 = v17->m128i_i64[0];
              if ( v10 == v17->m128i_i64[0] )
                goto LABEL_14;
              ++v14;
            }
            if ( v15 )
              v17 = v15;
          }
LABEL_14:
          v17->m128i_i64[0] = v10;
          v17->m128i_i32[2] = v9->m128i_i32[2];
          v17[1] = _mm_loadu_si128(v9 + 1);
          v17[2].m128i_i32[0] = v9[2].m128i_i32[0];
          v9[1].m128i_i64[0] = 0;
          v9[1].m128i_i64[1] = 0;
          v9[2].m128i_i32[0] = 0;
          ++*(_DWORD *)(a1 + 16);
          _libc_free(v9[1].m128i_u64[0]);
        }
        v9 = (const __m128i *)((char *)v9 + 40);
      }
      while ( v7 != v9 );
    }
    j___libc_free_0(v4);
  }
  else
  {
    v19 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &v6[5 * v19]; j != v6; v6 += 5 )
    {
      if ( v6 )
        *v6 = -8;
    }
  }
}
