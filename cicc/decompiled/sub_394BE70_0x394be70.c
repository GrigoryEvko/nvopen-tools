// Function: sub_394BE70
// Address: 0x394be70
//
void __fastcall sub_394BE70(__int64 a1, int a2)
{
  __int64 v3; // r12
  unsigned __int64 v4; // r15
  unsigned __int64 v5; // rax
  __int64 v6; // rax
  const __m128i *v7; // r11
  __int64 i; // rdx
  const __m128i *v9; // rax
  __int64 v10; // r8
  unsigned __int8 v11; // di
  int v12; // esi
  __int64 v13; // r10
  int v14; // esi
  __int64 v15; // r12
  int v16; // r14d
  __int64 v17; // r15
  unsigned int j; // ecx
  __int64 v19; // rdx
  __int64 v20; // r13
  __m128i v21; // xmm0
  __int64 v22; // rdx
  __int64 k; // rdx
  unsigned int v24; // ecx
  unsigned __int64 v25; // [rsp+8h] [rbp-38h]

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
  v6 = sub_22077B0(40LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = v6;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = (const __m128i *)(v4 + 40 * v3);
    for ( i = v6 + 40LL * *(unsigned int *)(a1 + 24); i != v6; v6 += 40 )
    {
      if ( v6 )
      {
        *(_BYTE *)v6 = 0;
        *(_QWORD *)(v6 + 8) = 0;
        *(_QWORD *)(v6 + 16) = 0;
      }
    }
    v9 = (const __m128i *)v4;
    if ( v7 != (const __m128i *)v4 )
    {
      v25 = v4;
      do
      {
        while ( 1 )
        {
          v10 = v9->m128i_i64[1];
          v11 = v9->m128i_i8[0];
          if ( v10 || v9[1].m128i_i64[0] )
            break;
          v9 = (const __m128i *)((char *)v9 + 40);
          if ( v7 == v9 )
            goto LABEL_22;
        }
        v12 = *(_DWORD *)(a1 + 24);
        if ( !v12 )
        {
          MEMORY[0] = _mm_loadu_si128(v9);
          MEMORY[0x10] = v9[1].m128i_i64[0];
          BUG();
        }
        v13 = v9[1].m128i_i64[0];
        v14 = v12 - 1;
        v15 = *(_QWORD *)(a1 + 8);
        v16 = 1;
        v17 = 0;
        for ( j = v14 & (v11 ^ v13 ^ v10); ; j = v14 & v24 )
        {
          v19 = v15 + 40LL * j;
          v20 = *(_QWORD *)(v19 + 8);
          if ( v11 == *(_BYTE *)v19 && v20 == v10 && v13 == *(_QWORD *)(v19 + 16) )
            break;
          if ( *(_BYTE *)v19 )
          {
            if ( !v20 && !(*(_QWORD *)(v19 + 16) | v17) )
              v17 = v15 + 40LL * j;
          }
          else if ( !v20 && !*(_QWORD *)(v19 + 16) )
          {
            if ( v17 )
              v19 = v17;
            break;
          }
          v24 = v16 + j;
          ++v16;
        }
        v21 = _mm_loadu_si128(v9);
        v9 = (const __m128i *)((char *)v9 + 40);
        *(__m128i *)v19 = v21;
        *(_QWORD *)(v19 + 16) = v9[-2].m128i_i64[1];
        *(__m128i *)(v19 + 24) = _mm_loadu_si128(v9 - 1);
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v7 != v9 );
LABEL_22:
      v4 = v25;
    }
    j___libc_free_0(v4);
  }
  else
  {
    v22 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = v6 + 40 * v22; k != v6; v6 += 40 )
    {
      if ( v6 )
      {
        *(_BYTE *)v6 = 0;
        *(_QWORD *)(v6 + 8) = 0;
        *(_QWORD *)(v6 + 16) = 0;
      }
    }
  }
}
