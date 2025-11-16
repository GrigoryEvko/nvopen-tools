// Function: sub_39CF780
// Address: 0x39cf780
//
void __fastcall sub_39CF780(__int64 a1, int a2)
{
  __int64 v3; // rbx
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // rax
  _DWORD *v6; // rax
  __int64 v7; // rdx
  const __m128i *v8; // r14
  _DWORD *i; // rdx
  const __m128i *v10; // rbx
  const __m128i *v11; // rax
  __int32 v12; // esi
  int v13; // eax
  int v14; // edx
  __int64 v15; // r8
  int *v16; // r9
  int v17; // r10d
  unsigned int v18; // ecx
  __int32 *v19; // rax
  int v20; // edi
  const __m128i *v21; // rdx
  __int32 v22; // edx
  unsigned __int64 v23; // rdi
  _DWORD *j; // rdx

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
  v6 = (_DWORD *)sub_22077B0(48LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = v6;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = (const __m128i *)(v4 + 48 * v3);
    for ( i = &v6[12 * v7]; i != v6; v6 += 12 )
    {
      if ( v6 )
        *v6 = 0x7FFFFFFF;
    }
    v10 = (const __m128i *)(v4 + 24);
    if ( v8 != (const __m128i *)v4 )
    {
      while ( 1 )
      {
        v12 = v10[-2].m128i_i32[2];
        if ( (unsigned int)(v12 + 0x7FFFFFFF) > 0xFFFFFFFD )
          goto LABEL_10;
        v13 = *(_DWORD *)(a1 + 24);
        if ( !v13 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v14 = v13 - 1;
        v15 = *(_QWORD *)(a1 + 8);
        v16 = 0;
        v17 = 1;
        v18 = (v13 - 1) & (37 * v12);
        v19 = (__int32 *)(v15 + 48LL * v18);
        v20 = *v19;
        if ( v12 != *v19 )
        {
          while ( v20 != 0x7FFFFFFF )
          {
            if ( !v16 && v20 == 0x80000000 )
              v16 = v19;
            v18 = v14 & (v17 + v18);
            v19 = (__int32 *)(v15 + 48LL * v18);
            v20 = *v19;
            if ( v12 == *v19 )
              goto LABEL_15;
            ++v17;
          }
          if ( v16 )
            v19 = v16;
        }
LABEL_15:
        *v19 = v12;
        *((_QWORD *)v19 + 1) = v19 + 6;
        v21 = (const __m128i *)v10[-1].m128i_i64[0];
        if ( v10 == v21 )
        {
          *(__m128i *)(v19 + 6) = _mm_loadu_si128(v10);
        }
        else
        {
          *((_QWORD *)v19 + 1) = v21;
          *((_QWORD *)v19 + 3) = v10->m128i_i64[0];
        }
        *((_QWORD *)v19 + 2) = v10[-1].m128i_i64[1];
        v22 = v10[1].m128i_i32[0];
        v10[-1].m128i_i64[0] = (__int64)v10;
        v10[-1].m128i_i64[1] = 0;
        v10->m128i_i8[0] = 0;
        v19[10] = v22;
        *((_BYTE *)v19 + 44) = v10[1].m128i_i8[4];
        ++*(_DWORD *)(a1 + 16);
        v23 = v10[-1].m128i_u64[0];
        if ( v10 == (const __m128i *)v23 )
        {
LABEL_10:
          v11 = v10 + 3;
          if ( v8 == (const __m128i *)&v10[1].m128i_u64[1] )
            break;
        }
        else
        {
          j_j___libc_free_0(v23);
          v11 = v10 + 3;
          if ( v8 == (const __m128i *)&v10[1].m128i_u64[1] )
            break;
        }
        v10 = v11;
      }
    }
    j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &v6[12 * *(unsigned int *)(a1 + 24)]; j != v6; v6 += 12 )
    {
      if ( v6 )
        *v6 = 0x7FFFFFFF;
    }
  }
}
