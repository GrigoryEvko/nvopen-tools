// Function: sub_F90890
// Address: 0xf90890
//
__int64 __fastcall sub_F90890(__int64 a1, __int64 a2, __int64 a3)
{
  const __m128i *v5; // rcx
  unsigned __int64 v6; // r14
  const __m128i *v7; // rax
  const __m128i *v8; // rdi
  __m128i *v9; // r15
  __int64 v10; // r8
  signed __int64 v11; // rbx
  __int64 v12; // rax
  __m128i *v13; // rdx
  __int64 v14; // rbx
  __m128i *v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rbx
  __int64 v22; // r14
  __int64 v23; // rax
  __m128i *v24; // rsi
  __int64 v25; // [rsp+8h] [rbp-48h]
  __m128i v26; // [rsp+10h] [rbp-40h] BYREF

  if ( *(_BYTE *)a2 == 32 )
  {
    v5 = *(const __m128i **)a3;
    v6 = ((*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1) - 1;
    v7 = *(const __m128i **)a3;
    if ( v6 > (__int64)(*(_QWORD *)(a3 + 16) - *(_QWORD *)a3) >> 4 )
    {
      v8 = *(const __m128i **)(a3 + 8);
      v9 = 0;
      v10 = 16 * v6;
      v11 = (char *)v8 - (char *)v5;
      if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1 != 1 )
      {
        v12 = sub_22077B0(16 * v6);
        v5 = *(const __m128i **)a3;
        v8 = *(const __m128i **)(a3 + 8);
        v10 = 16 * v6;
        v9 = (__m128i *)v12;
        v7 = *(const __m128i **)a3;
      }
      if ( v5 != v8 )
      {
        v13 = v9;
        do
        {
          if ( v13 )
            *v13 = _mm_loadu_si128(v7);
          ++v7;
          ++v13;
        }
        while ( v7 != v8 );
        v8 = v5;
      }
      if ( v8 )
      {
        v25 = v10;
        j_j___libc_free_0(v8, *(_QWORD *)(a3 + 16) - (_QWORD)v8);
        v10 = v25;
      }
      *(_QWORD *)a3 = v9;
      *(_QWORD *)(a3 + 8) = (char *)v9 + v11;
      *(_QWORD *)(a3 + 16) = (char *)v9 + v10;
      v6 = ((*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1) - 1;
    }
    v14 = 0;
    if ( v6 )
    {
      do
      {
        v17 = 32;
        if ( (_DWORD)v14 != -2 )
          v17 = 32LL * (unsigned int)(2 * v14 + 3);
        v18 = *(_QWORD *)(a2 - 8);
        ++v14;
        v16 = *(__m128i **)(a3 + 8);
        v19 = *(_QWORD *)(v18 + v17);
        v20 = *(_QWORD *)(v18 + 32LL * (unsigned int)(2 * v14));
        v26.m128i_i64[1] = v19;
        v26.m128i_i64[0] = v20;
        if ( v16 == *(__m128i **)(a3 + 16) )
        {
          sub_F8F320((const __m128i **)a3, v16, &v26);
        }
        else
        {
          if ( v16 )
          {
            *v16 = _mm_loadu_si128(&v26);
            v16 = *(__m128i **)(a3 + 8);
          }
          *(_QWORD *)(a3 + 8) = v16 + 1;
        }
      }
      while ( v6 != v14 );
    }
    return *(_QWORD *)(*(_QWORD *)(a2 - 8) + 32LL);
  }
  else
  {
    v21 = *(_QWORD *)(a2 - 96);
    v22 = *(_QWORD *)(a2 - 32LL * ((*(_WORD *)(v21 + 2) & 0x3F) == 33) - 32);
    v23 = sub_F8E510(*(_QWORD *)(v21 - 32), *(_QWORD *)(a1 + 24));
    v24 = *(__m128i **)(a3 + 8);
    v26.m128i_i64[0] = v23;
    v26.m128i_i64[1] = v22;
    if ( v24 == *(__m128i **)(a3 + 16) )
    {
      sub_F8F320((const __m128i **)a3, v24, &v26);
    }
    else
    {
      if ( v24 )
      {
        *v24 = _mm_loadu_si128(&v26);
        v24 = *(__m128i **)(a3 + 8);
      }
      *(_QWORD *)(a3 + 8) = v24 + 1;
    }
    return *(_QWORD *)(a2 - 32LL * ((*(_WORD *)(v21 + 2) & 0x3F) == 32) - 32);
  }
}
