// Function: sub_1B449D0
// Address: 0x1b449d0
//
__int64 __fastcall sub_1B449D0(__int64 a1, __int64 a2, __int64 a3)
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
  __int64 v15; // r13
  __int64 v17; // rdx
  __int64 v18; // rcx
  __m128i *v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rbx
  int v23; // eax
  __int64 v24; // r14
  __int64 v25; // rax
  __m128i *v26; // rsi
  int v27; // eax
  __int64 v28; // [rsp+8h] [rbp-48h]
  __m128i v29; // [rsp+10h] [rbp-40h] BYREF

  if ( *(_BYTE *)(a2 + 16) == 27 )
  {
    v5 = *(const __m128i **)a3;
    v6 = ((*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1) - 1;
    v7 = *(const __m128i **)a3;
    if ( v6 > (__int64)(*(_QWORD *)(a3 + 16) - *(_QWORD *)a3) >> 4 )
    {
      v8 = *(const __m128i **)(a3 + 8);
      v9 = 0;
      v10 = 16 * v6;
      v11 = (char *)v8 - (char *)v5;
      if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1 != 1 )
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
        v28 = v10;
        j_j___libc_free_0(v8, *(_QWORD *)(a3 + 16) - (_QWORD)v8);
        v10 = v28;
      }
      *(_QWORD *)a3 = v9;
      *(_QWORD *)(a3 + 8) = (char *)v9 + v11;
      *(_QWORD *)(a3 + 16) = (char *)v9 + v10;
      v6 = ((*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1) - 1;
    }
    v14 = 0;
    if ( v6 )
    {
      do
      {
        v21 = 24;
        if ( (_DWORD)v14 != -2 )
          v21 = 24LL * (unsigned int)(2 * v14 + 3);
        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
          v17 = *(_QWORD *)(a2 - 8);
        else
          v17 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
        ++v14;
        v18 = *(_QWORD *)(v17 + v21);
        v19 = *(__m128i **)(a3 + 8);
        v20 = *(_QWORD *)(v17 + 24LL * (unsigned int)(2 * v14));
        v29.m128i_i64[1] = v18;
        v29.m128i_i64[0] = v20;
        if ( v19 == *(__m128i **)(a3 + 16) )
        {
          sub_1B43170((const __m128i **)a3, v19, &v29);
        }
        else
        {
          if ( v19 )
          {
            *v19 = _mm_loadu_si128(&v29);
            v19 = *(__m128i **)(a3 + 8);
          }
          *(_QWORD *)(a3 + 8) = v19 + 1;
        }
      }
      while ( v6 != v14 );
    }
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v15 = *(_QWORD *)(a2 - 8);
    else
      v15 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    return *(_QWORD *)(v15 + 24);
  }
  else
  {
    v22 = *(_QWORD *)(a2 - 72);
    v23 = *(unsigned __int16 *)(v22 + 18);
    BYTE1(v23) &= ~0x80u;
    v24 = *(_QWORD *)(a2 - 24LL * (v23 == 33) - 24);
    v25 = sub_1B42400(*(__int64 ****)(v22 - 24), *(_QWORD *)(a1 + 8));
    v26 = *(__m128i **)(a3 + 8);
    v29.m128i_i64[0] = v25;
    v29.m128i_i64[1] = v24;
    if ( v26 == *(__m128i **)(a3 + 16) )
    {
      sub_1B43170((const __m128i **)a3, v26, &v29);
    }
    else
    {
      if ( v26 )
      {
        *v26 = _mm_loadu_si128(&v29);
        v26 = *(__m128i **)(a3 + 8);
      }
      *(_QWORD *)(a3 + 8) = v26 + 1;
    }
    v27 = *(unsigned __int16 *)(v22 + 18);
    BYTE1(v27) &= ~0x80u;
    return *(_QWORD *)(a2 - 24LL * (v27 == 32) - 24);
  }
}
