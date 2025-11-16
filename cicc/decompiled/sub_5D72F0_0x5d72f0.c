// Function: sub_5D72F0
// Address: 0x5d72f0
//
__int64 __fastcall sub_5D72F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r13d
  const __m128i *v8; // r14
  __int64 result; // rax
  const __m128i *v10; // rcx
  char *v11; // rbx
  int v12; // edi
  char *v13; // rbx
  int v14; // edi
  __m128i *v15; // rax
  __m128i v16; // xmm5
  int v17; // edi
  char *v18; // rbx
  __m128i *v19; // rax
  __int64 v20; // r15
  __int64 v21; // rsi
  __int64 v22; // rax
  __m128i *v23; // [rsp+8h] [rbp-38h] BYREF

  v6 = a2;
  v23 = (__m128i *)sub_724DC0(a1, a2, a3, a4, a5, a6);
  if ( (**(_BYTE **)(a1 + 176) & 1) == 0 )
  {
    v8 = 0;
    goto LABEL_23;
  }
  v8 = *(const __m128i **)(a1 + 168);
  if ( (*(_BYTE *)(a1 + 161) & 0x10) == 0 )
  {
    if ( v8 )
    {
      result = *(_QWORD *)(a1 + 168);
      goto LABEL_5;
    }
LABEL_23:
    if ( *(char *)(a1 + 88) < 0 )
    {
LABEL_24:
      v20 = 0;
      while ( 1 )
      {
        v21 = unk_4F04C50;
        if ( unk_4F04C50 )
          v21 = qword_4CF7E98;
        v22 = sub_732D20(a1, v21, 0, v20);
        v20 = v22;
        if ( !v22 )
          break;
        sub_5D52E0(v22, v21);
      }
    }
LABEL_10:
    v11 = "num ";
    sub_5D45D0((unsigned int *)(a1 + 64));
    v12 = 101;
    do
    {
      ++v11;
      putc(v12, stream);
      v12 = *(v11 - 1);
    }
    while ( *(v11 - 1) );
    dword_4CF7F40 += 5;
    v13 = "{";
    sub_5D71E0(a1);
    v14 = 32;
    do
    {
      ++v13;
      putc(v14, stream);
      v14 = *(v13 - 1);
    }
    while ( *(v13 - 1) );
    v15 = v23;
    v16 = _mm_loadu_si128(v8);
    dword_4CF7F40 += 2;
    *v23 = v16;
    v15[1] = _mm_loadu_si128(v8 + 1);
    v15[2] = _mm_loadu_si128(v8 + 2);
    v15[3] = _mm_loadu_si128(v8 + 3);
    v15[4] = _mm_loadu_si128(v8 + 4);
    v15[5] = _mm_loadu_si128(v8 + 5);
    v15[6] = _mm_loadu_si128(v8 + 6);
    v15[7] = _mm_loadu_si128(v8 + 7);
    v15[8] = _mm_loadu_si128(v8 + 8);
    v15[9] = _mm_loadu_si128(v8 + 9);
    v15[10] = _mm_loadu_si128(v8 + 10);
    v15[11] = _mm_loadu_si128(v8 + 11);
    v15[12] = _mm_loadu_si128(v8 + 12);
    sub_620D80(&v23[11], 0);
    while ( 1 )
    {
      sub_5D45D0((unsigned int *)&v8[4]);
      sub_5D5A80((__int64)v8, 0);
      if ( (v8[10].m128i_i8[9] & 1) != 0 || (unsigned int)sub_621060(v8, v23) )
      {
        v17 = 32;
        v18 = "= ";
        do
        {
          ++v18;
          putc(v17, stream);
          v17 = *(v18 - 1);
        }
        while ( *(v18 - 1) );
        dword_4CF7F40 += 3;
        sub_74EE00(v8, 1, 1, &qword_4CF7CE0);
        v19 = v23;
        *v23 = _mm_loadu_si128(v8);
        v19[1] = _mm_loadu_si128(v8 + 1);
        v19[2] = _mm_loadu_si128(v8 + 2);
        v19[3] = _mm_loadu_si128(v8 + 3);
        v19[4] = _mm_loadu_si128(v8 + 4);
        v19[5] = _mm_loadu_si128(v8 + 5);
        v19[6] = _mm_loadu_si128(v8 + 6);
        v19[7] = _mm_loadu_si128(v8 + 7);
        v19[8] = _mm_loadu_si128(v8 + 8);
        v19[9] = _mm_loadu_si128(v8 + 9);
        v19[10] = _mm_loadu_si128(v8 + 10);
        v19[11] = _mm_loadu_si128(v8 + 11);
        v19[12] = _mm_loadu_si128(v8 + 12);
      }
      v8 = (const __m128i *)v8[7].m128i_i64[1];
      if ( !v8 )
        break;
      putc(44, stream);
      ++dword_4CF7F40;
      sub_621300(&v23[11]);
    }
    putc(125, stream);
    ++dword_4CF7F40;
    sub_74F590(a1, 1, &qword_4CF7CE0);
    if ( v6 )
    {
      putc(59, stream);
      ++dword_4CF7F40;
    }
    return sub_724E30(&v23);
  }
  v10 = (const __m128i *)v8[6].m128i_i64[0];
  result = (__int64)v10;
  if ( !v10 )
  {
LABEL_9:
    v8 = v10;
    if ( *(char *)(a1 + 88) < 0 )
      goto LABEL_24;
    goto LABEL_10;
  }
LABEL_5:
  while ( *(_BYTE *)(result + 173) != 12 )
  {
    result = *(_QWORD *)(result + 120);
    if ( !result )
    {
      if ( (*(_BYTE *)(a1 + 161) & 0x10) == 0 )
        goto LABEL_23;
      v10 = (const __m128i *)v8[6].m128i_i64[0];
      goto LABEL_9;
    }
  }
  return result;
}
