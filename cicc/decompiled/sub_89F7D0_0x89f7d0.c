// Function: sub_89F7D0
// Address: 0x89f7d0
//
__int64 __fastcall sub_89F7D0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  __int64 v3; // r13
  __int64 v4; // rbx
  char v5; // al
  __int64 v6; // r15
  __int64 *v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rax
  __m128i v10; // xmm2
  const __m128i *v12; // rax
  unsigned __int64 v13; // rdx
  __m128i v14; // xmm1
  __m128i v15; // xmm0
  __int64 v16; // rcx
  __int64 v17; // rsi
  __int64 v18; // [rsp+8h] [rbp-58h]
  __m128i v19; // [rsp+10h] [rbp-50h] BYREF
  __int64 v20; // [rsp+20h] [rbp-40h]

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  v1 = sub_823970(24);
  *(_QWORD *)(a1 + 8) = 1;
  *(_QWORD *)a1 = v1;
  v2 = unk_4F04C48;
  if ( unk_4F04C48 == -1 )
    return a1;
  v3 = *(_QWORD *)(a1 + 16);
  do
  {
    while ( 1 )
    {
      v4 = qword_4F04C68[0] + 776 * v2;
      v5 = *(_BYTE *)(v4 + 4);
      if ( v5 == 9 )
        break;
      if ( (unsigned __int8)(v5 - 3) <= 1u || !v5 )
        goto LABEL_13;
LABEL_5:
      v2 = *(int *)(v4 + 552);
      if ( (_DWORD)v2 == -1 )
        goto LABEL_13;
    }
    v6 = *(_QWORD *)(v4 + 376);
    if ( !v6 )
      goto LABEL_5;
    v7 = *(__int64 **)(v4 + 408);
    v20 = 0;
    v19 = 0;
    v8 = *v7;
    if ( *(_QWORD *)(a1 + 8) == v3 )
    {
      v18 = *v7;
      sub_738390((const __m128i **)a1);
      v8 = v18;
    }
    v9 = *(_QWORD *)a1 + 24 * v3;
    if ( v9 )
    {
      v19.m128i_i64[0] = v8;
      v19.m128i_i64[1] = v6;
      v10 = _mm_loadu_si128(&v19);
      *(_QWORD *)(v9 + 16) = v20;
      *(__m128i *)v9 = v10;
    }
    *(_QWORD *)(a1 + 16) = ++v3;
    v2 = *(int *)(v4 + 552);
  }
  while ( (_DWORD)v2 != -1 );
LABEL_13:
  if ( v3 > 1 )
  {
    v12 = *(const __m128i **)a1;
    v13 = *(_QWORD *)a1 + 24 * v3 - 24;
    if ( *(_QWORD *)a1 < v13 )
    {
      do
      {
        v14 = _mm_loadu_si128((const __m128i *)v13);
        v15 = _mm_loadu_si128(v12);
        v13 -= 24LL;
        v12 = (const __m128i *)((char *)v12 + 24);
        v16 = v12[-1].m128i_i64[1];
        *(__m128i *)((char *)v12 - 24) = v14;
        v17 = *(_QWORD *)(v13 + 40);
        v20 = v16;
        v12[-1].m128i_i64[1] = v17;
        v19 = v15;
        *(__m128i *)(v13 + 24) = v15;
        *(_QWORD *)(v13 + 40) = v16;
      }
      while ( (unsigned __int64)v12 < v13 );
    }
  }
  return a1;
}
