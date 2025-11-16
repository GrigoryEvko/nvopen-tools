// Function: sub_30B96B0
// Address: 0x30b96b0
//
__m128i *__fastcall sub_30B96B0(__int64 a1, int *a2)
{
  int v4; // eax
  const void *v5; // rsi
  char *v6; // rdi
  __int64 v7; // rdx
  unsigned int v8; // eax
  unsigned __int64 v9; // r13
  char *v10; // rcx
  __int64 v11; // r13
  unsigned __int64 v12; // r13
  char *v13; // rcx
  __int64 v14; // r13
  unsigned __int64 v15; // r13
  __m128i *result; // rax
  const __m128i *v17; // rcx
  const __m128i *v18; // rdx
  __m128i *v19; // rcx
  char *v20; // rax
  __int64 v21; // rdx

  v4 = *a2;
  *(_QWORD *)(a1 + 8) = 0;
  v5 = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)a1 = v4;
  *(_QWORD *)(a1 + 24) = 0;
  *(_DWORD *)(a1 + 32) = 0;
  v6 = 0;
  sub_C7D6A0(0, 0, 8);
  v8 = a2[8];
  *(_DWORD *)(a1 + 32) = v8;
  if ( v8 )
  {
    v20 = (char *)sub_C7D670(16LL * v8, 8);
    v21 = *(unsigned int *)(a1 + 32);
    v5 = (const void *)*((_QWORD *)a2 + 2);
    *(_QWORD *)(a1 + 16) = v20;
    v6 = v20;
    *(_QWORD *)(a1 + 24) = *((_QWORD *)a2 + 3);
    memcpy(v20, v5, 16 * v21);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 24) = 0;
  }
  v9 = *((_QWORD *)a2 + 6) - *((_QWORD *)a2 + 5);
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  if ( v9 )
  {
    if ( v9 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_27;
    v6 = (char *)v9;
    v10 = (char *)sub_22077B0(v9);
  }
  else
  {
    v10 = 0;
  }
  *(_QWORD *)(a1 + 40) = v10;
  *(_QWORD *)(a1 + 56) = &v10[v9];
  *(_QWORD *)(a1 + 48) = v10;
  v5 = (const void *)*((_QWORD *)a2 + 5);
  v11 = *((_QWORD *)a2 + 6) - (_QWORD)v5;
  if ( *((const void **)a2 + 6) != v5 )
  {
    v6 = v10;
    v10 = (char *)memmove(v10, v5, *((_QWORD *)a2 + 6) - (_QWORD)v5);
  }
  *(_QWORD *)(a1 + 48) = &v10[v11];
  v12 = *((_QWORD *)a2 + 9) - *((_QWORD *)a2 + 8);
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  if ( v12 )
  {
    if ( v12 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_27;
    v6 = (char *)v12;
    v13 = (char *)sub_22077B0(v12);
  }
  else
  {
    v12 = 0;
    v13 = 0;
  }
  *(_QWORD *)(a1 + 64) = v13;
  *(_QWORD *)(a1 + 80) = &v13[v12];
  *(_QWORD *)(a1 + 72) = v13;
  v5 = (const void *)*((_QWORD *)a2 + 8);
  v14 = *((_QWORD *)a2 + 9) - (_QWORD)v5;
  if ( *((const void **)a2 + 9) != v5 )
  {
    v6 = v13;
    v13 = (char *)memmove(v13, v5, *((_QWORD *)a2 + 9) - (_QWORD)v5);
  }
  *(_QWORD *)(a1 + 72) = &v13[v14];
  v15 = *((_QWORD *)a2 + 12) - *((_QWORD *)a2 + 11);
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  if ( v15 )
  {
    if ( v15 <= 0x7FFFFFFFFFFFFFE0LL )
    {
      result = (__m128i *)sub_22077B0(v15);
      goto LABEL_16;
    }
LABEL_27:
    sub_4261EA(v6, v5, v7);
  }
  v15 = 0;
  result = 0;
LABEL_16:
  *(_QWORD *)(a1 + 88) = result;
  *(_QWORD *)(a1 + 96) = result;
  *(_QWORD *)(a1 + 104) = (char *)result + v15;
  v17 = (const __m128i *)*((_QWORD *)a2 + 12);
  v18 = (const __m128i *)*((_QWORD *)a2 + 11);
  if ( v17 == v18 )
  {
    *(_QWORD *)(a1 + 96) = result;
  }
  else
  {
    v19 = (__m128i *)((char *)result + (char *)v17 - (char *)v18);
    do
    {
      if ( result )
      {
        *result = _mm_loadu_si128(v18);
        result[1] = _mm_loadu_si128(v18 + 1);
      }
      result += 2;
      v18 += 2;
    }
    while ( v19 != result );
    *(_QWORD *)(a1 + 96) = v19;
  }
  return result;
}
