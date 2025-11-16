// Function: sub_B18B40
// Address: 0xb18b40
//
char *__fastcall sub_B18B40(__int64 *a1, __m128i *a2, char *a3, char *a4)
{
  signed __int64 v5; // r9
  __int64 v6; // r10
  __m128i *v9; // r13
  char *v10; // r12
  __int64 v11; // rsi
  __int64 v12; // rdi
  unsigned __int64 v13; // rcx
  char *result; // rax
  unsigned __int64 v15; // rdx
  __int64 v16; // r11
  __m128i *v17; // r8
  size_t v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rbx
  unsigned int v21; // esi
  __m128i *v22; // rdx
  __int64 v23; // r10
  const __m128i *v24; // rbx
  __int64 v25; // r11
  const __m128i *v26; // rcx
  __m128i *v27; // rsi
  __int64 v28; // [rsp+8h] [rbp-58h]
  signed __int64 v29; // [rsp+10h] [rbp-50h]
  __int64 v30; // [rsp+18h] [rbp-48h]
  int v31; // [rsp+18h] [rbp-48h]
  __int64 v32; // [rsp+18h] [rbp-48h]
  __m128i *v33; // [rsp+20h] [rbp-40h]
  signed __int64 v34; // [rsp+20h] [rbp-40h]
  int v35; // [rsp+20h] [rbp-40h]
  __m128i *v36; // [rsp+20h] [rbp-40h]
  size_t v37; // [rsp+28h] [rbp-38h]
  int v38; // [rsp+28h] [rbp-38h]
  char *v39; // [rsp+28h] [rbp-38h]
  signed __int64 v40; // [rsp+28h] [rbp-38h]
  signed __int64 v41; // [rsp+28h] [rbp-38h]
  char *v42; // [rsp+28h] [rbp-38h]

  v5 = a4 - a3;
  v6 = (a4 - a3) >> 4;
  v9 = a2;
  v10 = a3;
  v11 = *((unsigned int *)a1 + 2);
  v12 = *a1;
  v13 = *((unsigned int *)a1 + 3);
  result = &a2->m128i_i8[-v12];
  v15 = v11 + v6;
  v16 = 16 * v11;
  v17 = (__m128i *)(v12 + 16 * v11);
  if ( v9 == v17 )
  {
    if ( v15 > v13 )
    {
      v35 = v6;
      v41 = v5;
      result = (char *)sub_C8D5F0(a1, a1 + 2, v15, 16);
      LODWORD(v11) = *((_DWORD *)a1 + 2);
      LODWORD(v6) = v35;
      v5 = v41;
      v17 = (__m128i *)(*a1 + 16LL * (unsigned int)v11);
    }
    if ( v10 != a4 )
    {
      v38 = v6;
      result = (char *)memcpy(v17, v10, v5);
      LODWORD(v11) = *((_DWORD *)a1 + 2);
      LODWORD(v6) = v38;
    }
    *((_DWORD *)a1 + 2) = v6 + v11;
  }
  else
  {
    if ( v15 > v13 )
    {
      v31 = v6;
      v34 = v5;
      v39 = result;
      sub_C8D5F0(a1, a1 + 2, v15, 16);
      v11 = *((unsigned int *)a1 + 2);
      v12 = *a1;
      result = v39;
      LODWORD(v6) = v31;
      v5 = v34;
      v16 = 16 * v11;
      v9 = (__m128i *)&v39[*a1];
      v17 = (__m128i *)(*a1 + 16 * v11);
    }
    v18 = v16 - (_QWORD)result;
    v19 = (v16 - (__int64)result) >> 4;
    v20 = v19;
    if ( v16 - (__int64)result >= (unsigned __int64)v5 )
    {
      v22 = v17;
      v23 = v16 - v5;
      v24 = (const __m128i *)(v12 + v16 - v5);
      v25 = v5 >> 4;
      if ( v11 + (v5 >> 4) > (unsigned __int64)*((unsigned int *)a1 + 3) )
      {
        v28 = v5 >> 4;
        v29 = v5;
        v32 = v23;
        v36 = v17;
        v42 = result;
        sub_C8D5F0(a1, a1 + 2, v11 + (v5 >> 4), 16);
        LODWORD(v11) = *((_DWORD *)a1 + 2);
        LODWORD(v25) = v28;
        v5 = v29;
        v23 = v32;
        v17 = v36;
        result = v42;
        v22 = (__m128i *)(*a1 + 16LL * (unsigned int)v11);
      }
      if ( v24 != v17 )
      {
        v26 = v24;
        v27 = (__m128i *)((char *)v22 + (char *)v17 - (char *)v24);
        do
        {
          if ( v22 )
            *v22 = _mm_loadu_si128(v26);
          ++v22;
          ++v26;
        }
        while ( v27 != v22 );
        LODWORD(v11) = *((_DWORD *)a1 + 2);
      }
      *((_DWORD *)a1 + 2) = v11 + v25;
      if ( v24 != v9 )
      {
        v40 = v5;
        result = (char *)memmove((char *)v17 - (v23 - (_QWORD)result), v9, v23 - (_QWORD)result);
        v5 = v40;
      }
      if ( v10 != a4 )
        return (char *)memmove(v9, v10, v5);
    }
    else
    {
      v21 = v6 + v11;
      *((_DWORD *)a1 + 2) = v21;
      if ( v9 != v17 )
      {
        v30 = (v16 - (__int64)result) >> 4;
        v33 = v17;
        v37 = v16 - (_QWORD)result;
        result = (char *)memcpy((void *)(16LL * v21 - v18 + v12), v9, v18);
        v19 = v30;
        v17 = v33;
        v18 = v37;
      }
      if ( v19 )
      {
        result = 0;
        do
        {
          *(__m128i *)&result[(_QWORD)v9] = _mm_loadu_si128((const __m128i *)&result[(_QWORD)v10]);
          result += 16;
          --v20;
        }
        while ( v20 );
        v10 += v18;
      }
      if ( a4 != v10 )
        return (char *)memcpy(v17, v10, a4 - v10);
    }
  }
  return result;
}
