// Function: sub_2739AD0
// Address: 0x2739ad0
//
__int64 __fastcall sub_2739AD0(__int64 a1, __m128i *a2, char *a3, char *a4)
{
  signed __int64 v5; // r10
  __m128i *v8; // r13
  char *v9; // r12
  __int64 v10; // r9
  __int64 v11; // rdi
  __int64 v12; // rsi
  unsigned __int64 v13; // rcx
  unsigned __int64 v14; // rdx
  __int64 result; // rax
  __m128i *v16; // r8
  size_t v17; // rdx
  unsigned __int64 v18; // rcx
  unsigned __int64 v19; // rbx
  unsigned int v20; // esi
  __m128i *v21; // rdx
  __int64 v22; // r9
  const __m128i *v23; // rbx
  unsigned __int64 v24; // rax
  const __m128i *v25; // rcx
  unsigned __int64 v26; // [rsp+8h] [rbp-58h]
  signed __int64 v27; // [rsp+10h] [rbp-50h]
  unsigned __int64 v28; // [rsp+18h] [rbp-48h]
  int v29; // [rsp+18h] [rbp-48h]
  __int64 v30; // [rsp+18h] [rbp-48h]
  __m128i *v31; // [rsp+20h] [rbp-40h]
  signed __int64 v32; // [rsp+20h] [rbp-40h]
  signed __int64 v33; // [rsp+20h] [rbp-40h]
  int v34; // [rsp+20h] [rbp-40h]
  __m128i *v35; // [rsp+20h] [rbp-40h]
  __int8 *v36; // [rsp+28h] [rbp-38h]
  size_t v37; // [rsp+28h] [rbp-38h]
  int v38; // [rsp+28h] [rbp-38h]
  signed __int64 v39; // [rsp+28h] [rbp-38h]

  v5 = a4 - a3;
  v8 = a2;
  v9 = a3;
  v10 = 0xAAAAAAAAAAAAAAABLL * ((a4 - a3) >> 3);
  v11 = *(_QWORD *)a1;
  v12 = *(unsigned int *)(a1 + 8);
  v13 = *(unsigned int *)(a1 + 12);
  v14 = v12 + v10;
  v36 = &a2->m128i_i8[-v11];
  result = 24 * v12;
  v16 = (__m128i *)(v11 + 24 * v12);
  if ( v8 == v16 )
  {
    if ( v14 > v13 )
    {
      v34 = v10;
      v39 = v5;
      sub_C8D5F0(a1, (const void *)(a1 + 16), v14, 0x18u, (__int64)v16, v10);
      v12 = *(unsigned int *)(a1 + 8);
      result = *(_QWORD *)a1;
      LODWORD(v10) = v34;
      v5 = v39;
      v16 = (__m128i *)(*(_QWORD *)a1 + 24 * v12);
    }
    if ( v9 != a4 )
    {
      v38 = v10;
      result = (__int64)memcpy(v16, v9, v5);
      LODWORD(v12) = *(_DWORD *)(a1 + 8);
      LODWORD(v10) = v38;
    }
    *(_DWORD *)(a1 + 8) = v10 + v12;
  }
  else
  {
    if ( v14 > v13 )
    {
      v29 = v10;
      v32 = v5;
      sub_C8D5F0(a1, (const void *)(a1 + 16), v14, 0x18u, (__int64)v16, v10);
      v11 = *(_QWORD *)a1;
      v12 = *(unsigned int *)(a1 + 8);
      LODWORD(v10) = v29;
      v8 = (__m128i *)&v36[*(_QWORD *)a1];
      v5 = v32;
      result = 24 * v12;
      v16 = (__m128i *)(*(_QWORD *)a1 + 24 * v12);
    }
    v17 = result - (_QWORD)v36;
    v18 = 0xAAAAAAAAAAAAAAABLL * ((result - (__int64)v36) >> 3);
    v19 = v18;
    if ( result - (__int64)v36 >= (unsigned __int64)v5 )
    {
      v21 = v16;
      v22 = result - v5;
      v23 = (const __m128i *)(v11 + result - v5);
      v24 = 0xAAAAAAAAAAAAAAABLL * (v5 >> 3);
      if ( v12 - 0x5555555555555555LL * (v5 >> 3) > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        v26 = 0xAAAAAAAAAAAAAAABLL * (v5 >> 3);
        v27 = v5;
        v30 = v22;
        v35 = v16;
        sub_C8D5F0(a1, (const void *)(a1 + 16), v12 - 0x5555555555555555LL * (v5 >> 3), 0x18u, (__int64)v16, v22);
        v12 = *(unsigned int *)(a1 + 8);
        v24 = v26;
        v5 = v27;
        v22 = v30;
        v16 = v35;
        v21 = (__m128i *)(*(_QWORD *)a1 + 24 * v12);
      }
      if ( v23 != v16 )
      {
        v25 = v23;
        do
        {
          if ( v21 )
          {
            *v21 = _mm_loadu_si128(v25);
            v21[1].m128i_i64[0] = v25[1].m128i_i64[0];
          }
          v25 = (const __m128i *)((char *)v25 + 24);
          v21 = (__m128i *)((char *)v21 + 24);
        }
        while ( v25 != v16 );
        v12 = *(unsigned int *)(a1 + 8);
      }
      result = v12 + v24;
      *(_DWORD *)(a1 + 8) = result;
      if ( v23 != v8 )
      {
        v33 = v5;
        result = (__int64)memmove((char *)v16 - (v22 - (_QWORD)v36), v8, v22 - (_QWORD)v36);
        v5 = v33;
      }
      if ( v9 != a4 )
        return (__int64)memmove(v8, v9, v5);
    }
    else
    {
      v20 = v10 + v12;
      *(_DWORD *)(a1 + 8) = v20;
      if ( v8 != v16 )
      {
        v28 = 0xAAAAAAAAAAAAAAABLL * ((result - (__int64)v36) >> 3);
        v31 = v16;
        v37 = result - (_QWORD)v36;
        result = (__int64)memcpy((void *)(v11 + 24LL * v20 - v17), v8, v17);
        v18 = v28;
        v16 = v31;
        v17 = v37;
      }
      if ( v18 )
      {
        result = 0;
        do
        {
          *(__m128i *)((char *)v8 + result) = _mm_loadu_si128((const __m128i *)&v9[result]);
          v8[1].m128i_i8[result] = v9[result + 16];
          result += 24;
          --v19;
        }
        while ( v19 );
        v9 += v17;
      }
      if ( a4 != v9 )
        return (__int64)memcpy(v16, v9, a4 - v9);
    }
  }
  return result;
}
