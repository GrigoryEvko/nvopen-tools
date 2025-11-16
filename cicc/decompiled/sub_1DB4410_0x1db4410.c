// Function: sub_1DB4410
// Address: 0x1db4410
//
_QWORD *__fastcall sub_1DB4410(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  _QWORD *result; // rax
  int v9; // r8d
  int v10; // r9d
  __int64 v11; // rsi
  __int64 v12; // rdx
  unsigned __int64 v13; // rdx
  __m128i *v14; // rbx
  __int64 v15; // rsi
  unsigned int v16; // eax
  __int64 v17; // rdi
  __m128i *v18; // rcx
  unsigned __int64 v19; // r8
  const __m128i *v20; // rdx
  __int64 v21; // rax
  __m128i *v22; // rdx
  __m128i v23; // xmm0
  unsigned int v24; // edi
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int8 *v27; // rbx
  _QWORD *v28; // [rsp+8h] [rbp-48h]
  __m128i v29; // [rsp+10h] [rbp-40h] BYREF
  _QWORD *v30; // [rsp+20h] [rbp-30h]
  char v31; // [rsp+28h] [rbp-28h] BYREF

  result = (_QWORD *)sub_1DB3C70((__int64 *)a1, a2);
  v11 = result[2];
  if ( *result == a2 )
  {
    if ( a3 == result[1] )
    {
      v24 = *(_DWORD *)(a1 + 8);
      v25 = *(_QWORD *)a1;
      v26 = *(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8);
      if ( a4 )
      {
        if ( v26 == v25 )
        {
LABEL_25:
          v28 = result;
          sub_1DB4220(a1, v11);
          result = v28;
          v24 = *(_DWORD *)(a1 + 8);
          v26 = *(_QWORD *)a1 + 24LL * v24;
        }
        else
        {
          while ( result == (_QWORD *)v25 || *(_QWORD *)(v25 + 16) != v11 )
          {
            v25 += 24;
            if ( v25 == v26 )
              goto LABEL_25;
          }
        }
      }
      if ( result + 3 != (_QWORD *)v26 )
      {
        result = memmove(result, result + 3, v26 - (_QWORD)(result + 3));
        v24 = *(_DWORD *)(a1 + 8);
      }
      *(_DWORD *)(a1 + 8) = v24 - 1;
    }
    else
    {
      *result = a3;
    }
  }
  else
  {
    v12 = result[1];
    result[1] = a2;
    if ( a3 != v12 )
    {
      v29.m128i_i64[1] = v12;
      v13 = *(unsigned int *)(a1 + 8);
      v14 = (__m128i *)(result + 3);
      v30 = (_QWORD *)v11;
      v15 = *(_QWORD *)a1;
      v29.m128i_i64[0] = a3;
      v16 = v13;
      v17 = 24 * v13;
      v18 = (__m128i *)(v15 + 24 * v13);
      if ( v18 == v14 )
      {
        if ( (unsigned int)v13 >= *(_DWORD *)(a1 + 12) )
        {
          sub_16CD150(a1, (const void *)(a1 + 16), 0, 24, v9, v10);
          v18 = (__m128i *)(*(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8));
        }
        result = v30;
        *v18 = _mm_loadu_si128(&v29);
        v18[1].m128i_i64[0] = (__int64)result;
        ++*(_DWORD *)(a1 + 8);
      }
      else
      {
        v19 = *(unsigned int *)(a1 + 12);
        if ( v13 >= v19 )
        {
          v27 = &v14->m128i_i8[-v15];
          sub_16CD150(a1, (const void *)(a1 + 16), 0, 24, v19, v10);
          v15 = *(_QWORD *)a1;
          v14 = (__m128i *)&v27[*(_QWORD *)a1];
          v16 = *(_DWORD *)(a1 + 8);
          v17 = 24LL * v16;
          v18 = (__m128i *)(*(_QWORD *)a1 + v17);
        }
        v20 = (const __m128i *)(v15 + v17 - 24);
        if ( v18 )
        {
          *v18 = _mm_loadu_si128(v20);
          v18[1].m128i_i64[0] = v20[1].m128i_i64[0];
          v15 = *(_QWORD *)a1;
          v16 = *(_DWORD *)(a1 + 8);
          v17 = 24LL * v16;
          v20 = (const __m128i *)(*(_QWORD *)a1 + v17 - 24);
        }
        if ( v20 != v14 )
        {
          memmove((void *)(v15 + v17 - ((char *)v20 - (char *)v14)), v14, (char *)v20 - (char *)v14);
          v16 = *(_DWORD *)(a1 + 8);
        }
        v21 = v16 + 1;
        v22 = &v29;
        *(_DWORD *)(a1 + 8) = v21;
        if ( v14 <= &v29 && (unsigned __int64)&v29 < *(_QWORD *)a1 + 24 * v21 )
          v22 = (__m128i *)&v31;
        result = (_QWORD *)v22[1].m128i_i64[0];
        v23 = _mm_loadu_si128(v22);
        v14[1].m128i_i64[0] = (__int64)result;
        *v14 = v23;
      }
    }
  }
  return result;
}
