// Function: sub_FD5CC0
// Address: 0xfd5cc0
//
__m128i *__fastcall sub_FD5CC0(__int64 a1, __int64 a2, const __m128i *a3, char a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // r8
  __m128i *result; // rax
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 v15; // rsi
  signed __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // rdi
  const void *v19; // rsi
  __int8 *v20; // r14
  __int64 v21; // [rsp+8h] [rbp-38h]

  if ( (*(_BYTE *)(a1 + 67) & 0x40) == 0 && !a4 )
  {
    v13 = *(_QWORD *)(a1 + 24);
    v14 = 48LL * *(unsigned int *)(a1 + 32);
    v15 = v13 + v14;
    v16 = 0xAAAAAAAAAAAAAAABLL * (v14 >> 4);
    if ( v16 >> 2 )
    {
      v21 = v13 + 192 * (v16 >> 2);
      while ( (unsigned __int8)sub_CF4D50(**(_QWORD **)a2, (__int64)a3, v13, *(_QWORD *)a2 + 8LL, 0) != 3 )
      {
        v17 = v13;
        v13 += 48;
        if ( (unsigned __int8)sub_CF4D50(**(_QWORD **)a2, (__int64)a3, v13, *(_QWORD *)a2 + 8LL, 0) == 3 )
          break;
        v13 = v17 + 96;
        if ( (unsigned __int8)sub_CF4D50(**(_QWORD **)a2, (__int64)a3, v17 + 96, *(_QWORD *)a2 + 8LL, 0) == 3 )
          break;
        v13 = v17 + 144;
        if ( (unsigned __int8)sub_CF4D50(**(_QWORD **)a2, (__int64)a3, v17 + 144, *(_QWORD *)a2 + 8LL, 0) == 3 )
          break;
        v13 = v17 + 192;
        if ( v21 == v17 + 192 )
        {
          v16 = 0xAAAAAAAAAAAAAAABLL * ((v15 - v13) >> 4);
          goto LABEL_19;
        }
      }
LABEL_12:
      if ( v13 != v15 )
        goto LABEL_3;
      goto LABEL_13;
    }
LABEL_19:
    if ( v16 != 2 )
    {
      if ( v16 != 3 )
      {
        if ( v16 != 1 )
          goto LABEL_13;
        goto LABEL_22;
      }
      if ( (unsigned __int8)sub_CF4D50(**(_QWORD **)a2, (__int64)a3, v13, *(_QWORD *)a2 + 8LL, 0) == 3 )
        goto LABEL_12;
      v13 += 48;
    }
    if ( (unsigned __int8)sub_CF4D50(**(_QWORD **)a2, (__int64)a3, v13, *(_QWORD *)a2 + 8LL, 0) == 3 )
      goto LABEL_12;
    v13 += 48;
LABEL_22:
    if ( (unsigned __int8)sub_CF4D50(**(_QWORD **)a2, (__int64)a3, v13, *(_QWORD *)a2 + 8LL, 0) == 3 )
      goto LABEL_12;
LABEL_13:
    *(_BYTE *)(a1 + 67) |= 0x40u;
  }
LABEL_3:
  v9 = *(unsigned int *)(a1 + 32);
  v10 = *(_QWORD *)(a1 + 24);
  v11 = v9 + 1;
  if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 36) )
  {
    v18 = a1 + 24;
    v19 = (const void *)(a1 + 40);
    if ( v10 > (unsigned __int64)a3 || (unsigned __int64)a3 >= v10 + 48 * v9 )
    {
      sub_C8D5F0(v18, v19, v11, 0x30u, v11, a6);
      v10 = *(_QWORD *)(a1 + 24);
      v9 = *(unsigned int *)(a1 + 32);
    }
    else
    {
      v20 = &a3->m128i_i8[-v10];
      sub_C8D5F0(v18, v19, v11, 0x30u, v11, a6);
      v10 = *(_QWORD *)(a1 + 24);
      v9 = *(unsigned int *)(a1 + 32);
      a3 = (const __m128i *)&v20[v10];
    }
  }
  result = (__m128i *)(v10 + 48 * v9);
  *result = _mm_loadu_si128(a3);
  result[1] = _mm_loadu_si128(a3 + 1);
  result[2] = _mm_loadu_si128(a3 + 2);
  ++*(_DWORD *)(a1 + 32);
  ++*(_DWORD *)(a2 + 56);
  return result;
}
