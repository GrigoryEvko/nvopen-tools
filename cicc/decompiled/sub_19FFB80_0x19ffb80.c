// Function: sub_19FFB80
// Address: 0x19ffb80
//
void __fastcall sub_19FFB80(__int64 a1, __m128i *a2, const __m128i *a3, __int64 a4, __int64 a5, int a6)
{
  unsigned __int64 v9; // r8
  __int64 v10; // rdx
  unsigned __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rdi
  __m128i *v14; // rsi
  const __m128i *v15; // rcx
  __int64 v16; // rax
  __int8 *v17; // r12

  v9 = *(unsigned int *)(a1 + 8);
  v10 = *(_QWORD *)a1;
  v11 = *(unsigned int *)(a1 + 12);
  LODWORD(v12) = *(_DWORD *)(a1 + 8);
  v13 = 16 * v9;
  v14 = (__m128i *)(v10 + 16 * v9);
  if ( v14 == a2 )
  {
    if ( (unsigned int)v9 >= (unsigned int)v11 )
    {
      sub_16CD150(a1, (const void *)(a1 + 16), 0, 16, v9, a6);
      a2 = (__m128i *)(*(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8));
    }
    *a2 = _mm_loadu_si128(a3);
    ++*(_DWORD *)(a1 + 8);
  }
  else
  {
    if ( v9 >= v11 )
    {
      v17 = &a2->m128i_i8[-v10];
      sub_16CD150(a1, (const void *)(a1 + 16), 0, 16, v9, a6);
      v10 = *(_QWORD *)a1;
      v12 = *(unsigned int *)(a1 + 8);
      v13 = 16 * v12;
      a2 = (__m128i *)&v17[*(_QWORD *)a1];
      v14 = (__m128i *)(*(_QWORD *)a1 + 16 * v12);
    }
    v15 = (const __m128i *)(v10 + v13 - 16);
    if ( v14 )
    {
      *v14 = _mm_loadu_si128(v15);
      v10 = *(_QWORD *)a1;
      v12 = *(unsigned int *)(a1 + 8);
      v13 = 16 * v12;
      v15 = (const __m128i *)(*(_QWORD *)a1 + 16 * v12 - 16);
    }
    if ( a2 != v15 )
    {
      memmove((void *)(v10 + v13 - ((char *)v15 - (char *)a2)), a2, (char *)v15 - (char *)a2);
      LODWORD(v12) = *(_DWORD *)(a1 + 8);
    }
    v16 = (unsigned int)(v12 + 1);
    *(_DWORD *)(a1 + 8) = v16;
    if ( a3 >= a2 && (unsigned __int64)a3 < *(_QWORD *)a1 + 16 * v16 )
      ++a3;
    *a2 = _mm_loadu_si128(a3);
  }
}
