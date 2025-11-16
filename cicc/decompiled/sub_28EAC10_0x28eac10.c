// Function: sub_28EAC10
// Address: 0x28eac10
//
__int64 __fastcall sub_28EAC10(__int64 a1, __m128i *a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r13
  __int64 v10; // r15
  unsigned __int64 v11; // rsi
  __int64 result; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // rdi
  __m128i *v15; // rcx
  const __m128i *v16; // rdx

  v7 = a3;
  v10 = *(_QWORD *)a1;
  v11 = *(unsigned int *)(a1 + 12);
  result = *(unsigned int *)(a1 + 8);
  v13 = result + 1;
  v14 = 16 * result;
  v15 = (__m128i *)(v10 + 16 * result);
  if ( a2 == v15 )
  {
    if ( v13 > v11 )
    {
      result = sub_C8D5F0(a1, (const void *)(a1 + 16), v13, 0x10u, a5, a6);
      v15 = (__m128i *)(*(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8));
    }
    v15->m128i_i64[0] = v7;
    v15->m128i_i64[1] = a4;
    ++*(_DWORD *)(a1 + 8);
  }
  else
  {
    if ( v13 > v11 )
    {
      sub_C8D5F0(a1, (const void *)(a1 + 16), v13, 0x10u, a5, a6);
      result = *(unsigned int *)(a1 + 8);
      v14 = 16 * result;
      a2 = (__m128i *)((char *)a2 + *(_QWORD *)a1 - v10);
      v10 = *(_QWORD *)a1;
      v15 = (__m128i *)(*(_QWORD *)a1 + 16 * result);
    }
    v16 = (const __m128i *)(v10 + v14 - 16);
    if ( v15 )
    {
      *v15 = _mm_loadu_si128(v16);
      v10 = *(_QWORD *)a1;
      result = *(unsigned int *)(a1 + 8);
      v14 = 16 * result;
      v16 = (const __m128i *)(*(_QWORD *)a1 + 16 * result - 16);
    }
    if ( a2 != v16 )
    {
      memmove((void *)(v10 + v14 - ((char *)v16 - (char *)a2)), a2, (char *)v16 - (char *)a2);
      LODWORD(result) = *(_DWORD *)(a1 + 8);
    }
    result = (unsigned int)(result + 1);
    *(_DWORD *)(a1 + 8) = result;
    a2->m128i_i32[0] = v7;
    a2->m128i_i64[1] = a4;
  }
  return result;
}
