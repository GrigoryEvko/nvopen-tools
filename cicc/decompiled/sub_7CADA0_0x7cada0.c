// Function: sub_7CADA0
// Address: 0x7cada0
//
__m128i *__fastcall sub_7CADA0(unsigned __int64 a1, const __m128i *a2)
{
  __m128i *v4; // r12
  unsigned __int64 v5; // r14
  int v6; // esi
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // rdi
  __m128i **v10; // rax
  __m128i **v11; // r14
  __m128i *result; // rax
  __int64 v13[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( !dword_4F07590
    && (dword_4F04C44 != -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0)
    && (!dword_4F07570 || dword_4F04C64 == -1 || *(char *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) >= 0) )
  {
    return 0;
  }
  v4 = *(__m128i **)(a1 + 80);
  if ( (*(_BYTE *)(a1 + 89) & 1) != 0 || !*(_QWORD *)(a1 + 8) )
  {
    while ( v4 )
    {
      if ( sub_7BEF80((__int64)v4, (__int64)a2) )
        return v4;
      v4 = (__m128i *)v4->m128i_i64[0];
    }
    result = (__m128i *)sub_727560();
    *result = _mm_loadu_si128(a2);
    result[1] = _mm_loadu_si128(a2 + 1);
    result[2].m128i_i64[0] = a2[2].m128i_i64[0];
    result->m128i_i64[0] = *(_QWORD *)(a1 + 80);
    *(_QWORD *)(a1 + 80) = result;
  }
  else
  {
    v5 = a1 >> 3;
    if ( v4 )
    {
      v6 = *(_DWORD *)(qword_4F08490 + 8);
      v7 = v6 & v5;
      v8 = (__int64 *)(*(_QWORD *)qword_4F08490 + 16LL * (v6 & (unsigned int)v5));
      v9 = *v8;
      if ( *v8 == a1 )
      {
LABEL_24:
        v9 = v8[1];
      }
      else
      {
        while ( v9 )
        {
          v7 = v6 & (v7 + 1);
          v8 = (__int64 *)(*(_QWORD *)qword_4F08490 + 16LL * v7);
          v9 = *v8;
          if ( a1 == *v8 )
            goto LABEL_24;
        }
      }
      v13[0] = v9;
    }
    else
    {
      v13[0] = sub_881A70(0, 11, 40, 41);
      sub_7CAB40(qword_4F08490, a1, v13, a1 >> 3);
      v9 = v13[0];
    }
    v10 = (__m128i **)sub_881B20(v9, a2, 1);
    v4 = *v10;
    v11 = v10;
    if ( !*v10 )
    {
      v4 = (__m128i *)sub_727560();
      *v4 = _mm_loadu_si128(a2);
      v4[1] = _mm_loadu_si128(a2 + 1);
      v4[2].m128i_i64[0] = a2[2].m128i_i64[0];
      *v11 = v4;
      v4->m128i_i64[0] = *(_QWORD *)(a1 + 80);
      *(_QWORD *)(a1 + 80) = v4;
    }
    return v4;
  }
  return result;
}
