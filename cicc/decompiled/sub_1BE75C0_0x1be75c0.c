// Function: sub_1BE75C0
// Address: 0x1be75c0
//
__int64 *__fastcall sub_1BE75C0(__int64 a1)
{
  __m128i *v2; // rsi
  __int64 *result; // rax
  __int64 v4; // rbx
  __int64 *v5; // rax
  char v6; // dl
  __int64 v7; // rax
  __int64 *v8; // rcx
  unsigned int v9; // edi
  __int64 *v10; // rsi
  __m128i v11; // [rsp+0h] [rbp-20h] BYREF

LABEL_1:
  v2 = *(__m128i **)(a1 + 112);
  while ( 1 )
  {
    result = (__int64 *)v2[-1].m128i_i64[1];
    if ( result == (__int64 *)(*(_QWORD *)(v2[-1].m128i_i64[0] + 80) + 8LL * *(unsigned int *)(v2[-1].m128i_i64[0] + 88)) )
      return result;
    v2[-1].m128i_i64[1] = (__int64)(result + 1);
    v4 = *result;
    v5 = *(__int64 **)(a1 + 8);
    if ( *(__int64 **)(a1 + 16) == v5 )
    {
      v8 = &v5[*(unsigned int *)(a1 + 28)];
      v9 = *(_DWORD *)(a1 + 28);
      if ( v5 != v8 )
      {
        v10 = 0;
        while ( v4 != *v5 )
        {
          if ( *v5 == -2 )
          {
            v10 = v5;
            if ( v5 + 1 == v8 )
              goto LABEL_15;
            ++v5;
          }
          else if ( v8 == ++v5 )
          {
            if ( !v10 )
              goto LABEL_18;
LABEL_15:
            *v10 = v4;
            v2 = *(__m128i **)(a1 + 112);
            --*(_DWORD *)(a1 + 32);
            ++*(_QWORD *)a1;
            goto LABEL_5;
          }
        }
        goto LABEL_1;
      }
LABEL_18:
      if ( v9 < *(_DWORD *)(a1 + 24) )
      {
        *(_DWORD *)(a1 + 28) = v9 + 1;
        *v8 = v4;
        v2 = *(__m128i **)(a1 + 112);
        ++*(_QWORD *)a1;
        goto LABEL_5;
      }
    }
    sub_16CCBA0(a1, v4);
    v2 = *(__m128i **)(a1 + 112);
    if ( v6 )
    {
LABEL_5:
      v7 = *(_QWORD *)(v4 + 80);
      v11.m128i_i64[0] = v4;
      v11.m128i_i64[1] = v7;
      if ( v2 == *(__m128i **)(a1 + 120) )
      {
        sub_1BE7440((const __m128i **)(a1 + 104), v2, &v11);
        v2 = *(__m128i **)(a1 + 112);
      }
      else
      {
        if ( v2 )
        {
          *v2 = _mm_loadu_si128(&v11);
          v2 = *(__m128i **)(a1 + 112);
        }
        *(_QWORD *)(a1 + 112) = ++v2;
      }
    }
  }
}
