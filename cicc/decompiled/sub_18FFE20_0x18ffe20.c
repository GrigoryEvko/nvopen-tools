// Function: sub_18FFE20
// Address: 0x18ffe20
//
__int64 __fastcall sub_18FFE20(__int64 *a1)
{
  __int64 result; // rax
  __m128i *v2; // rbx
  char v4; // r8
  __int64 *v5; // rax
  int v6; // esi
  int v7; // edx
  unsigned int v8; // esi
  __int64 v9; // rdx
  __int64 v10; // r12
  __int64 *m128i_i64; // rsi
  __int64 v12; // rdi
  char v13; // r8
  __int64 *v14; // rax
  __int64 *v15[7]; // [rsp+8h] [rbp-38h] BYREF

  result = *a1;
  *(_QWORD *)(*a1 + 32) = a1[1];
  v2 = (__m128i *)a1[2];
  if ( v2 )
  {
    while ( 1 )
    {
      v10 = *a1;
      m128i_i64 = v2[1].m128i_i64;
      v12 = *a1;
      if ( !v2->m128i_i64[1] )
      {
        v13 = sub_18FEB70(v12, m128i_i64, v15);
        v14 = v15[0];
        if ( v13 )
        {
          *v15[0] = -16;
          v14[1] = 0;
          v14[2] = 0;
          v14[3] = 0;
          v14[4] = 0;
          --*(_DWORD *)(v10 + 16);
          ++*(_DWORD *)(v10 + 20);
        }
        goto LABEL_10;
      }
      v4 = sub_18FEB70(v12, m128i_i64, v15);
      v5 = v15[0];
      if ( !v4 )
        break;
LABEL_9:
      v5[5] = v2->m128i_i64[1];
LABEL_10:
      a1[2] = v2->m128i_i64[0];
      result = *a1;
      v2->m128i_i64[0] = *(_QWORD *)(*a1 + 40);
      *(_QWORD *)(result + 40) = v2;
      v2 = (__m128i *)a1[2];
      if ( !v2 )
        return result;
    }
    v6 = *(_DWORD *)(v10 + 16);
    ++*(_QWORD *)v10;
    v7 = v6 + 1;
    v8 = *(_DWORD *)(v10 + 24);
    if ( 4 * v7 >= 3 * v8 )
    {
      v8 *= 2;
    }
    else if ( v8 - *(_DWORD *)(v10 + 20) - v7 > v8 >> 3 )
    {
      goto LABEL_6;
    }
    sub_18FFA60(v10, v8);
    sub_18FEB70(v10, v2[1].m128i_i64, v15);
    v5 = v15[0];
    v7 = *(_DWORD *)(v10 + 16) + 1;
LABEL_6:
    *(_DWORD *)(v10 + 16) = v7;
    if ( *v5 != -8 || v5[1] || v5[2] || v5[3] || v5[4] )
      --*(_DWORD *)(v10 + 20);
    *(__m128i *)v5 = _mm_loadu_si128(v2 + 1);
    *((__m128i *)v5 + 1) = _mm_loadu_si128(v2 + 2);
    v9 = v2[3].m128i_i64[0];
    v5[5] = 0;
    v5[4] = v9;
    goto LABEL_9;
  }
  return result;
}
