// Function: sub_277FFC0
// Address: 0x277ffc0
//
__int64 **__fastcall sub_277FFC0(_QWORD *a1)
{
  __m128i *v1; // rbx
  __int64 **result; // rax
  bool v3; // zf
  __int64 *v4; // rax
  int v5; // esi
  int v6; // edx
  unsigned int v7; // esi
  __m128i v8; // xmm2
  __int64 v9; // r12
  __int64 v10; // r15
  _QWORD *v11; // rax
  __int64 *v12; // [rsp+10h] [rbp-40h] BYREF
  __int64 *v13; // [rsp+18h] [rbp-38h] BYREF

  *(_QWORD *)(*a1 + 136LL) = a1[1];
  v1 = (__m128i *)a1[2];
  result = &v12;
  if ( v1 )
  {
    while ( 1 )
    {
      v9 = *a1;
      v10 = *a1 + 104LL;
      if ( !v1->m128i_i64[1] )
      {
        v11 = sub_277D3C0(*a1 + 104LL, v1[1].m128i_i64);
        if ( v11 )
        {
          *v11 = -8192;
          v11[1] = -4;
          v11[2] = 0;
          v11[3] = 0;
          v11[4] = 0;
          v11[5] = 0;
          --*(_DWORD *)(v9 + 120);
          ++*(_DWORD *)(v9 + 124);
          v9 = *a1;
        }
        goto LABEL_10;
      }
      v3 = (unsigned __int8)sub_277DBB0(*a1 + 104LL, v1[1].m128i_i64, &v12) == 0;
      v4 = v12;
      if ( v3 )
        break;
LABEL_9:
      v4[6] = v1->m128i_i64[1];
      v9 = *a1;
LABEL_10:
      a1[2] = v1->m128i_i64[0];
      result = *(__int64 ***)v9;
      v1->m128i_i64[0] = *(_QWORD *)v9;
      *(_QWORD *)v9 = v1;
      v1 = (__m128i *)a1[2];
      if ( !v1 )
        return result;
    }
    v5 = *(_DWORD *)(v9 + 120);
    ++*(_QWORD *)(v9 + 104);
    v13 = v4;
    v6 = v5 + 1;
    v7 = *(_DWORD *)(v9 + 128);
    if ( 4 * v6 >= 3 * v7 )
    {
      v7 *= 2;
    }
    else if ( v7 - *(_DWORD *)(v9 + 124) - v6 > v7 >> 3 )
    {
      goto LABEL_6;
    }
    sub_277FD80(v10, v7);
    sub_277DBB0(v10, v1[1].m128i_i64, &v13);
    v6 = *(_DWORD *)(v9 + 120) + 1;
    v4 = v13;
LABEL_6:
    *(_DWORD *)(v9 + 120) = v6;
    if ( *v4 != -4096 || v4[1] != -3 || v4[2] || v4[3] || v4[4] || v4[5] )
      --*(_DWORD *)(v9 + 124);
    *(__m128i *)v4 = _mm_loadu_si128(v1 + 1);
    *((__m128i *)v4 + 1) = _mm_loadu_si128(v1 + 2);
    v8 = _mm_loadu_si128(v1 + 3);
    v4[6] = 0;
    *((__m128i *)v4 + 2) = v8;
    goto LABEL_9;
  }
  return result;
}
