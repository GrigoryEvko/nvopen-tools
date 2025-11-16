// Function: sub_18FFC60
// Address: 0x18ffc60
//
__int64 __fastcall sub_18FFC60(__int64 a1, __int64 a2, __m128i *a3, int *a4)
{
  char v7; // al
  __int64 *v8; // r13
  __int64 result; // rax
  __int64 v10; // r8
  __int64 v11; // rcx
  __m128i v12; // xmm0
  __m128i v13; // xmm1
  int v14; // edx
  unsigned int v15; // esi
  int v16; // eax
  int v17; // eax
  __int64 v18; // rax
  __int64 v19; // [rsp+0h] [rbp-50h]
  __int64 v20; // [rsp+8h] [rbp-48h]
  __int64 *v21[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = sub_18FEB70(a1, a3->m128i_i64, v21);
  v8 = v21[0];
  if ( !v7 )
  {
    v15 = *(_DWORD *)(a1 + 24);
    v16 = *(_DWORD *)(a1 + 16);
    ++*(_QWORD *)a1;
    v17 = v16 + 1;
    if ( 4 * v17 >= 3 * v15 )
    {
      v15 *= 2;
    }
    else if ( v15 - *(_DWORD *)(a1 + 20) - v17 > v15 >> 3 )
    {
LABEL_7:
      *(_DWORD *)(a1 + 16) = v17;
      if ( *v8 != -8 || v8[1] || v8[2] || v8[3] || v8[4] )
        --*(_DWORD *)(a1 + 20);
      v10 = 0;
      *(__m128i *)v8 = _mm_loadu_si128(a3);
      *((__m128i *)v8 + 1) = _mm_loadu_si128(a3 + 1);
      v18 = a3[2].m128i_i64[0];
      v8[5] = 0;
      v8[4] = v18;
      result = *(_QWORD *)(a1 + 40);
      v11 = *(_QWORD *)(a2 + 16);
      if ( result )
        goto LABEL_3;
LABEL_10:
      v19 = v10;
      v20 = v11;
      result = sub_145CBF0((__int64 *)(a1 + 48), 64, 8);
      v10 = v19;
      v11 = v20;
      goto LABEL_4;
    }
    sub_18FFA60(a1, v15);
    sub_18FEB70(a1, a3->m128i_i64, v21);
    v8 = v21[0];
    v17 = *(_DWORD *)(a1 + 16) + 1;
    goto LABEL_7;
  }
  result = *(_QWORD *)(a1 + 40);
  v10 = v21[0][5];
  v11 = *(_QWORD *)(a2 + 16);
  if ( !result )
    goto LABEL_10;
LABEL_3:
  *(_QWORD *)(a1 + 40) = *(_QWORD *)result;
LABEL_4:
  v12 = _mm_loadu_si128(a3);
  v13 = _mm_loadu_si128(a3 + 1);
  *(_QWORD *)(result + 48) = a3[2].m128i_i64[0];
  v14 = *a4;
  *(__m128i *)(result + 16) = v12;
  *(_DWORD *)(result + 56) = v14;
  *(_QWORD *)result = v11;
  *(_QWORD *)(result + 8) = v10;
  *(__m128i *)(result + 32) = v13;
  v8[5] = result;
  *(_QWORD *)(a2 + 16) = result;
  return result;
}
