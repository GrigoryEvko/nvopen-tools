// Function: sub_27808D0
// Address: 0x27808d0
//
unsigned __int64 __fastcall sub_27808D0(__int64 a1, __int64 a2, __m128i *a3, int *a4)
{
  char v8; // al
  __int64 v9; // rdi
  __int64 v10; // r8
  __int64 *v11; // r12
  unsigned __int64 result; // rax
  __int64 v13; // r9
  __m128i v14; // xmm0
  __m128i v15; // xmm1
  __m128i v16; // xmm2
  int v17; // edx
  unsigned int v18; // esi
  int v19; // eax
  __int64 *v20; // r12
  int v21; // eax
  __m128i v22; // xmm5
  __int64 v23; // rcx
  __int64 v24; // [rsp+0h] [rbp-50h]
  __int64 v25; // [rsp+8h] [rbp-48h]
  __int64 *v26; // [rsp+10h] [rbp-40h] BYREF
  __int64 *v27; // [rsp+18h] [rbp-38h] BYREF

  v8 = sub_277DBB0(a1 + 104, a3->m128i_i64, &v26);
  v9 = a1 + 104;
  if ( !v8 )
  {
    v18 = *(_DWORD *)(a1 + 128);
    v19 = *(_DWORD *)(a1 + 120);
    v20 = v26;
    ++*(_QWORD *)(a1 + 104);
    v21 = v19 + 1;
    v27 = v20;
    if ( 4 * v21 >= 3 * v18 )
    {
      v18 *= 2;
    }
    else if ( v18 - *(_DWORD *)(a1 + 124) - v21 > v18 >> 3 )
    {
LABEL_7:
      *(_DWORD *)(a1 + 120) = v21;
      if ( *v20 != -4096 || v20[1] != -3 || v20[2] || v20[3] || v20[4] || v20[5] )
        --*(_DWORD *)(a1 + 124);
      v11 = v20 + 6;
      *((__m128i *)v11 - 3) = _mm_loadu_si128(a3);
      *((__m128i *)v11 - 2) = _mm_loadu_si128(a3 + 1);
      v22 = _mm_loadu_si128(a3 + 2);
      *v11 = 0;
      *((__m128i *)v11 - 1) = v22;
      result = *(_QWORD *)a1;
      v13 = *v11;
      v10 = *(_QWORD *)(a2 + 16);
      if ( *(_QWORD *)a1 )
        goto LABEL_3;
      goto LABEL_10;
    }
    sub_277FD80(v9, v18);
    sub_277DBB0(v9, a3->m128i_i64, &v27);
    v20 = v27;
    v21 = *(_DWORD *)(a1 + 120) + 1;
    goto LABEL_7;
  }
  v10 = *(_QWORD *)(a2 + 16);
  v11 = v26 + 6;
  result = *(_QWORD *)a1;
  v13 = v26[6];
  if ( *(_QWORD *)a1 )
  {
LABEL_3:
    *(_QWORD *)a1 = *(_QWORD *)result;
    goto LABEL_4;
  }
LABEL_10:
  v23 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 88) += 72LL;
  result = (v23 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(_QWORD *)(a1 + 16) >= result + 72 && v23 )
  {
    *(_QWORD *)(a1 + 8) = result + 72;
    if ( !result )
    {
      MEMORY[0] = v10;
      BUG();
    }
  }
  else
  {
    v24 = v10;
    v25 = v13;
    result = sub_9D1E70(a1 + 8, 72, 72, 3);
    v10 = v24;
    v13 = v25;
  }
LABEL_4:
  v14 = _mm_loadu_si128(a3);
  v15 = _mm_loadu_si128(a3 + 1);
  v16 = _mm_loadu_si128(a3 + 2);
  v17 = *a4;
  *(_QWORD *)result = v10;
  *(_QWORD *)(result + 8) = v13;
  *(_DWORD *)(result + 64) = v17;
  *(__m128i *)(result + 16) = v14;
  *(__m128i *)(result + 32) = v15;
  *(__m128i *)(result + 48) = v16;
  *v11 = result;
  *(_QWORD *)(a2 + 16) = result;
  return result;
}
