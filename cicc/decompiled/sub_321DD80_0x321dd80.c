// Function: sub_321DD80
// Address: 0x321dd80
//
unsigned __int64 __fastcall sub_321DD80(__int64 a1, __int64 a2)
{
  unsigned __int64 result; // rax
  __int64 *v3; // r8
  __int64 v4; // r9
  unsigned __int64 v5; // r10
  __int64 v6; // rcx
  __int64 *v7; // rdx
  int v8; // edx
  __int64 v9; // r9
  unsigned __int64 v10; // rdx
  const __m128i *v11; // rcx
  __m128i *v12; // r8
  const void *v13; // rsi
  char *v14; // rbx
  unsigned __int64 v15; // [rsp+0h] [rbp-40h]
  __int64 v16; // [rsp+10h] [rbp-30h] BYREF
  __int64 v17; // [rsp+18h] [rbp-28h]
  __int64 v18; // [rsp+20h] [rbp-20h]
  __int64 v19; // [rsp+28h] [rbp-18h]

  result = *(unsigned int *)(a1 + 8);
  v3 = *(__int64 **)a1;
  v4 = *(unsigned int *)(a1 + 152);
  if ( !*(_DWORD *)(a1 + 8) )
  {
    if ( *(_DWORD *)(a1 + 12) )
    {
      v8 = 0;
      if ( !v3 )
        goto LABEL_4;
      v7 = *(__int64 **)a1;
      v6 = 0;
LABEL_3:
      *v7 = a2;
      v7[1] = 0;
      v7[2] = v4;
      v7[3] = v6;
      v8 = *(_DWORD *)(a1 + 8);
LABEL_4:
      *(_DWORD *)(a1 + 8) = v8 + 1;
      return result;
    }
    v16 = a2;
    v7 = v3;
    v17 = 0;
    v19 = 0;
    v18 = v4;
    v9 = 1;
    goto LABEL_12;
  }
  v5 = *(unsigned int *)(a1 + 12);
  v6 = *(_QWORD *)(a1 + 1192)
     + v3[4 * result - 1]
     + 16
     + 18 * (v4 - v3[4 * result - 2])
     - *(_QWORD *)(*(_QWORD *)(a1 + 144) + 32 * v3[4 * result - 2] + 16);
  v7 = &v3[4 * result];
  if ( result < v5 )
    goto LABEL_3;
  v18 = *(unsigned int *)(a1 + 152);
  v9 = result + 1;
  v16 = a2;
  v17 = 0;
  v19 = v6;
  if ( v5 < result + 1 )
  {
LABEL_12:
    v15 = result;
    v13 = (const void *)(a1 + 16);
    if ( v3 > &v16 || v7 <= &v16 )
    {
      sub_C8D5F0(a1, v13, v9, 0x20u, (__int64)v3, v9);
      result = v15;
      v11 = (const __m128i *)&v16;
      v3 = *(__int64 **)a1;
      v10 = *(unsigned int *)(a1 + 8);
    }
    else
    {
      v14 = (char *)((char *)&v16 - (char *)v3);
      sub_C8D5F0(a1, v13, v9, 0x20u, (__int64)v3, v9);
      result = v15;
      v3 = *(__int64 **)a1;
      v10 = *(unsigned int *)(a1 + 8);
      v11 = (const __m128i *)&v14[*(_QWORD *)a1];
    }
    goto LABEL_10;
  }
  v10 = result;
  v11 = (const __m128i *)&v16;
LABEL_10:
  v12 = (__m128i *)&v3[4 * v10];
  *v12 = _mm_loadu_si128(v11);
  v12[1] = _mm_loadu_si128(v11 + 1);
  ++*(_DWORD *)(a1 + 8);
  return result;
}
