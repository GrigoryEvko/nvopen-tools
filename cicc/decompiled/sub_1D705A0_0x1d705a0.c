// Function: sub_1D705A0
// Address: 0x1d705a0
//
__m128i *__fastcall sub_1D705A0(__int64 a1, __m128i *a2)
{
  __m128i *result; // rax
  int v5; // r8d
  int v6; // r9d
  __int64 *v7; // rdx
  unsigned int v8; // eax
  int v9; // eax
  unsigned int v10; // esi
  unsigned int v11; // edi
  __int64 v12; // rax
  __int64 *v13[5]; // [rsp+8h] [rbp-28h] BYREF

  result = (__m128i *)sub_1D67E80(a1, a2->m128i_i64, v13);
  v7 = v13[0];
  if ( (_BYTE)result )
    return result;
  v8 = *(_DWORD *)(a1 + 8);
  ++*(_QWORD *)a1;
  v9 = (v8 >> 1) + 1;
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v11 = 24;
    v10 = 8;
  }
  else
  {
    v10 = *(_DWORD *)(a1 + 24);
    v11 = 3 * v10;
  }
  if ( 4 * v9 >= v11 )
  {
    v10 *= 2;
  }
  else if ( v10 - (v9 + *(_DWORD *)(a1 + 12)) > v10 >> 3 )
  {
    goto LABEL_6;
  }
  sub_1D70130(a1, v10);
  sub_1D67E80(a1, a2->m128i_i64, v13);
  v7 = v13[0];
  v9 = (*(_DWORD *)(a1 + 8) >> 1) + 1;
LABEL_6:
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(a1 + 8) & 1 | (2 * v9);
  if ( *v7 != -8 || v7[1] != -8 )
    --*(_DWORD *)(a1 + 12);
  *v7 = a2->m128i_i64[0];
  v7[1] = a2->m128i_i64[1];
  v12 = *(unsigned int *)(a1 + 152);
  if ( (unsigned int)v12 >= *(_DWORD *)(a1 + 156) )
  {
    sub_16CD150(a1 + 144, (const void *)(a1 + 160), 0, 16, v5, v6);
    v12 = *(unsigned int *)(a1 + 152);
  }
  result = (__m128i *)(*(_QWORD *)(a1 + 144) + 16 * v12);
  *result = _mm_loadu_si128(a2);
  ++*(_DWORD *)(a1 + 152);
  return result;
}
