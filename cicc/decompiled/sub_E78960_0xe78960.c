// Function: sub_E78960
// Address: 0xe78960
//
__int64 __fastcall sub_E78960(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rsi
  __int64 v8; // r12
  __int64 result; // rax
  _QWORD *v10; // r14
  char *v11; // r14
  __int64 v12; // rdx
  __int64 v13; // rax
  int v14; // esi
  __int64 v15; // rsi
  _QWORD *v16; // r15
  int v17; // r15d
  unsigned __int64 v18[7]; // [rsp+8h] [rbp-38h] BYREF

  v7 = a1 + 16;
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0x50u, v18, a6);
  result = *(_QWORD *)a1;
  v10 = (_QWORD *)(*(_QWORD *)a1 + 80LL * *(unsigned int *)(a1 + 8));
  if ( *(_QWORD **)a1 != v10 )
  {
    v11 = (char *)v10 - result;
    v12 = result + 16;
    v13 = v8;
    do
    {
      if ( v13 )
      {
        *(_QWORD *)v13 = v13 + 16;
        v15 = *(_QWORD *)(v12 - 16);
        if ( v12 == v15 )
        {
          *(__m128i *)(v13 + 16) = _mm_loadu_si128((const __m128i *)v12);
        }
        else
        {
          *(_QWORD *)v13 = v15;
          *(_QWORD *)(v13 + 16) = *(_QWORD *)v12;
        }
        *(_QWORD *)(v13 + 8) = *(_QWORD *)(v12 - 8);
        v14 = *(_DWORD *)(v12 + 16);
        *(_QWORD *)(v12 - 16) = v12;
        *(_QWORD *)(v12 - 8) = 0;
        *(_BYTE *)v12 = 0;
        *(_DWORD *)(v13 + 32) = v14;
        *(__m128i *)(v13 + 36) = _mm_loadu_si128((const __m128i *)(v12 + 20));
        *(_BYTE *)(v13 + 52) = *(_BYTE *)(v12 + 36);
        *(__m128i *)(v13 + 56) = _mm_loadu_si128((const __m128i *)(v12 + 40));
        v7 = *(_QWORD *)(v12 + 56);
        *(_QWORD *)(v13 + 72) = v7;
      }
      v13 += 80;
      v12 += 80;
    }
    while ( v8 + 16 * (5 * ((0xCCCCCCCCCCCCCCDLL * ((unsigned __int64)(v11 - 80) >> 4)) & 0xFFFFFFFFFFFFFFFLL) + 5) != v13 );
    result = *(unsigned int *)(a1 + 8);
    v16 = *(_QWORD **)a1;
    v10 = (_QWORD *)(*(_QWORD *)a1 + 80 * result);
    if ( *(_QWORD **)a1 != v10 )
    {
      do
      {
        v10 -= 10;
        result = (__int64)(v10 + 2);
        if ( (_QWORD *)*v10 != v10 + 2 )
        {
          v7 = v10[2] + 1LL;
          result = j_j___libc_free_0(*v10, v7);
        }
      }
      while ( v10 != v16 );
      v10 = *(_QWORD **)a1;
    }
  }
  v17 = v18[0];
  if ( (_QWORD *)(a1 + 16) != v10 )
    result = _libc_free(v10, v7);
  *(_QWORD *)a1 = v8;
  *(_DWORD *)(a1 + 12) = v17;
  return result;
}
