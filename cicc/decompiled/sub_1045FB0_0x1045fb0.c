// Function: sub_1045FB0
// Address: 0x1045fb0
//
unsigned __int8 *__fastcall sub_1045FB0(__int64 a1, unsigned __int8 *a2, const __m128i *a3, __int64 a4, _DWORD *a5)
{
  __int64 v5; // rbp
  unsigned __int8 *result; // rax
  int v9; // eax
  unsigned __int64 v10; // rax
  __int64 v11; // rcx
  __m128i v12; // xmm0
  __m128i v13; // xmm1
  __m128i v14; // xmm2
  __int64 v15; // r10
  char v16; // [rsp-58h] [rbp-58h] BYREF
  __m128i v17; // [rsp-50h] [rbp-50h]
  __m128i v18; // [rsp-40h] [rbp-40h]
  __m128i v19; // [rsp-30h] [rbp-30h]
  __int64 v20; // [rsp-20h] [rbp-20h]
  unsigned __int8 *v21; // [rsp-18h] [rbp-18h]
  char v22; // [rsp-10h] [rbp-10h]
  __int64 v23; // [rsp-8h] [rbp-8h]

  if ( !a3->m128i_i64[0] )
    return a2;
  if ( (unsigned int)*a2 - 26 > 1 )
    goto LABEL_7;
  result = *(unsigned __int8 **)(*(_QWORD *)(a1 + 2392) + 128LL);
  if ( a2 != result )
  {
    v9 = **((unsigned __int8 **)a2 + 9);
    if ( (unsigned __int8)(v9 - 34) <= 0x33u )
    {
      v15 = 0x8000000000041LL;
      if ( _bittest64(&v15, (unsigned int)(v9 - 34)) )
        goto LABEL_7;
      v10 = (unsigned int)(v9 - 29);
LABEL_6:
      v11 = 0x110000800000220LL;
      if ( !_bittest64(&v11, v10) )
        goto LABEL_7;
      return a2;
    }
    v10 = (unsigned int)(v9 - 29);
    if ( (unsigned int)v10 <= 0x38 )
      goto LABEL_6;
LABEL_7:
    v23 = v5;
    v12 = _mm_loadu_si128(a3);
    v13 = _mm_loadu_si128(a3 + 1);
    v14 = _mm_loadu_si128(a3 + 2);
    v16 = 0;
    v20 = 0;
    v22 = 0;
    v21 = a2;
    v17 = v12;
    v18 = v13;
    v19 = v14;
    return sub_1044BF0(a1, a4, a2, (__int64)&v16, a5);
  }
  return result;
}
