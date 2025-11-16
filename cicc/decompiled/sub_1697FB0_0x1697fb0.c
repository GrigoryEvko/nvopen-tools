// Function: sub_1697FB0
// Address: 0x1697fb0
//
__int64 *__fastcall sub_1697FB0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  unsigned __int64 *v8; // r12
  __int64 *result; // rax
  __int64 v10; // rbx
  __int64 v11; // r12
  __int64 *v12; // r13
  __int64 v13; // r14
  __m128i *v14; // rax
  __int64 v15; // rsi
  __int64 v16; // [rsp+8h] [rbp-58h]
  __int64 v17; // [rsp+10h] [rbp-50h]
  __int64 v18; // [rsp+18h] [rbp-48h] BYREF
  __int64 v19[7]; // [rsp+28h] [rbp-38h] BYREF

  v18 = a4;
  if ( (_DWORD)a5 )
  {
    v7 = 0;
    v16 = (unsigned int)a5;
    v17 = 16LL * (unsigned int)a5;
    do
    {
      v8 = (unsigned __int64 *)(v7 + v18);
      v7 += 16;
      *v8 = sub_1697F90(a1, *v8, a2, a6, a5, a6);
    }
    while ( v17 != v7 );
    result = (__int64 *)sub_16946D0(a1, a2);
    v10 = v18;
    v11 = result[1];
    v12 = result;
    v13 = v18 + 16 * v16;
    v19[0] = v13;
    if ( v11 == result[2] )
    {
      return (__int64 *)sub_1696100((char **)result, (char *)v11, &v18, v19);
    }
    else
    {
      if ( v11 )
      {
        *(_QWORD *)(v11 + 8) = v11;
        *(_QWORD *)v11 = v11;
        *(_QWORD *)(v11 + 16) = 0;
        do
        {
          v10 += 16;
          v14 = (__m128i *)sub_22077B0(32);
          v14[1] = _mm_loadu_si128((const __m128i *)(v10 - 16));
          result = (__int64 *)sub_2208C80(v14, v11);
          ++*(_QWORD *)(v11 + 16);
        }
        while ( v13 != v10 );
        v11 = v12[1];
      }
      v12[1] = v11 + 24;
    }
  }
  else
  {
    result = (__int64 *)sub_16946D0(a1, a2);
    v15 = result[1];
    if ( v15 == result[2] )
    {
      return sub_1695EB0(result, (char *)v15);
    }
    else
    {
      if ( v15 )
      {
        *(_QWORD *)(v15 + 8) = v15;
        *(_QWORD *)v15 = v15;
        *(_QWORD *)(v15 + 16) = 0;
        v15 = result[1];
      }
      result[1] = v15 + 24;
    }
  }
  return result;
}
