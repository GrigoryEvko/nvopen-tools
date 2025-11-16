// Function: sub_1BDB370
// Address: 0x1bdb370
//
__int64 __fastcall sub_1BDB370(
        __int64 a1,
        __int64 a2,
        __m128i a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10,
        __int64 a11,
        __int64 a12,
        __int64 *a13,
        __int64 *a14)
{
  __int64 v14; // r15
  unsigned int v15; // r12d
  unsigned int v16; // r13d
  unsigned int i; // r14d
  __int64 v18; // rsi
  unsigned int v19; // edx
  __int64 v21; // [rsp+0h] [rbp-40h]

  v21 = *(_QWORD *)(a1 + 112);
  if ( *(_QWORD *)(a1 + 104) == v21 )
  {
    return 0;
  }
  else
  {
    v14 = *(_QWORD *)(a1 + 104);
    v15 = 0;
    do
    {
      v16 = *(_DWORD *)(v14 + 16);
      if ( v16 > 1 )
      {
        for ( i = 0; i < v16; i += 16 )
        {
          v18 = *(_QWORD *)(v14 + 8) + 8LL * i;
          v19 = v16 - i;
          if ( v16 - i > 0x10 )
            v19 = 16;
          v15 |= sub_1BD9F80(a1, v18, v19, a2, a13, a14, a3, a4, a5, a6, a7, a8, a9, a10);
        }
      }
      v14 += 88;
    }
    while ( v21 != v14 );
  }
  return v15;
}
