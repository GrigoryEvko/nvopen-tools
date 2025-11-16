// Function: sub_9825D0
// Address: 0x9825d0
//
__int64 __fastcall sub_9825D0(
        const __m128i *a1,
        unsigned __int64 a2,
        unsigned __int64 a3,
        unsigned __int8 (__fastcall *a4)(__m128i *))
{
  __int64 result; // rax
  __int64 v6; // r14
  __int64 i; // r12
  unsigned __int64 j; // r12
  __int128 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // r10
  __int64 v17; // r11
  __int128 v18; // [rsp-30h] [rbp-B0h]
  __int128 v19; // [rsp-20h] [rbp-A0h]
  __int128 v20; // [rsp-20h] [rbp-A0h]
  __int128 v21; // [rsp-10h] [rbp-90h]

  result = a2 - (_QWORD)a1;
  v6 = (__int64)(a2 - (_QWORD)a1) >> 6;
  if ( (__int64)(a2 - (_QWORD)a1) > 64 )
  {
    for ( i = (v6 - 2) / 2; ; --i )
    {
      v19 = (__int128)a1[4 * i + 2];
      result = sub_982320(
                 (__int64)a1,
                 i,
                 v6,
                 a4,
                 v19,
                 *((__int64 *)&v19 + 1),
                 *(_OWORD *)&a1[4 * i],
                 *(_OWORD *)&a1[4 * i + 1],
                 v19,
                 *(_OWORD *)&a1[4 * i + 3]);
      if ( !i )
        break;
    }
  }
  for ( j = a2; a3 > j; result = sub_982320((__int64)a1, 0, v6, a4, v14, v15, v11, v18, v20, v21) )
  {
    while ( 1 )
    {
      result = ((__int64 (__fastcall *)(unsigned __int64, const __m128i *))a4)(j, a1);
      if ( (_BYTE)result )
        break;
      j += 64LL;
      if ( a3 <= j )
        return result;
    }
    *((_QWORD *)&v11 + 1) = *(_QWORD *)(j + 8);
    j += 64LL;
    *(_QWORD *)&v11 = *(_QWORD *)(j - 64);
    v12 = *(_QWORD *)(j - 48);
    *(__m128i *)(j - 64) = _mm_loadu_si128(a1);
    v13 = *(_QWORD *)(j - 40);
    v14 = *(_QWORD *)(j - 32);
    v15 = *(_QWORD *)(j - 24);
    *(__m128i *)(j - 48) = _mm_loadu_si128(a1 + 1);
    v16 = *(_QWORD *)(j - 16);
    v17 = *(_QWORD *)(j - 8);
    *(__m128i *)(j - 32) = _mm_loadu_si128(a1 + 2);
    *(__m128i *)(j - 16) = _mm_loadu_si128(a1 + 3);
    *((_QWORD *)&v21 + 1) = v17;
    *(_QWORD *)&v21 = v16;
    *((_QWORD *)&v20 + 1) = v15;
    *(_QWORD *)&v20 = v14;
    *((_QWORD *)&v18 + 1) = v13;
    *(_QWORD *)&v18 = v12;
  }
  return result;
}
