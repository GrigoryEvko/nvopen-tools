// Function: sub_EFA340
// Address: 0xefa340
//
__int64 *__fastcall sub_EFA340(__int64 *a1, _QWORD *a2)
{
  __int64 v4; // r15
  __int64 v5; // r14
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  const __m128i *v10; // rsi
  const __m128i *v11; // r9
  unsigned __int64 v12; // r10
  __int64 v13; // rax
  __m128i *v14; // rdi
  __m128i *v15; // rdx
  const __m128i *v16; // rax
  __int64 v18; // [rsp+0h] [rbp-50h]
  unsigned __int64 v19; // [rsp+8h] [rbp-48h]
  __int64 v20; // [rsp+10h] [rbp-40h]
  __int64 v21; // [rsp+18h] [rbp-38h]

  sub_EF9EB0((__int64)a2);
  v4 = a2[12];
  v5 = a2[13];
  v6 = a2[16];
  v21 = a2[14];
  v20 = a2[15];
  v7 = sub_22077B0(88);
  v9 = v7;
  if ( v7 )
  {
    v10 = (const __m128i *)a2[10];
    v11 = (const __m128i *)a2[9];
    *(_DWORD *)v7 = 0;
    *(_QWORD *)(v7 + 8) = 0;
    *(_QWORD *)(v7 + 16) = 0;
    *(_QWORD *)(v7 + 24) = 0;
    v12 = (char *)v10 - (char *)v11;
    if ( v10 == v11 )
    {
      v14 = 0;
    }
    else
    {
      if ( v12 > 0x7FFFFFFFFFFFFFF8LL )
        sub_4261EA(88, v10, v8);
      v18 = v7;
      v19 = (char *)v10 - (char *)v11;
      v13 = sub_22077B0((char *)v10 - (char *)v11);
      v10 = (const __m128i *)a2[10];
      v11 = (const __m128i *)a2[9];
      v12 = v19;
      v9 = v18;
      v14 = (__m128i *)v13;
    }
    *(_QWORD *)(v9 + 8) = v14;
    *(_QWORD *)(v9 + 16) = v14;
    *(_QWORD *)(v9 + 24) = (char *)v14 + v12;
    if ( v11 != v10 )
    {
      v15 = v14;
      v16 = v11;
      do
      {
        if ( v15 )
        {
          *v15 = _mm_loadu_si128(v16);
          v15[1].m128i_i64[0] = v16[1].m128i_i64[0];
        }
        v16 = (const __m128i *)((char *)v16 + 24);
        v15 = (__m128i *)((char *)v15 + 24);
      }
      while ( v10 != v16 );
      v14 = (__m128i *)((char *)v14 + 8 * ((unsigned __int64)((char *)&v10[-2].m128i_u64[1] - (char *)v11) >> 3) + 24);
    }
    *(_QWORD *)(v9 + 16) = v14;
    *(_QWORD *)(v9 + 32) = v4;
    *(_QWORD *)(v9 + 56) = v21;
    *(_QWORD *)(v9 + 40) = v5;
    *(_QWORD *)(v9 + 48) = v6;
    *(_QWORD *)(v9 + 64) = v20;
    *(_BYTE *)(v9 + 72) = 0;
    *(_QWORD *)(v9 + 80) = 0;
  }
  *a1 = v9;
  return a1;
}
