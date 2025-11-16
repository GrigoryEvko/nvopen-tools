// Function: sub_EFA1E0
// Address: 0xefa1e0
//
__int64 *__fastcall sub_EFA1E0(__int64 *a1, _QWORD *a2)
{
  __int64 v3; // r14
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // r15
  const __m128i *v7; // rcx
  const __m128i *v8; // r9
  unsigned __int64 v9; // rdx
  __int64 v10; // rax
  __m128i *v11; // rdi
  __m128i *v12; // rdx
  const __m128i *v13; // rax
  unsigned __int64 v15; // [rsp+8h] [rbp-48h]
  __int64 v16; // [rsp+10h] [rbp-40h]
  __int64 v17; // [rsp+18h] [rbp-38h]

  sub_EF9EB0((__int64)a2);
  v3 = a2[12];
  v4 = a2[13];
  v17 = a2[14];
  v16 = a2[15];
  v5 = sub_22077B0(88);
  v6 = v5;
  if ( v5 )
  {
    v7 = (const __m128i *)a2[10];
    v8 = (const __m128i *)a2[9];
    *(_DWORD *)v5 = 2;
    *(_QWORD *)(v5 + 8) = 0;
    *(_QWORD *)(v5 + 16) = 0;
    *(_QWORD *)(v5 + 24) = 0;
    v9 = (char *)v7 - (char *)v8;
    if ( v7 == v8 )
    {
      v11 = 0;
    }
    else
    {
      if ( v9 > 0x7FFFFFFFFFFFFFF8LL )
        sub_4261EA(88, a2, v9);
      v15 = (char *)v7 - (char *)v8;
      v10 = sub_22077B0((char *)v7 - (char *)v8);
      v7 = (const __m128i *)a2[10];
      v8 = (const __m128i *)a2[9];
      v9 = v15;
      v11 = (__m128i *)v10;
    }
    *(_QWORD *)(v6 + 8) = v11;
    *(_QWORD *)(v6 + 16) = v11;
    *(_QWORD *)(v6 + 24) = (char *)v11 + v9;
    if ( v8 != v7 )
    {
      v12 = v11;
      v13 = v8;
      do
      {
        if ( v12 )
        {
          *v12 = _mm_loadu_si128(v13);
          v12[1].m128i_i64[0] = v13[1].m128i_i64[0];
        }
        v13 = (const __m128i *)((char *)v13 + 24);
        v12 = (__m128i *)((char *)v12 + 24);
      }
      while ( v7 != v13 );
      v11 = (__m128i *)((char *)v11 + 8 * ((unsigned __int64)((char *)&v7[-2].m128i_u64[1] - (char *)v8) >> 3) + 24);
    }
    *(_QWORD *)(v6 + 16) = v11;
    *(_QWORD *)(v6 + 32) = v3;
    *(_QWORD *)(v6 + 56) = v17;
    *(_QWORD *)(v6 + 40) = v4;
    *(_QWORD *)(v6 + 48) = 0;
    *(_QWORD *)(v6 + 64) = v16;
    *(_BYTE *)(v6 + 72) = 0;
    *(_QWORD *)(v6 + 80) = 0;
  }
  *a1 = v6;
  return a1;
}
