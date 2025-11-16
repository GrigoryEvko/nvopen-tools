// Function: sub_3990780
// Address: 0x3990780
//
__int64 __fastcall sub_3990780(const __m128i *a1, __m128i *a2, unsigned __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  __m128i *v6; // rbx
  __int64 i; // r14
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rsi
  __int64 result; // rax
  unsigned __int64 v12; // r15
  __int64 v13; // rcx
  __int64 v14; // rsi
  unsigned __int64 v15; // rdi
  __int64 v16; // r8
  __int64 v17; // r9
  __m128i v18; // xmm0
  __m128i v19; // xmm1
  __int128 v20; // [rsp-20h] [rbp-A0h]
  __int128 v21; // [rsp-20h] [rbp-A0h]
  __int128 v22; // [rsp-10h] [rbp-90h]
  __int128 v23; // [rsp-10h] [rbp-90h]
  __int64 v24; // [rsp+0h] [rbp-80h]
  char v26[8]; // [rsp+10h] [rbp-70h] BYREF
  unsigned __int64 v27; // [rsp+18h] [rbp-68h]
  __int64 v28; // [rsp+30h] [rbp-50h] BYREF
  unsigned __int64 v29; // [rsp+38h] [rbp-48h]
  __int64 v30; // [rsp+40h] [rbp-40h]
  __int64 v31; // [rsp+48h] [rbp-38h]

  v4 = (char *)a2 - (char *)a1;
  v6 = a2;
  if ( (char *)a2 - (char *)a1 > 32 )
  {
    for ( i = ((v4 >> 5) - 2) / 2; ; --i )
    {
      v8 = a1[2 * i + 1].m128i_i64[0];
      v9 = a1[2 * i + 1].m128i_i64[1];
      v10 = a1[2 * i].m128i_i64[0];
      v29 = a1[2 * i].m128i_u64[1];
      *((_QWORD *)&v22 + 1) = v9;
      *(_QWORD *)&v22 = v8;
      *((_QWORD *)&v20 + 1) = v29;
      *(_QWORD *)&v20 = v10;
      v28 = v10;
      v30 = v8;
      v31 = v9;
      result = sub_3986080((__int64)a1, i, v4 >> 5, a4, v8, v9, v20, v22);
      if ( !i )
        break;
    }
  }
  if ( (unsigned __int64)v6 < a3 )
  {
    v24 = v4 >> 5;
    do
    {
      while ( 1 )
      {
        sub_15B1350(
          (__int64)v26,
          *(unsigned __int64 **)(v6->m128i_i64[0] + 24),
          *(unsigned __int64 **)(v6->m128i_i64[0] + 32));
        v12 = v27;
        result = sub_15B1350(
                   (__int64)&v28,
                   *(unsigned __int64 **)(a1->m128i_i64[0] + 24),
                   *(unsigned __int64 **)(a1->m128i_i64[0] + 32));
        if ( v12 < v29 )
          break;
        v6 += 2;
        if ( a3 <= (unsigned __int64)v6 )
          return result;
      }
      v14 = v6->m128i_i64[0];
      v15 = v6->m128i_u64[1];
      v6 += 2;
      v16 = v6[-1].m128i_i64[0];
      v17 = v6[-1].m128i_i64[1];
      v18 = _mm_loadu_si128(a1);
      v28 = v14;
      *((_QWORD *)&v23 + 1) = v17;
      v6[-2] = v18;
      v19 = _mm_loadu_si128(a1 + 1);
      *(_QWORD *)&v23 = v16;
      *((_QWORD *)&v21 + 1) = v15;
      *(_QWORD *)&v21 = v14;
      v29 = v15;
      v6[-1] = v19;
      v30 = v16;
      v31 = v17;
      result = sub_3986080((__int64)a1, 0, v24, v13, v16, v17, v21, v23);
    }
    while ( a3 > (unsigned __int64)v6 );
  }
  return result;
}
