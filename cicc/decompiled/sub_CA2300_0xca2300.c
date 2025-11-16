// Function: sub_CA2300
// Address: 0xca2300
//
__m128i *__fastcall sub_CA2300(__m128i *a1, _QWORD *a2)
{
  __int64 v2; // rdx
  __int64 v3; // r8
  __int64 v4; // rdx
  _BYTE *v5; // rsi
  __m128i *v6; // rax
  _BYTE *v7; // rsi
  __m128i *v9; // [rsp+0h] [rbp-30h] BYREF
  __int64 v10; // [rsp+8h] [rbp-28h]
  __m128i v11[2]; // [rsp+10h] [rbp-20h] BYREF

  v2 = a2[14];
  if ( v2 )
  {
    v7 = (_BYTE *)a2[13];
    v9 = v11;
    sub_CA1F00((__int64 *)&v9, v7, (__int64)&v7[v2]);
    v6 = v9;
    v2 = v10;
  }
  else
  {
    v3 = a2[2];
    if ( !v3 )
    {
      a1[2].m128i_i8[0] &= ~1u;
      v11[0].m128i_i8[0] = 0;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      goto LABEL_9;
    }
    v4 = a2[3];
    v5 = (_BYTE *)a2[2];
    v9 = v11;
    sub_CA1FB0((__int64 *)&v9, v5, v3 + v4);
    v6 = v9;
    v2 = v10;
  }
  a1[2].m128i_i8[0] &= ~1u;
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( v6 == v11 )
  {
LABEL_9:
    a1[1] = _mm_load_si128(v11);
    goto LABEL_7;
  }
  a1->m128i_i64[0] = (__int64)v6;
  a1[1].m128i_i64[0] = v11[0].m128i_i64[0];
LABEL_7:
  a1->m128i_i64[1] = v2;
  return a1;
}
