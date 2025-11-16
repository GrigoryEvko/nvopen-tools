// Function: sub_E406B0
// Address: 0xe406b0
//
__m128i *__fastcall sub_E406B0(__m128i *a1, _BYTE *a2, unsigned __int64 a3)
{
  _BYTE *v4; // r8
  char *v5; // rax
  char *v6; // rcx
  __int64 v7; // rax
  void *v8; // [rsp+0h] [rbp-70h] BYREF
  unsigned __int64 v9; // [rsp+8h] [rbp-68h]
  __int64 v10[2]; // [rsp+10h] [rbp-60h] BYREF
  __m128i v11; // [rsp+20h] [rbp-50h] BYREF
  void *v12[4]; // [rsp+30h] [rbp-40h] BYREF
  __int16 v13; // [rsp+50h] [rbp-20h]

  v8 = a2;
  v9 = a3;
  if ( *a2 == 35 )
  {
    v4 = a2;
    if ( a3 )
    {
      v4 = a2 + 1;
      a2 += a3;
    }
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    sub_E40120(a1->m128i_i64, v4, (__int64)a2);
    a1[2].m128i_i8[0] = 1;
    return a1;
  }
  else
  {
    if ( *a2 != 63
      || (v5 = (char *)sub_C931B0((__int64 *)&v8, &word_3F7DA1C, 3u, 0), v5 == (char *)-1LL)
      || (v6 = v5 + 3, (unsigned __int64)(v5 + 3) >= v9) )
    {
      a1[2].m128i_i8[0] = 0;
    }
    else
    {
      v12[0] = v8;
      if ( (unsigned __int64)v5 > v9 )
        v5 = (char *)v9;
      v12[3] = (void *)(v9 - (_QWORD)v6);
      v12[2] = &v6[(_QWORD)v8];
      v12[1] = v5;
      v13 = 1285;
      sub_CA0F50(v10, v12);
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      if ( (__m128i *)v10[0] == &v11 )
      {
        a1[1] = _mm_load_si128(&v11);
      }
      else
      {
        a1->m128i_i64[0] = v10[0];
        a1[1].m128i_i64[0] = v11.m128i_i64[0];
      }
      v7 = v10[1];
      a1[2].m128i_i8[0] = 1;
      a1->m128i_i64[1] = v7;
    }
    return a1;
  }
}
