// Function: sub_3023AD0
// Address: 0x3023ad0
//
size_t __fastcall sub_3023AD0(__int64 a1, int a2, unsigned __int64 *a3)
{
  size_t v4; // r12
  char *v5; // rdi
  size_t v6; // rsi
  unsigned __int64 v7; // rax
  __m128i *v9; // rax
  unsigned __int64 v10; // rcx
  _QWORD *v11; // rdi
  size_t v12; // [rsp+8h] [rbp-88h]
  char *v13; // [rsp+10h] [rbp-80h] BYREF
  size_t v14; // [rsp+18h] [rbp-78h]
  __int64 v15; // [rsp+20h] [rbp-70h] BYREF
  unsigned __int64 v16[2]; // [rsp+30h] [rbp-60h] BYREF
  __m128i v17; // [rsp+40h] [rbp-50h] BYREF
  _QWORD *v18; // [rsp+50h] [rbp-40h] BYREF
  __int64 v19; // [rsp+58h] [rbp-38h]
  _BYTE v20[48]; // [rsp+60h] [rbp-30h] BYREF

  sub_30232C0((__int64)&v13, a1, a2);
  if ( a3 )
    sub_2240AE0(a3, (unsigned __int64 *)&v13);
  v12 = 0;
  v4 = v14;
  if ( v14 > 8 )
  {
    if ( !byte_502ADF8 )
    {
      byte_502ADF8 = 1;
      v19 = 0;
      v18 = v20;
      v20[0] = 0;
      sub_2240E30((__int64)&v18, v14 + 14);
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v19) <= 0xD
        || (sub_2241490((unsigned __int64 *)&v18, "Register name ", 0xEu),
            sub_2241490((unsigned __int64 *)&v18, v13, v14),
            (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v19) <= 0x3C) )
      {
        sub_4262D8((__int64)"basic_string::append");
      }
      v9 = (__m128i *)sub_2241490(
                        (unsigned __int64 *)&v18,
                        " is too large, generated debug information may be inaccurate.",
                        0x3Du);
      v16[0] = (unsigned __int64)&v17;
      if ( (__m128i *)v9->m128i_i64[0] == &v9[1] )
      {
        v17 = _mm_loadu_si128(v9 + 1);
      }
      else
      {
        v16[0] = v9->m128i_i64[0];
        v17.m128i_i64[0] = v9[1].m128i_i64[0];
      }
      v10 = v9->m128i_u64[1];
      v9[1].m128i_i8[0] = 0;
      v16[1] = v10;
      v9->m128i_i64[0] = (__int64)v9[1].m128i_i64;
      v11 = v18;
      v9->m128i_i64[1] = 0;
      if ( v11 != (_QWORD *)v20 )
        j_j___libc_free_0((unsigned __int64)v11);
      sub_CEB650(v16);
      if ( (__m128i *)v16[0] != &v17 )
        j_j___libc_free_0(v16[0]);
    }
    v5 = v13;
    v4 = 0;
  }
  else
  {
    v5 = v13;
    if ( v14 )
    {
      v5 = v13;
      v6 = v14 - 1;
      v7 = 0;
      while ( 1 )
      {
        *((_BYTE *)&v12 + v7) = v5[v6];
        if ( v4 <= ++v7 || v7 > 7 )
          break;
        if ( --v6 >= v4 )
          sub_222CF80("basic_string::at: __n (which is %zu) >= this->size() (which is %zu)", v6, v4);
      }
      v4 = v12;
    }
  }
  if ( v5 != (char *)&v15 )
    j_j___libc_free_0((unsigned __int64)v5);
  return v4;
}
