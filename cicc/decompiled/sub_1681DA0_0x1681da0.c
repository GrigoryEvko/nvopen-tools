// Function: sub_1681DA0
// Address: 0x1681da0
//
void __fastcall sub_1681DA0(__m128i **a1, _BYTE *a2, __int64 a3, char a4)
{
  char v5; // r13
  __m128i *v6; // rsi
  __m128i *v7; // rdi
  char *v9; // rcx
  __m128i *v10; // rax
  __int64 v11; // rcx
  _QWORD v12[2]; // [rsp+0h] [rbp-70h] BYREF
  _QWORD v13[2]; // [rsp+10h] [rbp-60h] BYREF
  __int64 v14; // [rsp+20h] [rbp-50h] BYREF
  __m128i v15; // [rsp+30h] [rbp-40h] BYREF
  __m128i v16[3]; // [rsp+40h] [rbp-30h] BYREF

  v12[0] = a2;
  v12[1] = a3;
  if ( a3 )
  {
    if ( ((*a2 - 43) & 0xFD) != 0 )
    {
      sub_16D2060(v13, v12);
      v9 = "+";
      if ( !a4 )
        v9 = "-";
      v10 = (__m128i *)sub_2241130(v13, 0, 0, v9, 1);
      v15.m128i_i64[0] = (__int64)v16;
      if ( (__m128i *)v10->m128i_i64[0] == &v10[1] )
      {
        v16[0] = _mm_loadu_si128(v10 + 1);
      }
      else
      {
        v15.m128i_i64[0] = v10->m128i_i64[0];
        v16[0].m128i_i64[0] = v10[1].m128i_i64[0];
      }
      v11 = v10->m128i_i64[1];
      v10[1].m128i_i8[0] = 0;
      v5 = 1;
      v15.m128i_i64[1] = v11;
      v10->m128i_i64[0] = (__int64)v10[1].m128i_i64;
      v10->m128i_i64[1] = 0;
    }
    else
    {
      v5 = 0;
      sub_16D2060(&v15, v12);
    }
    v6 = a1[1];
    if ( v6 == a1[2] )
    {
      sub_8F99A0(a1, v6, &v15);
      v7 = (__m128i *)v15.m128i_i64[0];
    }
    else
    {
      v7 = (__m128i *)v15.m128i_i64[0];
      if ( v6 )
      {
        v6->m128i_i64[0] = (__int64)v6[1].m128i_i64;
        if ( (__m128i *)v15.m128i_i64[0] == v16 )
        {
          v6[1] = _mm_load_si128(v16);
        }
        else
        {
          v6->m128i_i64[0] = v15.m128i_i64[0];
          v6[1].m128i_i64[0] = v16[0].m128i_i64[0];
        }
        v15.m128i_i64[0] = (__int64)v16;
        v7 = v16;
        v6->m128i_i64[1] = v15.m128i_i64[1];
        v6 = a1[1];
        v15.m128i_i64[1] = 0;
        v16[0].m128i_i8[0] = 0;
      }
      a1[1] = v6 + 2;
    }
    if ( v7 != v16 )
      j_j___libc_free_0(v7, v16[0].m128i_i64[0] + 1);
    if ( v5 )
    {
      if ( (__int64 *)v13[0] != &v14 )
        j_j___libc_free_0(v13[0], v14 + 1);
    }
  }
}
