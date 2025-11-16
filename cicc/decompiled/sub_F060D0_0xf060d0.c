// Function: sub_F060D0
// Address: 0xf060d0
//
void __fastcall sub_F060D0(__m128i **a1, _BYTE *a2, __int64 a3, char a4)
{
  char *v5; // rcx
  __m128i *v6; // rax
  __int64 v7; // rcx
  _QWORD v8[2]; // [rsp+0h] [rbp-70h] BYREF
  __int64 v9[2]; // [rsp+10h] [rbp-60h] BYREF
  __int64 v10; // [rsp+20h] [rbp-50h] BYREF
  __m128i v11; // [rsp+30h] [rbp-40h] BYREF
  _OWORD v12[3]; // [rsp+40h] [rbp-30h] BYREF

  v8[0] = a2;
  v8[1] = a3;
  if ( a3 )
  {
    if ( ((*a2 - 43) & 0xFD) != 0 )
    {
      sub_C93130(v9, (__int64)v8);
      v5 = "+";
      if ( !a4 )
        v5 = "-";
      v6 = (__m128i *)sub_2241130(v9, 0, 0, v5, 1);
      v11.m128i_i64[0] = (__int64)v12;
      if ( (__m128i *)v6->m128i_i64[0] == &v6[1] )
      {
        v12[0] = _mm_loadu_si128(v6 + 1);
      }
      else
      {
        v11.m128i_i64[0] = v6->m128i_i64[0];
        *(_QWORD *)&v12[0] = v6[1].m128i_i64[0];
      }
      v7 = v6->m128i_i64[1];
      v6[1].m128i_i8[0] = 0;
      v11.m128i_i64[1] = v7;
      v6->m128i_i64[0] = (__int64)v6[1].m128i_i64;
      v6->m128i_i64[1] = 0;
      sub_F06060(a1, &v11);
      if ( (_OWORD *)v11.m128i_i64[0] != v12 )
        j_j___libc_free_0(v11.m128i_i64[0], *(_QWORD *)&v12[0] + 1LL);
      if ( (__int64 *)v9[0] != &v10 )
        j_j___libc_free_0(v9[0], v10 + 1);
    }
    else
    {
      sub_C93130(v11.m128i_i64, (__int64)v8);
      sub_F06060(a1, &v11);
      if ( (_OWORD *)v11.m128i_i64[0] != v12 )
        j_j___libc_free_0(v11.m128i_i64[0], *(_QWORD *)&v12[0] + 1LL);
    }
  }
}
