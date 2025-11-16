// Function: sub_9C81F0
// Address: 0x9c81f0
//
__int64 *__fastcall sub_9C81F0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rsi
  __int64 v6; // rcx
  __m128i *v7; // rax
  __int64 v8; // rcx
  __m128i *v9; // rax
  __int64 v10; // rcx
  _QWORD v11[2]; // [rsp+0h] [rbp-C0h] BYREF
  __int64 v12; // [rsp+10h] [rbp-B0h] BYREF
  _QWORD *v13; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v14; // [rsp+28h] [rbp-98h]
  _QWORD v15[2]; // [rsp+30h] [rbp-90h] BYREF
  __m128i *v16; // [rsp+40h] [rbp-80h] BYREF
  __int64 v17; // [rsp+48h] [rbp-78h]
  __m128i v18; // [rsp+50h] [rbp-70h] BYREF
  __m128i *v19; // [rsp+60h] [rbp-60h] BYREF
  __int64 v20; // [rsp+68h] [rbp-58h]
  __m128i v21; // [rsp+70h] [rbp-50h] BYREF
  __int16 v22; // [rsp+80h] [rbp-40h]

  sub_CA0F50(v11, a3);
  v4 = *(_QWORD *)(a2 + 400);
  if ( v4 )
  {
    v14 = 0;
    v13 = v15;
    LOBYTE(v15[0]) = 0;
    sub_2240E30(&v13, v4 + 13);
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v14) <= 0xC )
      goto LABEL_20;
    sub_2241490(&v13, " (Producer: '", 13, 0x3FFFFFFFFFFFFFFFLL);
    sub_2241490(&v13, *(_QWORD *)(a2 + 392), *(_QWORD *)(a2 + 400), v6);
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v14) <= 0xF )
      goto LABEL_20;
    v7 = (__m128i *)sub_2241490(&v13, "' Reader: 'LLVM ", 16, 0x3FFFFFFFFFFFFFFFLL - v14);
    v16 = &v18;
    if ( (__m128i *)v7->m128i_i64[0] == &v7[1] )
    {
      v18 = _mm_loadu_si128(v7 + 1);
    }
    else
    {
      v16 = (__m128i *)v7->m128i_i64[0];
      v18.m128i_i64[0] = v7[1].m128i_i64[0];
    }
    v8 = v7->m128i_i64[1];
    v7[1].m128i_i8[0] = 0;
    v17 = v8;
    v7->m128i_i64[0] = (__int64)v7[1].m128i_i64;
    v7->m128i_i64[1] = 0;
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v17) <= 7 )
LABEL_20:
      sub_4262D8((__int64)"basic_string::append");
    v9 = (__m128i *)sub_2241490(&v16, "20.0.0')", 8, v8);
    v19 = &v21;
    if ( (__m128i *)v9->m128i_i64[0] == &v9[1] )
    {
      v21 = _mm_loadu_si128(v9 + 1);
    }
    else
    {
      v19 = (__m128i *)v9->m128i_i64[0];
      v21.m128i_i64[0] = v9[1].m128i_i64[0];
    }
    v20 = v9->m128i_i64[1];
    v10 = v20;
    v9->m128i_i64[0] = (__int64)v9[1].m128i_i64;
    v9->m128i_i64[1] = 0;
    v9[1].m128i_i8[0] = 0;
    sub_2241490(v11, v19, v20, v10);
    if ( v19 != &v21 )
      j_j___libc_free_0(v19, v21.m128i_i64[0] + 1);
    if ( v16 != &v18 )
      j_j___libc_free_0(v16, v18.m128i_i64[0] + 1);
    if ( v13 != v15 )
      j_j___libc_free_0(v13, v15[0] + 1LL);
  }
  v19 = (__m128i *)v11;
  v22 = 260;
  sub_9C8190(a1, (__int64)&v19);
  if ( (__int64 *)v11[0] != &v12 )
    j_j___libc_free_0(v11[0], v12 + 1);
  return a1;
}
