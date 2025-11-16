// Function: sub_23B1880
// Address: 0x23b1880
//
__m128i *__fastcall sub_23B1880(__m128i *a1, __int64 a2, __int64 a3, __int64 a4)
{
  bool v5; // zf
  __int64 v6; // rdx
  __m128i *v8; // rax
  __int64 v9; // rcx
  __m128i *v10; // rax
  unsigned __int64 v11; // rcx
  __m128i *v12; // rax
  __int64 v13; // rcx
  __m128i *v14; // rax
  __int64 v15; // rcx
  _OWORD *v16; // rdi
  _QWORD v17[2]; // [rsp+0h] [rbp-C0h] BYREF
  unsigned __int64 v18[4]; // [rsp+10h] [rbp-B0h] BYREF
  __m128i *v19; // [rsp+30h] [rbp-90h] BYREF
  __int64 v20; // [rsp+38h] [rbp-88h]
  __m128i v21; // [rsp+40h] [rbp-80h] BYREF
  unsigned __int64 v22[2]; // [rsp+50h] [rbp-70h] BYREF
  __m128i v23; // [rsp+60h] [rbp-60h] BYREF
  _OWORD *v24; // [rsp+70h] [rbp-50h] BYREF
  __int64 v25; // [rsp+78h] [rbp-48h]
  _OWORD v26[4]; // [rsp+80h] [rbp-40h] BYREF

  v5 = *(_QWORD *)(a2 + 8) == 0;
  v17[0] = a3;
  v17[1] = a4;
  if ( v5 )
  {
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      a1[1] = _mm_loadu_si128((const __m128i *)(a2 + 16));
    }
    else
    {
      a1->m128i_i64[0] = *(_QWORD *)a2;
      a1[1].m128i_i64[0] = *(_QWORD *)(a2 + 16);
    }
    v6 = *(_QWORD *)(a2 + 8);
    *(_QWORD *)a2 = a2 + 16;
    *(_QWORD *)(a2 + 8) = 0;
    a1->m128i_i64[1] = v6;
    *(_BYTE *)(a2 + 16) = 0;
  }
  else
  {
    sub_95CA80((__int64 *)v18, (__int64)v17);
    v8 = (__m128i *)sub_2241130(v18, 0, 0, "<FONT COLOR=\"", 0xDu);
    v19 = &v21;
    if ( (__m128i *)v8->m128i_i64[0] == &v8[1] )
    {
      v21 = _mm_loadu_si128(v8 + 1);
    }
    else
    {
      v19 = (__m128i *)v8->m128i_i64[0];
      v21.m128i_i64[0] = v8[1].m128i_i64[0];
    }
    v9 = v8->m128i_i64[1];
    v8[1].m128i_i8[0] = 0;
    v20 = v9;
    v8->m128i_i64[0] = (__int64)v8[1].m128i_i64;
    v8->m128i_i64[1] = 0;
    if ( v20 == 0x3FFFFFFFFFFFFFFFLL || v20 == 4611686018427387902LL )
      goto LABEL_26;
    v10 = (__m128i *)sub_2241490((unsigned __int64 *)&v19, "\">", 2u);
    v22[0] = (unsigned __int64)&v23;
    if ( (__m128i *)v10->m128i_i64[0] == &v10[1] )
    {
      v23 = _mm_loadu_si128(v10 + 1);
    }
    else
    {
      v22[0] = v10->m128i_i64[0];
      v23.m128i_i64[0] = v10[1].m128i_i64[0];
    }
    v11 = v10->m128i_u64[1];
    v10[1].m128i_i8[0] = 0;
    v22[1] = v11;
    v10->m128i_i64[0] = (__int64)v10[1].m128i_i64;
    v10->m128i_i64[1] = 0;
    v12 = (__m128i *)sub_2241490(v22, *(char **)a2, *(_QWORD *)(a2 + 8));
    v24 = v26;
    if ( (__m128i *)v12->m128i_i64[0] == &v12[1] )
    {
      v26[0] = _mm_loadu_si128(v12 + 1);
    }
    else
    {
      v24 = (_OWORD *)v12->m128i_i64[0];
      *(_QWORD *)&v26[0] = v12[1].m128i_i64[0];
    }
    v13 = v12->m128i_i64[1];
    v12[1].m128i_i8[0] = 0;
    v25 = v13;
    v12->m128i_i64[0] = (__int64)v12[1].m128i_i64;
    v12->m128i_i64[1] = 0;
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v25) <= 6 )
LABEL_26:
      sub_4262D8((__int64)"basic_string::append");
    v14 = (__m128i *)sub_2241490((unsigned __int64 *)&v24, "</FONT>", 7u);
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    if ( (__m128i *)v14->m128i_i64[0] == &v14[1] )
    {
      a1[1] = _mm_loadu_si128(v14 + 1);
    }
    else
    {
      a1->m128i_i64[0] = v14->m128i_i64[0];
      a1[1].m128i_i64[0] = v14[1].m128i_i64[0];
    }
    v15 = v14->m128i_i64[1];
    v14->m128i_i64[0] = (__int64)v14[1].m128i_i64;
    v16 = v24;
    v14->m128i_i64[1] = 0;
    a1->m128i_i64[1] = v15;
    v14[1].m128i_i8[0] = 0;
    if ( v16 != v26 )
      j_j___libc_free_0((unsigned __int64)v16);
    sub_2240A30(v22);
    if ( v19 != &v21 )
      j_j___libc_free_0((unsigned __int64)v19);
    sub_2240A30(v18);
  }
  return a1;
}
