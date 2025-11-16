// Function: sub_16356C0
// Address: 0x16356c0
//
__int64 __fastcall sub_16356C0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r15d
  _BYTE *v6; // rax
  __int64 v7; // rdx
  __m128i *v8; // rax
  __int64 v9; // rcx
  __m128i *v10; // rax
  size_t v11; // rcx
  _OWORD *v12; // rdi
  const char *v13; // rax
  size_t v14; // rdx
  __m128i *v15; // [rsp+0h] [rbp-90h]
  size_t v16; // [rsp+8h] [rbp-88h]
  __m128i v17; // [rsp+10h] [rbp-80h] BYREF
  __int64 v18[2]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD v19[2]; // [rsp+30h] [rbp-60h] BYREF
  _OWORD *v20; // [rsp+40h] [rbp-50h] BYREF
  __int64 v21; // [rsp+48h] [rbp-48h]
  _OWORD v22[4]; // [rsp+50h] [rbp-40h] BYREF

  v3 = 1;
  if ( *(_BYTE *)(a1 + 8) )
  {
    v6 = (_BYTE *)sub_1649960(a3);
    if ( v6 )
    {
      v18[0] = (__int64)v19;
      sub_1634F50(v18, v6, (__int64)&v6[v7]);
    }
    else
    {
      v18[1] = 0;
      v18[0] = (__int64)v19;
      LOBYTE(v19[0]) = 0;
    }
    v8 = (__m128i *)sub_2241130(v18, 0, 0, "function (", 10);
    v20 = v22;
    if ( (__m128i *)v8->m128i_i64[0] == &v8[1] )
    {
      v22[0] = _mm_loadu_si128(v8 + 1);
    }
    else
    {
      v20 = (_OWORD *)v8->m128i_i64[0];
      *(_QWORD *)&v22[0] = v8[1].m128i_i64[0];
    }
    v9 = v8->m128i_i64[1];
    v8[1].m128i_i8[0] = 0;
    v21 = v9;
    v8->m128i_i64[0] = (__int64)v8[1].m128i_i64;
    v8->m128i_i64[1] = 0;
    if ( v21 == 0x3FFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"basic_string::append");
    v10 = (__m128i *)sub_2241490(&v20, ")", 1, v9);
    v15 = &v17;
    if ( (__m128i *)v10->m128i_i64[0] == &v10[1] )
    {
      v17 = _mm_loadu_si128(v10 + 1);
    }
    else
    {
      v15 = (__m128i *)v10->m128i_i64[0];
      v17.m128i_i64[0] = v10[1].m128i_i64[0];
    }
    v11 = v10->m128i_u64[1];
    v10[1].m128i_i8[0] = 0;
    v16 = v11;
    v10->m128i_i64[0] = (__int64)v10[1].m128i_i64;
    v12 = v20;
    v10->m128i_i64[1] = 0;
    if ( v12 != v22 )
      j_j___libc_free_0(v12, *(_QWORD *)&v22[0] + 1LL);
    if ( (_QWORD *)v18[0] != v19 )
      j_j___libc_free_0(v18[0], v19[0] + 1LL);
    v13 = (const char *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 16LL))(a2);
    v3 = sub_1635030(a1, v13, v14, v15->m128i_i8, v16);
    if ( v15 != &v17 )
      j_j___libc_free_0(v15, v17.m128i_i64[0] + 1);
  }
  return v3;
}
