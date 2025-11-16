// Function: sub_16354C0
// Address: 0x16354c0
//
__int64 __fastcall sub_16354C0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r15d
  _BYTE *v6; // rsi
  __int64 v8; // rax
  __m128i *v9; // rax
  __int64 v10; // rcx
  __m128i *v11; // rax
  size_t v12; // rcx
  _OWORD *v13; // rdi
  const char *v14; // rax
  size_t v15; // rdx
  __m128i *v16; // [rsp+0h] [rbp-90h]
  size_t v17; // [rsp+8h] [rbp-88h]
  __m128i v18; // [rsp+10h] [rbp-80h] BYREF
  __int64 v19[2]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD v20[2]; // [rsp+30h] [rbp-60h] BYREF
  _OWORD *v21; // [rsp+40h] [rbp-50h] BYREF
  __int64 v22; // [rsp+48h] [rbp-48h]
  _OWORD v23[4]; // [rsp+50h] [rbp-40h] BYREF

  v3 = 1;
  if ( *(_BYTE *)(a1 + 8) )
  {
    v6 = *(_BYTE **)(a3 + 176);
    if ( v6 )
    {
      v8 = *(_QWORD *)(a3 + 184);
      v19[0] = (__int64)v20;
      sub_1634F50(v19, v6, (__int64)&v6[v8]);
    }
    else
    {
      v19[1] = 0;
      v19[0] = (__int64)v20;
      LOBYTE(v20[0]) = 0;
    }
    v9 = (__m128i *)sub_2241130(v19, 0, 0, "module (", 8);
    v21 = v23;
    if ( (__m128i *)v9->m128i_i64[0] == &v9[1] )
    {
      v23[0] = _mm_loadu_si128(v9 + 1);
    }
    else
    {
      v21 = (_OWORD *)v9->m128i_i64[0];
      *(_QWORD *)&v23[0] = v9[1].m128i_i64[0];
    }
    v10 = v9->m128i_i64[1];
    v9[1].m128i_i8[0] = 0;
    v22 = v10;
    v9->m128i_i64[0] = (__int64)v9[1].m128i_i64;
    v9->m128i_i64[1] = 0;
    if ( v22 == 0x3FFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"basic_string::append");
    v11 = (__m128i *)sub_2241490(&v21, ")", 1, v10);
    v16 = &v18;
    if ( (__m128i *)v11->m128i_i64[0] == &v11[1] )
    {
      v18 = _mm_loadu_si128(v11 + 1);
    }
    else
    {
      v16 = (__m128i *)v11->m128i_i64[0];
      v18.m128i_i64[0] = v11[1].m128i_i64[0];
    }
    v12 = v11->m128i_u64[1];
    v11[1].m128i_i8[0] = 0;
    v17 = v12;
    v11->m128i_i64[0] = (__int64)v11[1].m128i_i64;
    v13 = v21;
    v11->m128i_i64[1] = 0;
    if ( v13 != v23 )
      j_j___libc_free_0(v13, *(_QWORD *)&v23[0] + 1LL);
    if ( (_QWORD *)v19[0] != v20 )
      j_j___libc_free_0(v19[0], v20[0] + 1LL);
    v14 = (const char *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 16LL))(a2);
    v3 = sub_1635030(a1, v14, v15, v16->m128i_i8, v17);
    if ( v16 != &v18 )
      j_j___libc_free_0(v16, v18.m128i_i64[0] + 1);
  }
  return v3;
}
