// Function: sub_16358B0
// Address: 0x16358b0
//
__int64 __fastcall sub_16358B0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r14d
  _BYTE *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdx
  _BYTE *v11; // rsi
  __m128i *v12; // rax
  __int64 v13; // rcx
  __m128i *v14; // rax
  __int64 v15; // rcx
  __m128i *v16; // r9
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rcx
  __m128i *v20; // rax
  _OWORD *v21; // rsi
  __m128i *v22; // rdx
  __m128i *v23; // rax
  size_t v24; // rsi
  _OWORD *v25; // rdi
  const char *v26; // rax
  size_t v27; // rdx
  __m128i *v28; // [rsp+20h] [rbp-F0h]
  __m128i v29; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v30[2]; // [rsp+40h] [rbp-D0h] BYREF
  _QWORD v31[2]; // [rsp+50h] [rbp-C0h] BYREF
  __m128i *v32; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v33; // [rsp+68h] [rbp-A8h]
  __m128i v34; // [rsp+70h] [rbp-A0h] BYREF
  __m128i *v35; // [rsp+80h] [rbp-90h] BYREF
  __int64 v36; // [rsp+88h] [rbp-88h]
  __m128i v37; // [rsp+90h] [rbp-80h] BYREF
  _QWORD *v38; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v39; // [rsp+A8h] [rbp-68h]
  _QWORD v40[2]; // [rsp+B0h] [rbp-60h] BYREF
  _OWORD *v41; // [rsp+C0h] [rbp-50h] BYREF
  __int64 v42; // [rsp+C8h] [rbp-48h]
  _OWORD v43[4]; // [rsp+D0h] [rbp-40h] BYREF

  v3 = 1;
  if ( !*(_BYTE *)(a1 + 8) )
    return v3;
  v8 = (_BYTE *)sub_1649960(*(_QWORD *)(a3 + 56));
  if ( v8 )
  {
    v38 = v40;
    sub_1634F50((__int64 *)&v38, v8, (__int64)&v8[v9]);
    v11 = (_BYTE *)sub_1649960(a3);
    if ( v11 )
    {
LABEL_5:
      v30[0] = (__int64)v31;
      sub_1634F50(v30, v11, (__int64)&v11[v10]);
      goto LABEL_6;
    }
  }
  else
  {
    v39 = 0;
    v38 = v40;
    LOBYTE(v40[0]) = 0;
    v11 = (_BYTE *)sub_1649960(a3);
    if ( v11 )
      goto LABEL_5;
  }
  LOBYTE(v31[0]) = 0;
  v30[0] = (__int64)v31;
  v30[1] = 0;
LABEL_6:
  v12 = (__m128i *)sub_2241130(v30, 0, 0, "basic block (", 13);
  v32 = &v34;
  if ( (__m128i *)v12->m128i_i64[0] == &v12[1] )
  {
    v34 = _mm_loadu_si128(v12 + 1);
  }
  else
  {
    v32 = (__m128i *)v12->m128i_i64[0];
    v34.m128i_i64[0] = v12[1].m128i_i64[0];
  }
  v13 = v12->m128i_i64[1];
  v12[1].m128i_i8[0] = 0;
  v33 = v13;
  v12->m128i_i64[0] = (__int64)v12[1].m128i_i64;
  v12->m128i_i64[1] = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v33) <= 0xE )
LABEL_42:
    sub_4262D8((__int64)"basic_string::append");
  v14 = (__m128i *)sub_2241490(&v32, ") in function (", 15, v13);
  v35 = &v37;
  if ( (__m128i *)v14->m128i_i64[0] == &v14[1] )
  {
    v37 = _mm_loadu_si128(v14 + 1);
  }
  else
  {
    v35 = (__m128i *)v14->m128i_i64[0];
    v37.m128i_i64[0] = v14[1].m128i_i64[0];
  }
  v15 = v14->m128i_i64[1];
  v14[1].m128i_i8[0] = 0;
  v36 = v15;
  v14->m128i_i64[0] = (__int64)v14[1].m128i_i64;
  v16 = v35;
  v14->m128i_i64[1] = 0;
  v17 = 15;
  v18 = 15;
  if ( v16 != &v37 )
    v18 = v37.m128i_i64[0];
  v19 = v36 + v39;
  if ( v36 + v39 <= v18 )
    goto LABEL_17;
  if ( v38 != v40 )
    v17 = v40[0];
  if ( v19 <= v17 )
  {
    v20 = (__m128i *)sub_2241130(&v38, 0, 0, v16, v36);
    v41 = v43;
    v21 = (_OWORD *)v20->m128i_i64[0];
    v22 = v20 + 1;
    if ( (__m128i *)v20->m128i_i64[0] != &v20[1] )
      goto LABEL_18;
  }
  else
  {
LABEL_17:
    v20 = (__m128i *)sub_2241490(&v35, v38, v39, v19);
    v41 = v43;
    v21 = (_OWORD *)v20->m128i_i64[0];
    v22 = v20 + 1;
    if ( (__m128i *)v20->m128i_i64[0] != &v20[1] )
    {
LABEL_18:
      v41 = v21;
      *(_QWORD *)&v43[0] = v20[1].m128i_i64[0];
      goto LABEL_19;
    }
  }
  v43[0] = _mm_loadu_si128(v20 + 1);
LABEL_19:
  v42 = v20->m128i_i64[1];
  v20->m128i_i64[0] = (__int64)v22;
  v20->m128i_i64[1] = 0;
  v20[1].m128i_i8[0] = 0;
  if ( v42 == 0x3FFFFFFFFFFFFFFFLL )
    goto LABEL_42;
  v23 = (__m128i *)sub_2241490(&v41, ")", 1, v43);
  v28 = &v29;
  if ( (__m128i *)v23->m128i_i64[0] == &v23[1] )
  {
    v29 = _mm_loadu_si128(v23 + 1);
  }
  else
  {
    v28 = (__m128i *)v23->m128i_i64[0];
    v29.m128i_i64[0] = v23[1].m128i_i64[0];
  }
  v24 = v23->m128i_u64[1];
  v23[1].m128i_i8[0] = 0;
  v23->m128i_i64[0] = (__int64)v23[1].m128i_i64;
  v25 = v41;
  v23->m128i_i64[1] = 0;
  if ( v25 != v43 )
    j_j___libc_free_0(v25, *(_QWORD *)&v43[0] + 1LL);
  if ( v35 != &v37 )
    j_j___libc_free_0(v35, v37.m128i_i64[0] + 1);
  if ( v32 != &v34 )
    j_j___libc_free_0(v32, v34.m128i_i64[0] + 1);
  if ( (_QWORD *)v30[0] != v31 )
    j_j___libc_free_0(v30[0], v31[0] + 1LL);
  if ( v38 != v40 )
    j_j___libc_free_0(v38, v40[0] + 1LL);
  v26 = (const char *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 16LL))(a2);
  v3 = sub_1635030(a1, v26, v27, v28->m128i_i8, v24);
  if ( v28 != &v29 )
    j_j___libc_free_0(v28, v29.m128i_i64[0] + 1);
  return v3;
}
