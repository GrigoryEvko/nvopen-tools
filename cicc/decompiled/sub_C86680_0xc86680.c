// Function: sub_C86680
// Address: 0xc86680
//
void __fastcall sub_C86680(_QWORD *a1, __int64 a2, int a3)
{
  __int64 v4; // rsi
  __int64 v6; // rcx
  _BYTE *v7; // r15
  size_t v8; // r12
  _QWORD *v9; // rax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rcx
  __m128i *v13; // rax
  _OWORD *v14; // rcx
  __m128i *v15; // rdx
  _BYTE *v16; // rdi
  size_t v17; // rdx
  __int64 v18; // rsi
  __int64 v19; // rcx
  _QWORD *v20; // rdi
  _QWORD *v21; // [rsp-98h] [rbp-98h] BYREF
  size_t v22; // [rsp-90h] [rbp-90h]
  _QWORD v23[2]; // [rsp-88h] [rbp-88h] BYREF
  _QWORD *v24; // [rsp-78h] [rbp-78h] BYREF
  __int64 v25; // [rsp-70h] [rbp-70h]
  _QWORD v26[2]; // [rsp-68h] [rbp-68h] BYREF
  _OWORD *v27; // [rsp-58h] [rbp-58h] BYREF
  size_t v28; // [rsp-50h] [rbp-50h]
  _OWORD v29[4]; // [rsp-48h] [rbp-48h] BYREF

  if ( !a1 )
    return;
  v4 = 0;
  if ( a3 == -1 )
    v4 = (unsigned int)*__errno_location();
  sub_F03820(&v24, v4);
  v7 = *(_BYTE **)a2;
  v8 = *(_QWORD *)(a2 + 8);
  v21 = v23;
  if ( &v7[v8] && !v7 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v27 = (_OWORD *)v8;
  if ( v8 > 0xF )
  {
    v21 = (_QWORD *)sub_22409D0(&v21, &v27, 0);
    v20 = v21;
    v23[0] = v27;
  }
  else
  {
    if ( v8 == 1 )
    {
      LOBYTE(v23[0]) = *v7;
      v9 = v23;
      goto LABEL_9;
    }
    if ( !v8 )
    {
      v9 = v23;
      goto LABEL_9;
    }
    v20 = v23;
  }
  memcpy(v20, v7, v8);
  v8 = (size_t)v27;
  v9 = v21;
LABEL_9:
  v22 = v8;
  *((_BYTE *)v9 + v8) = 0;
  if ( v22 == 0x3FFFFFFFFFFFFFFFLL || v22 == 4611686018427387902LL )
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490(&v21, ": ", 2, v6);
  v10 = 15;
  v11 = 15;
  if ( v21 != v23 )
    v11 = v23[0];
  v12 = v22 + v25;
  if ( v22 + v25 <= v11 )
    goto LABEL_16;
  if ( v24 != v26 )
    v10 = v26[0];
  if ( v12 <= v10 )
  {
    v13 = (__m128i *)sub_2241130(&v24, 0, 0, v21, v22);
    v27 = v29;
    v14 = (_OWORD *)v13->m128i_i64[0];
    v15 = v13 + 1;
    if ( (__m128i *)v13->m128i_i64[0] != &v13[1] )
      goto LABEL_17;
  }
  else
  {
LABEL_16:
    v13 = (__m128i *)sub_2241490(&v21, v24, v25, v12);
    v27 = v29;
    v14 = (_OWORD *)v13->m128i_i64[0];
    v15 = v13 + 1;
    if ( (__m128i *)v13->m128i_i64[0] != &v13[1] )
    {
LABEL_17:
      v27 = v14;
      *(_QWORD *)&v29[0] = v13[1].m128i_i64[0];
      goto LABEL_18;
    }
  }
  v29[0] = _mm_loadu_si128(v13 + 1);
LABEL_18:
  v28 = v13->m128i_u64[1];
  v13->m128i_i64[0] = (__int64)v15;
  v13->m128i_i64[1] = 0;
  v13[1].m128i_i8[0] = 0;
  v16 = (_BYTE *)*a1;
  v17 = v28;
  if ( v27 == v29 )
  {
    if ( v28 )
    {
      if ( v28 == 1 )
        *v16 = v29[0];
      else
        memcpy(v16, v29, v28);
      v17 = v28;
      v16 = (_BYTE *)*a1;
    }
    a1[1] = v17;
    v16[v17] = 0;
    v16 = v27;
    goto LABEL_22;
  }
  v18 = *(_QWORD *)&v29[0];
  if ( v16 == (_BYTE *)(a1 + 2) )
  {
    *a1 = v27;
    a1[1] = v17;
    a1[2] = v18;
    goto LABEL_43;
  }
  v19 = a1[2];
  *a1 = v27;
  a1[1] = v17;
  a1[2] = v18;
  if ( !v16 )
  {
LABEL_43:
    v27 = v29;
    v16 = v29;
    goto LABEL_22;
  }
  v27 = v16;
  *(_QWORD *)&v29[0] = v19;
LABEL_22:
  v28 = 0;
  *v16 = 0;
  if ( v27 != v29 )
    j_j___libc_free_0(v27, *(_QWORD *)&v29[0] + 1LL);
  if ( v21 != v23 )
    j_j___libc_free_0(v21, v23[0] + 1LL);
  if ( v24 != v26 )
    j_j___libc_free_0(v24, v26[0] + 1LL);
}
