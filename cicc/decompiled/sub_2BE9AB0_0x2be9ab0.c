// Function: sub_2BE9AB0
// Address: 0x2be9ab0
//
void __fastcall sub_2BE9AB0(__int64 a1, char a2, char a3)
{
  __int64 v4; // rax
  size_t v5; // r8
  __int64 v6; // r9
  __m128i *v7; // rax
  unsigned __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // r12
  __m128i *v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __m128i *v14; // rdi
  __int64 v15; // rax
  __m128i *v16; // rdi
  __int64 v17; // [rsp+0h] [rbp-100h]
  __int8 *v18; // [rsp+10h] [rbp-F0h]
  size_t v19; // [rsp+18h] [rbp-E8h]
  __int64 v20; // [rsp+18h] [rbp-E8h]
  __int8 *src; // [rsp+20h] [rbp-E0h]
  unsigned __int64 *v22; // [rsp+28h] [rbp-D8h]
  __m128i *v23; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v24; // [rsp+38h] [rbp-C8h]
  __m128i v25; // [rsp+40h] [rbp-C0h] BYREF
  __m128i *v26; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v27; // [rsp+58h] [rbp-A8h]
  __m128i v28; // [rsp+60h] [rbp-A0h] BYREF
  __int8 *v29; // [rsp+70h] [rbp-90h] BYREF
  size_t n; // [rsp+78h] [rbp-88h]
  _QWORD v31[2]; // [rsp+80h] [rbp-80h] BYREF
  __m128i v32; // [rsp+90h] [rbp-70h] BYREF
  __m128i v33; // [rsp+A0h] [rbp-60h] BYREF
  __m128i *v34; // [rsp+B0h] [rbp-50h]
  __int64 v35; // [rsp+B8h] [rbp-48h]
  _OWORD v36[4]; // [rsp+C0h] [rbp-40h] BYREF

  if ( a2 > a3 )
    abort();
  v22 = (unsigned __int64 *)(a1 + 48);
  v29 = (__int8 *)v31;
  sub_2240A50((__int64 *)&v29, 1u, a3);
  v19 = n;
  v18 = v29;
  v4 = sub_221F880(*(_QWORD **)(a1 + 104), 1);
  v5 = v19;
  v32.m128i_i64[0] = (__int64)&v33;
  v6 = v4;
  v26 = (__m128i *)v19;
  if ( v19 > 0xF )
  {
    v17 = v4;
    v15 = sub_22409D0((__int64)&v32, (unsigned __int64 *)&v26, 0);
    v5 = v19;
    v32.m128i_i64[0] = v15;
    v16 = (__m128i *)v15;
    v6 = v17;
    v33.m128i_i64[0] = (__int64)v26;
  }
  else
  {
    if ( v19 == 1 )
    {
      v33.m128i_i8[0] = *v18;
      v7 = &v33;
      goto LABEL_5;
    }
    if ( !v19 )
    {
      v7 = &v33;
      goto LABEL_5;
    }
    v16 = &v33;
  }
  v20 = v6;
  memcpy(v16, v18, v5);
  v5 = (size_t)v26;
  v7 = (__m128i *)v32.m128i_i64[0];
  v6 = v20;
LABEL_5:
  v32.m128i_i64[1] = v5;
  v7->m128i_i8[v5] = 0;
  (*(void (__fastcall **)(__m128i **, __int64, __int64, __int64))(*(_QWORD *)v6 + 24LL))(
    &v26,
    v6,
    v32.m128i_i64[0],
    v32.m128i_i64[0] + v32.m128i_i64[1]);
  if ( (__m128i *)v32.m128i_i64[0] != &v33 )
    j_j___libc_free_0(v32.m128i_u64[0]);
  if ( v29 != (__int8 *)v31 )
    j_j___libc_free_0((unsigned __int64)v29);
  v29 = (__int8 *)v31;
  sub_2240A50((__int64 *)&v29, 1u, a2);
  v8 = n;
  src = v29;
  v9 = sub_221F880(*(_QWORD **)(a1 + 104), 1);
  v32.m128i_i64[0] = (__int64)&v33;
  v23 = (__m128i *)v8;
  v10 = v9;
  if ( v8 > 0xF )
  {
    v32.m128i_i64[0] = sub_22409D0((__int64)&v32, (unsigned __int64 *)&v23, 0);
    v14 = (__m128i *)v32.m128i_i64[0];
    v33.m128i_i64[0] = (__int64)v23;
  }
  else
  {
    if ( v8 == 1 )
    {
      v33.m128i_i8[0] = *src;
      v11 = &v33;
      goto LABEL_12;
    }
    if ( !v8 )
    {
      v11 = &v33;
      goto LABEL_12;
    }
    v14 = &v33;
  }
  memcpy(v14, src, v8);
  v8 = (unsigned __int64)v23;
  v11 = (__m128i *)v32.m128i_i64[0];
LABEL_12:
  v32.m128i_i64[1] = v8;
  v11->m128i_i8[v8] = 0;
  (*(void (__fastcall **)(__m128i **, __int64, __int64, __int64))(*(_QWORD *)v10 + 24LL))(
    &v23,
    v10,
    v32.m128i_i64[0],
    v32.m128i_i64[0] + v32.m128i_i64[1]);
  if ( (__m128i *)v32.m128i_i64[0] != &v33 )
    j_j___libc_free_0(v32.m128i_u64[0]);
  if ( v29 != (__int8 *)v31 )
    j_j___libc_free_0((unsigned __int64)v29);
  v32.m128i_i64[0] = (__int64)&v33;
  if ( v23 == &v25 )
  {
    v33 = _mm_load_si128(&v25);
  }
  else
  {
    v32.m128i_i64[0] = (__int64)v23;
    v33.m128i_i64[0] = v25.m128i_i64[0];
  }
  v12 = v24;
  v23 = &v25;
  v24 = 0;
  v32.m128i_i64[1] = v12;
  v25.m128i_i8[0] = 0;
  v34 = (__m128i *)v36;
  if ( v26 == &v28 )
  {
    v36[0] = _mm_load_si128(&v28);
  }
  else
  {
    v34 = v26;
    *(_QWORD *)&v36[0] = v28.m128i_i64[0];
  }
  v13 = v27;
  v26 = &v28;
  v27 = 0;
  v35 = v13;
  v28.m128i_i8[0] = 0;
  sub_2BE99F0(v22, &v32);
  if ( v34 != (__m128i *)v36 )
    j_j___libc_free_0((unsigned __int64)v34);
  if ( (__m128i *)v32.m128i_i64[0] != &v33 )
    j_j___libc_free_0(v32.m128i_u64[0]);
  if ( v23 != &v25 )
    j_j___libc_free_0((unsigned __int64)v23);
  if ( v26 != &v28 )
    j_j___libc_free_0((unsigned __int64)v26);
}
