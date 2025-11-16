// Function: sub_103F8F0
// Address: 0x103f8f0
//
__int64 __fastcall sub_103F8F0(__int64 a1, __int64 a2)
{
  char *v4; // rax
  __int64 v5; // rdx
  __m128i *v6; // rax
  __int64 v7; // rsi
  __m128i *v8; // rax
  __int64 v9; // rsi
  __m128i *v10; // rdi
  __int64 v11; // rdi
  __int64 v12; // r8
  __int64 result; // rax
  _QWORD *v14; // rsi
  __int64 v15; // rax
  size_t v16; // rdx
  unsigned __int8 *v17; // rsi
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // r13
  __int64 v21; // [rsp+8h] [rbp-98h]
  __int64 v22; // [rsp+8h] [rbp-98h]
  __int64 v23; // [rsp+8h] [rbp-98h]
  __m128i *v24; // [rsp+10h] [rbp-90h] BYREF
  __int64 v25; // [rsp+18h] [rbp-88h]
  __m128i v26; // [rsp+20h] [rbp-80h] BYREF
  __m128i *v27; // [rsp+30h] [rbp-70h] BYREF
  __int64 v28; // [rsp+38h] [rbp-68h]
  __m128i v29; // [rsp+40h] [rbp-60h] BYREF
  unsigned __int8 *v30; // [rsp+50h] [rbp-50h] BYREF
  size_t v31; // [rsp+58h] [rbp-48h]
  _QWORD v32[8]; // [rsp+60h] [rbp-40h] BYREF

  v4 = (char *)sub_BD5D20(***(_QWORD ***)(a1 + 8));
  if ( v4 )
  {
    v30 = (unsigned __int8 *)v32;
    sub_103ABA0((__int64 *)&v30, v4, (__int64)&v4[v5]);
  }
  else
  {
    v31 = 0;
    v30 = (unsigned __int8 *)v32;
    LOBYTE(v32[0]) = 0;
  }
  v6 = (__m128i *)sub_2241130(&v30, 0, 0, "MSSA CFG for '", 14);
  v27 = &v29;
  if ( (__m128i *)v6->m128i_i64[0] == &v6[1] )
  {
    v29 = _mm_loadu_si128(v6 + 1);
  }
  else
  {
    v27 = (__m128i *)v6->m128i_i64[0];
    v29.m128i_i64[0] = v6[1].m128i_i64[0];
  }
  v7 = v6->m128i_i64[1];
  v6[1].m128i_i8[0] = 0;
  v28 = v7;
  v6->m128i_i64[0] = (__int64)v6[1].m128i_i64;
  v6->m128i_i64[1] = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v28) <= 9 )
    sub_4262D8((__int64)"basic_string::append");
  v8 = (__m128i *)sub_2241490(&v27, "' function", 10, &v29);
  v24 = &v26;
  if ( (__m128i *)v8->m128i_i64[0] == &v8[1] )
  {
    v26 = _mm_loadu_si128(v8 + 1);
  }
  else
  {
    v24 = (__m128i *)v8->m128i_i64[0];
    v26.m128i_i64[0] = v8[1].m128i_i64[0];
  }
  v9 = v8->m128i_i64[1];
  v8[1].m128i_i8[0] = 0;
  v25 = v9;
  v8->m128i_i64[0] = (__int64)v8[1].m128i_i64;
  v10 = v27;
  v8->m128i_i64[1] = 0;
  if ( v10 != &v29 )
    j_j___libc_free_0(v10, v29.m128i_i64[0] + 1);
  if ( v30 != (unsigned __int8 *)v32 )
    j_j___libc_free_0(v30, v32[0] + 1LL);
  v11 = *(_QWORD *)a1;
  if ( *(_QWORD *)(a2 + 8) )
  {
    v14 = (_QWORD *)a2;
    v22 = sub_904010(v11, "digraph \"");
  }
  else
  {
    if ( !v25 )
    {
      sub_904010(v11, "digraph unnamed {\n");
      goto LABEL_15;
    }
    v14 = &v24;
    v22 = sub_904010(v11, "digraph \"");
  }
  sub_C67200((__int64 *)&v30, (__int64)v14);
  v15 = sub_CB6200(v22, v30, v31);
  sub_904010(v15, "\" {\n");
  if ( v30 != (unsigned __int8 *)v32 )
  {
    j_j___libc_free_0(v30, v32[0] + 1LL);
    v12 = *(_QWORD *)a1;
    if ( !*(_QWORD *)(a2 + 8) )
      goto LABEL_16;
LABEL_25:
    v23 = sub_904010(v12, "\tlabel=\"");
    sub_C67200((__int64 *)&v30, a2);
    v16 = v31;
    v17 = v30;
    v18 = v23;
    goto LABEL_26;
  }
LABEL_15:
  v12 = *(_QWORD *)a1;
  if ( *(_QWORD *)(a2 + 8) )
    goto LABEL_25;
LABEL_16:
  if ( !v25 )
    goto LABEL_17;
  v20 = sub_904010(v12, "\tlabel=\"");
  sub_C67200((__int64 *)&v30, (__int64)&v24);
  v16 = v31;
  v17 = v30;
  v18 = v20;
LABEL_26:
  v19 = sub_CB6200(v18, v17, v16);
  sub_904010(v19, "\";\n");
  if ( v30 != (unsigned __int8 *)v32 )
    j_j___libc_free_0(v30, v32[0] + 1LL);
  v12 = *(_QWORD *)a1;
LABEL_17:
  v21 = v12;
  v30 = (unsigned __int8 *)v32;
  sub_103ABA0((__int64 *)&v30, byte_3F871B3, (__int64)byte_3F871B3);
  sub_CB6200(v21, v30, v31);
  if ( v30 != (unsigned __int8 *)v32 )
    j_j___libc_free_0(v30, v32[0] + 1LL);
  result = sub_904010(*(_QWORD *)a1, "\n");
  if ( v24 != &v26 )
    return j_j___libc_free_0(v24, v26.m128i_i64[0] + 1);
  return result;
}
