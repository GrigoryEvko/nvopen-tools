// Function: sub_2531A30
// Address: 0x2531a30
//
void __fastcall sub_2531A30(__int64 a1)
{
  unsigned __int64 v1; // rax
  unsigned __int64 v2; // rdi
  __m128i *v3; // rax
  unsigned __int64 v4; // rsi
  __m128i *v5; // rdx
  unsigned __int64 *v6; // rax
  __int64 v7; // r8
  __int64 v8; // rdx
  _OWORD *v9; // rcx
  __int64 v10; // rsi
  char *v11; // rdi
  void *v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 (__fastcall **v15)(); // rax
  __int64 *v16; // rsi
  __int64 v18; // [rsp+28h] [rbp-128h] BYREF
  _BYTE *v19; // [rsp+30h] [rbp-120h] BYREF
  __int64 v20; // [rsp+38h] [rbp-118h]
  _BYTE v21[16]; // [rsp+40h] [rbp-110h] BYREF
  __m128i *v22; // [rsp+50h] [rbp-100h]
  size_t v23; // [rsp+58h] [rbp-F8h]
  __m128i v24; // [rsp+60h] [rbp-F0h] BYREF
  _BYTE *v25; // [rsp+70h] [rbp-E0h] BYREF
  __int64 (__fastcall **v26)(); // [rsp+78h] [rbp-D8h]
  _QWORD v27[2]; // [rsp+80h] [rbp-D0h] BYREF
  char *v28; // [rsp+90h] [rbp-C0h] BYREF
  size_t v29; // [rsp+98h] [rbp-B8h]
  _QWORD v30[2]; // [rsp+A0h] [rbp-B0h] BYREF
  __int16 v31; // [rsp+B0h] [rbp-A0h]
  _OWORD *v32; // [rsp+C0h] [rbp-90h] BYREF
  __int64 v33; // [rsp+C8h] [rbp-88h]
  _OWORD v34[8]; // [rsp+D0h] [rbp-80h] BYREF

  if ( !byte_4FEE4B0 && (unsigned int)sub_2207590((__int64)&byte_4FEE4B0) )
    sub_2207640((__int64)&byte_4FEE4B0);
  v20 = 0;
  v19 = v21;
  v21[0] = 0;
  if ( qword_4FEEAB0 )
    sub_2240AE0((unsigned __int64 *)&v19, (unsigned __int64 *)&qword_4FEEAA8);
  else
    sub_2241130((unsigned __int64 *)&v19, 0, 0, "dep_graph", 9u);
  sub_2509010((__int64 *)&v28, dword_4FEE4B8);
  v25 = v27;
  sub_2506C40((__int64 *)&v25, v19, (__int64)&v19[v20]);
  if ( v26 == (__int64 (__fastcall **)())0x3FFFFFFFFFFFFFFFLL )
LABEL_36:
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490((unsigned __int64 *)&v25, "_", 1u);
  v1 = 15;
  v2 = 15;
  if ( v25 != (_BYTE *)v27 )
    v2 = v27[0];
  if ( (unsigned __int64)v26 + v29 <= v2 )
    goto LABEL_11;
  if ( v28 != (char *)v30 )
    v1 = v30[0];
  if ( (unsigned __int64)v26 + v29 <= v1 )
  {
    v3 = (__m128i *)sub_2241130((unsigned __int64 *)&v28, 0, 0, v25, (size_t)v26);
    v32 = v34;
    v4 = v3->m128i_i64[0];
    v5 = v3 + 1;
    if ( (__m128i *)v3->m128i_i64[0] != &v3[1] )
      goto LABEL_12;
  }
  else
  {
LABEL_11:
    v3 = (__m128i *)sub_2241490((unsigned __int64 *)&v25, v28, v29);
    v32 = v34;
    v4 = v3->m128i_i64[0];
    v5 = v3 + 1;
    if ( (__m128i *)v3->m128i_i64[0] != &v3[1] )
    {
LABEL_12:
      v32 = (_OWORD *)v4;
      *(_QWORD *)&v34[0] = v3[1].m128i_i64[0];
      goto LABEL_13;
    }
  }
  v34[0] = _mm_loadu_si128(v3 + 1);
LABEL_13:
  v33 = v3->m128i_i64[1];
  v3->m128i_i64[0] = (__int64)v5;
  v3->m128i_i64[1] = 0;
  v3[1].m128i_i8[0] = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v33) <= 3 )
    goto LABEL_36;
  v6 = sub_2241490((unsigned __int64 *)&v32, ".dot", 4u);
  v22 = &v24;
  v8 = (__int64)(v6 + 2);
  v9 = v34;
  if ( (unsigned __int64 *)*v6 == v6 + 2 )
  {
    v24 = _mm_loadu_si128((const __m128i *)v6 + 1);
  }
  else
  {
    v22 = (__m128i *)*v6;
    v24.m128i_i64[0] = v6[2];
  }
  v10 = v6[1];
  *((_BYTE *)v6 + 16) = 0;
  v23 = v10;
  *v6 = v8;
  v6[1] = 0;
  if ( v32 != v34 )
  {
    v10 = *(_QWORD *)&v34[0] + 1LL;
    j_j___libc_free_0((unsigned __int64)v32);
  }
  if ( v25 != (_BYTE *)v27 )
  {
    v10 = v27[0] + 1LL;
    j_j___libc_free_0((unsigned __int64)v25);
  }
  v11 = v28;
  if ( v28 != (char *)v30 )
  {
    v10 = v30[0] + 1LL;
    j_j___libc_free_0((unsigned __int64)v28);
  }
  v12 = sub_CB7210((__int64)v11, v10, v8, (__int64)v9, v7);
  v13 = sub_904010((__int64)v12, "Dependency graph dump to ");
  v14 = sub_CB6200(v13, (unsigned __int8 *)v22, v23);
  sub_904010(v14, ".\n");
  LODWORD(v25) = 0;
  v15 = sub_2241E40();
  v16 = (__int64 *)v22;
  v26 = v15;
  sub_CB7060((__int64)&v32, v22, v23, (__int64)&v25, 3u);
  if ( !(_DWORD)v25 )
  {
    v31 = 257;
    v16 = &v18;
    v18 = a1;
    sub_2514F80((__int64)&v32, (__int64)&v18, 0, (void **)&v28);
  }
  _InterlockedAdd(&dword_4FEE4B8, 1u);
  sub_CB5B00((int *)&v32, (__int64)v16);
  if ( v22 != &v24 )
    j_j___libc_free_0((unsigned __int64)v22);
  if ( v19 != v21 )
    j_j___libc_free_0((unsigned __int64)v19);
}
