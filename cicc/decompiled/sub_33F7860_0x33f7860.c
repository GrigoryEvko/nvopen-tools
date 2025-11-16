// Function: sub_33F7860
// Address: 0x33f7860
//
void __fastcall sub_33F7860(const __m128i *a1)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // r8
  __int64 v5; // r9
  __m128i v6; // xmm0
  __int64 v7; // rax
  __int64 v8; // rbx
  const __m128i *v9; // r13
  _BYTE **v10; // rdi
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  __int64 v13; // rbx
  __int64 v14; // r13
  __m128i v15; // [rsp+0h] [rbp-510h] BYREF
  __int64 v16; // [rsp+10h] [rbp-500h]
  __int64 v17; // [rsp+18h] [rbp-4F8h]
  __int64 v18; // [rsp+20h] [rbp-4F0h]
  __int64 v19; // [rsp+28h] [rbp-4E8h]
  __m128i v20; // [rsp+30h] [rbp-4E0h]
  _QWORD v21[8]; // [rsp+40h] [rbp-4D0h] BYREF
  __int64 v22; // [rsp+80h] [rbp-490h]
  int v23; // [rsp+88h] [rbp-488h]
  __int64 v24; // [rsp+90h] [rbp-480h]
  __int64 v25; // [rsp+98h] [rbp-478h]
  __int64 v26; // [rsp+A0h] [rbp-470h] BYREF
  __int64 v27; // [rsp+A8h] [rbp-468h]
  _QWORD *v28; // [rsp+B0h] [rbp-460h]
  __int64 v29; // [rsp+B8h] [rbp-458h]
  __int64 v30; // [rsp+C0h] [rbp-450h] BYREF
  _BYTE *v31; // [rsp+D0h] [rbp-440h] BYREF
  __int64 i; // [rsp+D8h] [rbp-438h]
  _BYTE v33[1072]; // [rsp+E0h] [rbp-430h] BYREF

  v2 = a1[24].m128i_i64[0];
  v15 = _mm_loadu_si128(a1 + 24);
  v3 = sub_33ECD10(1u);
  v28 = v21;
  v6 = _mm_load_si128(&v15);
  v21[6] = v3;
  v22 = 0x100000000LL;
  v25 = 0xFFFFFFFFLL;
  v20 = v6;
  v30 = 0;
  v21[7] = 0;
  v23 = 0;
  v24 = 0;
  v29 = 0;
  LODWORD(v27) = v6.m128i_i32[2];
  v26 = v6.m128i_i64[0];
  v7 = *(_QWORD *)(v2 + 56);
  memset(v21, 0, 24);
  v21[3] = 328;
  v21[4] = -65536;
  v30 = v7;
  if ( v7 )
    *(_QWORD *)(v7 + 24) = &v30;
  v29 = v2 + 56;
  *(_QWORD *)(v2 + 56) = &v26;
  v8 = a1[25].m128i_i64[1];
  v9 = a1 + 25;
  v10 = &v31;
  v21[5] = &v26;
  LODWORD(v22) = 1;
  v31 = v33;
  for ( i = 0x8000000000LL; v9 != (const __m128i *)v8; v8 = *(_QWORD *)(v8 + 8) )
  {
    while ( 1 )
    {
      if ( !v8 )
        BUG();
      if ( !*(_QWORD *)(v8 + 48) )
        break;
      v8 = *(_QWORD *)(v8 + 8);
      if ( v9 == (const __m128i *)v8 )
        goto LABEL_11;
    }
    v11 = (unsigned int)i;
    v12 = (unsigned int)i + 1LL;
    if ( v12 > HIDWORD(i) )
    {
      v15.m128i_i64[0] = (__int64)v10;
      sub_C8D5F0((__int64)v10, v33, v12, 8u, v4, v5);
      v11 = (unsigned int)i;
      v10 = (_BYTE **)v15.m128i_i64[0];
    }
    *(_QWORD *)&v31[8 * v11] = v8 - 8;
    LODWORD(i) = i + 1;
  }
LABEL_11:
  sub_33EBD60((__int64)a1, (__int64)&v31);
  v13 = v26;
  v14 = v27;
  if ( v26 )
  {
    nullsub_1875();
    a1[24].m128i_i64[0] = v13;
    a1[24].m128i_i32[2] = v14;
    v18 = v13;
    v19 = v14;
    sub_33E2B60();
  }
  else
  {
    v16 = 0;
    v17 = v27;
    a1[24].m128i_i64[0] = 0;
    a1[24].m128i_i32[2] = v14;
  }
  if ( v31 != v33 )
    _libc_free((unsigned __int64)v31);
  sub_33CF710((__int64)v21);
}
