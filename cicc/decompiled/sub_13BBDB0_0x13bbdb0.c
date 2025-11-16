// Function: sub_13BBDB0
// Address: 0x13bbdb0
//
__int64 __fastcall sub_13BBDB0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  _BYTE *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rcx
  __m128i *v12; // rax
  _OWORD *v13; // rsi
  __m128i *v14; // rdx
  __m128i *v15; // rax
  __int64 v16; // rsi
  __int64 v18; // [rsp+18h] [rbp-D8h] BYREF
  _BYTE *v19; // [rsp+20h] [rbp-D0h]
  __int64 v20; // [rsp+28h] [rbp-C8h]
  _QWORD v21[2]; // [rsp+30h] [rbp-C0h] BYREF
  _QWORD v22[2]; // [rsp+40h] [rbp-B0h] BYREF
  __m128i v23; // [rsp+50h] [rbp-A0h] BYREF
  _QWORD *v24; // [rsp+60h] [rbp-90h] BYREF
  __int64 v25; // [rsp+68h] [rbp-88h]
  _QWORD v26[2]; // [rsp+70h] [rbp-80h] BYREF
  _QWORD *v27; // [rsp+80h] [rbp-70h] BYREF
  __int64 v28; // [rsp+88h] [rbp-68h]
  _QWORD v29[2]; // [rsp+90h] [rbp-60h] BYREF
  _OWORD *v30; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v31; // [rsp+A8h] [rbp-48h]
  _OWORD v32[4]; // [rsp+B0h] [rbp-40h] BYREF

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_36:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F9E06C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_36;
  }
  v18 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F9E06C)
      + 160;
  v19 = v21;
  strcpy((char *)v21, "Dominator tree");
  v20 = 14;
  v6 = (_BYTE *)sub_1649960(a2);
  if ( v6 )
  {
    v27 = v29;
    sub_13B5840((__int64 *)&v27, v6, (__int64)&v6[v7]);
  }
  else
  {
    v28 = 0;
    v27 = v29;
    LOBYTE(v29[0]) = 0;
  }
  v24 = v26;
  sub_13B5790((__int64 *)&v24, v19, (__int64)&v19[v20]);
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v25) <= 5 )
LABEL_35:
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490(&v24, " for '", 6, v8);
  v9 = 15;
  v10 = 15;
  if ( v24 != v26 )
    v10 = v26[0];
  v11 = v25 + v28;
  if ( v25 + v28 <= v10 )
    goto LABEL_14;
  if ( v27 != v29 )
    v9 = v29[0];
  if ( v11 <= v9 )
  {
    v12 = (__m128i *)sub_2241130(&v27, 0, 0, v24, v25);
    v30 = v32;
    v13 = (_OWORD *)v12->m128i_i64[0];
    v14 = v12 + 1;
    if ( (__m128i *)v12->m128i_i64[0] != &v12[1] )
      goto LABEL_15;
  }
  else
  {
LABEL_14:
    v12 = (__m128i *)sub_2241490(&v24, v27, v28, v11);
    v30 = v32;
    v13 = (_OWORD *)v12->m128i_i64[0];
    v14 = v12 + 1;
    if ( (__m128i *)v12->m128i_i64[0] != &v12[1] )
    {
LABEL_15:
      v30 = v13;
      *(_QWORD *)&v32[0] = v12[1].m128i_i64[0];
      goto LABEL_16;
    }
  }
  v32[0] = _mm_loadu_si128(v12 + 1);
LABEL_16:
  v31 = v12->m128i_i64[1];
  v12->m128i_i64[0] = (__int64)v14;
  v12->m128i_i64[1] = 0;
  v12[1].m128i_i8[0] = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v31) <= 9 )
    goto LABEL_35;
  v15 = (__m128i *)sub_2241490(&v30, "' function", 10, v32);
  v22[0] = &v23;
  if ( (__m128i *)v15->m128i_i64[0] == &v15[1] )
  {
    v23 = _mm_loadu_si128(v15 + 1);
  }
  else
  {
    v22[0] = v15->m128i_i64[0];
    v23.m128i_i64[0] = v15[1].m128i_i64[0];
  }
  v16 = v15->m128i_i64[1];
  v15[1].m128i_i8[0] = 0;
  v22[1] = v16;
  v15->m128i_i64[0] = (__int64)v15[1].m128i_i64;
  v15->m128i_i64[1] = 0;
  if ( v30 != v32 )
    j_j___libc_free_0(v30, *(_QWORD *)&v32[0] + 1LL);
  if ( v24 != v26 )
    j_j___libc_free_0(v24, v26[0] + 1LL);
  if ( v27 != v29 )
    j_j___libc_free_0(v27, v29[0] + 1LL);
  LOWORD(v32[0]) = 260;
  LOWORD(v29[0]) = 260;
  v30 = v22;
  v27 = (_QWORD *)(a1 + 160);
  sub_13BB990((__int64)&v18, (__int64)&v27, 1, (__int64)&v30, 0);
  if ( (__m128i *)v22[0] != &v23 )
    j_j___libc_free_0(v22[0], v23.m128i_i64[0] + 1);
  if ( v19 != (_BYTE *)v21 )
    j_j___libc_free_0(v19, v21[0] + 1LL);
  return 0;
}
