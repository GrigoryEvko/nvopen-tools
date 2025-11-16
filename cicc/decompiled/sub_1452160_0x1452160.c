// Function: sub_1452160
// Address: 0x1452160
//
__int64 __fastcall sub_1452160(__int64 a1, __int64 a2)
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
  _BYTE *v19[2]; // [rsp+20h] [rbp-D0h] BYREF
  _QWORD v20[2]; // [rsp+30h] [rbp-C0h] BYREF
  _QWORD v21[2]; // [rsp+40h] [rbp-B0h] BYREF
  __m128i v22; // [rsp+50h] [rbp-A0h] BYREF
  _QWORD *v23; // [rsp+60h] [rbp-90h] BYREF
  __int64 v24; // [rsp+68h] [rbp-88h]
  _QWORD v25[2]; // [rsp+70h] [rbp-80h] BYREF
  _QWORD *v26; // [rsp+80h] [rbp-70h] BYREF
  __int64 v27; // [rsp+88h] [rbp-68h]
  _QWORD v28[2]; // [rsp+90h] [rbp-60h] BYREF
  _OWORD *v29; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v30; // [rsp+A8h] [rbp-48h]
  _OWORD v31[4]; // [rsp+B0h] [rbp-40h] BYREF

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_36:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F9A04C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_36;
  }
  v18 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F9A04C)
      + 160;
  v19[0] = v20;
  sub_144C6E0((__int64 *)v19, "Region Graph", (__int64)"");
  v6 = (_BYTE *)sub_1649960(a2);
  if ( v6 )
  {
    v26 = v28;
    sub_144C6E0((__int64 *)&v26, v6, (__int64)&v6[v7]);
  }
  else
  {
    v27 = 0;
    v26 = v28;
    LOBYTE(v28[0]) = 0;
  }
  v23 = v25;
  sub_144C790((__int64 *)&v23, v19[0], (__int64)&v19[0][(unsigned __int64)v19[1]]);
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v24) <= 5 )
LABEL_35:
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490(&v23, " for '", 6, v8);
  v9 = 15;
  v10 = 15;
  if ( v23 != v25 )
    v10 = v25[0];
  v11 = v24 + v27;
  if ( v24 + v27 <= v10 )
    goto LABEL_14;
  if ( v26 != v28 )
    v9 = v28[0];
  if ( v11 <= v9 )
  {
    v12 = (__m128i *)sub_2241130(&v26, 0, 0, v23, v24);
    v29 = v31;
    v13 = (_OWORD *)v12->m128i_i64[0];
    v14 = v12 + 1;
    if ( (__m128i *)v12->m128i_i64[0] != &v12[1] )
      goto LABEL_15;
  }
  else
  {
LABEL_14:
    v12 = (__m128i *)sub_2241490(&v23, v26, v27, v11);
    v29 = v31;
    v13 = (_OWORD *)v12->m128i_i64[0];
    v14 = v12 + 1;
    if ( (__m128i *)v12->m128i_i64[0] != &v12[1] )
    {
LABEL_15:
      v29 = v13;
      *(_QWORD *)&v31[0] = v12[1].m128i_i64[0];
      goto LABEL_16;
    }
  }
  v31[0] = _mm_loadu_si128(v12 + 1);
LABEL_16:
  v30 = v12->m128i_i64[1];
  v12->m128i_i64[0] = (__int64)v14;
  v12->m128i_i64[1] = 0;
  v12[1].m128i_i8[0] = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v30) <= 9 )
    goto LABEL_35;
  v15 = (__m128i *)sub_2241490(&v29, "' function", 10, v31);
  v21[0] = &v22;
  if ( (__m128i *)v15->m128i_i64[0] == &v15[1] )
  {
    v22 = _mm_loadu_si128(v15 + 1);
  }
  else
  {
    v21[0] = v15->m128i_i64[0];
    v22.m128i_i64[0] = v15[1].m128i_i64[0];
  }
  v16 = v15->m128i_i64[1];
  v15[1].m128i_i8[0] = 0;
  v21[1] = v16;
  v15->m128i_i64[0] = (__int64)v15[1].m128i_i64;
  v15->m128i_i64[1] = 0;
  if ( v29 != v31 )
    j_j___libc_free_0(v29, *(_QWORD *)&v31[0] + 1LL);
  if ( v23 != v25 )
    j_j___libc_free_0(v23, v25[0] + 1LL);
  if ( v26 != v28 )
    j_j___libc_free_0(v26, v28[0] + 1LL);
  LOWORD(v31[0]) = 260;
  LOWORD(v28[0]) = 260;
  v29 = v21;
  v26 = (_QWORD *)(a1 + 160);
  sub_1451D60(&v18, (__int64)&v26, 1, (__int64)&v29, 0);
  if ( (__m128i *)v21[0] != &v22 )
    j_j___libc_free_0(v21[0], v22.m128i_i64[0] + 1);
  if ( (_QWORD *)v19[0] != v20 )
    j_j___libc_free_0(v19[0], v20[0] + 1LL);
  return 0;
}
