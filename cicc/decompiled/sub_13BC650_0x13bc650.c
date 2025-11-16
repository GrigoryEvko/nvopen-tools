// Function: sub_13BC650
// Address: 0x13bc650
//
__int64 __fastcall sub_13BC650(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v6; // rax
  __int64 v7; // rax
  _BYTE *v8; // rdx
  _BYTE *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rcx
  __m128i *v15; // rax
  __int64 v16; // rsi
  __m128i *v17; // rdx
  __m128i *v18; // rax
  __int64 v19; // rsi
  __int64 v21; // [rsp+18h] [rbp-D8h] BYREF
  _BYTE *v22; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v23; // [rsp+28h] [rbp-C8h]
  _QWORD v24[2]; // [rsp+30h] [rbp-C0h] BYREF
  _QWORD v25[2]; // [rsp+40h] [rbp-B0h] BYREF
  __m128i v26; // [rsp+50h] [rbp-A0h] BYREF
  _QWORD *v27; // [rsp+60h] [rbp-90h] BYREF
  __int64 v28; // [rsp+68h] [rbp-88h]
  _QWORD v29[2]; // [rsp+70h] [rbp-80h] BYREF
  _QWORD *v30; // [rsp+80h] [rbp-70h] BYREF
  __int64 v31; // [rsp+88h] [rbp-68h]
  _QWORD v32[2]; // [rsp+90h] [rbp-60h] BYREF
  __int64 v33; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v34; // [rsp+A8h] [rbp-48h]
  _OWORD v35[4]; // [rsp+B0h] [rbp-40h] BYREF

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_36:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F99CCC )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_36;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F99CCC);
  v33 = 19;
  v21 = v6 + 160;
  v22 = v24;
  v7 = sub_22409D0(&v22, &v33, 0);
  v22 = (_BYTE *)v7;
  v24[0] = v33;
  *(__m128i *)v7 = _mm_load_si128((const __m128i *)&xmmword_4289C10);
  v8 = v22;
  *(_WORD *)(v7 + 16) = 25970;
  *(_BYTE *)(v7 + 18) = 101;
  v23 = v33;
  v8[v33] = 0;
  v9 = (_BYTE *)sub_1649960(a2);
  if ( v9 )
  {
    v30 = v32;
    sub_13B5840((__int64 *)&v30, v9, (__int64)&v9[v10]);
  }
  else
  {
    v31 = 0;
    v30 = v32;
    LOBYTE(v32[0]) = 0;
  }
  v27 = v29;
  sub_13B5790((__int64 *)&v27, v22, (__int64)&v22[v23]);
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v28) <= 5 )
LABEL_35:
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490(&v27, " for '", 6, v11);
  v12 = 15;
  v13 = 15;
  if ( v27 != v29 )
    v13 = v29[0];
  v14 = v28 + v31;
  if ( v28 + v31 <= v13 )
    goto LABEL_14;
  if ( v30 != v32 )
    v12 = v32[0];
  if ( v14 <= v12 )
  {
    v15 = (__m128i *)sub_2241130(&v30, 0, 0, v27, v28);
    v33 = (__int64)v35;
    v16 = v15->m128i_i64[0];
    v17 = v15 + 1;
    if ( (__m128i *)v15->m128i_i64[0] != &v15[1] )
      goto LABEL_15;
  }
  else
  {
LABEL_14:
    v15 = (__m128i *)sub_2241490(&v27, v30, v31, v14);
    v33 = (__int64)v35;
    v16 = v15->m128i_i64[0];
    v17 = v15 + 1;
    if ( (__m128i *)v15->m128i_i64[0] != &v15[1] )
    {
LABEL_15:
      v33 = v16;
      *(_QWORD *)&v35[0] = v15[1].m128i_i64[0];
      goto LABEL_16;
    }
  }
  v35[0] = _mm_loadu_si128(v15 + 1);
LABEL_16:
  v34 = v15->m128i_i64[1];
  v15->m128i_i64[0] = (__int64)v17;
  v15->m128i_i64[1] = 0;
  v15[1].m128i_i8[0] = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v34) <= 9 )
    goto LABEL_35;
  v18 = (__m128i *)sub_2241490(&v33, "' function", 10, v35);
  v25[0] = &v26;
  if ( (__m128i *)v18->m128i_i64[0] == &v18[1] )
  {
    v26 = _mm_loadu_si128(v18 + 1);
  }
  else
  {
    v25[0] = v18->m128i_i64[0];
    v26.m128i_i64[0] = v18[1].m128i_i64[0];
  }
  v19 = v18->m128i_i64[1];
  v18[1].m128i_i8[0] = 0;
  v25[1] = v19;
  v18->m128i_i64[0] = (__int64)v18[1].m128i_i64;
  v18->m128i_i64[1] = 0;
  if ( (_OWORD *)v33 != v35 )
    j_j___libc_free_0(v33, *(_QWORD *)&v35[0] + 1LL);
  if ( v27 != v29 )
    j_j___libc_free_0(v27, v29[0] + 1LL);
  if ( v30 != v32 )
    j_j___libc_free_0(v30, v32[0] + 1LL);
  LOWORD(v35[0]) = 260;
  LOWORD(v32[0]) = 260;
  v33 = (__int64)v25;
  v30 = (_QWORD *)(a1 + 160);
  sub_13BC5F0((__int64)&v21, (__int64)&v30, 0, (__int64)&v33, 0);
  if ( (__m128i *)v25[0] != &v26 )
    j_j___libc_free_0(v25[0], v26.m128i_i64[0] + 1);
  if ( v22 != (_BYTE *)v24 )
    j_j___libc_free_0(v22, v24[0] + 1LL);
  return 0;
}
