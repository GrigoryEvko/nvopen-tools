// Function: sub_1C3EA60
// Address: 0x1c3ea60
//
void __fastcall sub_1C3EA60(__int64 a1, char *a2, char a3)
{
  bool v5; // zf
  char v6; // al
  __int64 v7; // r8
  char *v8; // rcx
  const void *v9; // r9
  size_t v10; // r8
  _QWORD *v11; // rax
  __m128i *v12; // rax
  __int64 v13; // r8
  __int64 v14; // r9
  size_t v15; // rcx
  _BYTE *v16; // rdi
  __int64 v17; // rsi
  _QWORD *v18; // r12
  __int64 v19; // rax
  _QWORD *v20; // rdi
  size_t v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  void *src; // [rsp+0h] [rbp-C0h]
  size_t n; // [rsp+8h] [rbp-B8h]
  void *dest; // [rsp+10h] [rbp-B0h] BYREF
  size_t v29; // [rsp+18h] [rbp-A8h]
  _QWORD v30[2]; // [rsp+20h] [rbp-A0h] BYREF
  _QWORD *v31; // [rsp+30h] [rbp-90h] BYREF
  unsigned __int64 v32; // [rsp+38h] [rbp-88h]
  _QWORD v33[2]; // [rsp+40h] [rbp-80h] BYREF
  _QWORD *v34; // [rsp+50h] [rbp-70h] BYREF
  size_t v35; // [rsp+58h] [rbp-68h]
  _QWORD v36[2]; // [rsp+60h] [rbp-60h] BYREF
  _OWORD *v37; // [rsp+70h] [rbp-50h] BYREF
  size_t v38; // [rsp+78h] [rbp-48h]
  _OWORD v39[4]; // [rsp+80h] [rbp-40h] BYREF

  v5 = a2[1] == 0;
  dest = v30;
  v29 = 0;
  LOBYTE(v30[0]) = 0;
  if ( v5 )
  {
    if ( !*(_QWORD *)(a1 + 8) )
      return;
    sub_2240AE0(&dest, a1);
    goto LABEL_29;
  }
  v6 = *a2;
  LOBYTE(v33[0]) = 0;
  v31 = v33;
  v32 = 0;
  if ( v6 == 2 )
  {
    v7 = 6;
    v8 = "remark";
  }
  else if ( v6 > 2 )
  {
    v7 = 4;
    v8 = "note";
    if ( v6 != 3 )
      goto LABEL_26;
  }
  else
  {
    if ( v6 )
    {
      if ( v6 == 1 )
      {
        v7 = 7;
        v8 = "warning";
        goto LABEL_7;
      }
LABEL_26:
      v10 = 0;
      v34 = v36;
      v11 = v36;
      goto LABEL_11;
    }
    v7 = 5;
    v8 = "error";
  }
LABEL_7:
  sub_2241130(&v31, 0, 0, v8, v7);
  v9 = v31;
  v10 = v32;
  v34 = v36;
  if ( v31 == 0 && (_QWORD *)((char *)v31 + v32) != 0 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v37 = (_OWORD *)v32;
  if ( v32 > 0xF )
  {
    src = v31;
    n = v32;
    v19 = sub_22409D0(&v34, &v37, 0);
    v10 = n;
    v9 = src;
    v34 = (_QWORD *)v19;
    v20 = (_QWORD *)v19;
    v36[0] = v37;
LABEL_39:
    memcpy(v20, v9, v10);
    v10 = (size_t)v37;
    v11 = v34;
    goto LABEL_11;
  }
  if ( v32 == 1 )
  {
    LOBYTE(v36[0]) = *(_BYTE *)v31;
    v11 = v36;
    goto LABEL_11;
  }
  if ( v32 )
  {
    v20 = v36;
    goto LABEL_39;
  }
  v11 = v36;
LABEL_11:
  v35 = v10;
  *((_BYTE *)v11 + v10) = 0;
  if ( v35 == 0x3FFFFFFFFFFFFFFFLL || v35 == 4611686018427387902LL )
    goto LABEL_57;
  sub_2241490(&v34, ": ", 2);
  v12 = (__m128i *)sub_2241490(&v34, *(const char **)a1, *(_QWORD *)(a1 + 8));
  v37 = v39;
  if ( (__m128i *)v12->m128i_i64[0] == &v12[1] )
  {
    v39[0] = _mm_loadu_si128(v12 + 1);
  }
  else
  {
    v37 = (_OWORD *)v12->m128i_i64[0];
    *(_QWORD *)&v39[0] = v12[1].m128i_i64[0];
  }
  v15 = v12->m128i_u64[1];
  v38 = v15;
  v12->m128i_i64[0] = (__int64)v12[1].m128i_i64;
  v16 = dest;
  v12->m128i_i64[1] = 0;
  v12[1].m128i_i8[0] = 0;
  if ( v37 == v39 )
  {
    v21 = v38;
    if ( v38 )
    {
      if ( v38 == 1 )
        *v16 = v39[0];
      else
        memcpy(v16, v39, v38);
      v21 = v38;
      v16 = dest;
    }
    v29 = v21;
    v16[v21] = 0;
    v16 = v37;
  }
  else
  {
    v15 = v38;
    if ( v16 == (_BYTE *)v30 )
    {
      dest = v37;
      v29 = v38;
      v30[0] = *(_QWORD *)&v39[0];
    }
    else
    {
      v17 = v30[0];
      dest = v37;
      v29 = v38;
      v30[0] = *(_QWORD *)&v39[0];
      if ( v16 )
      {
        v37 = v16;
        *(_QWORD *)&v39[0] = v17;
        goto LABEL_18;
      }
    }
    v37 = v39;
    v16 = v39;
  }
LABEL_18:
  v38 = 0;
  *v16 = 0;
  if ( v37 != v39 )
    j_j___libc_free_0(v37, *(_QWORD *)&v39[0] + 1LL);
  if ( v34 != v36 )
    j_j___libc_free_0(v34, v36[0] + 1LL);
  if ( v31 != v33 )
  {
    j_j___libc_free_0(v31, v33[0] + 1LL);
    if ( qword_4FBA5B0 )
      goto LABEL_30;
    goto LABEL_24;
  }
LABEL_29:
  if ( !qword_4FBA5B0 )
LABEL_24:
    sub_16C1EA0((__int64)&qword_4FBA5B0, (__int64 (*)(void))sub_1C3E6D0, (__int64)sub_1C3E470, v15, v13, v14);
LABEL_30:
  v18 = sub_16D40F0(qword_4FBA5B0);
  if ( !v18 )
  {
    v22 = sub_22077B0(32);
    v18 = (_QWORD *)v22;
    if ( v22 )
    {
      *(_BYTE *)(v22 + 16) = 0;
      *(_QWORD *)v22 = v22 + 16;
      *(_QWORD *)(v22 + 8) = 0;
    }
    if ( !qword_4FBA5B0 )
      sub_16C1EA0((__int64)&qword_4FBA5B0, (__int64 (*)(void))sub_1C3E6D0, (__int64)sub_1C3E470, v23, v24, v25);
    sub_16D40E0(qword_4FBA5B0, v18);
  }
  sub_2241490(v18, (const char *)dest, v29);
  if ( !a3 )
    goto LABEL_32;
  if ( v18[1] == 0x3FFFFFFFFFFFFFFFLL )
LABEL_57:
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490(v18, "\n", 1);
LABEL_32:
  if ( dest != v30 )
    j_j___libc_free_0(dest, v30[0] + 1LL);
}
