// Function: sub_CEB020
// Address: 0xceb020
//
__int16 __fastcall sub_CEB020(_QWORD *a1, unsigned __int16 a2, char a3, char *a4)
{
  __int16 result; // ax
  __int64 v7; // r8
  const void *v8; // r9
  size_t v9; // r8
  _QWORD *v10; // rax
  __int64 v11; // rcx
  __m128i *v12; // rax
  _BYTE *v13; // rdi
  __int64 v14; // rsi
  __int64 v15; // rcx
  _QWORD *v16; // r12
  __int64 v17; // rcx
  __int64 v18; // rax
  _QWORD *v19; // rdi
  size_t v20; // rdx
  __int64 v21; // rax
  void *src; // [rsp+0h] [rbp-C0h]
  size_t n; // [rsp+8h] [rbp-B8h]
  void *dest; // [rsp+10h] [rbp-B0h] BYREF
  size_t v25; // [rsp+18h] [rbp-A8h]
  _QWORD v26[2]; // [rsp+20h] [rbp-A0h] BYREF
  _QWORD *v27; // [rsp+30h] [rbp-90h] BYREF
  unsigned __int64 v28; // [rsp+38h] [rbp-88h]
  _QWORD v29[2]; // [rsp+40h] [rbp-80h] BYREF
  _QWORD *v30; // [rsp+50h] [rbp-70h] BYREF
  size_t v31; // [rsp+58h] [rbp-68h]
  _QWORD v32[2]; // [rsp+60h] [rbp-60h] BYREF
  _OWORD *v33; // [rsp+70h] [rbp-50h] BYREF
  size_t v34; // [rsp+78h] [rbp-48h]
  _OWORD v35[4]; // [rsp+80h] [rbp-40h] BYREF

  result = HIBYTE(a2);
  dest = v26;
  v25 = 0;
  LOBYTE(v26[0]) = 0;
  if ( !HIBYTE(a2) )
  {
    if ( !a1[1] )
      return result;
    sub_2240AE0(&dest, a1);
    goto LABEL_29;
  }
  LOBYTE(v29[0]) = 0;
  v27 = v29;
  v28 = 0;
  if ( (_BYTE)a2 == 2 )
  {
    v7 = 6;
    a4 = "remark";
  }
  else if ( (char)a2 > 2 )
  {
    v7 = 4;
    a4 = "note";
    if ( (_BYTE)a2 != 3 )
      goto LABEL_26;
  }
  else
  {
    if ( (_BYTE)a2 )
    {
      if ( (_BYTE)a2 == 1 )
      {
        v7 = 7;
        a4 = "warning";
        goto LABEL_7;
      }
LABEL_26:
      v9 = 0;
      v30 = v32;
      v10 = v32;
      goto LABEL_11;
    }
    v7 = 5;
    a4 = "error";
  }
LABEL_7:
  sub_2241130(&v27, 0, 0, a4, v7);
  v8 = v27;
  v9 = v28;
  v30 = v32;
  if ( v27 == 0 && (_QWORD *)((char *)v27 + v28) != 0 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v33 = (_OWORD *)v28;
  if ( v28 > 0xF )
  {
    src = v27;
    n = v28;
    v18 = sub_22409D0(&v30, &v33, 0);
    v9 = n;
    v8 = src;
    v30 = (_QWORD *)v18;
    v19 = (_QWORD *)v18;
    v32[0] = v33;
LABEL_39:
    memcpy(v19, v8, v9);
    v9 = (size_t)v33;
    v10 = v30;
    goto LABEL_11;
  }
  if ( v28 == 1 )
  {
    LOBYTE(v32[0]) = *(_BYTE *)v27;
    v10 = v32;
    goto LABEL_11;
  }
  if ( v28 )
  {
    v19 = v32;
    goto LABEL_39;
  }
  v10 = v32;
LABEL_11:
  v31 = v9;
  *((_BYTE *)v10 + v9) = 0;
  if ( v31 == 0x3FFFFFFFFFFFFFFFLL || v31 == 4611686018427387902LL )
    goto LABEL_57;
  sub_2241490(&v30, ": ", 2, a4);
  v12 = (__m128i *)sub_2241490(&v30, *a1, a1[1], v11);
  v33 = v35;
  if ( (__m128i *)v12->m128i_i64[0] == &v12[1] )
  {
    v35[0] = _mm_loadu_si128(v12 + 1);
  }
  else
  {
    v33 = (_OWORD *)v12->m128i_i64[0];
    *(_QWORD *)&v35[0] = v12[1].m128i_i64[0];
  }
  v34 = v12->m128i_u64[1];
  v12->m128i_i64[0] = (__int64)v12[1].m128i_i64;
  v13 = dest;
  v12->m128i_i64[1] = 0;
  v12[1].m128i_i8[0] = 0;
  if ( v33 == v35 )
  {
    v20 = v34;
    if ( v34 )
    {
      if ( v34 == 1 )
        *v13 = v35[0];
      else
        memcpy(v13, v35, v34);
      v20 = v34;
      v13 = dest;
    }
    v25 = v20;
    v13[v20] = 0;
    v13 = v33;
  }
  else
  {
    if ( v13 == (_BYTE *)v26 )
    {
      dest = v33;
      v25 = v34;
      v26[0] = *(_QWORD *)&v35[0];
    }
    else
    {
      v14 = v26[0];
      dest = v33;
      v25 = v34;
      v26[0] = *(_QWORD *)&v35[0];
      if ( v13 )
      {
        v33 = v13;
        *(_QWORD *)&v35[0] = v14;
        goto LABEL_18;
      }
    }
    v33 = v35;
    v13 = v35;
  }
LABEL_18:
  v34 = 0;
  *v13 = 0;
  if ( v33 != v35 )
    j_j___libc_free_0(v33, *(_QWORD *)&v35[0] + 1LL);
  if ( v30 != v32 )
    j_j___libc_free_0(v30, v32[0] + 1LL);
  if ( v27 != v29 )
  {
    j_j___libc_free_0(v27, v29[0] + 1LL);
    if ( qword_4F85270 )
      goto LABEL_30;
    goto LABEL_24;
  }
LABEL_29:
  if ( !qword_4F85270 )
LABEL_24:
    sub_C7D570(&qword_4F85270, (__int64 (*)(void))sub_CEAC80, (__int64)sub_CEAA20);
LABEL_30:
  v16 = sub_C94E20(qword_4F85270);
  if ( !v16 )
  {
    v21 = sub_22077B0(32);
    v16 = (_QWORD *)v21;
    if ( v21 )
    {
      *(_BYTE *)(v21 + 16) = 0;
      *(_QWORD *)v21 = v21 + 16;
      *(_QWORD *)(v21 + 8) = 0;
    }
    if ( !qword_4F85270 )
      sub_C7D570(&qword_4F85270, (__int64 (*)(void))sub_CEAC80, (__int64)sub_CEAA20);
    sub_C94E10(qword_4F85270, v16);
  }
  result = sub_2241490(v16, dest, v25, v15);
  if ( !a3 )
    goto LABEL_32;
  if ( v16[1] == 0x3FFFFFFFFFFFFFFFLL )
LABEL_57:
    sub_4262D8((__int64)"basic_string::append");
  result = sub_2241490(v16, "\n", 1, v17);
LABEL_32:
  if ( dest != v26 )
    return j_j___libc_free_0(dest, v26[0] + 1LL);
  return result;
}
