// Function: sub_3716850
// Address: 0x3716850
//
__int64 *__fastcall sub_3716850(__int64 *a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  _QWORD *v5; // r13
  __int64 *v6; // rax
  __int64 v7; // rdx
  char *v8; // rax
  __int64 *v9; // r9
  __int64 v10; // rdx
  const __m128i *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r9
  unsigned __int64 v14; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  unsigned int v18; // r8d
  bool v19; // zf
  __int64 v20; // rax
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // rax
  __int64 *v25; // [rsp+20h] [rbp-F0h]
  char v26; // [rsp+3Fh] [rbp-D1h] BYREF
  unsigned __int64 v27; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v28; // [rsp+48h] [rbp-C8h] BYREF
  _QWORD *v29; // [rsp+50h] [rbp-C0h]
  _QWORD v30[2]; // [rsp+60h] [rbp-B0h] BYREF
  __m128i v31; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v32; // [rsp+80h] [rbp-90h] BYREF
  __int64 v33[2]; // [rsp+90h] [rbp-80h] BYREF
  _QWORD v34[2]; // [rsp+A0h] [rbp-70h] BYREF
  __m128i v35[2]; // [rsp+B0h] [rbp-60h] BYREF
  __int16 v36; // [rsp+D0h] [rbp-40h]

  v5 = a2 + 2;
  v6 = sub_3707A80();
  v8 = (char *)sub_370C9C0(a2 + 2, *(_BYTE *)(a4 + 6), v6, v7);
  v25 = v9;
  v29 = v30;
  sub_370CD40(v9, v8, (__int64)&v8[v10]);
  v11 = (const __m128i *)sub_3707A90();
  sub_3715FB0(&v31, a2 + 2, *(unsigned __int8 *)(a4 + 7), v11, v12, v13);
  v35[0].m128i_i64[0] = (__int64)"ReturnType";
  v36 = 259;
  sub_37011E0((unsigned __int64 *)v33, a2 + 2, (unsigned int *)(a4 + 2), v35[0].m128i_i64);
  v14 = v33[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v33[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v33[0] = 0;
    *a1 = v14 | 1;
    sub_9C66B0(v33);
    goto LABEL_3;
  }
  v33[0] = 0;
  sub_9C66B0(v33);
  sub_8FD6D0((__int64)v33, "CallingConvention: ", v25);
  v19 = a2[9] == 0;
  v35[0].m128i_i64[0] = (__int64)v33;
  v36 = 260;
  if ( !v19 && !a2[7] && !a2[8] )
    goto LABEL_23;
  if ( (unsigned int)sub_3700ED0((__int64)v5, (__int64)"CallingConvention: ", v16, v17, v18) )
  {
    v20 = a2[9];
    if ( a2[8] )
    {
      if ( v20 )
        goto LABEL_13;
    }
    else if ( !v20 )
    {
LABEL_13:
      sub_3702900((unsigned __int64 *)&v28, v5, &v26, v35);
      v21 = v28 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v28 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        v28 = 0;
        v27 = v21 | 1;
        sub_9C66B0(&v28);
      }
      else
      {
        v28 = 0;
        sub_9C66B0(&v28);
        if ( a2[7] && !a2[9] && !a2[8] )
          *(_BYTE *)(a4 + 6) = v26;
        v27 = 1;
        v28 = 0;
        sub_9C66B0(&v28);
      }
      goto LABEL_15;
    }
LABEL_23:
    if ( !a2[7] )
      v26 = *(_BYTE *)(a4 + 6);
    goto LABEL_13;
  }
  sub_370CCD0(&v27, 2u);
LABEL_15:
  if ( (_QWORD *)v33[0] != v34 )
    j_j___libc_free_0(v33[0]);
  v22 = v27 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v27 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v27 = 0;
    *a1 = v22 | 1;
    sub_9C66B0((__int64 *)&v27);
  }
  else
  {
    v27 = 0;
    sub_9C66B0((__int64 *)&v27);
    sub_8FD6D0((__int64)v33, "FunctionOptions", &v31);
    v36 = 260;
    v35[0].m128i_i64[0] = (__int64)v33;
    sub_370FA10((unsigned __int64 *)&v28, v5, (char *)(a4 + 7), v35, (unsigned int)&v28);
    if ( (_QWORD *)v33[0] != v34 )
      j_j___libc_free_0(v33[0]);
    v23 = v28 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v28 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      v28 = 0;
      *a1 = v23 | 1;
      sub_9C66B0(&v28);
    }
    else
    {
      v28 = 0;
      sub_9C66B0(&v28);
      v35[0].m128i_i64[0] = (__int64)"NumParameters";
      v36 = 259;
      sub_370BC10((unsigned __int64 *)v33, v5, (unsigned __int16 *)(a4 + 8), v35);
      v24 = v33[0] & 0xFFFFFFFFFFFFFFFELL;
      if ( (v33[0] & 0xFFFFFFFFFFFFFFFELL) != 0
        || (v33[0] = 0,
            sub_9C66B0(v33),
            v35[0].m128i_i64[0] = (__int64)"ArgListType",
            v36 = 259,
            sub_37011E0((unsigned __int64 *)v33, v5, (unsigned int *)(a4 + 10), v35[0].m128i_i64),
            v24 = v33[0] & 0xFFFFFFFFFFFFFFFELL,
            (v33[0] & 0xFFFFFFFFFFFFFFFELL) != 0) )
      {
        v33[0] = v24 | 1;
        *a1 = 0;
        sub_9C6670(a1, v33);
        sub_9C66B0(v33);
      }
      else
      {
        v33[0] = 0;
        sub_9C66B0(v33);
        v35[0].m128i_i64[0] = 0;
        *a1 = 1;
        sub_9C66B0(v35[0].m128i_i64);
      }
    }
  }
LABEL_3:
  if ( (__int64 *)v31.m128i_i64[0] != &v32 )
    j_j___libc_free_0(v31.m128i_u64[0]);
  if ( v29 != v30 )
    j_j___libc_free_0((unsigned __int64)v29);
  return a1;
}
