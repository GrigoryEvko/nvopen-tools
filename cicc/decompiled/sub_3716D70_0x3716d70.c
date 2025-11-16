// Function: sub_3716D70
// Address: 0x3716d70
//
__int64 *__fastcall sub_3716D70(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v5; // r12
  __int64 *v6; // rax
  __int64 v7; // rdx
  char *v8; // rax
  __int64 v9; // rdx
  const __m128i *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r9
  unsigned __int64 v13; // rax
  unsigned int v15; // r8d
  unsigned __int64 v16; // rax
  unsigned int v17; // r8d
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v20[2]; // [rsp+30h] [rbp-C0h] BYREF
  _QWORD v21[2]; // [rsp+40h] [rbp-B0h] BYREF
  __m128i v22; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v23; // [rsp+60h] [rbp-90h] BYREF
  __int64 v24[4]; // [rsp+70h] [rbp-80h] BYREF
  __m128i v25[2]; // [rsp+90h] [rbp-60h] BYREF
  __int16 v26; // [rsp+B0h] [rbp-40h]

  v5 = (_QWORD *)(a2 + 16);
  v6 = sub_3707A80();
  v8 = (char *)sub_370C9C0((_QWORD *)(a2 + 16), *(_BYTE *)(a4 + 14), v6, v7);
  v20[0] = (__int64)v21;
  sub_370CD40(v20, v8, (__int64)&v8[v9]);
  v10 = (const __m128i *)sub_3707A90();
  sub_3715FB0(&v22, (_QWORD *)(a2 + 16), *(unsigned __int8 *)(a4 + 15), v10, v11, v12);
  v25[0].m128i_i64[0] = (__int64)"ReturnType";
  v26 = 259;
  sub_37011E0((unsigned __int64 *)v24, (_QWORD *)(a2 + 16), (unsigned int *)(a4 + 2), v25[0].m128i_i64);
  v13 = v24[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v24[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    goto LABEL_3;
  v25[0].m128i_i64[0] = (__int64)"ClassType";
  v26 = 259;
  sub_37011E0((unsigned __int64 *)v24, v5, (unsigned int *)(a4 + 6), v25[0].m128i_i64);
  v13 = v24[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v24[0] & 0xFFFFFFFFFFFFFFFELL) != 0
    || (v24[0] = 0,
        sub_9C66B0(v24),
        v25[0].m128i_i64[0] = (__int64)"ThisType",
        v26 = 259,
        sub_37011E0((unsigned __int64 *)v24, v5, (unsigned int *)(a4 + 10), v25[0].m128i_i64),
        v13 = v24[0] & 0xFFFFFFFFFFFFFFFELL,
        (v24[0] & 0xFFFFFFFFFFFFFFFELL) != 0) )
  {
LABEL_3:
    v24[0] = 0;
    *a1 = v13 | 1;
    sub_9C66B0(v24);
  }
  else
  {
    v24[0] = 0;
    sub_9C66B0(v24);
    sub_8FD6D0((__int64)v24, "CallingConvention: ", v20);
    v26 = 260;
    v25[0].m128i_i64[0] = (__int64)v24;
    sub_370F8F0(&v19, v5, (char *)(a4 + 14), v25, v15);
    sub_2240A30((unsigned __int64 *)v24);
    v16 = v19 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v19 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      goto LABEL_16;
    v19 = 0;
    sub_9C66B0((__int64 *)&v19);
    sub_8FD6D0((__int64)v24, "FunctionOptions", &v22);
    v26 = 260;
    v25[0].m128i_i64[0] = (__int64)v24;
    sub_370FA10(&v19, v5, (char *)(a4 + 15), v25, v17);
    sub_2240A30((unsigned __int64 *)v24);
    v16 = v19 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v19 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
LABEL_16:
      v19 = v16 | 1;
      *a1 = 0;
      sub_9C6670(a1, &v19);
      sub_9C66B0((__int64 *)&v19);
    }
    else
    {
      v19 = 0;
      sub_9C66B0((__int64 *)&v19);
      v25[0].m128i_i64[0] = (__int64)"NumParameters";
      v26 = 259;
      sub_370BC10((unsigned __int64 *)v24, v5, (unsigned __int16 *)(a4 + 16), v25);
      v18 = v24[0] & 0xFFFFFFFFFFFFFFFELL;
      if ( (v24[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
        goto LABEL_17;
      v24[0] = 0;
      sub_9C66B0(v24);
      v25[0].m128i_i64[0] = (__int64)"ArgListType";
      v26 = 259;
      sub_37011E0((unsigned __int64 *)v24, v5, (unsigned int *)(a4 + 18), v25[0].m128i_i64);
      v18 = v24[0] & 0xFFFFFFFFFFFFFFFELL;
      if ( (v24[0] & 0xFFFFFFFFFFFFFFFELL) != 0
        || (v24[0] = 0,
            sub_9C66B0(v24),
            v25[0].m128i_i64[0] = (__int64)"ThisAdjustment",
            v26 = 259,
            sub_370BFD0((unsigned __int64 *)v24, v5, (unsigned int *)(a4 + 24), v25),
            v18 = v24[0] & 0xFFFFFFFFFFFFFFFELL,
            (v24[0] & 0xFFFFFFFFFFFFFFFELL) != 0) )
      {
LABEL_17:
        v24[0] = v18 | 1;
        *a1 = 0;
        sub_9C6670(a1, v24);
        sub_9C66B0(v24);
      }
      else
      {
        v24[0] = 0;
        sub_9C66B0(v24);
        v25[0].m128i_i64[0] = 0;
        *a1 = 1;
        sub_9C66B0(v25[0].m128i_i64);
      }
    }
  }
  if ( (__int64 *)v22.m128i_i64[0] != &v23 )
    j_j___libc_free_0(v22.m128i_u64[0]);
  if ( (_QWORD *)v20[0] != v21 )
    j_j___libc_free_0(v20[0]);
  return a1;
}
