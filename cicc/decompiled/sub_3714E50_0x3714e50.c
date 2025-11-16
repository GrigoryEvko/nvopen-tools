// Function: sub_3714E50
// Address: 0x3714e50
//
unsigned __int64 *__fastcall sub_3714E50(unsigned __int64 *a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  _QWORD *v5; // r14
  __int64 *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  unsigned int v10; // r8d
  bool v11; // zf
  __int64 v12; // rax
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rax
  unsigned __int16 v16; // [rsp+1Eh] [rbp-B2h] BYREF
  unsigned __int64 v17; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v18; // [rsp+28h] [rbp-A8h] BYREF
  __m128i v19; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v20; // [rsp+40h] [rbp-90h] BYREF
  unsigned __int64 v21[2]; // [rsp+50h] [rbp-80h] BYREF
  __int64 v22; // [rsp+60h] [rbp-70h] BYREF
  __m128i v23[2]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v24; // [rsp+90h] [rbp-40h]

  v5 = a2 + 2;
  v6 = sub_3707A70();
  sub_37145A0(&v19, a2 + 2, *(_WORD *)(a4 + 6), (unsigned __int64)v6, v7, (__int64)&v19);
  v23[0].m128i_i64[0] = (__int64)"ModifiedType";
  v24 = 259;
  sub_37011E0(v21, a2 + 2, (unsigned int *)(a4 + 2), v23[0].m128i_i64);
  if ( (v21[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v21[0] & 0xFFFFFFFFFFFFFFFELL | 1;
    goto LABEL_14;
  }
  sub_8FD6D0((__int64)v21, "Modifiers", &v19);
  v11 = a2[9] == 0;
  v24 = 260;
  v23[0].m128i_i64[0] = (__int64)v21;
  if ( !v11 && !a2[7] && !a2[8] )
    goto LABEL_21;
  if ( (unsigned int)sub_3700ED0((__int64)v5, (__int64)"Modifiers", v8, v9, v10) > 1 )
  {
    v12 = a2[9];
    if ( a2[8] )
    {
      if ( v12 )
        goto LABEL_7;
    }
    else if ( !v12 )
    {
LABEL_7:
      sub_370BC10((unsigned __int64 *)&v18, v5, &v16, v23);
      v13 = v18 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        v18 = 0;
        v17 = v13 | 1;
        sub_9C66B0(&v18);
      }
      else
      {
        v18 = 0;
        sub_9C66B0(&v18);
        if ( a2[7] && !a2[9] && !a2[8] )
          *(_WORD *)(a4 + 6) = v16;
        v17 = 1;
        v18 = 0;
        sub_9C66B0(&v18);
      }
      goto LABEL_9;
    }
LABEL_21:
    if ( !a2[7] )
      v16 = *(_WORD *)(a4 + 6);
    goto LABEL_7;
  }
  sub_370CCD0(&v17, 2u);
LABEL_9:
  if ( (__int64 *)v21[0] != &v22 )
    j_j___libc_free_0(v21[0]);
  v14 = v17 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v17 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v17 = 0;
    *a1 = v14 | 1;
    sub_9C66B0((__int64 *)&v17);
  }
  else
  {
    v17 = 0;
    sub_9C66B0((__int64 *)&v17);
    *a1 = 1;
    v23[0].m128i_i64[0] = 0;
    sub_9C66B0(v23[0].m128i_i64);
  }
LABEL_14:
  if ( (__int64 *)v19.m128i_i64[0] != &v20 )
    j_j___libc_free_0(v19.m128i_u64[0]);
  return a1;
}
