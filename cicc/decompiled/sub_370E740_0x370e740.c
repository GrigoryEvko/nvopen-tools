// Function: sub_370E740
// Address: 0x370e740
//
unsigned __int64 *__fastcall sub_370E740(unsigned __int64 *a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  _QWORD *v6; // r15
  __int64 *v7; // rax
  __int64 v8; // rdx
  char *v9; // rax
  __int64 *v10; // r8
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rcx
  unsigned int v14; // r8d
  bool v15; // zf
  __int64 v16; // rax
  unsigned __int64 v17; // rax
  _QWORD *v18; // rdi
  unsigned __int64 v19; // rax
  __int64 *v21; // [rsp+8h] [rbp-C8h]
  unsigned __int16 v22; // [rsp+1Eh] [rbp-B2h] BYREF
  __int64 v23; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v24; // [rsp+28h] [rbp-A8h] BYREF
  _QWORD *v25; // [rsp+30h] [rbp-A0h]
  _QWORD v26[2]; // [rsp+40h] [rbp-90h] BYREF
  unsigned __int64 v27[2]; // [rsp+50h] [rbp-80h] BYREF
  __int64 v28; // [rsp+60h] [rbp-70h] BYREF
  __m128i v29[2]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v30; // [rsp+90h] [rbp-40h]

  v6 = a2 + 2;
  v7 = sub_3707AA0();
  v9 = (char *)sub_370CAA0(a2 + 2, *(_WORD *)(a4 + 2), v7, v8);
  v21 = v10;
  v25 = v26;
  sub_370CD40(v10, v9, (__int64)&v9[v11]);
  sub_8FD6D0((__int64)v27, "Mode: ", v21);
  v15 = a2[9] == 0;
  v30 = 260;
  v29[0].m128i_i64[0] = (__int64)v27;
  if ( !v15 && !a2[7] && !a2[8] )
  {
LABEL_19:
    if ( !a2[7] )
      v22 = *(_WORD *)(a4 + 2);
    goto LABEL_6;
  }
  if ( (unsigned int)sub_3700ED0((__int64)v6, (__int64)"Mode: ", v12, v13, v14) <= 1 )
  {
    sub_370CCD0((unsigned __int64 *)&v23, 2u);
    goto LABEL_8;
  }
  v16 = a2[9];
  if ( a2[8] )
  {
    if ( !v16 )
      goto LABEL_19;
  }
  else if ( v16 )
  {
    goto LABEL_19;
  }
LABEL_6:
  sub_370BC10((unsigned __int64 *)&v24, v6, &v22, v29);
  v17 = v24 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v24 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v24 = 0;
    v23 = v17 | 1;
    sub_9C66B0(&v24);
  }
  else
  {
    if ( a2[7] && !a2[9] && !a2[8] )
      *(_WORD *)(a4 + 2) = v22;
    v23 = 1;
    v24 = 0;
    sub_9C66B0(&v24);
  }
LABEL_8:
  if ( (__int64 *)v27[0] != &v28 )
    j_j___libc_free_0(v27[0]);
  v18 = v25;
  v19 = v23 | 1;
  if ( (v23 & 0xFFFFFFFFFFFFFFFELL) == 0 )
    v19 = 1;
  *a1 = v19;
  if ( v18 != v26 )
    j_j___libc_free_0((unsigned __int64)v18);
  return a1;
}
