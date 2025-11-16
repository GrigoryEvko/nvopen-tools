// Function: sub_DD1F00
// Address: 0xdd1f00
//
__int64 __fastcall sub_DD1F00(__int64 *a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v5; // rax
  unsigned int v6; // ebx
  __int64 v7; // rax
  _QWORD *v8; // r15
  __int64 v9; // rax
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  _QWORD *v12; // rax
  __int64 v13; // rax
  unsigned int v14; // eax
  unsigned int v15; // ebx
  unsigned __int64 v16; // r13
  unsigned int v17; // r12d
  __int64 v19; // rax
  unsigned __int64 v20; // rax
  _QWORD *v21; // rax
  __int64 v22; // rax
  unsigned int v23; // eax
  unsigned int v24; // ebx
  unsigned __int64 v26; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v27; // [rsp+18h] [rbp-78h]
  unsigned __int64 v28; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v29; // [rsp+28h] [rbp-68h]
  __int64 v30; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v31; // [rsp+38h] [rbp-58h]
  unsigned __int64 v32; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v33; // [rsp+48h] [rbp-48h]
  unsigned __int64 v34; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v35; // [rsp+58h] [rbp-38h]

  v5 = sub_D95540(a2);
  v6 = sub_D97050((__int64)a1, v5);
  v7 = sub_D95540(a3);
  v8 = sub_DA2C50((__int64)a1, v7, 1, 0);
  if ( a4 )
  {
    v9 = sub_DBB9F0((__int64)a1, a2, 1u, 0);
    sub_AB13A0((__int64)&v26, v9);
    v29 = v6;
    v10 = ~(1LL << ((unsigned __int8)v6 - 1));
    if ( v6 > 0x40 )
    {
      sub_C43690((__int64)&v28, -1, 1);
      v10 = ~(1LL << ((unsigned __int8)v6 - 1));
      if ( v29 > 0x40 )
      {
        *(_QWORD *)(v28 + 8LL * ((v6 - 1) >> 6)) &= ~(1LL << ((unsigned __int8)v6 - 1));
LABEL_7:
        v12 = sub_DCC810(a1, a3, (__int64)v8, 0, 0);
        v13 = sub_DBB9F0((__int64)a1, (__int64)v12, 1u, 0);
        sub_AB13A0((__int64)&v30, v13);
        v14 = v29;
        v29 = 0;
        v33 = v14;
        v32 = v28;
        sub_C46B40((__int64)&v32, &v30);
        v15 = v33;
        v16 = v32;
        v33 = 0;
        v35 = v15;
        v34 = v32;
        v17 = (unsigned int)sub_C4C880((__int64)&v34, (__int64)&v26) >> 31;
        if ( v15 <= 0x40 )
          goto LABEL_12;
        goto LABEL_8;
      }
    }
    else
    {
      v11 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v6;
      if ( !v6 )
        v11 = 0;
      v28 = v11;
    }
    v28 &= v10;
    goto LABEL_7;
  }
  v19 = sub_DBB9F0((__int64)a1, a2, 0, 0);
  sub_AB0910((__int64)&v26, v19);
  v29 = v6;
  if ( v6 > 0x40 )
  {
    sub_C43690((__int64)&v28, -1, 1);
  }
  else
  {
    v20 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v6;
    if ( !v6 )
      v20 = 0;
    v28 = v20;
  }
  v21 = sub_DCC810(a1, a3, (__int64)v8, 0, 0);
  v22 = sub_DBB9F0((__int64)a1, (__int64)v21, 0, 0);
  sub_AB0910((__int64)&v30, v22);
  v23 = v29;
  v29 = 0;
  v33 = v23;
  v32 = v28;
  sub_C46B40((__int64)&v32, &v30);
  v24 = v33;
  v16 = v32;
  v33 = 0;
  v35 = v24;
  v34 = v32;
  v17 = (unsigned int)sub_C49970((__int64)&v34, &v26) >> 31;
  if ( v24 > 0x40 )
  {
LABEL_8:
    if ( v16 )
    {
      j_j___libc_free_0_0(v16);
      if ( v33 > 0x40 )
      {
        if ( v32 )
          j_j___libc_free_0_0(v32);
      }
    }
  }
LABEL_12:
  if ( v31 > 0x40 && v30 )
    j_j___libc_free_0_0(v30);
  if ( v29 > 0x40 && v28 )
    j_j___libc_free_0_0(v28);
  if ( v27 > 0x40 && v26 )
    j_j___libc_free_0_0(v26);
  return v17;
}
