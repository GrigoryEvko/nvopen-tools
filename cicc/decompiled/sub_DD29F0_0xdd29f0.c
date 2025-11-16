// Function: sub_DD29F0
// Address: 0xdd29f0
//
__int64 __fastcall sub_DD29F0(__int64 *a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v5; // rax
  unsigned int v6; // ebx
  __int64 v7; // rax
  _QWORD *v8; // r15
  __int64 v9; // rax
  __int64 v10; // rax
  _QWORD *v11; // rax
  __int64 v12; // rax
  unsigned int v13; // eax
  unsigned int v14; // ebx
  __int64 v15; // r13
  __int64 *v16; // r12
  __int64 v18; // rax
  _QWORD *v19; // rax
  __int64 v20; // rax
  unsigned int v21; // eax
  unsigned int v22; // ebx
  unsigned __int64 v24; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v25; // [rsp+18h] [rbp-78h]
  __int64 v26; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v27; // [rsp+28h] [rbp-68h]
  __int64 v28; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v29; // [rsp+38h] [rbp-58h]
  __int64 v30; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v31; // [rsp+48h] [rbp-48h]
  __int64 v32; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v33; // [rsp+58h] [rbp-38h]

  v5 = sub_D95540(a2);
  v6 = sub_D97050((__int64)a1, v5);
  v7 = sub_D95540(a3);
  v8 = sub_DA2C50((__int64)a1, v7, 1, 0);
  if ( !a4 )
  {
    v18 = sub_DBB9F0((__int64)a1, a2, 0, 0);
    sub_AB0A00((__int64)&v24, v18);
    v27 = v6;
    if ( v6 > 0x40 )
      sub_C43690((__int64)&v26, 0, 0);
    else
      v26 = 0;
    v19 = sub_DCC810(a1, a3, (__int64)v8, 0, 0);
    v16 = &v28;
    v20 = sub_DBB9F0((__int64)a1, (__int64)v19, 0, 0);
    sub_AB0910((__int64)&v28, v20);
    v21 = v27;
    v27 = 0;
    v31 = v21;
    v30 = v26;
    sub_C45EE0((__int64)&v30, &v28);
    v22 = v31;
    v15 = v30;
    v31 = 0;
    v33 = v22;
    v32 = v30;
    LOBYTE(v16) = (int)sub_C49970((__int64)&v32, &v24) > 0;
    if ( v22 <= 0x40 )
      goto LABEL_10;
    goto LABEL_6;
  }
  v9 = sub_DBB9F0((__int64)a1, a2, 1u, 0);
  sub_AB14C0((__int64)&v24, v9);
  v27 = v6;
  v10 = 1LL << ((unsigned __int8)v6 - 1);
  if ( v6 <= 0x40 )
  {
    v26 = 0;
LABEL_4:
    v26 |= v10;
    goto LABEL_5;
  }
  sub_C43690((__int64)&v26, 0, 0);
  v10 = 1LL << ((unsigned __int8)v6 - 1);
  if ( v27 <= 0x40 )
    goto LABEL_4;
  *(_QWORD *)(v26 + 8LL * ((v6 - 1) >> 6)) |= 1LL << ((unsigned __int8)v6 - 1);
LABEL_5:
  v11 = sub_DCC810(a1, a3, (__int64)v8, 0, 0);
  v16 = &v28;
  v12 = sub_DBB9F0((__int64)a1, (__int64)v11, 1u, 0);
  sub_AB13A0((__int64)&v28, v12);
  v13 = v27;
  v27 = 0;
  v31 = v13;
  v30 = v26;
  sub_C45EE0((__int64)&v30, &v28);
  v14 = v31;
  v15 = v30;
  v31 = 0;
  v33 = v14;
  v32 = v30;
  LOBYTE(v16) = (int)sub_C4C880((__int64)&v32, (__int64)&v24) > 0;
  if ( v14 <= 0x40 )
    goto LABEL_10;
LABEL_6:
  if ( v15 )
  {
    j_j___libc_free_0_0(v15);
    if ( v31 > 0x40 )
    {
      if ( v30 )
        j_j___libc_free_0_0(v30);
    }
  }
LABEL_10:
  if ( v29 > 0x40 && v28 )
    j_j___libc_free_0_0(v28);
  if ( v27 > 0x40 && v26 )
    j_j___libc_free_0_0(v26);
  if ( v25 > 0x40 && v24 )
    j_j___libc_free_0_0(v24);
  return (unsigned int)v16;
}
