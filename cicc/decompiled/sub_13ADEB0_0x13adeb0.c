// Function: sub_13ADEB0
// Address: 0x13adeb0
//
__int64 __fastcall sub_13ADEB0(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4)
{
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r15
  __int64 v14; // [rsp+8h] [rbp-78h]
  __int64 v16; // [rsp+10h] [rbp-70h]
  __int64 v18; // [rsp+20h] [rbp-60h]
  __int64 v19; // [rsp+20h] [rbp-60h]
  __int64 v20; // [rsp+28h] [rbp-58h]
  __int64 v21; // [rsp+28h] [rbp-58h]
  __int64 v22; // [rsp+28h] [rbp-58h]
  __int64 *v23; // [rsp+30h] [rbp-50h] BYREF
  __int64 v24; // [rsp+38h] [rbp-48h]
  __int64 v25; // [rsp+40h] [rbp-40h] BYREF
  __int64 v26; // [rsp+48h] [rbp-38h]

  v5 = sub_13A62B0(a4);
  v14 = sub_13AD540(a1, *a2, v5);
  v20 = sub_13AD540(a1, *a3, v5);
  v18 = *(_QWORD *)(a1 + 8);
  v26 = sub_13A6250(a4);
  v25 = v14;
  v23 = &v25;
  v24 = 0x200000002LL;
  v6 = sub_147EE30(v18, &v23, 0, 0);
  v7 = a4;
  v19 = v6;
  if ( v23 != &v25 )
  {
    _libc_free((unsigned __int64)v23);
    v7 = a4;
  }
  v16 = *(_QWORD *)(a1 + 8);
  v26 = sub_13A6260(v7);
  v25 = v20;
  v23 = &v25;
  v24 = 0x200000002LL;
  v8 = sub_147EE30(v16, &v23, 0, 0);
  v9 = v8;
  if ( v23 != &v25 )
  {
    v21 = v8;
    _libc_free((unsigned __int64)v23);
    v9 = v21;
  }
  v22 = *(_QWORD *)(a1 + 8);
  v10 = sub_14806B0(v22, v19, v9, 0, 0);
  v11 = *a2;
  v26 = v10;
  v25 = v11;
  v23 = &v25;
  v24 = 0x200000002LL;
  v12 = sub_147DD40(v22, &v23, 0, 0);
  if ( v23 != &v25 )
    _libc_free((unsigned __int64)v23);
  *a2 = v12;
  *a2 = sub_13AD5A0(a1, v12, v5);
  *a3 = sub_13AD5A0(a1, *a3, v5);
  return 1;
}
