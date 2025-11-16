// Function: sub_1C3DD40
// Address: 0x1c3dd40
//
__int64 __fastcall sub_1C3DD40(_QWORD **a1, int a2, int a3, int a4, __int64 a5)
{
  int v6; // eax
  unsigned int v7; // r12d
  unsigned __int8 v9; // [rsp+Fh] [rbp-131h] BYREF
  _QWORD v10[2]; // [rsp+10h] [rbp-130h] BYREF
  _QWORD v11[2]; // [rsp+20h] [rbp-120h] BYREF
  void *v12; // [rsp+30h] [rbp-110h] BYREF
  __int64 v13; // [rsp+38h] [rbp-108h]
  __int64 v14; // [rsp+40h] [rbp-100h]
  __int64 v15; // [rsp+48h] [rbp-F8h]
  int v16; // [rsp+50h] [rbp-F0h]
  _QWORD *v17; // [rsp+58h] [rbp-E8h]
  _DWORD v18[4]; // [rsp+60h] [rbp-E0h] BYREF
  unsigned __int8 *v19; // [rsp+70h] [rbp-D0h]
  void **v20; // [rsp+78h] [rbp-C8h]
  _QWORD v21[2]; // [rsp+80h] [rbp-C0h] BYREF
  _QWORD v22[2]; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v23[4]; // [rsp+A0h] [rbp-A0h] BYREF
  int v24; // [rsp+C0h] [rbp-80h]
  _QWORD *v25; // [rsp+C8h] [rbp-78h]
  __int64 v26; // [rsp+D0h] [rbp-70h]
  __int64 v27; // [rsp+D8h] [rbp-68h]
  __int64 v28; // [rsp+E0h] [rbp-60h]
  __int64 v29; // [rsp+E8h] [rbp-58h]
  __int64 v30; // [rsp+F0h] [rbp-50h]
  __int64 v31; // [rsp+F8h] [rbp-48h]
  __int64 v32; // [rsp+100h] [rbp-40h]
  __int64 v33; // [rsp+108h] [rbp-38h]

  v18[0] = a2;
  v18[1] = a3;
  v18[2] = a4;
  v12 = &unk_49EFBE0;
  v23[0] = (__int64)&unk_49EFBE0;
  v25 = v21;
  v19 = &v9;
  v10[0] = v11;
  v10[1] = 0;
  LOBYTE(v11[0]) = 0;
  v16 = 1;
  v15 = 0;
  v14 = 0;
  v13 = 0;
  v17 = v10;
  v20 = &v12;
  v21[0] = v22;
  v21[1] = 0;
  LOBYTE(v22[0]) = 0;
  v24 = 1;
  memset(&v23[1], 0, 24);
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v9 = 1;
  sub_1C3BC10((__int64)v18, a1);
  v6 = v9;
  if ( a5 && !v9 )
  {
    if ( v15 != v13 )
      sub_16E7BA0((__int64 *)&v12);
    sub_2240AE0(a5, v10);
    v6 = v9;
  }
  v7 = v6 ^ 1;
  j___libc_free_0(v31);
  j___libc_free_0(v27);
  sub_16E7BC0(v23);
  if ( (_QWORD *)v21[0] != v22 )
    j_j___libc_free_0(v21[0], v22[0] + 1LL);
  sub_16E7BC0((__int64 *)&v12);
  if ( (_QWORD *)v10[0] != v11 )
    j_j___libc_free_0(v10[0], v11[0] + 1LL);
  return v7;
}
