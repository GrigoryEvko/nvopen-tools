// Function: sub_2A38DB0
// Address: 0x2a38db0
//
_BYTE *__fastcall sub_2A38DB0(__int64 **a1, unsigned __int8 *a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r15
  __int64 v6; // rdx
  int v7; // eax
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v10; // rax
  _BYTE *result; // rax
  __int64 v12; // rax
  __int64 v13; // [rsp+0h] [rbp-C0h]
  __int64 v14; // [rsp+8h] [rbp-B8h]
  __int64 v15; // [rsp+10h] [rbp-B0h]
  char v16; // [rsp+1Eh] [rbp-A2h]
  char v17; // [rsp+1Fh] [rbp-A1h]
  __int8 *v18[2]; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v19; // [rsp+30h] [rbp-90h] BYREF
  __int64 v20[2]; // [rsp+40h] [rbp-80h] BYREF
  __int64 v21; // [rsp+50h] [rbp-70h] BYREF
  __int64 *v22; // [rsp+60h] [rbp-60h]
  __int64 v23; // [rsp+70h] [rbp-50h] BYREF

  v16 = a2[2] & 1;
  v17 = sub_B46500(a2);
  v3 = sub_9208B0((__int64)a1[4], *(_QWORD *)(*((_QWORD *)a2 - 8) + 8LL));
  v20[1] = v4;
  v20[0] = (unsigned __int64)(v3 + 7) >> 3;
  v5 = sub_CA1930(v20);
  v14 = ((__int64 (__fastcall *)(__int64 **, _QWORD))(*a1)[3])(a1, 0);
  v13 = v6;
  v15 = (__int64)a1[2];
  v7 = ((__int64 (__fastcall *)(__int64 **))(*a1)[4])(a1);
  if ( v7 == 14 )
  {
    v8 = sub_22077B0(0x1B0u);
    v9 = v8;
    if ( v8 )
      sub_B176B0(v8, v15, v14, v13, (__int64)a2);
  }
  else
  {
    if ( v7 != 15 )
      BUG();
    v12 = sub_22077B0(0x1B0u);
    v9 = v12;
    if ( v12 )
      sub_B178C0(v12, v15, v14, v13, (__int64)a2);
  }
  ((void (__fastcall *)(__int8 **, __int64 **, char *, __int64))(*a1)[2])(v18, a1, "Store", 5);
  sub_B18290(v9, v18[0], (size_t)v18[1]);
  sub_B18290(v9, "\nStore size: ", 0xDu);
  sub_B167F0(v20, "StoreSize", 9, v5);
  v10 = sub_2A38130(v9, (__int64)v20);
  sub_B18290(v10, " bytes.", 7u);
  if ( v22 != &v23 )
    j_j___libc_free_0((unsigned __int64)v22);
  if ( (__int64 *)v20[0] != &v21 )
    j_j___libc_free_0(v20[0]);
  if ( (__int64 *)v18[0] != &v19 )
    j_j___libc_free_0((unsigned __int64)v18[0]);
  sub_2A38830((__int64)a1, *((unsigned __int8 **)a2 - 4), 0, v9);
  sub_2A381E0(0, v16, v17, v9);
  result = sub_1049740(a1[1], v9);
  if ( v9 )
    return (_BYTE *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v9 + 16LL))(v9);
  return result;
}
