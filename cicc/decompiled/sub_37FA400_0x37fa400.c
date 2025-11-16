// Function: sub_37FA400
// Address: 0x37fa400
//
_QWORD *__fastcall sub_37FA400(_QWORD *a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  _QWORD v10[8]; // [rsp+10h] [rbp-1D0h] BYREF
  _QWORD v11[4]; // [rsp+50h] [rbp-190h] BYREF
  char v12; // [rsp+70h] [rbp-170h]
  void *v13; // [rsp+78h] [rbp-168h] BYREF
  int v14; // [rsp+80h] [rbp-160h]
  void **v15; // [rsp+88h] [rbp-158h] BYREF
  char *v16[3]; // [rsp+90h] [rbp-150h] BYREF
  _BYTE v17[312]; // [rsp+A8h] [rbp-138h] BYREF

  v4 = a4[2];
  if ( !v4 )
    v4 = a4[4] - a4[3];
  v14 = v4;
  v11[2] = &v15;
  v15 = &v13;
  v10[5] = 0x100000000LL;
  v11[0] = "<vftable {0} methods>";
  v13 = &unk_49E65E8;
  v10[0] = &unk_49DD288;
  v11[1] = 21;
  v11[3] = 1;
  v12 = 1;
  v16[0] = v17;
  v16[1] = 0;
  v16[2] = (char *)256;
  v10[1] = 2;
  memset(&v10[2], 0, 24);
  v10[6] = v16;
  sub_CB5980((__int64)v10, 0, 0, 0);
  sub_CB6840((__int64)v10, (__int64)v11);
  v10[0] = &unk_49DD388;
  sub_CB5840((__int64)v10);
  sub_37FA2C0(a2 + 24, v16, v5, v6, v7, v8);
  if ( v16[0] != v17 )
    _libc_free((unsigned __int64)v16[0]);
  *a1 = 1;
  return a1;
}
