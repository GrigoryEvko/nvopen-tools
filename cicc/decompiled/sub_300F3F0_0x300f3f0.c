// Function: sub_300F3F0
// Address: 0x300f3f0
//
__int64 __fastcall sub_300F3F0(__int64 a1, __int64 **a2)
{
  __int64 *v3; // rdi
  __int64 v4; // r12
  _QWORD **v5; // rax
  _QWORD *v6; // rdi
  __int64 v8; // [rsp+8h] [rbp-E8h]
  _QWORD v9[4]; // [rsp+10h] [rbp-E0h] BYREF
  _BYTE *v10; // [rsp+30h] [rbp-C0h]
  __int64 v11; // [rsp+38h] [rbp-B8h]
  _BYTE v12[32]; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v13; // [rsp+60h] [rbp-90h]
  __int64 v14; // [rsp+68h] [rbp-88h]
  __int16 v15; // [rsp+70h] [rbp-80h]
  __int64 *v16; // [rsp+78h] [rbp-78h]
  void **v17; // [rsp+80h] [rbp-70h]
  void **v18; // [rsp+88h] [rbp-68h]
  __int64 v19; // [rsp+90h] [rbp-60h]
  int v20; // [rsp+98h] [rbp-58h]
  __int16 v21; // [rsp+9Ch] [rbp-54h]
  char v22; // [rsp+9Eh] [rbp-52h]
  __int64 v23; // [rsp+A0h] [rbp-50h]
  __int64 v24; // [rsp+A8h] [rbp-48h]
  void *v25; // [rsp+B0h] [rbp-40h] BYREF
  void *v26; // [rsp+B8h] [rbp-38h] BYREF

  v3 = *a2;
  v11 = 0x200000000LL;
  v21 = 512;
  v18 = &v26;
  v15 = 0;
  v16 = v3;
  v10 = v12;
  v17 = &v25;
  v19 = 0;
  v20 = 0;
  v22 = 7;
  v23 = 0;
  v24 = 0;
  v13 = 0;
  v14 = 0;
  v25 = &unk_49DA100;
  v26 = &unk_49DA0B0;
  v4 = sub_BCB2D0(v3);
  v8 = sub_BCE3C0(v16, 0);
  v5 = (_QWORD **)sub_BCB2D0(v16);
  v6 = *v5;
  v9[0] = v5;
  v9[1] = v8;
  v9[2] = v4;
  *(_QWORD *)(a1 + 176) = sub_BD0B90(v6, v9, 3, 0);
  nullsub_61();
  v25 = &unk_49DA100;
  nullsub_63();
  if ( v10 != v12 )
    _libc_free((unsigned __int64)v10);
  return 0;
}
