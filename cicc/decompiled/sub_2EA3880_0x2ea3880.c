// Function: sub_2EA3880
// Address: 0x2ea3880
//
__int64 __fastcall sub_2EA3880(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  void *v4; // r15
  __int64 *v5; // r8
  char v6; // r13
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  _BYTE v12[8]; // [rsp+10h] [rbp-5F0h] BYREF
  unsigned __int64 v13; // [rsp+18h] [rbp-5E8h]
  char v14; // [rsp+2Ch] [rbp-5D4h]
  _BYTE v15[16]; // [rsp+30h] [rbp-5D0h] BYREF
  _BYTE v16[8]; // [rsp+40h] [rbp-5C0h] BYREF
  unsigned __int64 v17; // [rsp+48h] [rbp-5B8h]
  char v18; // [rsp+5Ch] [rbp-5A4h]
  _BYTE v19[1440]; // [rsp+60h] [rbp-5A0h] BYREF

  v4 = (void *)(a1 + 32);
  sub_2E98450((__int64)v12, 1, 0, a4);
  v6 = sub_2EA0A20((__int64)v12, v5);
  sub_2E981F0((__int64)v12);
  if ( v6 )
  {
    sub_2EAFFB0(v12);
    sub_2E98B00((__int64)v12, (__int64)&unk_50208B0, v8, v9, v10, v11);
    sub_C8CF70(a1, v4, 2, (__int64)v15, (__int64)v12);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v19, (__int64)v16);
    if ( !v18 )
      _libc_free(v17);
    if ( !v14 )
      _libc_free(v13);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v4;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  return a1;
}
