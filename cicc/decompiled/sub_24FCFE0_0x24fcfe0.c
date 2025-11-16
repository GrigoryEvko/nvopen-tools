// Function: sub_24FCFE0
// Address: 0x24fcfe0
//
__int64 __fastcall sub_24FCFE0(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  char v7; // al
  void *v8; // rsi
  __int64 v10; // [rsp+0h] [rbp-A0h] BYREF
  __int64 v11; // [rsp+8h] [rbp-98h] BYREF
  __int64 v12; // [rsp+10h] [rbp-90h] BYREF
  _QWORD *v13; // [rsp+18h] [rbp-88h]
  int v14; // [rsp+20h] [rbp-80h]
  int v15; // [rsp+24h] [rbp-7Ch]
  int v16; // [rsp+28h] [rbp-78h]
  char v17; // [rsp+2Ch] [rbp-74h]
  _QWORD v18[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v19; // [rsp+40h] [rbp-60h] BYREF
  _BYTE *v20; // [rsp+48h] [rbp-58h]
  __int64 v21; // [rsp+50h] [rbp-50h]
  int v22; // [rsp+58h] [rbp-48h]
  char v23; // [rsp+5Ch] [rbp-44h]
  _BYTE v24[64]; // [rsp+60h] [rbp-40h] BYREF

  v10 = *(_QWORD *)(sub_BC0510(a4, &unk_4F82418, a3) + 8);
  v11 = v10;
  v6 = sub_BC0510(a4, &unk_4F87C68, a3);
  v7 = sub_24FB8F0(
         a3,
         *a2,
         v6 + 8,
         v10,
         (__int64)sub_24FB2C0,
         (__int64)&v10,
         (__int64 (__fastcall *)(__int64, unsigned __int8 *))sub_24FB2E0,
         (__int64)&v11);
  v8 = (void *)(a1 + 32);
  if ( v7 )
  {
    v13 = v18;
    v14 = 2;
    v18[0] = &unk_4F82420;
    v16 = 0;
    v17 = 1;
    v19 = 0;
    v20 = v24;
    v21 = 2;
    v22 = 0;
    v23 = 1;
    v15 = 1;
    v12 = 1;
    sub_C8CF70(a1, v8, 2, (__int64)v18, (__int64)&v12);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v24, (__int64)&v19);
    if ( !v23 )
      _libc_free((unsigned __int64)v20);
    if ( !v17 )
      _libc_free((unsigned __int64)v13);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v8;
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
