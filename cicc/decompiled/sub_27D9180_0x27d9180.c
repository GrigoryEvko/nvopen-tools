// Function: sub_27D9180
// Address: 0x27d9180
//
__int64 __fastcall sub_27D9180(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r15
  void *v7; // rsi
  __int64 v9; // [rsp+0h] [rbp-F0h]
  __int64 v10; // [rsp+8h] [rbp-E8h]
  _QWORD v11[8]; // [rsp+10h] [rbp-E0h] BYREF
  __int16 v12; // [rsp+50h] [rbp-A0h]
  __int64 v13; // [rsp+60h] [rbp-90h] BYREF
  _QWORD *v14; // [rsp+68h] [rbp-88h]
  int v15; // [rsp+70h] [rbp-80h]
  int v16; // [rsp+74h] [rbp-7Ch]
  int v17; // [rsp+78h] [rbp-78h]
  char v18; // [rsp+7Ch] [rbp-74h]
  _QWORD v19[2]; // [rsp+80h] [rbp-70h] BYREF
  __int64 v20; // [rsp+90h] [rbp-60h] BYREF
  _BYTE *v21; // [rsp+98h] [rbp-58h]
  __int64 v22; // [rsp+A0h] [rbp-50h]
  int v23; // [rsp+A8h] [rbp-48h]
  char v24; // [rsp+ACh] [rbp-44h]
  _BYTE v25[64]; // [rsp+B0h] [rbp-40h] BYREF

  v9 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v6 = sub_BC1CD0(a4, &unk_4F6D3F8, a3);
  v10 = sub_BC1CD0(a4, &unk_4F86630, a3);
  v11[0] = sub_B2BEC0(a3);
  v11[1] = v6 + 8;
  v11[2] = sub_BC1CD0(a4, &unk_4F89C30, a3) + 8;
  v11[3] = v9 + 8;
  v11[4] = v10 + 8;
  memset(&v11[5], 0, 24);
  v12 = 257;
  v7 = (void *)(a1 + 32);
  if ( (unsigned __int8)sub_27D8A00(a3, (__int64)v11) )
  {
    v14 = v19;
    v15 = 2;
    v19[0] = &unk_4F82408;
    v17 = 0;
    v18 = 1;
    v20 = 0;
    v21 = v25;
    v22 = 2;
    v23 = 0;
    v24 = 1;
    v16 = 1;
    v13 = 1;
    sub_C8CF70(a1, v7, 2, (__int64)v19, (__int64)&v13);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v25, (__int64)&v20);
    if ( !v24 )
      _libc_free((unsigned __int64)v21);
    if ( !v18 )
      _libc_free((unsigned __int64)v14);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v7;
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
