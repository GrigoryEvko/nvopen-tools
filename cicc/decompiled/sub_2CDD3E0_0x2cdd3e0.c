// Function: sub_2CDD3E0
// Address: 0x2cdd3e0
//
__int64 __fastcall sub_2CDD3E0(__int64 a1, __int64 a2, __int64 a3, char a4, unsigned __int8 a5)
{
  __int64 v7; // r15
  __int64 v8; // rax
  char v9; // al
  void *v10; // rsi
  __int64 v13; // [rsp+10h] [rbp-A0h]
  __int64 v14; // [rsp+18h] [rbp-98h]
  __int64 v15; // [rsp+20h] [rbp-90h] BYREF
  _QWORD *v16; // [rsp+28h] [rbp-88h]
  int v17; // [rsp+30h] [rbp-80h]
  int v18; // [rsp+34h] [rbp-7Ch]
  int v19; // [rsp+38h] [rbp-78h]
  char v20; // [rsp+3Ch] [rbp-74h]
  _QWORD v21[2]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v22; // [rsp+50h] [rbp-60h] BYREF
  _BYTE *v23; // [rsp+58h] [rbp-58h]
  __int64 v24; // [rsp+60h] [rbp-50h]
  int v25; // [rsp+68h] [rbp-48h]
  char v26; // [rsp+6Ch] [rbp-44h]
  _BYTE v27[64]; // [rsp+70h] [rbp-40h] BYREF

  v14 = sub_BC1CD0(a3, &unk_4F81450, a2);
  v7 = sub_BC1CD0(a3, &unk_4F86540, a2);
  v13 = sub_BC1CD0(a3, &unk_5035D48, a2);
  v8 = sub_BC1CD0(a3, &unk_5010CC8, a2);
  LOBYTE(v16) = a4;
  v9 = sub_2CDCF50(&v15, a2, a5, (__int64 **)(v14 + 8), v7 + 8, v13 + 8, v8 + 8);
  v10 = (void *)(a1 + 32);
  if ( v9 )
  {
    v16 = v21;
    v17 = 2;
    v21[0] = &unk_4F82408;
    v19 = 0;
    v20 = 1;
    v22 = 0;
    v23 = v27;
    v24 = 2;
    v25 = 0;
    v26 = 1;
    v18 = 1;
    v15 = 1;
    sub_C8CF70(a1, v10, 2, (__int64)v21, (__int64)&v15);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v27, (__int64)&v22);
    if ( !v26 )
      _libc_free((unsigned __int64)v23);
    if ( !v20 )
      _libc_free((unsigned __int64)v16);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v10;
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
