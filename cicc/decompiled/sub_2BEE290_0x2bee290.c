// Function: sub_2BEE290
// Address: 0x2bee290
//
__int64 __fastcall sub_2BEE290(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  char v9; // al
  void *v10; // rsi
  __int64 v12; // [rsp+0h] [rbp-80h] BYREF
  _QWORD *v13; // [rsp+8h] [rbp-78h]
  int v14; // [rsp+10h] [rbp-70h]
  int v15; // [rsp+14h] [rbp-6Ch]
  int v16; // [rsp+18h] [rbp-68h]
  char v17; // [rsp+1Ch] [rbp-64h]
  _QWORD v18[2]; // [rsp+20h] [rbp-60h] BYREF
  __int64 v19; // [rsp+30h] [rbp-50h] BYREF
  _BYTE *v20; // [rsp+38h] [rbp-48h]
  __int64 v21; // [rsp+40h] [rbp-40h]
  int v22; // [rsp+48h] [rbp-38h]
  char v23; // [rsp+4Ch] [rbp-34h]
  _BYTE v24[48]; // [rsp+50h] [rbp-30h] BYREF

  *a2 = sub_BC1CD0(a4, &unk_4F89C30, a3) + 8;
  a2[1] = sub_BC1CD0(a4, &unk_4F86540, a3) + 8;
  a2[2] = sub_BC1CD0(a4, &unk_4F881D0, a3) + 8;
  v9 = sub_2BEE040((__int64)a2, a3, v6, v7, v8);
  v10 = (void *)(a1 + 32);
  if ( v9 )
  {
    v13 = v18;
    v14 = 2;
    v18[0] = &unk_4F82408;
    v16 = 0;
    v17 = 1;
    v19 = 0;
    v20 = v24;
    v21 = 2;
    v22 = 0;
    v23 = 1;
    v15 = 1;
    v12 = 1;
    sub_C8CF70(a1, v10, 2, (__int64)v18, (__int64)&v12);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v24, (__int64)&v19);
    if ( !v23 )
      _libc_free((unsigned __int64)v20);
    if ( !v17 )
      _libc_free((unsigned __int64)v13);
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
