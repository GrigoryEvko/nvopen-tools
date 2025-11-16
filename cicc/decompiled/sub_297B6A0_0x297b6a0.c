// Function: sub_297B6A0
// Address: 0x297b6a0
//
__int64 __fastcall sub_297B6A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  char v6; // al
  void *v7; // rsi
  __int64 v9; // [rsp+0h] [rbp-80h] BYREF
  _QWORD *v10; // [rsp+8h] [rbp-78h]
  int v11; // [rsp+10h] [rbp-70h]
  int v12; // [rsp+14h] [rbp-6Ch]
  int v13; // [rsp+18h] [rbp-68h]
  char v14; // [rsp+1Ch] [rbp-64h]
  _QWORD v15[2]; // [rsp+20h] [rbp-60h] BYREF
  __int64 v16; // [rsp+30h] [rbp-50h] BYREF
  _BYTE *v17; // [rsp+38h] [rbp-48h]
  __int64 v18; // [rsp+40h] [rbp-40h]
  int v19; // [rsp+48h] [rbp-38h]
  char v20; // [rsp+4Ch] [rbp-34h]
  _BYTE v21[48]; // [rsp+50h] [rbp-30h] BYREF

  v5 = sub_BC1CD0(a4, &unk_4F89C30, a3);
  v6 = sub_297B1E0(a2, a3, v5 + 8);
  v7 = (void *)(a1 + 32);
  if ( v6 )
  {
    v10 = v15;
    v11 = 2;
    v15[0] = &unk_4F82408;
    v13 = 0;
    v14 = 1;
    v16 = 0;
    v17 = v21;
    v18 = 2;
    v19 = 0;
    v20 = 1;
    v12 = 1;
    v9 = 1;
    sub_C8CF70(a1, v7, 2, (__int64)v15, (__int64)&v9);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v21, (__int64)&v16);
    if ( !v20 )
      _libc_free((unsigned __int64)v17);
    if ( !v14 )
      _libc_free((unsigned __int64)v10);
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
