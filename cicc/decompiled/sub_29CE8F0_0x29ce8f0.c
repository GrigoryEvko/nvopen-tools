// Function: sub_29CE8F0
// Address: 0x29ce8f0
//
__int64 __fastcall sub_29CE8F0(__int64 a1, char *a2, __int64 a3)
{
  char v4; // bl
  char v5; // al
  __int64 v6; // r9
  char v7; // al
  __int64 v9; // [rsp+10h] [rbp-90h] BYREF
  _QWORD *v10; // [rsp+18h] [rbp-88h]
  int v11; // [rsp+20h] [rbp-80h]
  int v12; // [rsp+24h] [rbp-7Ch]
  int v13; // [rsp+28h] [rbp-78h]
  char v14; // [rsp+2Ch] [rbp-74h]
  _QWORD v15[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v16; // [rsp+40h] [rbp-60h] BYREF
  _BYTE *v17; // [rsp+48h] [rbp-58h]
  __int64 v18; // [rsp+50h] [rbp-50h]
  int v19; // [rsp+58h] [rbp-48h]
  char v20; // [rsp+5Ch] [rbp-44h]
  _BYTE v21[64]; // [rsp+60h] [rbp-40h] BYREF

  v4 = *a2;
  v5 = sub_B2D610(a3, 20);
  v6 = a1 + 32;
  if ( v5 || (*(_BYTE *)(a3 + 32) & 0xF) == 1 || (v7 = sub_29CE1B0(a3, v4), v6 = a1 + 32, !v7) )
  {
    *(_QWORD *)(a1 + 8) = v6;
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
    return a1;
  }
  v10 = v15;
  v15[0] = &unk_4F82408;
  v11 = 2;
  v13 = 0;
  v14 = 1;
  v16 = 0;
  v17 = v21;
  v18 = 2;
  v19 = 0;
  v20 = 1;
  v12 = 1;
  v9 = 1;
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v15, (__int64)&v9);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v21, (__int64)&v16);
  if ( !v20 )
  {
    _libc_free((unsigned __int64)v17);
    if ( v14 )
      return a1;
    goto LABEL_8;
  }
  if ( !v14 )
LABEL_8:
    _libc_free((unsigned __int64)v10);
  return a1;
}
