// Function: sub_29CCFB0
// Address: 0x29ccfb0
//
__int64 __fastcall sub_29CCFB0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // r13
  bool v6; // zf
  __int64 v7; // [rsp+10h] [rbp-90h] BYREF
  _QWORD *v8; // [rsp+18h] [rbp-88h]
  int v9; // [rsp+20h] [rbp-80h]
  int v10; // [rsp+24h] [rbp-7Ch]
  int v11; // [rsp+28h] [rbp-78h]
  char v12; // [rsp+2Ch] [rbp-74h]
  _QWORD v13[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v14; // [rsp+40h] [rbp-60h] BYREF
  _BYTE *v15; // [rsp+48h] [rbp-58h]
  __int64 v16; // [rsp+50h] [rbp-50h]
  int v17; // [rsp+58h] [rbp-48h]
  char v18; // [rsp+5Ch] [rbp-44h]
  _BYTE v19[64]; // [rsp+60h] [rbp-40h] BYREF

  v12 = 1;
  v8 = v13;
  v9 = 2;
  v11 = 0;
  v14 = 0;
  v15 = v19;
  v16 = 2;
  v17 = 0;
  v18 = 1;
  v10 = 1;
  v13[0] = &unk_4F82418;
  v7 = 1;
  if ( &unk_4F82418 != (_UNKNOWN *)&qword_4F82400 && &unk_4F82418 != &unk_4F82420 )
  {
    v10 = 2;
    v7 = 2;
    v13[1] = &unk_4F82420;
  }
  v3 = sub_BA8DC0(a3, (__int64)"dx.valver", 9);
  v4 = v3;
  if ( v3 )
  {
    sub_B91A30(v3);
    sub_B91A20(v4);
    sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v13, (__int64)&v7);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v19, (__int64)&v14);
    if ( v18 )
      goto LABEL_6;
LABEL_9:
    _libc_free((unsigned __int64)v15);
    if ( v12 )
      return a1;
LABEL_10:
    _libc_free((unsigned __int64)v8);
    return a1;
  }
  v6 = v18 == 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
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
  if ( v6 )
    goto LABEL_9;
LABEL_6:
  if ( !v12 )
    goto LABEL_10;
  return a1;
}
