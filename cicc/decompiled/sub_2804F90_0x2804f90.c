// Function: sub_2804F90
// Address: 0x2804f90
//
__int64 __fastcall sub_2804F90(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r15
  __int64 v7; // rbx
  void *v8; // rsi
  __int64 v10; // [rsp+8h] [rbp-D8h]
  __int64 v11; // [rsp+10h] [rbp-D0h]
  __int64 v12; // [rsp+18h] [rbp-C8h]
  _QWORD v13[6]; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v14; // [rsp+50h] [rbp-90h] BYREF
  _QWORD *v15; // [rsp+58h] [rbp-88h]
  int v16; // [rsp+60h] [rbp-80h]
  int v17; // [rsp+64h] [rbp-7Ch]
  int v18; // [rsp+68h] [rbp-78h]
  char v19; // [rsp+6Ch] [rbp-74h]
  _QWORD v20[2]; // [rsp+70h] [rbp-70h] BYREF
  __int64 v21; // [rsp+80h] [rbp-60h] BYREF
  _BYTE *v22; // [rsp+88h] [rbp-58h]
  __int64 v23; // [rsp+90h] [rbp-50h]
  int v24; // [rsp+98h] [rbp-48h]
  char v25; // [rsp+9Ch] [rbp-44h]
  _BYTE v26[64]; // [rsp+A0h] [rbp-40h] BYREF

  v10 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v11 = sub_BC1CD0(a4, &unk_4F875F0, a3);
  v12 = sub_BC1CD0(a4, &unk_4F881D0, a3);
  v6 = sub_BC1CD0(a4, &unk_4F86630, a3);
  v7 = sub_BC1CD0(a4, &unk_4F8FAE8, a3) + 8;
  v13[0] = v6 + 8;
  v13[4] = sub_BC1CD0(a4, &unk_4F89C30, a3) + 8;
  v13[1] = v10 + 8;
  v13[3] = v12 + 8;
  v13[2] = v11 + 8;
  v13[5] = v7;
  v8 = (void *)(a1 + 32);
  if ( !(unsigned __int8)sub_2804C20(v13) )
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
    return a1;
  }
  v15 = v20;
  v16 = 2;
  v18 = 0;
  v19 = 1;
  v21 = 0;
  v22 = v26;
  v23 = 2;
  v24 = 0;
  v25 = 1;
  v17 = 1;
  v20[0] = &unk_4F81450;
  v14 = 1;
  if ( &unk_4F81450 != (_UNKNOWN *)&qword_4F82400 && &unk_4F81450 != &unk_4F875F0 )
  {
    v17 = 2;
    v14 = 2;
    v20[1] = &unk_4F875F0;
  }
  sub_C8CF70(a1, v8, 2, (__int64)v20, (__int64)&v14);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v26, (__int64)&v21);
  if ( !v25 )
  {
    _libc_free((unsigned __int64)v22);
    if ( v19 )
      return a1;
    goto LABEL_9;
  }
  if ( !v19 )
LABEL_9:
    _libc_free((unsigned __int64)v15);
  return a1;
}
