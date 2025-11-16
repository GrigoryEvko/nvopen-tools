// Function: sub_29772B0
// Address: 0x29772b0
//
__int64 __fastcall sub_29772B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 *v4; // r15
  __int64 v8; // rax
  __int64 v9; // r9
  void *v10; // rsi
  bool v11; // zf
  __int64 **v13; // [rsp+8h] [rbp-98h]
  __int64 v14; // [rsp+10h] [rbp-90h] BYREF
  _QWORD *v15; // [rsp+18h] [rbp-88h]
  __int64 v16; // [rsp+20h] [rbp-80h]
  int v17; // [rsp+28h] [rbp-78h]
  char v18; // [rsp+2Ch] [rbp-74h]
  _QWORD v19[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v20; // [rsp+40h] [rbp-60h] BYREF
  _BYTE *v21; // [rsp+48h] [rbp-58h]
  __int64 v22; // [rsp+50h] [rbp-50h]
  int v23; // [rsp+58h] [rbp-48h]
  char v24; // [rsp+5Ch] [rbp-44h]
  _BYTE v25[64]; // [rsp+60h] [rbp-40h] BYREF

  v4 = 0;
  v13 = (__int64 **)(sub_BC1CD0(a4, &unk_4F89C30, a3) + 8);
  *(_QWORD *)(a2 + 16) = sub_BC1CD0(a4, &unk_4F86630, a3) + 8;
  if ( LOBYTE(qword_4F8D3A0[17]) )
    v4 = (unsigned __int64 *)(sub_BC1CD0(a4, &unk_4F81450, a3) + 8);
  v8 = sub_BC1CD0(a4, &unk_4F8FC88, a3);
  v10 = (void *)(a1 + 32);
  if ( !(unsigned __int8)sub_29760C0(a3, v13, v4, v8 + 8, a2, v9) )
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
    return a1;
  }
  v14 = 0;
  v15 = v19;
  v11 = LOBYTE(qword_4F8D3A0[17]) == 0;
  v16 = 2;
  v17 = 0;
  v18 = 1;
  v20 = 0;
  v21 = v25;
  v22 = 2;
  v23 = 0;
  v24 = 1;
  if ( !v11 )
  {
    HIDWORD(v16) = 1;
    v14 = 1;
    v19[0] = &unk_4F81450;
  }
  sub_C8CF70(a1, v10, 2, (__int64)v19, (__int64)&v14);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v25, (__int64)&v20);
  if ( !v24 )
  {
    _libc_free((unsigned __int64)v21);
    if ( v18 )
      return a1;
    goto LABEL_9;
  }
  if ( !v18 )
LABEL_9:
    _libc_free((unsigned __int64)v15);
  return a1;
}
