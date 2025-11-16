// Function: sub_280EFE0
// Address: 0x280efe0
//
__int64 __fastcall sub_280EFE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  void *v8; // rsi
  __int64 v10; // [rsp+8h] [rbp-A8h]
  __int64 v11; // [rsp+10h] [rbp-A0h]
  __int64 v12; // [rsp+18h] [rbp-98h]
  __int64 v13; // [rsp+20h] [rbp-90h] BYREF
  _QWORD *v14; // [rsp+28h] [rbp-88h]
  int v15; // [rsp+30h] [rbp-80h]
  int v16; // [rsp+34h] [rbp-7Ch]
  int v17; // [rsp+38h] [rbp-78h]
  char v18; // [rsp+3Ch] [rbp-74h]
  _QWORD v19[2]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v20; // [rsp+50h] [rbp-60h] BYREF
  _BYTE *v21; // [rsp+58h] [rbp-58h]
  __int64 v22; // [rsp+60h] [rbp-50h]
  int v23; // [rsp+68h] [rbp-48h]
  char v24; // [rsp+6Ch] [rbp-44h]
  _BYTE v25[64]; // [rsp+70h] [rbp-40h] BYREF

  v10 = sub_BC1CD0(a4, &unk_4F875F0, a3);
  v11 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v12 = sub_BC1CD0(a4, &unk_4F881D0, a3);
  v6 = sub_BC1CD0(a4, &unk_4F8FAE8, a3);
  v7 = sub_BC1CD0(a4, &unk_4F86D28, a3);
  v8 = (void *)(a1 + 32);
  if ( (unsigned __int8)sub_280BCF0(a3, v10 + 8, v11 + 8, v12 + 8, (__int64 *)(v6 + 8), v7 + 8) )
  {
    v15 = 2;
    v14 = v19;
    v17 = 0;
    v18 = 1;
    v20 = 0;
    v21 = v25;
    v22 = 2;
    v23 = 0;
    v24 = 1;
    v16 = 1;
    v19[0] = &unk_4F875F0;
    v13 = 1;
    if ( &unk_4F875F0 != (_UNKNOWN *)&qword_4F82400 && &unk_4F875F0 != &unk_4F81450 )
    {
      v16 = 2;
      v13 = 2;
      v19[1] = &unk_4F81450;
    }
    sub_C8CF70(a1, v8, 2, (__int64)v19, (__int64)&v13);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v25, (__int64)&v20);
    if ( v24 )
    {
      if ( v18 )
        return a1;
    }
    else
    {
      _libc_free((unsigned __int64)v21);
      if ( v18 )
        return a1;
    }
    _libc_free((unsigned __int64)v14);
    return a1;
  }
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
