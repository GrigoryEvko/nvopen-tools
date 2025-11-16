// Function: sub_2618720
// Address: 0x2618720
//
__int64 __fastcall sub_2618720(__int64 a1, _DWORD *a2, __int64 a3, __int64 a4)
{
  char v6; // al
  void *v7; // rsi
  __int64 v9; // [rsp+8h] [rbp-A8h] BYREF
  __int64 v10; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v11; // [rsp+18h] [rbp-98h] BYREF
  __int64 v12; // [rsp+20h] [rbp-90h] BYREF
  __int64 (__fastcall *v13)(__int64 *, __int64); // [rsp+28h] [rbp-88h]
  __int64 v14; // [rsp+30h] [rbp-80h]
  __int64 (__fastcall *v15)(__int64 *, __int64); // [rsp+38h] [rbp-78h]
  _QWORD v16[2]; // [rsp+40h] [rbp-70h] BYREF
  __int64 *v17; // [rsp+50h] [rbp-60h] BYREF
  _BYTE *v18; // [rsp+58h] [rbp-58h]
  __int64 v19; // [rsp+60h] [rbp-50h]
  int v20; // [rsp+68h] [rbp-48h]
  char v21; // [rsp+6Ch] [rbp-44h]
  _BYTE v22[64]; // [rsp+70h] [rbp-40h] BYREF

  v9 = *(_QWORD *)(sub_BC0510(a4, &unk_4F82418, a3) + 8);
  v10 = v9;
  v11 = v9;
  LODWORD(v12) = *a2;
  v13 = sub_26176B0;
  v14 = (__int64)&v9;
  v15 = sub_26176D0;
  v16[0] = &v10;
  v16[1] = sub_26177B0;
  v17 = &v11;
  v6 = sub_2617DD0((__int64)&v12, a3);
  v7 = (void *)(a1 + 32);
  if ( v6 )
  {
    v13 = (__int64 (__fastcall *)(__int64 *, __int64))v16;
    v14 = 0x100000002LL;
    v16[0] = &unk_4F875F0;
    LODWORD(v15) = 0;
    BYTE4(v15) = 1;
    v17 = 0;
    v18 = v22;
    v19 = 2;
    v20 = 0;
    v21 = 1;
    v12 = 1;
    sub_C8CF70(a1, v7, 2, (__int64)v16, (__int64)&v12);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v22, (__int64)&v17);
    if ( v21 )
    {
      if ( BYTE4(v15) )
        return a1;
    }
    else
    {
      _libc_free((unsigned __int64)v18);
      if ( BYTE4(v15) )
        return a1;
    }
    _libc_free((unsigned __int64)v13);
    return a1;
  }
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
  return a1;
}
