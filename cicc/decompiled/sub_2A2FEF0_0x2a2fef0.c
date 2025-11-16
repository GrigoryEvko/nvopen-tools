// Function: sub_2A2FEF0
// Address: 0x2a2fef0
//
__int64 __fastcall sub_2A2FEF0(__int64 a1, __int64 a2, __int64 **a3)
{
  void *v3; // rsi
  __int64 v5; // [rsp+0h] [rbp-80h] BYREF
  _QWORD *v6; // [rsp+8h] [rbp-78h]
  int v7; // [rsp+10h] [rbp-70h]
  int v8; // [rsp+14h] [rbp-6Ch]
  int v9; // [rsp+18h] [rbp-68h]
  char v10; // [rsp+1Ch] [rbp-64h]
  _QWORD v11[2]; // [rsp+20h] [rbp-60h] BYREF
  __int64 v12; // [rsp+30h] [rbp-50h] BYREF
  _BYTE *v13; // [rsp+38h] [rbp-48h]
  __int64 v14; // [rsp+40h] [rbp-40h]
  int v15; // [rsp+48h] [rbp-38h]
  char v16; // [rsp+4Ch] [rbp-34h]
  _BYTE v17[48]; // [rsp+50h] [rbp-30h] BYREF

  v3 = (void *)(a1 + 32);
  if ( (unsigned __int8)sub_2A2E910(a3) )
  {
    v6 = v11;
    v7 = 2;
    v11[0] = &unk_4F82408;
    v9 = 0;
    v10 = 1;
    v12 = 0;
    v13 = v17;
    v14 = 2;
    v15 = 0;
    v16 = 1;
    v8 = 1;
    v5 = 1;
    sub_C8CF70(a1, v3, 2, (__int64)v11, (__int64)&v5);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v17, (__int64)&v12);
    if ( !v16 )
      _libc_free((unsigned __int64)v13);
    if ( !v10 )
      _libc_free((unsigned __int64)v6);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v3;
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
