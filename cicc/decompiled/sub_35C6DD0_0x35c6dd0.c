// Function: sub_35C6DD0
// Address: 0x35c6dd0
//
__int64 __fastcall sub_35C6DD0(__int64 a1, __int64 a2)
{
  __int64 (*v2)(); // rax
  __int64 v3; // rax
  void (*v4)(); // rax
  __int64 v6; // [rsp+8h] [rbp-138h]
  _BYTE *v7; // [rsp+10h] [rbp-130h] BYREF
  __int64 v8; // [rsp+18h] [rbp-128h]
  _BYTE v9[48]; // [rsp+20h] [rbp-120h] BYREF
  int v10; // [rsp+50h] [rbp-F0h]
  __int64 v11; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v12; // [rsp+68h] [rbp-D8h]
  __int64 v13; // [rsp+70h] [rbp-D0h]
  __int64 v14; // [rsp+78h] [rbp-C8h]
  __int64 v15; // [rsp+80h] [rbp-C0h]
  _BYTE *v16; // [rsp+88h] [rbp-B8h]
  __int64 v17; // [rsp+90h] [rbp-B0h]
  _BYTE v18[32]; // [rsp+98h] [rbp-A8h] BYREF
  __int64 v19; // [rsp+B8h] [rbp-88h]
  _BYTE *v20; // [rsp+C0h] [rbp-80h]
  __int64 v21; // [rsp+C8h] [rbp-78h]
  _BYTE v22[48]; // [rsp+D0h] [rbp-70h] BYREF
  int v23; // [rsp+100h] [rbp-40h]

  v2 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 136LL);
  if ( v2 == sub_2DD19D0 )
  {
    v11 = 0;
    v16 = v18;
    v12 = 0;
    v17 = 0x200000000LL;
    v20 = v22;
    v13 = 0;
    v14 = 0;
    v15 = 0;
    v19 = 0;
    v21 = 0x600000000LL;
    v23 = 0;
    v7 = v9;
    v8 = 0x600000000LL;
    v10 = 0;
    BUG();
  }
  v3 = v2();
  v20 = v22;
  v11 = 0;
  v17 = 0x200000000LL;
  v12 = 0;
  v13 = 0;
  v14 = 0;
  v15 = 0;
  v16 = v18;
  v19 = 0;
  v21 = 0x600000000LL;
  v23 = 0;
  v7 = v9;
  v8 = 0x600000000LL;
  v10 = 0;
  v6 = v3;
  (*(void (__fastcall **)(__int64, __int64, _BYTE **, __int64 *))(*(_QWORD *)v3 + 264LL))(v3, a2, &v7, &v11);
  v4 = *(void (**)())(*(_QWORD *)v6 + 272LL);
  if ( v4 != nullsub_1676 )
    ((void (__fastcall *)(__int64, __int64, __int64 *))v4)(v6, a2, &v11);
  sub_35C6D20((_QWORD *)a2, (__int64)&v11);
  if ( v7 != v9 )
    _libc_free((unsigned __int64)v7);
  if ( v20 != v22 )
    _libc_free((unsigned __int64)v20);
  if ( v16 != v18 )
    _libc_free((unsigned __int64)v16);
  return 1;
}
