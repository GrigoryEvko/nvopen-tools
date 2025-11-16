// Function: sub_2221440
// Address: 0x2221440
//
void __fastcall sub_2221440(__int64 *a1, __int64 a2)
{
  int v3; // eax
  __int64 v4; // rax
  __int64 v5; // r14
  __int64 v6; // r13
  _BYTE *v7; // rdi
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // r13
  _BYTE *v11; // rdi
  __int64 v12; // r15
  unsigned __int64 v13; // r13
  __int64 v14; // rbp
  _BYTE *v15; // rdi
  __int64 v16; // [rsp+8h] [rbp-60h]
  _BYTE *v17; // [rsp+10h] [rbp-58h] BYREF
  __int64 v18; // [rsp+18h] [rbp-50h]
  _BYTE v19[72]; // [rsp+20h] [rbp-48h] BYREF

  *(_DWORD *)(a2 + 72) = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 16))(a1);
  v3 = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 24))(a1);
  *(_BYTE *)(a2 + 328) = 1;
  *(_DWORD *)(a2 + 76) = v3;
  v4 = *a1;
  *(_QWORD *)(a2 + 16) = 0;
  *(_QWORD *)(a2 + 40) = 0;
  *(_QWORD *)(a2 + 56) = 0;
  (*(void (__fastcall **)(_BYTE **, __int64 *))(v4 + 32))(&v17, a1);
  v5 = v18;
  v6 = sub_2207820(v18 + 1);
  sub_2241570(&v17, v6, v5, 0);
  v7 = v17;
  *(_BYTE *)(v6 + v5) = 0;
  *(_QWORD *)(a2 + 24) = v5;
  *(_QWORD *)(a2 + 16) = v6;
  if ( v7 != v19 )
    j___libc_free_0((unsigned __int64)v7);
  (*(void (__fastcall **)(_BYTE **, __int64 *))(*a1 + 40))(&v17, a1);
  v8 = v18;
  v9 = v18 + 1;
  if ( (unsigned __int64)(v18 + 1) > 0x1FFFFFFFFFFFFFFELL )
    sub_426640();
  v16 = 4 * v9;
  v10 = sub_2207820(4 * v9);
  sub_2251FC0(&v17, v10, v8, 0);
  v11 = v17;
  *(_QWORD *)(a2 + 40) = v10;
  *(_QWORD *)(a2 + 48) = v8;
  *(_DWORD *)(v10 + v16 - 4) = 0;
  if ( v11 != v19 )
    j___libc_free_0((unsigned __int64)v11);
  (*(void (__fastcall **)(_BYTE **, __int64 *))(*a1 + 48))(&v17, a1);
  v12 = v18;
  if ( (unsigned __int64)(v18 + 1) > 0x1FFFFFFFFFFFFFFELL )
    sub_426640();
  v13 = 4 * (v18 + 1);
  v14 = sub_2207820(v13);
  sub_2251FC0(&v17, v14, v12, 0);
  v15 = v17;
  *(_QWORD *)(a2 + 56) = v14;
  *(_DWORD *)(v14 + v13 - 4) = 0;
  *(_QWORD *)(a2 + 64) = v12;
  if ( v15 != v19 )
    j___libc_free_0((unsigned __int64)v15);
}
