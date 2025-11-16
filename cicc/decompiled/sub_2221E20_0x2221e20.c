// Function: sub_2221E20
// Address: 0x2221e20
//
__int64 __fastcall sub_2221E20(__int64 *a1, __int64 a2)
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
  __int64 v13; // rax
  __int64 v14; // r13
  _BYTE *v15; // rdi
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // r13
  _BYTE *v19; // rdi
  __int64 result; // rax
  __int64 v21; // [rsp+8h] [rbp-60h]
  __int64 v22; // [rsp+8h] [rbp-60h]
  __int64 v23; // [rsp+8h] [rbp-60h]
  _BYTE *v24; // [rsp+10h] [rbp-58h] BYREF
  __int64 v25; // [rsp+18h] [rbp-50h]
  _BYTE v26[72]; // [rsp+20h] [rbp-48h] BYREF

  *(_DWORD *)(a2 + 36) = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 16))(a1);
  *(_DWORD *)(a2 + 40) = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 24))(a1);
  v3 = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 64))(a1);
  *(_QWORD *)(a2 + 16) = 0;
  *(_DWORD *)(a2 + 96) = v3;
  v4 = *a1;
  *(_QWORD *)(a2 + 48) = 0;
  *(_QWORD *)(a2 + 64) = 0;
  *(_QWORD *)(a2 + 80) = 0;
  *(_BYTE *)(a2 + 152) = 1;
  (*(void (__fastcall **)(_BYTE **, __int64 *))(v4 + 32))(&v24, a1);
  v5 = v25;
  v6 = sub_2207820(v25 + 1);
  sub_2241570(&v24, v6, v5, 0);
  v7 = v24;
  *(_BYTE *)(v6 + v5) = 0;
  *(_QWORD *)(a2 + 24) = v5;
  *(_QWORD *)(a2 + 16) = v6;
  if ( v7 != v26 )
    j___libc_free_0((unsigned __int64)v7);
  (*(void (__fastcall **)(_BYTE **, __int64 *))(*a1 + 40))(&v24, a1);
  v8 = v25;
  v9 = v25 + 1;
  if ( (unsigned __int64)(v25 + 1) > 0x1FFFFFFFFFFFFFFELL )
    sub_426640();
  v21 = 4 * v9;
  v10 = sub_2207820(4 * v9);
  sub_2251FC0(&v24, v10, v8, 0);
  v11 = v24;
  *(_QWORD *)(a2 + 48) = v10;
  *(_QWORD *)(a2 + 56) = v8;
  *(_DWORD *)(v10 + v21 - 4) = 0;
  if ( v11 != v26 )
    j___libc_free_0((unsigned __int64)v11);
  (*(void (__fastcall **)(_BYTE **, __int64 *))(*a1 + 48))(&v24, a1);
  v12 = v25;
  v13 = v25 + 1;
  if ( (unsigned __int64)(v25 + 1) > 0x1FFFFFFFFFFFFFFELL )
    sub_426640();
  v22 = 4 * v13;
  v14 = sub_2207820(4 * v13);
  sub_2251FC0(&v24, v14, v12, 0);
  v15 = v24;
  *(_QWORD *)(a2 + 64) = v14;
  *(_QWORD *)(a2 + 72) = v12;
  *(_DWORD *)(v14 + v22 - 4) = 0;
  if ( v15 != v26 )
    j___libc_free_0((unsigned __int64)v15);
  (*(void (__fastcall **)(_BYTE **, __int64 *))(*a1 + 56))(&v24, a1);
  v16 = v25;
  v17 = v25 + 1;
  if ( (unsigned __int64)(v25 + 1) > 0x1FFFFFFFFFFFFFFELL )
    sub_426640();
  v23 = 4 * v17;
  v18 = sub_2207820(4 * v17);
  sub_2251FC0(&v24, v18, v16, 0);
  v19 = v24;
  *(_QWORD *)(a2 + 80) = v18;
  *(_QWORD *)(a2 + 88) = v16;
  *(_DWORD *)(v18 + v23 - 4) = 0;
  if ( v19 != v26 )
    j___libc_free_0((unsigned __int64)v19);
  *(_DWORD *)(a2 + 100) = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 72))(a1);
  result = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 80))(a1);
  *(_DWORD *)(a2 + 104) = result;
  return result;
}
