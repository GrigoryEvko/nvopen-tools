// Function: sub_22212D0
// Address: 0x22212d0
//
void __fastcall sub_22212D0(__int64 *a1, __int64 a2)
{
  char v3; // al
  __int64 v4; // rax
  __int64 v5; // r14
  __int64 v6; // r13
  _BYTE *v7; // rdi
  __int64 v8; // r15
  __int64 v9; // r14
  _BYTE *v10; // rdi
  __int64 v11; // r14
  __int64 v12; // rbp
  _BYTE *v13; // rdi
  _BYTE *v14; // [rsp+0h] [rbp-58h] BYREF
  __int64 v15; // [rsp+8h] [rbp-50h]
  _BYTE v16[72]; // [rsp+10h] [rbp-48h] BYREF

  *(_BYTE *)(a2 + 72) = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 16))(a1);
  v3 = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 24))(a1);
  *(_BYTE *)(a2 + 136) = 1;
  *(_BYTE *)(a2 + 73) = v3;
  v4 = *a1;
  *(_QWORD *)(a2 + 16) = 0;
  *(_QWORD *)(a2 + 40) = 0;
  *(_QWORD *)(a2 + 56) = 0;
  (*(void (__fastcall **)(_BYTE **, __int64 *))(v4 + 32))(&v14, a1);
  v5 = v15;
  v6 = sub_2207820(v15 + 1);
  sub_2241570(&v14, v6, v5, 0);
  v7 = v14;
  *(_BYTE *)(v6 + v5) = 0;
  *(_QWORD *)(a2 + 16) = v6;
  *(_QWORD *)(a2 + 24) = v5;
  if ( v7 != v16 )
    j___libc_free_0((unsigned __int64)v7);
  (*(void (__fastcall **)(_BYTE **, __int64 *))(*a1 + 40))(&v14, a1);
  v8 = v15;
  v9 = sub_2207820(v15 + 1);
  sub_2241570(&v14, v9, v8, 0);
  v10 = v14;
  *(_BYTE *)(v9 + v8) = 0;
  *(_QWORD *)(a2 + 40) = v9;
  *(_QWORD *)(a2 + 48) = v8;
  if ( v10 != v16 )
    j___libc_free_0((unsigned __int64)v10);
  (*(void (__fastcall **)(_BYTE **, __int64 *))(*a1 + 48))(&v14, a1);
  v11 = v15;
  v12 = sub_2207820(v15 + 1);
  sub_2241570(&v14, v12, v11, 0);
  v13 = v14;
  *(_BYTE *)(v12 + v11) = 0;
  *(_QWORD *)(a2 + 56) = v12;
  *(_QWORD *)(a2 + 64) = v11;
  if ( v13 != v16 )
    j___libc_free_0((unsigned __int64)v13);
}
