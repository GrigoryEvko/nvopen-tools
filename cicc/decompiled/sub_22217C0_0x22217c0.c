// Function: sub_22217C0
// Address: 0x22217c0
//
__int64 __fastcall sub_22217C0(__int64 *a1, __int64 a2)
{
  int v3; // eax
  __int64 v4; // rax
  __int64 v5; // r14
  __int64 v6; // r13
  _BYTE *v7; // rdi
  __int64 v8; // r15
  __int64 v9; // r13
  _BYTE *v10; // rdi
  __int64 v11; // r15
  __int64 v12; // r13
  _BYTE *v13; // rdi
  __int64 v14; // r15
  __int64 v15; // r13
  _BYTE *v16; // rdi
  __int64 result; // rax
  _BYTE *v18; // [rsp+0h] [rbp-58h] BYREF
  __int64 v19; // [rsp+8h] [rbp-50h]
  _BYTE v20[72]; // [rsp+10h] [rbp-48h] BYREF

  *(_BYTE *)(a2 + 33) = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 16))(a1);
  *(_BYTE *)(a2 + 34) = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 24))(a1);
  v3 = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 64))(a1);
  *(_BYTE *)(a2 + 111) = 1;
  *(_DWORD *)(a2 + 88) = v3;
  v4 = *a1;
  *(_QWORD *)(a2 + 16) = 0;
  *(_QWORD *)(a2 + 40) = 0;
  *(_QWORD *)(a2 + 56) = 0;
  *(_QWORD *)(a2 + 72) = 0;
  (*(void (__fastcall **)(_BYTE **, __int64 *))(v4 + 32))(&v18, a1);
  v5 = v19;
  v6 = sub_2207820(v19 + 1);
  sub_2241570(&v18, v6, v5, 0);
  v7 = v18;
  *(_BYTE *)(v6 + v5) = 0;
  *(_QWORD *)(a2 + 24) = v5;
  *(_QWORD *)(a2 + 16) = v6;
  if ( v7 != v20 )
    j___libc_free_0((unsigned __int64)v7);
  (*(void (__fastcall **)(_BYTE **, __int64 *))(*a1 + 40))(&v18, a1);
  v8 = v19;
  v9 = sub_2207820(v19 + 1);
  sub_2241570(&v18, v9, v8, 0);
  v10 = v18;
  *(_BYTE *)(v9 + v8) = 0;
  *(_QWORD *)(a2 + 40) = v9;
  *(_QWORD *)(a2 + 48) = v8;
  if ( v10 != v20 )
    j___libc_free_0((unsigned __int64)v10);
  (*(void (__fastcall **)(_BYTE **, __int64 *))(*a1 + 48))(&v18, a1);
  v11 = v19;
  v12 = sub_2207820(v19 + 1);
  sub_2241570(&v18, v12, v11, 0);
  v13 = v18;
  *(_BYTE *)(v12 + v11) = 0;
  *(_QWORD *)(a2 + 56) = v12;
  *(_QWORD *)(a2 + 64) = v11;
  if ( v13 != v20 )
    j___libc_free_0((unsigned __int64)v13);
  (*(void (__fastcall **)(_BYTE **, __int64 *))(*a1 + 56))(&v18, a1);
  v14 = v19;
  v15 = sub_2207820(v19 + 1);
  sub_2241570(&v18, v15, v14, 0);
  v16 = v18;
  *(_BYTE *)(v15 + v14) = 0;
  *(_QWORD *)(a2 + 72) = v15;
  *(_QWORD *)(a2 + 80) = v14;
  if ( v16 != v20 )
    j___libc_free_0((unsigned __int64)v16);
  *(_DWORD *)(a2 + 92) = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 72))(a1);
  result = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 80))(a1);
  *(_DWORD *)(a2 + 96) = result;
  return result;
}
