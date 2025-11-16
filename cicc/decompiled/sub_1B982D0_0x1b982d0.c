// Function: sub_1B982D0
// Address: 0x1b982d0
//
__int64 __fastcall sub_1B982D0(__int64 a1, unsigned int a2, _BYTE *a3)
{
  __int64 v5; // rax
  _QWORD *v6; // rdi
  __int64 *v7; // r15
  unsigned int v8; // r14d
  bool v9; // zf
  __int64 v10; // rax
  __int64 v11; // rax
  _BYTE *v12; // r15
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // r12
  _QWORD *v17; // r13
  int *v18; // rdx
  __int64 *v19; // r13
  __int64 v20; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // [rsp+8h] [rbp-58h]
  _BYTE v25[16]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v26; // [rsp+20h] [rbp-40h]

  v5 = *(_QWORD *)(a1 + 8);
  v6 = *(_QWORD **)(v5 + 120);
  v7 = (__int64 *)(v5 + 96);
  v8 = a2 * *(_DWORD *)(v5 + 88);
  v9 = **(_BYTE **)a1 == 0;
  v26 = 257;
  if ( v9 )
  {
    v22 = sub_1643350(v6);
    v23 = sub_159C470(v22, v8, 0);
    v16 = sub_12815B0(v7, 0, a3, v23, (__int64)v25);
    sub_15FA2E0(v16, **(unsigned __int8 **)(a1 + 16));
  }
  else
  {
    v10 = sub_1643350(v6);
    v11 = sub_159C470(v10, -v8, 0);
    v12 = (_BYTE *)sub_12815B0(v7, 0, a3, v11, (__int64)v25);
    sub_15FA2E0((__int64)v12, **(unsigned __int8 **)(a1 + 16));
    v13 = *(_QWORD *)(a1 + 8);
    v26 = 257;
    v24 = (unsigned int)(1 - *(_DWORD *)(v13 + 88));
    v14 = sub_1643350(*(_QWORD **)(v13 + 120));
    v15 = sub_159C470(v14, v24, 0);
    v16 = sub_12815B0((__int64 *)(v13 + 96), 0, v12, v15, (__int64)v25);
    sub_15FA2E0(v16, **(unsigned __int8 **)(a1 + 16));
    if ( **(_BYTE **)(a1 + 24) )
    {
      v17 = (_QWORD *)(**(_QWORD **)(a1 + 32) + 8LL * a2);
      *v17 = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a1 + 8) + 32LL))(*(_QWORD *)(a1 + 8), *v17);
    }
  }
  v18 = *(int **)(a1 + 48);
  v19 = (__int64 *)(*(_QWORD *)(a1 + 8) + 96LL);
  v26 = 257;
  v20 = sub_1647190(**(__int64 ***)(a1 + 40), *v18);
  return sub_12AA3B0(v19, 0x2Fu, v16, v20, (__int64)v25);
}
