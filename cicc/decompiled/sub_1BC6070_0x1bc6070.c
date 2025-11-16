// Function: sub_1BC6070
// Address: 0x1bc6070
//
__int64 __fastcall sub_1BC6070(__int64 a1, __int64 a2, __int64 a3)
{
  char v5; // al
  int v6; // r9d
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned int v14; // esi
  int v15; // eax
  int v16; // eax
  int v17; // r8d
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  _QWORD v25[2]; // [rsp+0h] [rbp-50h] BYREF
  int v26; // [rsp+10h] [rbp-40h]

  v5 = sub_1BC2430(a2, a3, v25);
  v7 = v25[0];
  if ( v5 )
  {
    v8 = *(unsigned int *)(a2 + 24);
    *(_QWORD *)a1 = a2;
    *(_QWORD *)(a1 + 16) = v7;
    v9 = 5 * v8;
    v10 = *(_QWORD *)(a2 + 8);
    *(_BYTE *)(a1 + 32) = 0;
    v11 = v10 + 8 * v9;
    v12 = *(_QWORD *)a2;
    *(_QWORD *)(a1 + 24) = v11;
    *(_QWORD *)(a1 + 8) = v12;
    return a1;
  }
  v14 = *(_DWORD *)(a2 + 24);
  v15 = *(_DWORD *)(a2 + 16);
  ++*(_QWORD *)a2;
  v16 = v15 + 1;
  v17 = 2 * v14;
  if ( 4 * v16 >= 3 * v14 )
  {
    v14 *= 2;
  }
  else
  {
    v18 = v14 - *(_DWORD *)(a2 + 20) - v16;
    v19 = v14 >> 3;
    if ( (unsigned int)v18 > (unsigned int)v19 )
      goto LABEL_6;
  }
  sub_1BC5DD0(a2, v14);
  sub_1BC2430(a2, a3, v25);
  v7 = v25[0];
  v16 = *(_DWORD *)(a2 + 16) + 1;
LABEL_6:
  *(_DWORD *)(a2 + 16) = v16;
  v26 = -2;
  if ( *(_DWORD *)(v7 + 8) != 1 || **(_DWORD **)v7 != -2 )
    --*(_DWORD *)(a2 + 20);
  sub_1BB9EE0(v7, a3, v18, v19, v17, v6);
  *(_DWORD *)(v7 + 32) = 0;
  v20 = *(unsigned int *)(a2 + 24);
  *(_QWORD *)a1 = a2;
  v21 = 5 * v20;
  v22 = *(_QWORD *)(a2 + 8);
  *(_QWORD *)(a1 + 16) = v7;
  *(_BYTE *)(a1 + 32) = 1;
  v23 = v22 + 8 * v21;
  v24 = *(_QWORD *)a2;
  *(_QWORD *)(a1 + 24) = v23;
  *(_QWORD *)(a1 + 8) = v24;
  return a1;
}
