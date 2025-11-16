// Function: sub_152E930
// Address: 0x152e930
//
void __fastcall sub_152E930(_DWORD **a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v5; // r14
  __int64 v6; // rax
  unsigned __int64 v7; // r9
  __int64 v8; // rax
  unsigned __int64 v9; // r9
  __int64 v10; // rax
  unsigned __int64 v11; // r8
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // r12
  __int64 v16; // rax
  unsigned __int64 v17; // [rsp+0h] [rbp-50h]
  unsigned __int64 v18; // [rsp+0h] [rbp-50h]
  unsigned __int64 v19; // [rsp+0h] [rbp-50h]
  __int64 v20; // [rsp+0h] [rbp-50h]
  _QWORD v22[7]; // [rsp+18h] [rbp-38h] BYREF

  v22[0] = *(_BYTE *)(a2 + 1) == 1;
  sub_1525CA0(a3, v22);
  v5 = *(unsigned __int16 *)(a2 + 2);
  v6 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v6 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v6 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v6) = v5;
  ++*(_DWORD *)(a3 + 8);
  v7 = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 + 8 * (2LL - *(unsigned int *)(a2 + 8)))) >> 32;
  v8 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v8 >= *(_DWORD *)(a3 + 12) )
  {
    v17 = v7;
    sub_16CD150(a3, a3 + 16, 0, 8);
    v8 = *(unsigned int *)(a3 + 8);
    v7 = v17;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v8) = v7;
  ++*(_DWORD *)(a3 + 8);
  v9 = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 + 8 * (3LL - *(unsigned int *)(a2 + 8)))) >> 32;
  v10 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v10 >= *(_DWORD *)(a3 + 12) )
  {
    v18 = v9;
    sub_16CD150(a3, a3 + 16, 0, 8);
    v10 = *(unsigned int *)(a3 + 8);
    v9 = v18;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v10) = v9;
  ++*(_DWORD *)(a3 + 8);
  v11 = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 + 8 * (4LL - *(unsigned int *)(a2 + 8)))) >> 32;
  v12 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v12 >= *(_DWORD *)(a3 + 12) )
  {
    v19 = v11;
    sub_16CD150(a3, a3 + 16, 0, 8);
    v12 = *(unsigned int *)(a3 + 8);
    v11 = v19;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v12) = v11;
  v13 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v13;
  v14 = *(_QWORD *)(a2 + 32);
  if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v13 )
  {
    v20 = *(_QWORD *)(a2 + 32);
    sub_16CD150(a3, a3 + 16, 0, 8);
    v13 = *(unsigned int *)(a3 + 8);
    v14 = v20;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v13) = v14;
  ++*(_DWORD *)(a3 + 8);
  v22[0] = *(unsigned int *)(a2 + 48);
  sub_1525CA0(a3, v22);
  v15 = *(unsigned int *)(a2 + 52);
  v16 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v16 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v16 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v16) = v15;
  ++*(_DWORD *)(a3 + 8);
  sub_152B6B0(*a1, 0x29u, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
