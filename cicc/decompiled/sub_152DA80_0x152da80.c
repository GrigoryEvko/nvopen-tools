// Function: sub_152DA80
// Address: 0x152da80
//
void __fastcall sub_152DA80(_DWORD **a1, __int64 a2, __int64 a3, unsigned int a4)
{
  unsigned __int64 v6; // r9
  __int64 v7; // rax
  unsigned __int64 v8; // r9
  __int64 v9; // rax
  unsigned __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // rbx
  __int64 v15; // rax
  unsigned __int64 v16; // [rsp+0h] [rbp-50h]
  unsigned __int64 v17; // [rsp+0h] [rbp-50h]
  _QWORD v19[7]; // [rsp+18h] [rbp-38h] BYREF

  v19[0] = (*(_BYTE *)(a2 + 1) == 1) | 2LL;
  sub_1525CA0(a3, v19);
  v19[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8))) >> 32;
  sub_1525CA0(a3, v19);
  v6 = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8)))) >> 32;
  v7 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v7 >= *(_DWORD *)(a3 + 12) )
  {
    v16 = v6;
    sub_16CD150(a3, a3 + 16, 0, 8);
    v7 = *(unsigned int *)(a3 + 8);
    v6 = v16;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v7) = v6;
  ++*(_DWORD *)(a3 + 8);
  v8 = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 + 8 * (2LL - *(unsigned int *)(a2 + 8)))) >> 32;
  v9 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v9 >= *(_DWORD *)(a3 + 12) )
  {
    v17 = v8;
    sub_16CD150(a3, a3 + 16, 0, 8);
    v9 = *(unsigned int *)(a3 + 8);
    v8 = v17;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v9) = v8;
  ++*(_DWORD *)(a3 + 8);
  v19[0] = *(unsigned int *)(a2 + 24);
  sub_1525CA0(a3, v19);
  v10 = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 + 8 * (3LL - *(unsigned int *)(a2 + 8)))) >> 32;
  v11 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v11 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v11 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v11) = v10;
  v12 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v12;
  v13 = *(unsigned __int16 *)(a2 + 32);
  if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v12 )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v12 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v12) = v13;
  ++*(_DWORD *)(a3 + 8);
  v19[0] = *(unsigned int *)(a2 + 36);
  sub_1525CA0(a3, v19);
  v14 = *(unsigned int *)(a2 + 28);
  v15 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v15 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v15 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v15) = v14;
  ++*(_DWORD *)(a3 + 8);
  sub_152B6B0(*a1, 0x1Cu, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
