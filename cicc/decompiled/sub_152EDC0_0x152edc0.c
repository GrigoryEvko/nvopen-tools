// Function: sub_152EDC0
// Address: 0x152edc0
//
void __fastcall sub_152EDC0(_DWORD **a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // r14
  unsigned __int64 v11; // r9
  __int64 v12; // rax
  unsigned __int64 v13; // r12
  __int64 v14; // rax
  unsigned __int64 v15; // [rsp+0h] [rbp-50h]
  _QWORD v17[7]; // [rsp+18h] [rbp-38h] BYREF

  v17[0] = *(_BYTE *)(a2 + 1) == 1;
  sub_1525CA0(a3, v17);
  v5 = *(_QWORD *)(a2 + 24);
  v6 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v6 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v6 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v6) = v5;
  v7 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v7;
  v8 = *(_QWORD *)(a2 + 32);
  if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v7 )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v7 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v7) = v8;
  v9 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v9;
  v10 = *(unsigned __int8 *)(a2 + 40);
  if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v9 )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v9 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v9) = v10;
  ++*(_DWORD *)(a3 + 8);
  v17[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8))) >> 32;
  sub_1525CA0(a3, v17);
  v11 = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8)))) >> 32;
  v12 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v12 >= *(_DWORD *)(a3 + 12) )
  {
    v15 = v11;
    sub_16CD150(a3, a3 + 16, 0, 8);
    v12 = *(unsigned int *)(a3 + 8);
    v11 = v15;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v12) = v11;
  ++*(_DWORD *)(a3 + 8);
  v17[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 + 8 * (2LL - *(unsigned int *)(a2 + 8)))) >> 32;
  sub_1525CA0(a3, v17);
  v13 = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 + 8 * (3LL - *(unsigned int *)(a2 + 8)))) >> 32;
  v14 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v14 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v14 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v14) = v13;
  ++*(_DWORD *)(a3 + 8);
  sub_152B6B0(*a1, 0x2Bu, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
