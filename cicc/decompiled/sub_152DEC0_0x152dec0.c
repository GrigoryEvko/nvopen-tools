// Function: sub_152DEC0
// Address: 0x152dec0
//
void __fastcall sub_152DEC0(_DWORD **a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v5; // rax
  _BOOL8 v6; // r14
  unsigned __int64 v7; // r9
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r9
  unsigned __int64 v11; // r12
  __int64 v12; // rax
  unsigned __int64 v13; // [rsp+0h] [rbp-50h]
  __int64 v14; // [rsp+0h] [rbp-50h]
  _QWORD v16[7]; // [rsp+18h] [rbp-38h] BYREF

  v5 = *(unsigned int *)(a3 + 8);
  v6 = *(_BYTE *)(a2 + 1) == 1;
  if ( (unsigned int)v5 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v5 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v5) = v6;
  ++*(_DWORD *)(a3 + 8);
  v16[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8))) >> 32;
  sub_1525CA0(a3, v16);
  v7 = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8)))) >> 32;
  v8 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v8 >= *(_DWORD *)(a3 + 12) )
  {
    v13 = v7;
    sub_16CD150(a3, a3 + 16, 0, 8);
    v8 = *(unsigned int *)(a3 + 8);
    v7 = v13;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v8) = v7;
  v9 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v9;
  v10 = *(unsigned int *)(a2 + 24);
  if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v9 )
  {
    v14 = *(unsigned int *)(a2 + 24);
    sub_16CD150(a3, a3 + 16, 0, 8);
    v9 = *(unsigned int *)(a3 + 8);
    v10 = v14;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v9) = v10;
  ++*(_DWORD *)(a3 + 8);
  v16[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 + 8 * (3LL - *(unsigned int *)(a2 + 8)))) >> 32;
  sub_1525CA0(a3, v16);
  v16[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 + 8 * (2LL - *(unsigned int *)(a2 + 8)))) >> 32;
  sub_1525CA0(a3, v16);
  v16[0] = *(unsigned int *)(a2 + 28);
  sub_1525CA0(a3, v16);
  v11 = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 + 8 * (4LL - *(unsigned int *)(a2 + 8)))) >> 32;
  v12 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v12 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v12 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v12) = v11;
  ++*(_DWORD *)(a3 + 8);
  sub_152B6B0(*a1, 0x1Eu, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
