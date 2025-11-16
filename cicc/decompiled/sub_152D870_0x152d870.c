// Function: sub_152D870
// Address: 0x152d870
//
void __fastcall sub_152D870(_DWORD **a1, __int64 a2, __int64 a3, unsigned int a4)
{
  unsigned __int64 v6; // r9
  __int64 v7; // rax
  unsigned __int64 v8; // [rsp+0h] [rbp-50h]
  _QWORD v10[7]; // [rsp+18h] [rbp-38h] BYREF

  v10[0] = (*(_BYTE *)(a2 + 1) == 1) | 2LL;
  sub_1525CA0(a3, v10);
  v10[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8))) >> 32;
  sub_1525CA0(a3, v10);
  v10[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8)))) >> 32;
  sub_1525CA0(a3, v10);
  v10[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 + 8 * (5LL - *(unsigned int *)(a2 + 8)))) >> 32;
  sub_1525CA0(a3, v10);
  v6 = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 + 8 * (2LL - *(unsigned int *)(a2 + 8)))) >> 32;
  v7 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v7 >= *(_DWORD *)(a3 + 12) )
  {
    v8 = v6;
    sub_16CD150(a3, a3 + 16, 0, 8);
    v7 = *(unsigned int *)(a3 + 8);
    v6 = v8;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v7) = v6;
  ++*(_DWORD *)(a3 + 8);
  v10[0] = *(unsigned int *)(a2 + 24);
  sub_1525CA0(a3, v10);
  v10[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 + 8 * (3LL - *(unsigned int *)(a2 + 8)))) >> 32;
  sub_1525CA0(a3, v10);
  v10[0] = *(unsigned __int8 *)(a2 + 32);
  sub_1525CA0(a3, v10);
  v10[0] = *(unsigned __int8 *)(a2 + 33);
  sub_1525CA0(a3, v10);
  v10[0] = 0;
  sub_1525CA0(a3, v10);
  v10[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 + 8 * (6LL - *(unsigned int *)(a2 + 8)))) >> 32;
  sub_1525CA0(a3, v10);
  v10[0] = *(unsigned int *)(a2 + 28);
  sub_1525CA0(a3, v10);
  sub_152B6B0(*a1, 0x1Bu, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
