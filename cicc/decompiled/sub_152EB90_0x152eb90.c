// Function: sub_152EB90
// Address: 0x152eb90
//
void __fastcall sub_152EB90(_DWORD **a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v4; // r12
  __int64 v6; // rax
  _BOOL8 v7; // r13
  __int64 v8; // rax
  __int64 v9; // r13
  _QWORD v11[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = a2;
  v6 = *(unsigned int *)(a3 + 8);
  v7 = *(_BYTE *)(a2 + 1) == 1;
  if ( (unsigned int)v6 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v6 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v6) = v7;
  v8 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v8;
  v9 = *(unsigned __int16 *)(a2 + 2);
  if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v8 )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v8 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v8) = v9;
  ++*(_DWORD *)(a3 + 8);
  v11[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 + 8 * (2LL - *(unsigned int *)(a2 + 8)))) >> 32;
  sub_1525CA0(a3, v11);
  if ( *(_BYTE *)a2 != 15 )
    a2 = *(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8));
  v11[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), a2) >> 32;
  sub_1525CA0(a3, v11);
  v11[0] = *(unsigned int *)(v4 + 24);
  sub_1525CA0(a3, v11);
  v11[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(v4 + 8 * (1LL - *(unsigned int *)(v4 + 8)))) >> 32;
  sub_1525CA0(a3, v11);
  v11[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(v4 + 8 * (3LL - *(unsigned int *)(v4 + 8)))) >> 32;
  sub_1525CA0(a3, v11);
  v11[0] = *(_QWORD *)(v4 + 32);
  sub_1525CA0(a3, v11);
  v11[0] = *(unsigned int *)(v4 + 48);
  sub_1525CA0(a3, v11);
  v11[0] = *(_QWORD *)(v4 + 40);
  sub_1525CA0(a3, v11);
  v11[0] = *(unsigned int *)(v4 + 28);
  sub_1525CA0(a3, v11);
  v11[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(v4 + 8 * (4LL - *(unsigned int *)(v4 + 8)))) >> 32;
  sub_1525CA0(a3, v11);
  sub_152B6B0(*a1, 0x2Au, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
