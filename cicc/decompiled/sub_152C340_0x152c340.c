// Function: sub_152C340
// Address: 0x152c340
//
void __fastcall sub_152C340(_DWORD **a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v5; // rbx
  _QWORD v7[7]; // [rsp+18h] [rbp-38h] BYREF

  v5 = a2;
  v7[0] = (*(_BYTE *)(a2 + 1) == 1) | 2u;
  sub_1525CA0(a3, v7);
  v7[0] = *(unsigned __int16 *)(a2 + 2);
  sub_1525CA0(a3, v7);
  v7[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(v5 + 8 * (2LL - *(unsigned int *)(v5 + 8)))) >> 32;
  sub_1525CA0(a3, v7);
  if ( *(_BYTE *)a2 != 15 )
    a2 = *(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8));
  v7[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), a2) >> 32;
  sub_1525CA0(a3, v7);
  v7[0] = *(unsigned int *)(v5 + 24);
  sub_1525CA0(a3, v7);
  v7[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(v5 + 8 * (1LL - *(unsigned int *)(v5 + 8)))) >> 32;
  sub_1525CA0(a3, v7);
  v7[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(v5 + 8 * (3LL - *(unsigned int *)(v5 + 8)))) >> 32;
  sub_1525CA0(a3, v7);
  v7[0] = *(_QWORD *)(v5 + 32);
  sub_1525CA0(a3, v7);
  v7[0] = *(unsigned int *)(v5 + 48);
  sub_1525CA0(a3, v7);
  v7[0] = *(_QWORD *)(v5 + 40);
  sub_1525CA0(a3, v7);
  v7[0] = *(unsigned int *)(v5 + 28);
  sub_1525CA0(a3, v7);
  v7[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(v5 + 8 * (4LL - *(unsigned int *)(v5 + 8)))) >> 32;
  sub_1525CA0(a3, v7);
  v7[0] = *(unsigned int *)(v5 + 52);
  sub_1525CA0(a3, v7);
  v7[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(v5 + 8 * (5LL - *(unsigned int *)(v5 + 8)))) >> 32;
  sub_1525CA0(a3, v7);
  v7[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(v5 + 8 * (6LL - *(unsigned int *)(v5 + 8)))) >> 32;
  sub_1525CA0(a3, v7);
  v7[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(v5 + 8 * (7LL - *(unsigned int *)(v5 + 8)))) >> 32;
  sub_1525CA0(a3, v7);
  v7[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(v5 + 8 * (8LL - *(unsigned int *)(v5 + 8)))) >> 32;
  sub_1525CA0(a3, v7);
  sub_152B6B0(*a1, 0x12u, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
