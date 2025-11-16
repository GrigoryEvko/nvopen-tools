// Function: sub_152CBC0
// Address: 0x152cbc0
//
void __fastcall sub_152CBC0(_DWORD **a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rsi
  _QWORD v13[7]; // [rsp+18h] [rbp-38h] BYREF

  v5 = a2;
  v13[0] = (*(_BYTE *)(a2 + 1) == 1) | 2LL;
  sub_1525CA0(a3, v13);
  v13[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(v5 + 8 * (1LL - *(unsigned int *)(v5 + 8)))) >> 32;
  sub_1525CA0(a3, v13);
  v13[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(v5 + 8 * (2LL - *(unsigned int *)(v5 + 8)))) >> 32;
  sub_1525CA0(a3, v13);
  v13[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(v5 + 8 * (3LL - *(unsigned int *)(v5 + 8)))) >> 32;
  sub_1525CA0(a3, v13);
  if ( *(_BYTE *)a2 != 15 )
    a2 = *(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8));
  v13[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), a2) >> 32;
  sub_1525CA0(a3, v13);
  v13[0] = *(unsigned int *)(v5 + 24);
  sub_1525CA0(a3, v13);
  v13[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(v5 + 8 * (4LL - *(unsigned int *)(v5 + 8)))) >> 32;
  sub_1525CA0(a3, v13);
  v13[0] = (*(_BYTE *)(v5 + 40) & 4) != 0;
  sub_1525CA0(a3, v13);
  v13[0] = (*(_BYTE *)(v5 + 40) & 8) != 0;
  sub_1525CA0(a3, v13);
  v13[0] = *(unsigned int *)(v5 + 28);
  sub_1525CA0(a3, v13);
  v6 = *(unsigned int *)(v5 + 8);
  v7 = 0;
  if ( (unsigned int)v6 > 8 )
    v7 = *(_QWORD *)(v5 + 8 * (8 - v6));
  v13[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), v7) >> 32;
  sub_1525CA0(a3, v13);
  v13[0] = *(_BYTE *)(v5 + 40) & 3;
  sub_1525CA0(a3, v13);
  v13[0] = *(unsigned int *)(v5 + 32);
  sub_1525CA0(a3, v13);
  v13[0] = *(unsigned int *)(v5 + 44);
  sub_1525CA0(a3, v13);
  v13[0] = (*(_BYTE *)(v5 + 40) & 0x10) != 0;
  sub_1525CA0(a3, v13);
  v13[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(v5 + 8 * (5LL - *(unsigned int *)(v5 + 8)))) >> 32;
  sub_1525CA0(a3, v13);
  v8 = *(unsigned int *)(v5 + 8);
  v9 = 0;
  if ( (unsigned int)v8 > 9 )
    v9 = *(_QWORD *)(v5 + 8 * (9 - v8));
  v13[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), v9) >> 32;
  sub_1525CA0(a3, v13);
  v13[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(v5 + 8 * (6LL - *(unsigned int *)(v5 + 8)))) >> 32;
  sub_1525CA0(a3, v13);
  v13[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(v5 + 8 * (7LL - *(unsigned int *)(v5 + 8)))) >> 32;
  sub_1525CA0(a3, v13);
  v13[0] = *(int *)(v5 + 36);
  sub_1525CA0(a3, v13);
  v10 = *(unsigned int *)(v5 + 8);
  v11 = 0;
  if ( (unsigned int)v10 > 0xA )
    v11 = *(_QWORD *)(v5 + 8 * (10 - v10));
  v13[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), v11) >> 32;
  sub_1525CA0(a3, v13);
  sub_152B6B0(*a1, 0x15u, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
