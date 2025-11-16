// Function: sub_152C120
// Address: 0x152c120
//
void __fastcall sub_152C120(_DWORD **a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 v7; // rax
  _QWORD v9[7]; // [rsp+18h] [rbp-38h] BYREF

  v5 = a2;
  v9[0] = *(_BYTE *)(a2 + 1) == 1;
  sub_1525CA0(a3, v9);
  v6 = *(unsigned __int16 *)(a2 + 2);
  v7 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v7 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v7 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v7) = v6;
  ++*(_DWORD *)(a3 + 8);
  v9[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 + 8 * (2LL - *(unsigned int *)(a2 + 8)))) >> 32;
  sub_1525CA0(a3, v9);
  if ( *(_BYTE *)a2 != 15 )
    a2 = *(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8));
  v9[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), a2) >> 32;
  sub_1525CA0(a3, v9);
  v9[0] = *(unsigned int *)(v5 + 24);
  sub_1525CA0(a3, v9);
  v9[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(v5 + 8 * (1LL - *(unsigned int *)(v5 + 8)))) >> 32;
  sub_1525CA0(a3, v9);
  v9[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(v5 + 8 * (3LL - *(unsigned int *)(v5 + 8)))) >> 32;
  sub_1525CA0(a3, v9);
  v9[0] = *(_QWORD *)(v5 + 32);
  sub_1525CA0(a3, v9);
  v9[0] = *(unsigned int *)(v5 + 48);
  sub_1525CA0(a3, v9);
  v9[0] = *(_QWORD *)(v5 + 40);
  sub_1525CA0(a3, v9);
  v9[0] = *(unsigned int *)(v5 + 28);
  sub_1525CA0(a3, v9);
  v9[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(v5 + 8 * (4LL - *(unsigned int *)(v5 + 8)))) >> 32;
  sub_1525CA0(a3, v9);
  if ( *(_BYTE *)(v5 + 56) )
    v9[0] = (unsigned int)(*(_DWORD *)(v5 + 52) + 1);
  else
    v9[0] = 0;
  sub_1525CA0(a3, v9);
  sub_152B6B0(*a1, 0x11u, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
