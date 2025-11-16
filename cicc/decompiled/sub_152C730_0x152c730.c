// Function: sub_152C730
// Address: 0x152c730
//
void __fastcall sub_152C730(_DWORD **a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // rax
  _BOOL8 v7; // r12
  unsigned __int64 v8; // r14
  __int64 v9; // rax
  unsigned __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // [rsp+0h] [rbp-50h]
  _QWORD v15[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = *(unsigned int *)(a3 + 8);
  v7 = *(_BYTE *)(a2 + 1) == 1;
  if ( (unsigned int)v6 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v6 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v6) = v7;
  ++*(_DWORD *)(a3 + 8);
  v8 = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8))) >> 32;
  v9 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v9 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v9 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v9) = v8;
  ++*(_DWORD *)(a3 + 8);
  v10 = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8)))) >> 32;
  v11 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v11 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v11 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v11) = v10;
  ++*(_DWORD *)(a3 + 8);
  if ( *(_BYTE *)(a2 + 40) )
  {
    v15[0] = *(unsigned int *)(a2 + 24);
    sub_1525CA0(a3, v15);
    if ( *(_BYTE *)(a2 + 40) )
      v13 = *(_QWORD *)(a2 + 32);
    v12 = v13;
  }
  else
  {
    v15[0] = 0;
    sub_1525CA0(a3, v15);
    v12 = 0;
  }
  v15[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), v12) >> 32;
  sub_1525CA0(a3, v15);
  if ( *(_BYTE *)(a2 + 56) )
  {
    v15[0] = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 + 48)) >> 32;
    sub_1525CA0(a3, v15);
  }
  sub_152B6B0(*a1, 0x10u, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
