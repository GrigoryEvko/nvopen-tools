// Function: sub_152E0F0
// Address: 0x152e0f0
//
void __fastcall sub_152E0F0(_DWORD **a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // rax
  _BOOL8 v7; // r15
  __int64 v8; // rax
  __int64 v9; // r15
  unsigned __int64 v10; // r8
  __int64 v11; // rax
  unsigned __int64 v12; // r8
  __int64 v13; // rax
  unsigned __int64 v14; // r8
  __int64 v15; // rax
  unsigned __int64 v16; // r12
  __int64 v17; // rax
  unsigned __int64 v18; // [rsp+8h] [rbp-48h]
  unsigned __int64 v19; // [rsp+8h] [rbp-48h]
  unsigned __int64 v20; // [rsp+8h] [rbp-48h]
  _QWORD v21[7]; // [rsp+18h] [rbp-38h] BYREF

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
  v10 = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8))) >> 32;
  v11 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v11 >= *(_DWORD *)(a3 + 12) )
  {
    v18 = v10;
    sub_16CD150(a3, a3 + 16, 0, 8);
    v11 = *(unsigned int *)(a3 + 8);
    v10 = v18;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v11) = v10;
  ++*(_DWORD *)(a3 + 8);
  v12 = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8)))) >> 32;
  v13 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v13 >= *(_DWORD *)(a3 + 12) )
  {
    v19 = v12;
    sub_16CD150(a3, a3 + 16, 0, 8);
    v13 = *(unsigned int *)(a3 + 8);
    v12 = v19;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v13) = v12;
  ++*(_DWORD *)(a3 + 8);
  v21[0] = *(unsigned int *)(a2 + 24);
  sub_1525CA0(a3, v21);
  v14 = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 + 8 * (2LL - *(unsigned int *)(a2 + 8)))) >> 32;
  v15 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v15 >= *(_DWORD *)(a3 + 12) )
  {
    v20 = v14;
    sub_16CD150(a3, a3 + 16, 0, 8);
    v15 = *(unsigned int *)(a3 + 8);
    v14 = v20;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v15) = v14;
  ++*(_DWORD *)(a3 + 8);
  v16 = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 + 8 * (3LL - *(unsigned int *)(a2 + 8)))) >> 32;
  v17 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v17 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v17 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v17) = v16;
  ++*(_DWORD *)(a3 + 8);
  sub_152B6B0(*a1, 0x1Fu, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
