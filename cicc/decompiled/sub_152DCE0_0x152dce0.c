// Function: sub_152DCE0
// Address: 0x152dce0
//
void __fastcall sub_152DCE0(_DWORD **a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // rax
  _BOOL8 v7; // r15
  unsigned __int64 v8; // r8
  __int64 v9; // rax
  unsigned __int64 v10; // r8
  __int64 v11; // rax
  unsigned __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r12
  unsigned __int64 v16; // [rsp+8h] [rbp-38h]
  unsigned __int64 v17; // [rsp+8h] [rbp-38h]

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
    v16 = v8;
    sub_16CD150(a3, a3 + 16, 0, 8);
    v9 = *(unsigned int *)(a3 + 8);
    v8 = v16;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v9) = v8;
  ++*(_DWORD *)(a3 + 8);
  v10 = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8)))) >> 32;
  v11 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v11 >= *(_DWORD *)(a3 + 12) )
  {
    v17 = v10;
    sub_16CD150(a3, a3 + 16, 0, 8);
    v11 = *(unsigned int *)(a3 + 8);
    v10 = v17;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v11) = v10;
  ++*(_DWORD *)(a3 + 8);
  v12 = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 + 8 * (2LL - *(unsigned int *)(a2 + 8)))) >> 32;
  v13 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v13 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v13 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v13) = v12;
  v14 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v14;
  v15 = *(unsigned int *)(a2 + 24);
  if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v14 )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v14 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v14) = v15;
  ++*(_DWORD *)(a3 + 8);
  sub_152B6B0(*a1, 0x28u, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
