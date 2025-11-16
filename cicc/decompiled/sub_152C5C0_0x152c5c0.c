// Function: sub_152C5C0
// Address: 0x152c5c0
//
void __fastcall sub_152C5C0(_DWORD **a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // r15
  unsigned __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r13

  v6 = *(unsigned int *)(a3 + 8);
  v7 = (*(_BYTE *)(a2 + 1) == 1) | 2LL;
  if ( (unsigned int)v6 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v6 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v6) = v7;
  v8 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v8;
  v9 = *(unsigned int *)(a2 + 28);
  if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v8 )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v8 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v8) = v9;
  ++*(_DWORD *)(a3 + 8);
  v10 = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 + 8 * (3LL - *(unsigned int *)(a2 + 8)))) >> 32;
  v11 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v11 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v11 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v11) = v10;
  v12 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v12;
  v13 = *(unsigned __int8 *)(a2 + 52);
  if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v12 )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v12 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v12) = v13;
  ++*(_DWORD *)(a3 + 8);
  sub_152B6B0(*a1, 0x13u, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
