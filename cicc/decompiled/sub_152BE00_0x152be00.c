// Function: sub_152BE00
// Address: 0x152be00
//
void __fastcall sub_152BE00(_DWORD **a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r13
  unsigned __int64 v11; // r12
  __int64 v12; // rax

  v6 = *(unsigned int *)(a3 + 8);
  v7 = (*(_BYTE *)(a2 + 1) == 1) | (2 * (*(_DWORD *)(a2 + 4) != 0));
  if ( (unsigned int)v6 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v6 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v6) = v7;
  v8 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v8;
  v9 = *(_QWORD *)(a2 + 24);
  v10 = ~(2 * v9);
  if ( v9 >= 0 )
    v10 = 2 * v9;
  if ( (unsigned int)v8 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v8 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v8) = v10;
  ++*(_DWORD *)(a3 + 8);
  v11 = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8))) >> 32;
  v12 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v12 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v12 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v12) = v11;
  ++*(_DWORD *)(a3 + 8);
  sub_152B6B0(*a1, 0xEu, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
