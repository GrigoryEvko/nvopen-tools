// Function: sub_152BF40
// Address: 0x152bf40
//
void __fastcall sub_152BF40(_DWORD **a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // rax
  _BOOL8 v7; // r15
  __int64 v8; // rax
  __int64 v9; // r15
  unsigned __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // r12

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
  v10 = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 + 8 * (2LL - *(unsigned int *)(a2 + 8)))) >> 32;
  v11 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v11 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v11 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v11) = v10;
  v12 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v12;
  v13 = *(_QWORD *)(a2 + 32);
  if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v12 )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v12 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v12) = v13;
  v14 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v14;
  v15 = *(unsigned int *)(a2 + 48);
  if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v14 )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v14 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v14) = v15;
  v16 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v16;
  v17 = *(unsigned int *)(a2 + 52);
  if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v16 )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v16 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v16) = v17;
  ++*(_DWORD *)(a3 + 8);
  sub_152B6B0(*a1, 0xFu, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
