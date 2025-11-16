// Function: sub_152F000
// Address: 0x152f000
//
void __fastcall sub_152F000(_QWORD **a1, __int64 a2, __int64 a3, unsigned int *a4)
{
  __int64 v7; // rax
  _BOOL8 v8; // r15
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // r8
  __int64 v14; // rax
  __int64 v15; // rsi
  unsigned __int64 v16; // r12
  __int64 v17; // rax
  __int64 v18; // [rsp+8h] [rbp-38h]

  if ( !*a4 )
    *a4 = sub_1527610(a1);
  v7 = *(unsigned int *)(a3 + 8);
  v8 = *(_BYTE *)(a2 + 1) == 1;
  if ( (unsigned int)v7 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v7 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v7) = v8;
  v9 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v9;
  v10 = *(unsigned int *)(a2 + 4);
  if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v9 )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v9 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v9) = v10;
  v11 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v11;
  v12 = *(unsigned __int16 *)(a2 + 2);
  if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v11 )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v11 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v11) = v12;
  ++*(_DWORD *)(a3 + 8);
  v13 = (unsigned int)(((unsigned __int64)sub_1525AD0(
                                            (__int64)(a1 + 35),
                                            *(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8))) >> 32)
                     - 1);
  v14 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v14 >= *(_DWORD *)(a3 + 12) )
  {
    v18 = v13;
    sub_16CD150(a3, a3 + 16, 0, 8);
    v14 = *(unsigned int *)(a3 + 8);
    v13 = v18;
  }
  v15 = 0;
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v14) = v13;
  ++*(_DWORD *)(a3 + 8);
  if ( *(_DWORD *)(a2 + 8) == 2 )
    v15 = *(_QWORD *)(a2 - 8);
  v16 = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), v15) >> 32;
  v17 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v17 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v17 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v17) = v16;
  ++*(_DWORD *)(a3 + 8);
  sub_152B6B0(*a1, 7u, a3, *a4);
  *(_DWORD *)(a3 + 8) = 0;
}
