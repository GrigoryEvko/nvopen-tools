// Function: sub_152CEF0
// Address: 0x152cef0
//
void __fastcall sub_152CEF0(_DWORD **a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // rax
  _BOOL8 v7; // r15
  unsigned __int64 v8; // rax
  __int64 v9; // r9
  unsigned __int64 v10; // r8
  __int64 v11; // rax
  unsigned __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // r12
  unsigned __int64 v18; // [rsp+8h] [rbp-38h]

  v6 = *(unsigned int *)(a3 + 8);
  v7 = *(_BYTE *)(a2 + 1) == 1;
  if ( (unsigned int)v6 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v6 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v6) = v7;
  ++*(_DWORD *)(a3 + 8);
  v8 = sub_1525AD0((__int64)(a1 + 35), *(_QWORD *)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8))));
  v9 = a2;
  v10 = HIDWORD(v8);
  v11 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v11 >= *(_DWORD *)(a3 + 12) )
  {
    v18 = v10;
    sub_16CD150(a3, a3 + 16, 0, 8);
    v11 = *(unsigned int *)(a3 + 8);
    v9 = a2;
    v10 = v18;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v11) = v10;
  ++*(_DWORD *)(a3 + 8);
  if ( *(_BYTE *)a2 != 15 )
    v9 = *(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8));
  v12 = (unsigned __int64)sub_1525AD0((__int64)(a1 + 35), v9) >> 32;
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
  v16 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v16;
  v17 = *(unsigned __int16 *)(a2 + 28);
  if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v16 )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v16 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v16) = v17;
  ++*(_DWORD *)(a3 + 8);
  sub_152B6B0(*a1, 0x16u, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
