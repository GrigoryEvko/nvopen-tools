// Function: sub_152D540
// Address: 0x152d540
//
void __fastcall sub_152D540(_DWORD **a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // rax
  _BOOL8 v7; // r15
  unsigned __int64 v8; // r8
  __int64 v9; // rax
  unsigned __int64 v10; // r12
  __int64 v11; // rax
  unsigned __int64 v12; // [rsp+8h] [rbp-38h]

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
    v12 = v8;
    sub_16CD150(a3, a3 + 16, 0, 8);
    v9 = *(unsigned int *)(a3 + 8);
    v8 = v12;
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
  sub_152B6B0(*a1, 0x19u, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
