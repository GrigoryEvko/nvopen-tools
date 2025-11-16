// Function: sub_152BA30
// Address: 0x152ba30
//
void __fastcall sub_152BA30(_DWORD **a1, __int64 a2, __int64 a3, unsigned int a4)
{
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdx
  unsigned int v8; // ecx
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // rcx
  __int64 v12; // rdx
  _BYTE *v13; // rax
  _BYTE *v14; // r15
  size_t v15; // r8
  unsigned __int64 v16; // r14
  size_t v17; // [rsp+0h] [rbp-40h]
  _BYTE *v18; // [rsp+8h] [rbp-38h]

  v6 = *(unsigned int *)(a3 + 12);
  v7 = ((__int64)(*(_QWORD *)(a2 + 32) - *(_QWORD *)(a2 + 24)) >> 3) + 1;
  v8 = v6;
  if ( v7 > v6 )
  {
    sub_16CD150(a3, a3 + 16, v7, 8);
    v8 = *(_DWORD *)(a3 + 12);
  }
  v9 = *(unsigned int *)(a3 + 8);
  v10 = (*(_BYTE *)(a2 + 1) == 1) | 6LL;
  if ( (unsigned int)v9 >= v8 )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v9 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v9) = v10;
  v11 = *(unsigned int *)(a3 + 12);
  v12 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v12;
  v13 = *(_BYTE **)(a2 + 32);
  v14 = *(_BYTE **)(a2 + 24);
  v15 = v13 - v14;
  v16 = (v13 - v14) >> 3;
  if ( v16 > v11 - v12 )
  {
    v17 = *(_QWORD *)(a2 + 32) - (_QWORD)v14;
    v18 = *(_BYTE **)(a2 + 32);
    sub_16CD150(a3, a3 + 16, v16 + v12, 8);
    v12 = *(unsigned int *)(a3 + 8);
    v15 = v17;
    v13 = v18;
  }
  if ( v13 != v14 )
  {
    memcpy((void *)(*(_QWORD *)a3 + 8 * v12), v14, v15);
    LODWORD(v12) = *(_DWORD *)(a3 + 8);
  }
  *(_DWORD *)(a3 + 8) = v16 + v12;
  sub_152B6B0(*a1, 0x1Du, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
