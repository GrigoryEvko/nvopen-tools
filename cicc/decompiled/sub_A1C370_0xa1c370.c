// Function: sub_A1C370
// Address: 0xa1c370
//
void __fastcall sub_A1C370(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  unsigned __int64 v6; // rcx
  unsigned __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // r15
  unsigned __int64 v10; // rcx
  __int64 v11; // rax
  _BYTE *v12; // r8
  _BYTE *v13; // r15
  size_t v14; // r9
  __int64 v15; // r14
  size_t v16; // [rsp+0h] [rbp-40h]
  _BYTE *v17; // [rsp+8h] [rbp-38h]

  v6 = *(unsigned int *)(a3 + 12);
  v7 = ((__int64)(*(_QWORD *)(a2 + 24) - *(_QWORD *)(a2 + 16)) >> 3) + 1;
  if ( v7 > v6 )
  {
    sub_C8D5F0(a3, a3 + 16, v7, 8);
    v6 = *(unsigned int *)(a3 + 12);
  }
  v8 = *(unsigned int *)(a3 + 8);
  v9 = ((*(_BYTE *)(a2 + 1) & 0x7F) == 1) | 6LL;
  if ( v8 + 1 > v6 )
  {
    sub_C8D5F0(a3, a3 + 16, v8 + 1, 8);
    v8 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v8) = v9;
  v10 = *(unsigned int *)(a3 + 12);
  v11 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v11;
  v12 = *(_BYTE **)(a2 + 24);
  v13 = *(_BYTE **)(a2 + 16);
  v14 = v12 - v13;
  v15 = (v12 - v13) >> 3;
  if ( v15 + v11 > v10 )
  {
    v16 = *(_QWORD *)(a2 + 24) - (_QWORD)v13;
    v17 = *(_BYTE **)(a2 + 24);
    sub_C8D5F0(a3, a3 + 16, v15 + v11, 8);
    v11 = *(unsigned int *)(a3 + 8);
    v14 = v16;
    v12 = v17;
  }
  if ( v12 != v13 )
  {
    memcpy((void *)(*(_QWORD *)a3 + 8 * v11), v13, v14);
    LODWORD(v11) = *(_DWORD *)(a3 + 8);
  }
  *(_DWORD *)(a3 + 8) = v15 + v11;
  sub_A1BFB0(*a1, 0x1Du, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
