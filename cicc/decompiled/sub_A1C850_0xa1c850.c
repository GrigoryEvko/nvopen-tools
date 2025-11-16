// Function: sub_A1C850
// Address: 0xa1c850
//
void __fastcall sub_A1C850(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  unsigned __int64 v6; // r15
  __int64 v7; // rax
  unsigned __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // r15
  unsigned int v11; // r15d
  unsigned __int8 v12; // al
  __int64 *v13; // rdx
  unsigned __int64 v14; // r8
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // [rsp+8h] [rbp-38h]

  v6 = ((*(_BYTE *)(a2 + 1) & 0x7F) == 1) | (unsigned __int64)((2 * (*(_DWORD *)(a2 + 4) != 0)) | 4u);
  v7 = *(unsigned int *)(a3 + 8);
  if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, a3 + 16, v7 + 1, 8);
    v7 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v7) = v6;
  v8 = *(unsigned int *)(a3 + 12);
  v9 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v9;
  v10 = *(unsigned int *)(a2 + 24);
  if ( v9 + 1 > v8 )
  {
    sub_C8D5F0(a3, a3 + 16, v9 + 1, 8);
    v9 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v9) = v10;
  v11 = *(_DWORD *)(a3 + 8) + 1;
  *(_DWORD *)(a3 + 8) = v11;
  v12 = *(_BYTE *)(a2 - 16);
  if ( (v12 & 2) != 0 )
    v13 = *(__int64 **)(a2 - 32);
  else
    v13 = (__int64 *)(a2 - 16 - 8LL * ((v12 >> 2) & 0xF));
  v14 = (unsigned __int64)sub_A18650((__int64)(a1 + 35), *v13) >> 32;
  v15 = v11;
  v16 = v11 + 1LL;
  if ( v16 > *(unsigned int *)(a3 + 12) )
  {
    v17 = v14;
    sub_C8D5F0(a3, a3 + 16, v16, 8);
    v15 = *(unsigned int *)(a3 + 8);
    v14 = v17;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v15) = v14;
  ++*(_DWORD *)(a3 + 8);
  sub_A16F20((__int64 *)a3, a2 + 16);
  sub_A1BFB0(*a1, 0xEu, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
