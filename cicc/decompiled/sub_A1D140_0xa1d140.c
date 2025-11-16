// Function: sub_A1D140
// Address: 0xa1d140
//
void __fastcall sub_A1D140(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // rax
  __int64 v7; // r13
  unsigned __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // r13
  unsigned int v11; // r13d
  unsigned __int8 v12; // al
  __int64 v13; // rdx
  unsigned __int64 v14; // r8
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // r13
  unsigned __int64 v20; // [rsp+8h] [rbp-38h]

  v6 = *(unsigned int *)(a3 + 8);
  v7 = ((*(_BYTE *)(a2 + 1) & 0x7F) == 1) | 2LL;
  if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, a3 + 16, v6 + 1, 8);
    v6 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v6) = v7;
  v8 = *(unsigned int *)(a3 + 12);
  v9 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v9;
  v10 = *(unsigned int *)(a2 + 20);
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
    v13 = *(_QWORD *)(a2 - 32);
  else
    v13 = a2 - 16 - 8LL * ((v12 >> 2) & 0xF);
  v14 = (unsigned __int64)sub_A18650((__int64)(a1 + 35), *(_QWORD *)(v13 + 24)) >> 32;
  v15 = v11;
  v16 = v11 + 1LL;
  if ( v16 > *(unsigned int *)(a3 + 12) )
  {
    v20 = v14;
    sub_C8D5F0(a3, a3 + 16, v16, 8);
    v15 = *(unsigned int *)(a3 + 8);
    v14 = v20;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v15) = v14;
  v17 = *(unsigned int *)(a3 + 12);
  v18 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v18;
  v19 = *(unsigned __int8 *)(a2 + 44);
  if ( v18 + 1 > v17 )
  {
    sub_C8D5F0(a3, a3 + 16, v18 + 1, 8);
    v18 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v18) = v19;
  ++*(_DWORD *)(a3 + 8);
  sub_A1BFB0(*a1, 0x13u, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
