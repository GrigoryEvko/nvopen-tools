// Function: sub_A1F030
// Address: 0xa1f030
//
void __fastcall sub_A1F030(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // rax
  _BOOL8 v7; // r15
  unsigned __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // r15
  unsigned __int8 v12; // al
  __int64 *v13; // rdx
  unsigned __int64 v14; // rax
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  __int64 v17; // r8
  int v18; // edx
  unsigned __int8 v19; // al
  __int64 v20; // r15
  unsigned __int64 v21; // r12
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  unsigned int v24; // [rsp+8h] [rbp-38h]
  unsigned __int64 v25; // [rsp+8h] [rbp-38h]

  v6 = *(unsigned int *)(a3 + 8);
  v7 = (*(_BYTE *)(a2 + 1) & 0x7F) == 1;
  if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, a3 + 16, v6 + 1, 8);
    v6 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v6) = v7;
  v8 = *(unsigned int *)(a3 + 12);
  v9 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v9;
  v10 = *(unsigned __int16 *)(a2 + 2);
  if ( v9 + 1 > v8 )
  {
    sub_C8D5F0(a3, a3 + 16, v9 + 1, 8);
    v9 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v9) = v10;
  v11 = a2 - 16;
  ++*(_DWORD *)(a3 + 8);
  sub_A188E0(a3, *(unsigned int *)(a2 + 4));
  v12 = *(_BYTE *)(a2 - 16);
  if ( (v12 & 2) != 0 )
    v13 = *(__int64 **)(a2 - 32);
  else
    v13 = (__int64 *)(v11 - 8LL * ((v12 >> 2) & 0xF));
  v14 = sub_A18650((__int64)(a1 + 35), *v13);
  v15 = *(unsigned int *)(a3 + 8);
  v16 = HIDWORD(v14);
  v17 = (__int64)(a1 + 35);
  if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    v25 = v16;
    sub_C8D5F0(a3, a3 + 16, v15 + 1, 8);
    v15 = *(unsigned int *)(a3 + 8);
    v17 = (__int64)(a1 + 35);
    v16 = v25;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v15) = v16;
  v18 = *(_DWORD *)(a3 + 8) + 1;
  *(_DWORD *)(a3 + 8) = v18;
  v19 = *(_BYTE *)(a2 - 16);
  if ( (v19 & 2) != 0 )
    v20 = *(_QWORD *)(a2 - 32);
  else
    v20 = v11 - 8LL * ((v19 >> 2) & 0xF);
  v24 = v18;
  v21 = (unsigned __int64)sub_A18650(v17, *(_QWORD *)(v20 + 8)) >> 32;
  v22 = v24;
  v23 = v24 + 1LL;
  if ( v23 > *(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, a3 + 16, v23, 8);
    v22 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v22) = v21;
  ++*(_DWORD *)(a3 + 8);
  sub_A1BFB0(*a1, 0x21u, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
