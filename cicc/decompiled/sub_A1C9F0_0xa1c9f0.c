// Function: sub_A1C9F0
// Address: 0xa1c9f0
//
void __fastcall sub_A1C9F0(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  unsigned __int16 v6; // ax
  unsigned __int8 v7; // al
  __int64 v8; // rdx
  unsigned __int64 v9; // r15
  __int64 v10; // rax
  unsigned __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // r15
  unsigned int v14; // eax
  __int64 v15; // rax
  __int64 v16; // r15
  unsigned __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // r15

  sub_A188E0(a3, (*(_BYTE *)(a2 + 1) & 0x7F) == 1);
  v6 = sub_AF18C0(a2);
  sub_A188E0(a3, v6);
  v7 = *(_BYTE *)(a2 - 16);
  if ( (v7 & 2) != 0 )
    v8 = *(_QWORD *)(a2 - 32);
  else
    v8 = a2 - 16 - 8LL * ((v7 >> 2) & 0xF);
  v9 = (unsigned __int64)sub_A18650((__int64)(a1 + 35), *(_QWORD *)(v8 + 16)) >> 32;
  v10 = *(unsigned int *)(a3 + 8);
  if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, a3 + 16, v10 + 1, 8);
    v10 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v10) = v9;
  v11 = *(unsigned int *)(a3 + 12);
  v12 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v12;
  v13 = *(_QWORD *)(a2 + 24);
  if ( v12 + 1 > v11 )
  {
    sub_C8D5F0(a3, a3 + 16, v12 + 1, 8);
    v12 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v12) = v13;
  ++*(_DWORD *)(a3 + 8);
  v14 = sub_AF18D0(a2);
  sub_A188E0(a3, v14);
  v15 = *(unsigned int *)(a3 + 8);
  v16 = *(unsigned int *)(a2 + 44);
  if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, a3 + 16, v15 + 1, 8);
    v15 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v15) = v16;
  v17 = *(unsigned int *)(a3 + 12);
  v18 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v18;
  v19 = *(unsigned int *)(a2 + 20);
  if ( v18 + 1 > v17 )
  {
    sub_C8D5F0(a3, a3 + 16, v18 + 1, 8);
    v18 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v18) = v19;
  ++*(_DWORD *)(a3 + 8);
  sub_A188E0(a3, *(unsigned int *)(a2 + 40));
  sub_A1BFB0(*a1, 0xFu, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
