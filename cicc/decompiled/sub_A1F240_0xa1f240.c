// Function: sub_A1F240
// Address: 0xa1f240
//
void __fastcall sub_A1F240(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // rax
  _BOOL8 v7; // r15
  unsigned __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // r15
  unsigned __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 v14; // r15
  unsigned __int8 v15; // al
  __int64 *v16; // rdx
  unsigned __int64 v17; // rax
  unsigned __int8 v18; // al
  __int64 v19; // r15
  unsigned __int64 v20; // r12
  __int64 v21; // rax

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
  v11 = *(unsigned int *)(a3 + 12);
  v12 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v12;
  v13 = *(unsigned int *)(a2 + 4);
  if ( v12 + 1 > v11 )
  {
    sub_C8D5F0(a3, a3 + 16, v12 + 1, 8);
    v12 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v12) = v13;
  v14 = a2 - 16;
  ++*(_DWORD *)(a3 + 8);
  v15 = *(_BYTE *)(a2 - 16);
  if ( (v15 & 2) != 0 )
    v16 = *(__int64 **)(a2 - 32);
  else
    v16 = (__int64 *)(v14 - 8LL * ((v15 >> 2) & 0xF));
  v17 = sub_A18650((__int64)(a1 + 35), *v16);
  sub_A188E0(a3, HIDWORD(v17));
  v18 = *(_BYTE *)(a2 - 16);
  if ( (v18 & 2) != 0 )
    v19 = *(_QWORD *)(a2 - 32);
  else
    v19 = v14 - 8LL * ((v18 >> 2) & 0xF);
  v20 = (unsigned __int64)sub_A18650((__int64)(a1 + 35), *(_QWORD *)(v19 + 8)) >> 32;
  v21 = *(unsigned int *)(a3 + 8);
  if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, a3 + 16, v21 + 1, 8);
    v21 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v21) = v20;
  ++*(_DWORD *)(a3 + 8);
  sub_A1BFB0(*a1, 0x22u, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
