// Function: sub_A1EAC0
// Address: 0xa1eac0
//
void __fastcall sub_A1EAC0(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v5; // rax
  _BOOL8 v6; // r14
  __int64 v7; // r14
  unsigned __int8 v8; // al
  __int64 *v9; // rdx
  unsigned __int64 v10; // rax
  unsigned __int8 v11; // al
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  unsigned __int8 v14; // al
  __int64 v15; // r14
  unsigned __int64 v16; // r14
  __int64 v17; // rax

  v5 = *(unsigned int *)(a3 + 8);
  v6 = (*(_BYTE *)(a2 + 1) & 0x7F) == 1;
  if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, a3 + 16, v5 + 1, 8);
    v5 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v5) = v6;
  v7 = a2 - 16;
  ++*(_DWORD *)(a3 + 8);
  v8 = *(_BYTE *)(a2 - 16);
  if ( (v8 & 2) != 0 )
    v9 = *(__int64 **)(a2 - 32);
  else
    v9 = (__int64 *)(v7 - 8LL * ((v8 >> 2) & 0xF));
  v10 = sub_A18650((__int64)(a1 + 35), *v9);
  sub_A188E0(a3, HIDWORD(v10));
  v11 = *(_BYTE *)(a2 - 16);
  if ( (v11 & 2) != 0 )
    v12 = *(_QWORD *)(a2 - 32);
  else
    v12 = v7 - 8LL * ((v11 >> 2) & 0xF);
  v13 = sub_A18650((__int64)(a1 + 35), *(_QWORD *)(v12 + 8));
  sub_A188E0(a3, HIDWORD(v13));
  v14 = *(_BYTE *)(a2 - 16);
  if ( (v14 & 2) != 0 )
    v15 = *(_QWORD *)(a2 - 32);
  else
    v15 = v7 - 8LL * ((v14 >> 2) & 0xF);
  v16 = (unsigned __int64)sub_A18650((__int64)(a1 + 35), *(_QWORD *)(v15 + 16)) >> 32;
  v17 = *(unsigned int *)(a3 + 8);
  if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, a3 + 16, v17 + 1, 8);
    v17 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v17) = v16;
  ++*(_DWORD *)(a3 + 8);
  sub_A188E0(a3, *(unsigned int *)(a2 + 4));
  sub_A1BFB0(*a1, 0x28u, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
