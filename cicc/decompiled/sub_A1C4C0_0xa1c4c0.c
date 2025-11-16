// Function: sub_A1C4C0
// Address: 0xa1c4c0
//
void __fastcall sub_A1C4C0(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // rax
  _BOOL8 v7; // r12
  unsigned int v8; // r12d
  unsigned __int8 v9; // al
  __int64 *v10; // rdx
  unsigned __int64 v11; // rax
  __int64 v12; // r9
  unsigned __int64 v13; // r10
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  unsigned int v16; // r12d
  unsigned __int8 v17; // al
  __int64 v18; // r15
  unsigned __int64 v19; // r15
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // [rsp+10h] [rbp-40h]
  __int64 v23; // [rsp+18h] [rbp-38h]

  v6 = *(unsigned int *)(a3 + 8);
  v7 = (*(_BYTE *)(a2 + 1) & 0x7F) == 1;
  if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, a3 + 16, v6 + 1, 8);
    v6 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v6) = v7;
  v8 = *(_DWORD *)(a3 + 8) + 1;
  *(_DWORD *)(a3 + 8) = v8;
  v23 = a2 - 16;
  v9 = *(_BYTE *)(a2 - 16);
  if ( (v9 & 2) != 0 )
    v10 = *(__int64 **)(a2 - 32);
  else
    v10 = (__int64 *)(v23 - 8LL * ((v9 >> 2) & 0xF));
  v11 = sub_A18650((__int64)(a1 + 35), *v10);
  v12 = (__int64)(a1 + 35);
  v13 = HIDWORD(v11);
  v14 = v8;
  v15 = v8 + 1LL;
  if ( v15 > *(unsigned int *)(a3 + 12) )
  {
    v22 = v13;
    sub_C8D5F0(a3, a3 + 16, v15, 8);
    v14 = *(unsigned int *)(a3 + 8);
    v12 = (__int64)(a1 + 35);
    v13 = v22;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v14) = v13;
  v16 = *(_DWORD *)(a3 + 8) + 1;
  *(_DWORD *)(a3 + 8) = v16;
  v17 = *(_BYTE *)(a2 - 16);
  if ( (v17 & 2) != 0 )
    v18 = *(_QWORD *)(a2 - 32);
  else
    v18 = v23 - 8LL * ((v17 >> 2) & 0xF);
  v19 = (unsigned __int64)sub_A18650(v12, *(_QWORD *)(v18 + 8)) >> 32;
  v20 = v16;
  v21 = v16 + 1LL;
  if ( v21 > *(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, a3 + 16, v21, 8);
    v20 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v20) = v19;
  ++*(_DWORD *)(a3 + 8);
  sub_A1BFB0(*a1, 0x25u, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
