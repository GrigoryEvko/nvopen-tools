// Function: sub_A1E2F0
// Address: 0xa1e2f0
//
void __fastcall sub_A1E2F0(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // rax
  _BOOL8 v7; // r15
  __int64 v8; // r15
  int v9; // edx
  unsigned __int8 v10; // al
  __int64 *v11; // rcx
  unsigned __int64 v12; // rax
  __int64 v13; // r8
  unsigned __int64 v14; // r9
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  int v17; // edx
  unsigned __int8 v18; // al
  __int64 v19; // r15
  unsigned __int64 v20; // r15
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rcx
  __int64 v24; // rax
  unsigned __int64 v25; // r12
  unsigned int v26; // [rsp+0h] [rbp-40h]
  unsigned int v27; // [rsp+8h] [rbp-38h]
  unsigned __int64 v28; // [rsp+8h] [rbp-38h]

  v6 = *(unsigned int *)(a3 + 8);
  v7 = (*(_BYTE *)(a2 + 1) & 0x7F) == 1;
  if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, a3 + 16, v6 + 1, 8);
    v6 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v6) = v7;
  v8 = a2 - 16;
  v9 = *(_DWORD *)(a3 + 8) + 1;
  *(_DWORD *)(a3 + 8) = v9;
  v10 = *(_BYTE *)(a2 - 16);
  if ( (v10 & 2) != 0 )
    v11 = *(__int64 **)(a2 - 32);
  else
    v11 = (__int64 *)(v8 - 8LL * ((v10 >> 2) & 0xF));
  v26 = v9;
  v12 = sub_A18650((__int64)(a1 + 35), *v11);
  v13 = (__int64)(a1 + 35);
  v14 = HIDWORD(v12);
  v15 = v26;
  v16 = v26 + 1LL;
  if ( v16 > *(unsigned int *)(a3 + 12) )
  {
    v28 = v14;
    sub_C8D5F0(a3, a3 + 16, v16, 8);
    v15 = *(unsigned int *)(a3 + 8);
    v13 = (__int64)(a1 + 35);
    v14 = v28;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v15) = v14;
  v17 = *(_DWORD *)(a3 + 8) + 1;
  *(_DWORD *)(a3 + 8) = v17;
  v18 = *(_BYTE *)(a2 - 16);
  if ( (v18 & 2) != 0 )
    v19 = *(_QWORD *)(a2 - 32);
  else
    v19 = v8 - 8LL * ((v18 >> 2) & 0xF);
  v27 = v17;
  v20 = (unsigned __int64)sub_A18650(v13, *(_QWORD *)(v19 + 8)) >> 32;
  v21 = v27;
  v22 = v27 + 1LL;
  if ( v22 > *(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, a3 + 16, v22, 8);
    v21 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v21) = v20;
  v23 = *(unsigned int *)(a3 + 12);
  v24 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v24;
  v25 = (unsigned __int64)*(char *)(a2 + 1) >> 63;
  if ( v24 + 1 > v23 )
  {
    sub_C8D5F0(a3, a3 + 16, v24 + 1, 8);
    v24 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v24) = v25;
  ++*(_DWORD *)(a3 + 8);
  sub_A1BFB0(*a1, 0x19u, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
