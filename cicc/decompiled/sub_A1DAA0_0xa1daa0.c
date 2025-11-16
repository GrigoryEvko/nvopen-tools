// Function: sub_A1DAA0
// Address: 0xa1daa0
//
void __fastcall sub_A1DAA0(__int64 *a1, _BYTE *a2, __int64 a3, unsigned int a4)
{
  _BYTE *v4; // r12
  __int64 v6; // rax
  _BOOL8 v7; // r15
  unsigned int v8; // r15d
  unsigned __int8 v9; // al
  __int64 v10; // rdx
  unsigned __int64 v11; // rax
  _BYTE *v12; // r8
  unsigned __int64 v13; // r10
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  unsigned int v16; // r15d
  unsigned __int64 v17; // r8
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // r15
  unsigned __int64 v23; // [rsp+10h] [rbp-40h]
  unsigned __int64 v24; // [rsp+10h] [rbp-40h]

  v4 = a2;
  v6 = *(unsigned int *)(a3 + 8);
  v7 = (a2[1] & 0x7F) == 1;
  if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, a3 + 16, v6 + 1, 8);
    v6 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v6) = v7;
  v8 = *(_DWORD *)(a3 + 8) + 1;
  *(_DWORD *)(a3 + 8) = v8;
  v9 = *(a2 - 16);
  if ( (v9 & 2) != 0 )
    v10 = *((_QWORD *)a2 - 4);
  else
    v10 = (__int64)&a2[-8 * ((v9 >> 2) & 0xF) - 16];
  v11 = sub_A18650((__int64)(a1 + 35), *(_QWORD *)(v10 + 8));
  v12 = a2 - 16;
  v13 = HIDWORD(v11);
  v14 = v8;
  v15 = v8 + 1LL;
  if ( v15 > *(unsigned int *)(a3 + 12) )
  {
    v23 = v13;
    sub_C8D5F0(a3, a3 + 16, v15, 8);
    v14 = *(unsigned int *)(a3 + 8);
    v12 = a2 - 16;
    v13 = v23;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v14) = v13;
  v16 = *(_DWORD *)(a3 + 8) + 1;
  *(_DWORD *)(a3 + 8) = v16;
  if ( *a2 != 16 )
    a2 = *(_BYTE **)sub_A17150(v12);
  v17 = (unsigned __int64)sub_A18650((__int64)(a1 + 35), (__int64)a2) >> 32;
  v18 = v16;
  v19 = v16 + 1LL;
  if ( v19 > *(unsigned int *)(a3 + 12) )
  {
    v24 = v17;
    sub_C8D5F0(a3, a3 + 16, v19, 8);
    v18 = *(unsigned int *)(a3 + 8);
    v17 = v24;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v18) = v17;
  v20 = *(unsigned int *)(a3 + 12);
  v21 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v21;
  v22 = *((unsigned int *)v4 + 1);
  if ( v21 + 1 > v20 )
  {
    sub_C8D5F0(a3, a3 + 16, v21 + 1, 8);
    v21 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v21) = v22;
  ++*(_DWORD *)(a3 + 8);
  sub_A188E0(a3, *((unsigned __int16 *)v4 + 8));
  sub_A1BFB0(*a1, 0x16u, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}
