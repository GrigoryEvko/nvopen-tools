// Function: sub_33E3600
// Address: 0x33e3600
//
__int64 __fastcall sub_33E3600(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  int v7; // ebx
  unsigned __int64 v8; // rcx
  __int64 v9; // rax
  unsigned __int64 v10; // rbx
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // rdx
  __int64 i; // r13
  unsigned __int64 v18; // r8
  __int64 v19; // r8
  unsigned __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned int v24; // [rsp+8h] [rbp-38h]
  int v25; // [rsp+8h] [rbp-38h]
  unsigned __int64 v26; // [rsp+8h] [rbp-38h]

  v6 = *(unsigned int *)(a2 + 8);
  v7 = *(_DWORD *)(a1 + 24);
  if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
  {
    sub_C8D5F0(a2, (const void *)(a2 + 16), v6 + 1, 4u, a5, a6);
    v6 = *(unsigned int *)(a2 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a2 + 4 * v6) = v7;
  v8 = *(unsigned int *)(a2 + 12);
  v9 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
  *(_DWORD *)(a2 + 8) = v9;
  v10 = *(_QWORD *)(a1 + 48);
  if ( v9 + 1 > v8 )
  {
    sub_C8D5F0(a2, (const void *)(a2 + 16), v9 + 1, 4u, a5, a6);
    v9 = *(unsigned int *)(a2 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a2 + 4 * v9) = v10;
  v11 = HIDWORD(v10);
  v12 = *(unsigned int *)(a2 + 12);
  v13 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
  *(_DWORD *)(a2 + 8) = v13;
  if ( v13 + 1 > v12 )
  {
    sub_C8D5F0(a2, (const void *)(a2 + 16), v13 + 1, 4u, a5, a6);
    v13 = *(unsigned int *)(a2 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a2 + 4 * v13) = v11;
  v14 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
  *(_DWORD *)(a2 + 8) = v14;
  v15 = *(_QWORD *)(a1 + 40);
  v16 = 5LL * *(unsigned int *)(a1 + 64);
  for ( i = v15 + 40LL * *(unsigned int *)(a1 + 64); i != v15; *(_DWORD *)(a2 + 8) = v14 )
  {
    v18 = *(_QWORD *)v15;
    if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
    {
      v26 = *(_QWORD *)v15;
      sub_C8D5F0(a2, (const void *)(a2 + 16), v14 + 1, 4u, v18, a6);
      v14 = *(unsigned int *)(a2 + 8);
      v18 = v26;
    }
    *(_DWORD *)(*(_QWORD *)a2 + 4 * v14) = v18;
    v19 = HIDWORD(v18);
    v20 = *(unsigned int *)(a2 + 12);
    v21 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
    *(_DWORD *)(a2 + 8) = v21;
    if ( v21 + 1 > v20 )
    {
      v25 = v19;
      sub_C8D5F0(a2, (const void *)(a2 + 16), v21 + 1, 4u, v19, a6);
      v21 = *(unsigned int *)(a2 + 8);
      LODWORD(v19) = v25;
    }
    *(_DWORD *)(*(_QWORD *)a2 + 4 * v21) = v19;
    v12 = *(unsigned int *)(a2 + 12);
    v22 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
    *(_DWORD *)(a2 + 8) = v22;
    a5 = *(unsigned int *)(v15 + 8);
    if ( v22 + 1 > v12 )
    {
      v24 = *(_DWORD *)(v15 + 8);
      sub_C8D5F0(a2, (const void *)(a2 + 16), v22 + 1, 4u, a5, a6);
      v22 = *(unsigned int *)(a2 + 8);
      a5 = v24;
    }
    v16 = *(_QWORD *)a2;
    v15 += 40;
    *(_DWORD *)(*(_QWORD *)a2 + 4 * v22) = a5;
    v14 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
  }
  return sub_33E2C00(a2, a1, v16, v12, a5, a6);
}
