// Function: sub_E38CF0
// Address: 0xe38cf0
//
void __fastcall sub_E38CF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v9; // rdx
  char v10; // r13
  __int64 v11; // r8
  int v12; // ecx
  __int64 v13; // rsi
  int v14; // ecx
  unsigned int v15; // edx
  __int64 *v16; // rax
  __int64 v17; // rdi
  unsigned int *v18; // rcx
  unsigned int v19; // edx
  int v20; // eax
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rbx
  __int64 v24; // rax
  __int64 v25; // [rsp+0h] [rbp-40h]
  __int64 v26; // [rsp+8h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 16);
  if ( !v6 )
    return;
  while ( 1 )
  {
    v9 = *(_QWORD *)(v6 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v9 - 30) <= 0xAu )
      break;
    v6 = *(_QWORD *)(v6 + 8);
    if ( !v6 )
      return;
  }
  v10 = 0;
LABEL_5:
  v11 = *(_QWORD *)(v9 + 40);
  v12 = *(_DWORD *)(*(_QWORD *)a1 + 32LL);
  v13 = *(_QWORD *)(*(_QWORD *)a1 + 16LL);
  if ( !v12 )
    goto LABEL_18;
  v14 = v12 - 1;
  v15 = v14 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
  v16 = (__int64 *)(v13 + 16LL * v15);
  v17 = *v16;
  if ( v11 != *v16 )
  {
    v20 = 1;
    while ( v17 != -4096 )
    {
      a6 = (unsigned int)(v20 + 1);
      v15 = v14 & (v20 + v15);
      v16 = (__int64 *)(v13 + 16LL * v15);
      v17 = *v16;
      if ( v11 == *v16 )
        goto LABEL_7;
      v20 = a6;
    }
LABEL_18:
    if ( **(_DWORD **)(a1 + 8) )
      goto LABEL_11;
    goto LABEL_19;
  }
LABEL_7:
  v18 = *(unsigned int **)(a1 + 8);
  v19 = *((_DWORD *)v16 + 2);
  if ( *v18 > v19 || v18[1] < *((_DWORD *)v16 + 3) )
  {
    if ( v19 )
      v10 = 1;
    goto LABEL_11;
  }
LABEL_19:
  v21 = *(_QWORD *)(a1 + 16);
  v22 = *(unsigned int *)(v21 + 8);
  a6 = v22 + 1;
  if ( v22 + 1 > (unsigned __int64)*(unsigned int *)(v21 + 12) )
  {
    v25 = v11;
    v26 = *(_QWORD *)(a1 + 16);
    sub_C8D5F0(v26, (const void *)(v21 + 16), v22 + 1, 8u, v11, a6);
    v21 = v26;
    v11 = v25;
    v22 = *(unsigned int *)(v26 + 8);
  }
  *(_QWORD *)(*(_QWORD *)v21 + 8 * v22) = v11;
  ++*(_DWORD *)(v21 + 8);
LABEL_11:
  while ( 1 )
  {
    v6 = *(_QWORD *)(v6 + 8);
    if ( !v6 )
      break;
    v9 = *(_QWORD *)(v6 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v9 - 30) <= 0xAu )
      goto LABEL_5;
  }
  if ( v10 )
  {
    v23 = **(_QWORD **)(a1 + 24);
    v24 = *(unsigned int *)(v23 + 16);
    if ( v24 + 1 > (unsigned __int64)*(unsigned int *)(v23 + 20) )
    {
      sub_C8D5F0(v23 + 8, (const void *)(v23 + 24), v24 + 1, 8u, v11, a6);
      v24 = *(unsigned int *)(v23 + 16);
    }
    *(_QWORD *)(*(_QWORD *)(v23 + 8) + 8 * v24) = a2;
    *(_DWORD *)(v23 + 184) = 0;
    ++*(_DWORD *)(v23 + 16);
  }
}
