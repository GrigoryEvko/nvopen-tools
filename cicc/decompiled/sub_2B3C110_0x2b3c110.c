// Function: sub_2B3C110
// Address: 0x2b3c110
//
void __fastcall sub_2B3C110(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  int v9; // edx
  int v10; // eax
  int v11; // edx
  int v12; // eax
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // r14
  int v15; // eax
  unsigned __int64 v16; // r15
  unsigned __int64 v17; // r13
  __int64 v18; // rax
  __int64 *v19; // rdx
  __int64 *v20; // rcx
  __int64 v21; // rsi
  __int64 v22; // rdi
  __int64 v23; // rsi
  const void *v24; // rsi
  int v25; // ecx
  const void *v26; // rsi

  if ( a1 == a2 )
    return;
  v8 = *(_QWORD *)a1;
  if ( a1 + 16 != *(_QWORD *)a1 && *(_QWORD *)a2 != a2 + 16 )
  {
    *(_QWORD *)a1 = *(_QWORD *)a2;
    v9 = *(_DWORD *)(a2 + 8);
    *(_QWORD *)a2 = v8;
    v10 = *(_DWORD *)(a1 + 8);
    *(_DWORD *)(a1 + 8) = v9;
    v11 = *(_DWORD *)(a2 + 12);
    *(_DWORD *)(a2 + 8) = v10;
    v12 = *(_DWORD *)(a1 + 12);
    *(_DWORD *)(a1 + 12) = v11;
    *(_DWORD *)(a2 + 12) = v12;
    return;
  }
  v13 = *(unsigned int *)(a2 + 8);
  if ( v13 > *(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v13, 0x10u, a5, a6);
    v14 = *(unsigned int *)(a1 + 8);
    v15 = v14;
    if ( *(_DWORD *)(a2 + 12) >= (unsigned int)v14 )
      goto LABEL_8;
    goto LABEL_22;
  }
  v14 = *(unsigned int *)(a1 + 8);
  v15 = v14;
  if ( *(_DWORD *)(a2 + 12) < (unsigned int)v14 )
  {
LABEL_22:
    sub_C8D5F0(a2, (const void *)(a2 + 16), v14, 0x10u, a5, a6);
    v14 = *(unsigned int *)(a1 + 8);
    v15 = *(_DWORD *)(a1 + 8);
  }
LABEL_8:
  v16 = *(unsigned int *)(a2 + 8);
  v17 = v14;
  if ( v16 <= v14 )
    v17 = *(unsigned int *)(a2 + 8);
  if ( v17 )
  {
    v18 = 0;
    do
    {
      v19 = (__int64 *)(v18 + *(_QWORD *)a2);
      v20 = (__int64 *)(v18 + *(_QWORD *)a1);
      v18 += 16;
      v21 = *v20;
      *v20 = *v19;
      v22 = v19[1];
      *v19 = v21;
      v23 = v20[1];
      v20[1] = v22;
      v19[1] = v23;
    }
    while ( 16 * v17 != v18 );
    v14 = *(unsigned int *)(a1 + 8);
    v16 = *(unsigned int *)(a2 + 8);
    v15 = *(_DWORD *)(a1 + 8);
  }
  if ( v16 >= v14 )
  {
    if ( v16 > v14 )
    {
      v25 = v14;
      v26 = (const void *)(*(_QWORD *)a2 + 16 * v17);
      if ( v26 != (const void *)(16 * v16 + *(_QWORD *)a2) )
      {
        memcpy((void *)(*(_QWORD *)a1 + 16 * v14), v26, 16 * v16 - 16 * v17);
        v25 = *(_DWORD *)(a1 + 8);
      }
      *(_DWORD *)(a1 + 8) = v25 + v16 - v14;
      *(_DWORD *)(a2 + 8) = v17;
    }
  }
  else
  {
    v24 = (const void *)(*(_QWORD *)a1 + 16 * v17);
    if ( v24 != (const void *)(16 * v14 + *(_QWORD *)a1) )
    {
      memcpy((void *)(*(_QWORD *)a2 + 16 * v16), v24, 16 * v14 - 16 * v17);
      v15 = v14 + *(_DWORD *)(a2 + 8) - v16;
    }
    *(_DWORD *)(a2 + 8) = v15;
    *(_DWORD *)(a1 + 8) = v17;
  }
}
