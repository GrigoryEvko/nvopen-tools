// Function: sub_2B3BB10
// Address: 0x2b3bb10
//
void __fastcall sub_2B3BB10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  int v8; // edx
  int v9; // eax
  int v10; // edx
  int v11; // eax
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // r13
  int v14; // eax
  unsigned __int64 v15; // r15
  unsigned __int64 v16; // r14
  __int64 v17; // rax
  __int64 *v18; // rdx
  __int64 *v19; // rcx
  __int64 v20; // rsi
  const void *v21; // rsi
  int v22; // ecx
  const void *v23; // rsi

  if ( a1 == a2 )
    return;
  v7 = *(_QWORD *)a1;
  if ( a1 + 16 != *(_QWORD *)a1 && *(_QWORD *)a2 != a2 + 16 )
  {
    *(_QWORD *)a1 = *(_QWORD *)a2;
    v8 = *(_DWORD *)(a2 + 8);
    *(_QWORD *)a2 = v7;
    v9 = *(_DWORD *)(a1 + 8);
    *(_DWORD *)(a1 + 8) = v8;
    v10 = *(_DWORD *)(a2 + 12);
    *(_DWORD *)(a2 + 8) = v9;
    v11 = *(_DWORD *)(a1 + 12);
    *(_DWORD *)(a1 + 12) = v10;
    *(_DWORD *)(a2 + 12) = v11;
    return;
  }
  v12 = *(unsigned int *)(a2 + 8);
  if ( v12 > *(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v12, 8u, a5, a6);
    v13 = *(unsigned int *)(a1 + 8);
    v14 = v13;
    if ( *(_DWORD *)(a2 + 12) >= (unsigned int)v13 )
      goto LABEL_8;
    goto LABEL_22;
  }
  v13 = *(unsigned int *)(a1 + 8);
  v14 = v13;
  if ( *(_DWORD *)(a2 + 12) < (unsigned int)v13 )
  {
LABEL_22:
    sub_C8D5F0(a2, (const void *)(a2 + 16), v13, 8u, a5, a6);
    v13 = *(unsigned int *)(a1 + 8);
    v14 = *(_DWORD *)(a1 + 8);
  }
LABEL_8:
  v15 = *(unsigned int *)(a2 + 8);
  v16 = v13;
  if ( v15 <= v13 )
    v16 = *(unsigned int *)(a2 + 8);
  if ( v16 )
  {
    v17 = 0;
    do
    {
      v18 = (__int64 *)(v17 + *(_QWORD *)a2);
      v19 = (__int64 *)(v17 + *(_QWORD *)a1);
      v17 += 8;
      v20 = *v19;
      *v19 = *v18;
      *v18 = v20;
    }
    while ( 8 * v16 != v17 );
    v13 = *(unsigned int *)(a1 + 8);
    v15 = *(unsigned int *)(a2 + 8);
    v14 = *(_DWORD *)(a1 + 8);
  }
  if ( v15 >= v13 )
  {
    if ( v15 > v13 )
    {
      v22 = v13;
      v23 = (const void *)(*(_QWORD *)a2 + 8 * v16);
      if ( v23 != (const void *)(8 * v15 + *(_QWORD *)a2) )
      {
        memcpy((void *)(*(_QWORD *)a1 + 8 * v13), v23, 8 * v15 - 8 * v16);
        v22 = *(_DWORD *)(a1 + 8);
      }
      *(_DWORD *)(a1 + 8) = v22 + v15 - v13;
      *(_DWORD *)(a2 + 8) = v16;
    }
  }
  else
  {
    v21 = (const void *)(*(_QWORD *)a1 + 8 * v16);
    if ( v21 != (const void *)(8 * v13 + *(_QWORD *)a1) )
    {
      memcpy((void *)(*(_QWORD *)a2 + 8 * v15), v21, 8 * v13 - 8 * v16);
      v14 = v13 + *(_DWORD *)(a2 + 8) - v15;
    }
    *(_DWORD *)(a2 + 8) = v14;
    *(_DWORD *)(a1 + 8) = v16;
  }
}
