// Function: sub_26F7B50
// Address: 0x26f7b50
//
void __fastcall sub_26F7B50(__int64 **a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r15
  __int64 *i; // r14
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 *v11; // rax
  bool v12; // dl
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 *v15; // r14
  __int64 *j; // r13
  __int64 v17; // r12
  __int64 v18; // r15
  __int64 v19; // rax
  __int64 v20; // r8
  __int64 *v21; // rax
  bool v22; // dl
  __int64 v23; // rax
  __int64 v24; // r8
  __int64 v25; // [rsp+0h] [rbp-40h]
  __int64 v26; // [rsp+8h] [rbp-38h]
  __int64 v27; // [rsp+8h] [rbp-38h]

  v6 = (__int64 *)a2[5];
  for ( i = (__int64 *)a2[4]; v6 != i; *(_BYTE *)a1[2] |= v12 )
  {
    v8 = *i;
    a6 = *(unsigned int *)a1[1];
    v9 = *(unsigned int *)(*i + 72);
    v10 = **a1;
    if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(*i + 76) )
    {
      v25 = **a1;
      v27 = *(unsigned int *)a1[1];
      sub_C8D5F0(v8 + 64, (const void *)(v8 + 80), v9 + 1, 0x10u, v10, a6);
      v9 = *(unsigned int *)(v8 + 72);
      v10 = v25;
      a6 = v27;
    }
    v11 = (__int64 *)(*(_QWORD *)(v8 + 64) + 16 * v9);
    v12 = 1;
    *v11 = v10;
    v11[1] = a6;
    ++*(_DWORD *)(v8 + 72);
    v13 = *a1[3];
    v14 = *(_QWORD *)(v13 + 32);
    if ( *(_QWORD *)(v8 + 32) == v14 )
    {
      v12 = 0;
      if ( v14 )
        v12 = memcmp(*(const void **)(v13 + 24), *(const void **)(v8 + 24), *(_QWORD *)(v13 + 32)) != 0;
    }
    ++i;
  }
  v15 = (__int64 *)a2[8];
  for ( j = (__int64 *)a2[7]; v15 != j; *(_BYTE *)a1[2] |= v22 )
  {
    v17 = *j;
    v18 = *(unsigned int *)a1[1];
    v19 = *(unsigned int *)(*j + 72);
    v20 = **a1;
    if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(*j + 76) )
    {
      v26 = **a1;
      sub_C8D5F0(v17 + 64, (const void *)(v17 + 80), v19 + 1, 0x10u, v20, a6);
      v19 = *(unsigned int *)(v17 + 72);
      v20 = v26;
    }
    v21 = (__int64 *)(*(_QWORD *)(v17 + 64) + 16 * v19);
    v22 = 1;
    *v21 = v20;
    v21[1] = v18;
    ++*(_DWORD *)(v17 + 72);
    v23 = *a1[3];
    v24 = *(_QWORD *)(v23 + 32);
    if ( *(_QWORD *)(v17 + 32) == v24 )
    {
      v22 = 0;
      if ( v24 )
        v22 = memcmp(*(const void **)(v23 + 24), *(const void **)(v17 + 24), *(_QWORD *)(v23 + 32)) != 0;
    }
    ++j;
  }
}
