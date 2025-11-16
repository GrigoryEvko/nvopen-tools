// Function: sub_3120550
// Address: 0x3120550
//
__int64 __fastcall sub_3120550(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  _DWORD *v9; // rax
  _DWORD *v10; // r13
  _DWORD *v11; // rbx
  unsigned __int64 v12; // rcx
  unsigned int v13; // edx
  __int64 v14; // rax
  _QWORD *v15; // rax
  int *v16; // r14
  __int64 v17; // rbx
  int *v18; // r13
  unsigned __int64 v19; // rax
  int *v20; // rbx
  __int64 v21; // rdi
  unsigned __int64 v22; // rdx
  __int64 v23; // r15
  __int64 v24; // r8
  __int64 *v25; // rax
  __int64 v26; // [rsp+8h] [rbp-38h]

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x300000000LL;
  v7 = *(_QWORD *)(a2 + 24);
  if ( !*(_DWORD *)(v7 + 16) )
    return a1;
  v9 = *(_DWORD **)(v7 + 8);
  v10 = &v9[4 * *(unsigned int *)(v7 + 24)];
  if ( v9 == v10 )
    return a1;
  while ( 1 )
  {
    v11 = v9;
    if ( *v9 != -1 )
      break;
    if ( v9[1] != -1 )
      goto LABEL_6;
LABEL_31:
    v9 += 4;
    if ( v10 == v9 )
      return a1;
  }
  if ( *v9 == -2 && v9[1] == -2 )
    goto LABEL_31;
LABEL_6:
  if ( v10 == v9 )
    return a1;
  v12 = 3;
  v13 = 0;
LABEL_11:
  v14 = v13;
  if ( v13 >= v12 )
  {
    v22 = v13 + 1LL;
    v23 = *((_QWORD *)v11 + 1);
    v24 = *(_QWORD *)v11;
    if ( v12 < v14 + 1 )
    {
      v26 = *(_QWORD *)v11;
      sub_C8D5F0(a1, (const void *)(a1 + 16), v22, 0x10u, v24, a6);
      v14 = *(unsigned int *)(a1 + 8);
      v24 = v26;
    }
    v25 = (__int64 *)(*(_QWORD *)a1 + 16 * v14);
    *v25 = v24;
    v25[1] = v23;
    v13 = *(_DWORD *)(a1 + 8) + 1;
    *(_DWORD *)(a1 + 8) = v13;
  }
  else
  {
    v15 = (_QWORD *)(*(_QWORD *)a1 + 16LL * v13);
    if ( v15 )
    {
      *v15 = *(_QWORD *)v11;
      v15[1] = *((_QWORD *)v11 + 1);
      v13 = *(_DWORD *)(a1 + 8);
    }
    *(_DWORD *)(a1 + 8) = ++v13;
  }
  for ( v11 += 4; v10 != v11; v11 += 4 )
  {
    if ( *v11 == -1 )
    {
      if ( v11[1] != -1 )
        goto LABEL_9;
    }
    else if ( *v11 != -2 || v11[1] != -2 )
    {
LABEL_9:
      if ( v10 == v11 )
        break;
      v12 = *(unsigned int *)(a1 + 12);
      goto LABEL_11;
    }
  }
  v16 = *(int **)a1;
  v17 = 16LL * v13;
  v18 = (int *)(*(_QWORD *)a1 + v17);
  if ( v18 != *(int **)a1 )
  {
    _BitScanReverse64(&v19, v17 >> 4);
    sub_31201D0(*(_QWORD *)a1, *(_QWORD *)a1 + v17, 2LL * (int)(63 - (v19 ^ 0x3F)));
    if ( (unsigned __int64)v17 <= 0x100 )
    {
      sub_311D250(v16, v18);
    }
    else
    {
      v20 = v16 + 64;
      sub_311D250(v16, v16 + 64);
      if ( v18 != v16 + 64 )
      {
        do
        {
          v21 = (__int64)v20;
          v20 += 4;
          sub_311D1F0(v21);
        }
        while ( v18 != v20 );
      }
    }
  }
  return a1;
}
