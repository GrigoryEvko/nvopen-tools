// Function: sub_1370980
// Address: 0x1370980
//
__int64 __fastcall sub_1370980(char *a1, unsigned int *a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int *v4; // r9
  __int64 v6; // r12
  unsigned int *v7; // r13
  unsigned int v8; // edx
  unsigned int v9; // ecx
  char *v10; // rdi
  unsigned int v11; // esi
  unsigned int v12; // eax
  unsigned int v13; // edi
  unsigned int v14; // edx
  unsigned int *v15; // rcx
  unsigned int *v16; // rbx
  unsigned int *v17; // rax
  unsigned int *v18; // r14
  __int64 v19; // rbx
  __int64 v20; // r12
  unsigned int v21; // ecx
  unsigned int *v22; // [rsp-40h] [rbp-40h]

  result = (char *)a2 - a1;
  if ( (char *)a2 - a1 <= 64 )
    return result;
  v4 = a2;
  v6 = a3;
  if ( !a3 )
  {
    v18 = a2;
    goto LABEL_23;
  }
  v7 = (unsigned int *)(a1 + 4);
  v22 = (unsigned int *)(a1 + 8);
  while ( 2 )
  {
    v8 = *((_DWORD *)a1 + 1);
    v9 = *(v4 - 1);
    --v6;
    v10 = &a1[4 * (result >> 3)];
    v11 = *(_DWORD *)a1;
    v12 = *(_DWORD *)v10;
    if ( v8 >= *(_DWORD *)v10 )
    {
      if ( v8 < v9 )
        goto LABEL_7;
      if ( v12 < v9 )
      {
LABEL_17:
        *(_DWORD *)a1 = v9;
        v13 = v11;
        *(v4 - 1) = v11;
        v11 = *((_DWORD *)a1 + 1);
        goto LABEL_8;
      }
LABEL_21:
      *(_DWORD *)a1 = v12;
      *(_DWORD *)v10 = v11;
      v13 = *(v4 - 1);
      v11 = *((_DWORD *)a1 + 1);
      goto LABEL_8;
    }
    if ( v12 < v9 )
      goto LABEL_21;
    if ( v8 < v9 )
      goto LABEL_17;
LABEL_7:
    *(_DWORD *)a1 = v8;
    *((_DWORD *)a1 + 1) = v11;
    v13 = *(v4 - 1);
LABEL_8:
    v14 = *(_DWORD *)a1;
    v15 = v22;
    v16 = v7;
    v17 = v4;
    while ( 1 )
    {
      v18 = v16;
      if ( v11 < v14 )
        goto LABEL_14;
      --v17;
      if ( v13 > v14 )
      {
        do
          --v17;
        while ( *v17 > v14 );
      }
      if ( v16 >= v17 )
        break;
      *v16 = *v17;
      v13 = *(v17 - 1);
      *v17 = v11;
      v14 = *(_DWORD *)a1;
LABEL_14:
      v11 = *v15;
      ++v16;
      ++v15;
    }
    sub_1370980(v16, v4, v6, v15);
    result = (char *)v16 - a1;
    if ( (char *)v16 - a1 > 64 )
    {
      if ( v6 )
      {
        v4 = v16;
        continue;
      }
LABEL_23:
      v19 = result >> 2;
      v20 = ((result >> 2) - 2) >> 1;
      sub_13705D0((__int64)a1, v20, result >> 2, *(_DWORD *)&a1[4 * v20]);
      do
      {
        --v20;
        sub_13705D0((__int64)a1, v20, v19, *(_DWORD *)&a1[4 * v20]);
      }
      while ( v20 );
      do
      {
        v21 = *--v18;
        *v18 = *(_DWORD *)a1;
        result = sub_13705D0((__int64)a1, 0, ((char *)v18 - a1) >> 2, v21);
      }
      while ( (char *)v18 - a1 > 4 );
    }
    return result;
  }
}
