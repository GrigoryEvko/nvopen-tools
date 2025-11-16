// Function: sub_EF82A0
// Address: 0xef82a0
//
__int64 __fastcall sub_EF82A0(char *a1, char *a2, __int64 a3)
{
  __int64 result; // rax
  char *v5; // r14
  __int64 v6; // rbx
  char *v7; // r10
  char *v8; // r12
  unsigned int v9; // esi
  unsigned int v10; // r8d
  unsigned int v11; // edx
  char *v12; // rcx
  unsigned int v13; // eax
  unsigned int v14; // edx
  unsigned int *v15; // rdi
  char *v16; // r13
  char *v17; // rcx
  char *v18; // rax
  __int64 v19; // r12
  __int64 i; // rbx
  unsigned int *v21; // r14
  unsigned int v22; // ecx
  __int64 v23; // rbx
  unsigned int *v24; // [rsp-40h] [rbp-40h]

  result = a2 - a1;
  if ( a2 - a1 <= 64 )
    return result;
  v5 = a2;
  v6 = a3;
  if ( !a3 )
    goto LABEL_24;
  v7 = a2;
  v8 = a1 + 4;
  v24 = (unsigned int *)(a1 + 8);
  while ( 2 )
  {
    v9 = *((_DWORD *)a1 + 1);
    v10 = *(_DWORD *)a1;
    --v6;
    v11 = *((_DWORD *)v7 - 1);
    v12 = &a1[4 * ((__int64)(((v7 - a1) >> 2) + ((unsigned __int64)(v7 - a1) >> 63)) >> 1)];
    v13 = *(_DWORD *)v12;
    if ( v9 >= *(_DWORD *)v12 )
    {
      if ( v11 > v9 )
        goto LABEL_7;
      if ( v11 > v13 )
      {
LABEL_18:
        *(_DWORD *)a1 = v11;
        v14 = v10;
        *((_DWORD *)v7 - 1) = v10;
        v9 = *(_DWORD *)a1;
        v10 = *((_DWORD *)a1 + 1);
        goto LABEL_8;
      }
LABEL_23:
      *(_DWORD *)a1 = v13;
      *(_DWORD *)v12 = v10;
      v10 = *((_DWORD *)a1 + 1);
      v9 = *(_DWORD *)a1;
      v14 = *((_DWORD *)v7 - 1);
      goto LABEL_8;
    }
    if ( v11 > v13 )
      goto LABEL_23;
    if ( v11 > v9 )
      goto LABEL_18;
LABEL_7:
    *(_DWORD *)a1 = v9;
    *((_DWORD *)a1 + 1) = v10;
    v14 = *((_DWORD *)v7 - 1);
LABEL_8:
    v15 = v24;
    v16 = v8;
    v17 = v7;
    while ( 1 )
    {
      v5 = v16;
      if ( v9 > v10 )
        goto LABEL_15;
      if ( v14 <= v9 )
      {
        v17 -= 4;
      }
      else
      {
        v18 = v17 - 8;
        do
        {
          v17 = v18;
          v14 = *(_DWORD *)v18;
          v18 -= 4;
        }
        while ( v9 < v14 );
      }
      if ( v16 >= v17 )
        break;
      *(_DWORD *)v16 = v14;
      v14 = *((_DWORD *)v17 - 1);
      *(_DWORD *)v17 = v10;
      v9 = *(_DWORD *)a1;
LABEL_15:
      v10 = *v15;
      v16 += 4;
      ++v15;
    }
    sub_EF82A0(v16, v7, v6);
    result = v16 - a1;
    if ( v16 - a1 > 64 )
    {
      if ( v6 )
      {
        v7 = v16;
        continue;
      }
LABEL_24:
      v19 = result >> 2;
      for ( i = ((result >> 2) - 2) >> 1; ; --i )
      {
        sub_EF80D0((__int64)a1, i, v19, *(_DWORD *)&a1[4 * i]);
        if ( !i )
          break;
      }
      v21 = (unsigned int *)(v5 - 4);
      do
      {
        v22 = *v21;
        v23 = (char *)v21-- - a1;
        v21[1] = *(_DWORD *)a1;
        result = sub_EF80D0((__int64)a1, 0, v23 >> 2, v22);
      }
      while ( v23 > 4 );
    }
    return result;
  }
}
