// Function: sub_2C4D720
// Address: 0x2c4d720
//
__int64 __fastcall sub_2C4D720(char *a1, unsigned int *a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int *v4; // r9
  __int64 v6; // rbx
  unsigned int *v7; // r12
  unsigned int v8; // ecx
  unsigned int v9; // edx
  char *v10; // rsi
  unsigned int v11; // edi
  unsigned int v12; // eax
  unsigned int v13; // edx
  unsigned int *v14; // rsi
  unsigned int *v15; // r13
  unsigned int *v16; // rax
  unsigned int *v17; // r14
  __int64 v18; // rbx
  __int64 v19; // r12
  unsigned int v20; // ecx
  unsigned int *v21; // [rsp-40h] [rbp-40h]

  result = (char *)a2 - a1;
  if ( (char *)a2 - a1 <= 64 )
    return result;
  v4 = a2;
  v6 = a3;
  if ( !a3 )
  {
    v17 = a2;
    goto LABEL_23;
  }
  v7 = (unsigned int *)(a1 + 4);
  v21 = (unsigned int *)(a1 + 8);
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
        v8 = *(_DWORD *)a1;
        v11 = *((_DWORD *)a1 + 1);
        goto LABEL_8;
      }
LABEL_21:
      *(_DWORD *)a1 = v12;
      *(_DWORD *)v10 = v11;
      v13 = *(v4 - 1);
      v8 = *(_DWORD *)a1;
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
    v14 = v21;
    v15 = v7;
    v16 = v4;
    while ( 1 )
    {
      v17 = v15;
      if ( v11 < v8 )
        goto LABEL_14;
      for ( --v16; v8 < v13; --v16 )
        v13 = *(v16 - 1);
      if ( v15 >= v16 )
        break;
      *v15 = v13;
      v13 = *(v16 - 1);
      *v16 = v11;
      v8 = *(_DWORD *)a1;
LABEL_14:
      v11 = *v14;
      ++v15;
      ++v14;
    }
    sub_2C4D720(v15, v4, v6);
    result = (char *)v15 - a1;
    if ( (char *)v15 - a1 > 64 )
    {
      if ( v6 )
      {
        v4 = v15;
        continue;
      }
LABEL_23:
      v18 = result >> 2;
      v19 = ((result >> 2) - 2) >> 1;
      sub_2C4CCC0((__int64)a1, v19, result >> 2, *(_DWORD *)&a1[4 * v19]);
      do
      {
        --v19;
        sub_2C4CCC0((__int64)a1, v19, v18, *(_DWORD *)&a1[4 * v19]);
      }
      while ( v19 );
      do
      {
        v20 = *--v17;
        *v17 = *(_DWORD *)a1;
        result = sub_2C4CCC0((__int64)a1, 0, ((char *)v17 - a1) >> 2, v20);
      }
      while ( (char *)v17 - a1 > 4 );
    }
    return result;
  }
}
