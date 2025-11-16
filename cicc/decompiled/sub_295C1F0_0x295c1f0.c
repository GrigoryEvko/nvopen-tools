// Function: sub_295C1F0
// Address: 0x295c1f0
//
char *__fastcall sub_295C1F0(_QWORD *a1, _QWORD *a2, __int64 a3, __int64 a4, char *a5, __int64 a6)
{
  char *result; // rax
  char *v10; // r15
  char *v11; // r12
  _QWORD *v12; // rbx
  __int64 v13; // rdi
  __int64 v14; // rsi
  _QWORD *v15; // rax
  _QWORD *v16; // rcx
  size_t v17; // rdx
  __int64 v18; // rcx
  char *v19; // rbx
  __int64 v20; // r12
  __int64 v21; // rdx
  __int64 v22; // rcx
  char *v23; // r12
  __int64 v24; // rax
  __int64 v25; // rcx
  __int64 i; // rdi
  char *v27; // rsi
  __int64 j; // rdx
  __int64 v29; // r8
  __int64 v30; // rdx
  char *v31; // r8
  __int64 v32; // rdx
  __int64 v33; // rsi
  __int64 v34; // r9
  __int64 *v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rbx
  __int64 v39; // rax
  __int64 v40; // r13
  char *v41; // rbx
  char *dest; // [rsp+8h] [rbp-48h]

  result = (char *)a1;
  if ( a4 == 1 )
    return result;
  if ( a4 > a6 )
  {
    v18 = a4 / 2;
    v19 = (char *)&a1[v18];
    v20 = a4 - v18;
    v10 = v19;
    dest = (char *)sub_295C1F0(a1, v19, a3, v18, a5, a6);
    if ( v20 )
    {
      while ( (unsigned __int8)sub_B19060(a3, **(_QWORD **)(*(_QWORD *)v10 + 32LL), v21, v22) )
      {
        v10 += 8;
        if ( !--v20 )
          goto LABEL_21;
      }
      v10 = (char *)sub_295C1F0(v10, a2, a3, v20, a5, a6);
    }
LABEL_21:
    v23 = dest;
    if ( dest == v19 )
      return v10;
    if ( v19 == v10 )
      return dest;
    v24 = (v10 - dest) >> 3;
    v10 = &dest[v10 - v19];
    v25 = (v19 - dest) >> 3;
    if ( v25 == v24 - v25 )
    {
      v36 = 0;
      do
      {
        v37 = *(_QWORD *)&dest[v36];
        *(_QWORD *)&dest[v36] = *(_QWORD *)&v19[v36];
        *(_QWORD *)&v19[v36] = v37;
        v36 += 8;
      }
      while ( v19 - dest != v36 );
      return v19;
    }
    while ( 1 )
    {
      for ( i = v24 - v25; v25 < v24 - v25; i = v24 - v25 )
      {
        if ( v25 == 1 )
        {
          v39 = 8 * v24;
          v40 = *(_QWORD *)v23;
          v41 = &v23[v39];
          if ( &v23[v39] != v23 + 8 )
            memmove(v23, v23 + 8, v39 - 8);
          *((_QWORD *)v41 - 1) = v40;
          return v10;
        }
        v27 = &v23[8 * v25];
        if ( i > 0 )
        {
          for ( j = 0; j != i; ++j )
          {
            v29 = *(_QWORD *)&v23[8 * j];
            *(_QWORD *)&v23[8 * j] = *(_QWORD *)&v27[8 * j];
            *(_QWORD *)&v27[8 * j] = v29;
          }
          v23 += 8 * i;
        }
        v30 = v24 % v25;
        if ( !(v24 % v25) )
          return v10;
        v24 = v25;
        v25 -= v30;
      }
      v31 = &v23[8 * v24];
      if ( i == 1 )
      {
        v38 = *((_QWORD *)v31 - 1);
        if ( v23 != v31 - 8 )
          memmove(v23 + 8, v23, 8 * v24 - 8);
        *(_QWORD *)v23 = v38;
        return v10;
      }
      v23 = &v31[-8 * i];
      if ( v25 > 0 )
      {
        v32 = -8;
        v33 = 0;
        do
        {
          v34 = *(_QWORD *)&v23[v32];
          ++v33;
          *(_QWORD *)&v23[v32] = *(_QWORD *)&v31[v32];
          *(_QWORD *)&v31[v32] = v34;
          v32 -= 8;
        }
        while ( v25 != v33 );
        v23 -= 8 * v25;
      }
      v25 = v24 % i;
      if ( !(v24 % i) )
        return v10;
      v24 = i;
    }
  }
  v10 = (char *)a1;
  *(_QWORD *)a5 = *a1;
  v11 = a5 + 8;
  v12 = a1 + 1;
  if ( a1 + 1 == a2 )
  {
    v17 = 8;
    goto LABEL_12;
  }
  do
  {
    v13 = *v12;
    v14 = **(_QWORD **)(*v12 + 32LL);
    if ( *(_BYTE *)(a3 + 28) )
    {
      v15 = *(_QWORD **)(a3 + 8);
      v16 = &v15[*(unsigned int *)(a3 + 20)];
      if ( v15 != v16 )
      {
        while ( v14 != *v15 )
        {
          if ( v16 == ++v15 )
            goto LABEL_40;
        }
LABEL_9:
        *(_QWORD *)v10 = v13;
        v10 += 8;
        goto LABEL_10;
      }
    }
    else
    {
      v35 = sub_C8CA60(a3, v14);
      v13 = *v12;
      if ( v35 )
        goto LABEL_9;
    }
LABEL_40:
    *(_QWORD *)v11 = v13;
    v11 += 8;
LABEL_10:
    ++v12;
  }
  while ( a2 != v12 );
  v17 = v11 - a5;
LABEL_12:
  if ( a5 != v11 )
    memmove(v10, a5, v17);
  return v10;
}
