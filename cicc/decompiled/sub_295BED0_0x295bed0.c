// Function: sub_295BED0
// Address: 0x295bed0
//
__int64 *__fastcall sub_295BED0(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 *a5, __int64 a6)
{
  __int64 *result; // rax
  __int64 *v10; // r15
  __int64 *v11; // r12
  __int64 *v12; // rbx
  __int64 v13; // rsi
  __int64 *v14; // rax
  __int64 *v15; // rcx
  __int64 v16; // rdx
  size_t v17; // rdx
  __int64 v18; // rcx
  __int64 *v19; // rbx
  __int64 v20; // r12
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 *v23; // r12
  __int64 v24; // rax
  __int64 v25; // rcx
  __int64 i; // rdi
  __int64 *v27; // rsi
  __int64 j; // rdx
  __int64 v29; // r8
  __int64 v30; // rdx
  __int64 *v31; // r8
  __int64 v32; // rdx
  __int64 k; // rsi
  __int64 v34; // r9
  unsigned __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rbx
  __int64 v38; // rax
  __int64 v39; // r13
  __int64 *v40; // rbx
  __int64 *dest; // [rsp+8h] [rbp-48h]

  result = a1;
  if ( a4 == 1 )
    return result;
  if ( a4 > a6 )
  {
    v18 = a4 / 2;
    v19 = &a1[v18];
    v20 = a4 - v18;
    v10 = v19;
    dest = (__int64 *)sub_295BED0(a1, v19, a3, v18, a5, a6);
    if ( v20 )
    {
      while ( (unsigned __int8)sub_B19060(a3, *v10, v21, v22) )
      {
        ++v10;
        if ( !--v20 )
          goto LABEL_21;
      }
      v10 = (__int64 *)sub_295BED0(v10, a2, a3, v20, a5, a6);
    }
LABEL_21:
    v23 = dest;
    if ( dest == v19 )
      return v10;
    if ( v19 == v10 )
      return dest;
    v24 = v10 - dest;
    v10 = (__int64 *)((char *)dest + (char *)v10 - (char *)v19);
    v25 = v19 - dest;
    if ( v25 == v24 - v25 )
    {
      v35 = 0;
      do
      {
        v36 = dest[v35 / 8];
        dest[v35 / 8] = v19[v35 / 8];
        v19[v35 / 8] = v36;
        v35 += 8LL;
      }
      while ( v35 != (char *)v19 - (char *)dest );
      return v19;
    }
    while ( 1 )
    {
      for ( i = v24 - v25; v25 < v24 - v25; i = v24 - v25 )
      {
        if ( v25 == 1 )
        {
          v38 = v24;
          v39 = *v23;
          v40 = &v23[v38];
          if ( &v23[v38] != v23 + 1 )
            memmove(v23, v23 + 1, v38 * 8 - 8);
          *(v40 - 1) = v39;
          return v10;
        }
        v27 = &v23[v25];
        if ( i > 0 )
        {
          for ( j = 0; j != i; ++j )
          {
            v29 = v23[j];
            v23[j] = v27[j];
            v27[j] = v29;
          }
          v23 += i;
        }
        v30 = v24 % v25;
        if ( !(v24 % v25) )
          return v10;
        v24 = v25;
        v25 -= v30;
      }
      v31 = &v23[v24];
      if ( i == 1 )
      {
        v37 = *(v31 - 1);
        if ( v23 != v31 - 1 )
          memmove(v23 + 1, v23, 8 * v24 - 8);
        *v23 = v37;
        return v10;
      }
      v23 = &v31[-i];
      if ( v25 > 0 )
      {
        v32 = 0x1FFFFFFFFFFFFFFFLL;
        for ( k = 0; k != v25; ++k )
        {
          v34 = v23[v32];
          v23[v32] = v31[v32];
          v31[v32--] = v34;
        }
        v23 -= v25;
      }
      v25 = v24 % i;
      if ( !(v24 % i) )
        return v10;
      v24 = i;
    }
  }
  v10 = a1;
  *a5 = *a1;
  v11 = a5 + 1;
  v12 = a1 + 1;
  if ( a1 + 1 == a2 )
  {
    v17 = 8;
    goto LABEL_12;
  }
  do
  {
    v13 = *v12;
    if ( *(_BYTE *)(a3 + 28) )
    {
      v14 = *(__int64 **)(a3 + 8);
      v15 = &v14[*(unsigned int *)(a3 + 20)];
      if ( v14 == v15 )
        goto LABEL_40;
      while ( 1 )
      {
        v16 = *v14;
        if ( v13 == *v14 )
          break;
        if ( v15 == ++v14 )
          goto LABEL_40;
      }
LABEL_9:
      *v10++ = v16;
      goto LABEL_10;
    }
    if ( sub_C8CA60(a3, v13) )
    {
      v16 = *v12;
      goto LABEL_9;
    }
    v13 = *v12;
LABEL_40:
    *v11++ = v13;
LABEL_10:
    ++v12;
  }
  while ( a2 != v12 );
  v17 = (char *)v11 - (char *)a5;
LABEL_12:
  if ( a5 != v11 )
    memmove(v10, a5, v17);
  return v10;
}
