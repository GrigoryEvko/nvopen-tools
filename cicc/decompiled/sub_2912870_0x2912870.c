// Function: sub_2912870
// Address: 0x2912870
//
void __fastcall sub_2912870(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5, __int64 a6)
{
  __int64 v7; // rdx
  unsigned __int64 v8; // rax
  __int64 *v9; // r13
  __int64 *v10; // rsi
  __int64 *v11; // rbx
  __int64 v12; // rax
  unsigned __int64 v13; // rax
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rsi
  __int64 v19; // rcx
  __int64 v20; // rdx
  _QWORD *v21; // rdi
  _QWORD *v22; // rdx
  _QWORD *v23; // rdx
  size_t v24; // rdx
  char *v25; // rbx

  v7 = *(unsigned int *)(a1 + 40);
  if ( (_DWORD)v7 )
  {
    v8 = *(_QWORD *)(a1 + 8);
    if ( v8 >= *(_QWORD *)(a1 + 88) )
    {
      *(_DWORD *)(a1 + 40) = 0;
      *(_QWORD *)(a1 + 88) = 0;
      goto LABEL_4;
    }
    v17 = 8 * v7;
    v18 = *(_QWORD *)(a1 + 32);
    a5 = (_QWORD *)(v18 + v17);
    v19 = v17 >> 3;
    v20 = v17 >> 5;
    if ( v20 )
    {
      v21 = *(_QWORD **)(a1 + 32);
      v22 = (_QWORD *)(v18 + 32 * v20);
      while ( v8 < *(_QWORD *)(*v21 + 8LL) )
      {
        if ( v8 >= *(_QWORD *)(v21[1] + 8LL) )
        {
          ++v21;
          break;
        }
        if ( v8 >= *(_QWORD *)(v21[2] + 8LL) )
        {
          v21 += 2;
          break;
        }
        if ( v8 >= *(_QWORD *)(v21[3] + 8LL) )
        {
          v21 += 3;
          break;
        }
        v21 += 4;
        if ( v22 == v21 )
        {
          v19 = a5 - v21;
          goto LABEL_64;
        }
      }
LABEL_51:
      if ( a5 != v21 )
      {
        v23 = v21 + 1;
        if ( a5 != v21 + 1 )
        {
          while ( 1 )
          {
            if ( *(_QWORD *)(*v23 + 8LL) > v8 )
              *v21++ = *v23;
            if ( a5 == ++v23 )
              break;
            v8 = *(_QWORD *)(a1 + 8);
          }
          v18 = *(_QWORD *)(a1 + 32);
          v24 = v18 + 8LL * *(unsigned int *)(a1 + 40) - (_QWORD)a5;
          v25 = (char *)v21 + v24;
          if ( a5 != (_QWORD *)(v18 + 8LL * *(unsigned int *)(a1 + 40)) )
          {
            memmove(v21, a5, v24);
            v18 = *(_QWORD *)(a1 + 32);
          }
          goto LABEL_60;
        }
      }
LABEL_68:
      v25 = (char *)v21;
LABEL_60:
      *(_DWORD *)(a1 + 40) = (__int64)&v25[-v18] >> 3;
      goto LABEL_4;
    }
    v21 = *(_QWORD **)(a1 + 32);
LABEL_64:
    if ( v19 != 2 )
    {
      if ( v19 != 3 )
      {
        if ( v19 != 1 )
          goto LABEL_67;
        goto LABEL_76;
      }
      if ( v8 >= *(_QWORD *)(*v21 + 8LL) )
        goto LABEL_51;
      ++v21;
    }
    if ( v8 >= *(_QWORD *)(*v21 + 8LL) )
      goto LABEL_51;
    ++v21;
LABEL_76:
    if ( v8 >= *(_QWORD *)(*v21 + 8LL) )
      goto LABEL_51;
LABEL_67:
    v21 = a5;
    goto LABEL_68;
  }
LABEL_4:
  v9 = *(__int64 **)(a1 + 16);
  v10 = *(__int64 **)(a1 + 80);
  if ( v9 == v10 )
    return;
  v11 = *(__int64 **)(a1 + 24);
  if ( v9 == v11 )
  {
    if ( *(_DWORD *)(a1 + 40) )
    {
      v14 = *(_QWORD *)(a1 + 8);
      goto LABEL_23;
    }
LABEL_22:
    v14 = *v9;
    goto LABEL_23;
  }
  do
  {
    while ( (v9[2] & 4) == 0 || *(_QWORD *)(a1 + 8) >= (unsigned __int64)v9[1] )
    {
      v9 += 3;
      if ( v11 == v9 )
        goto LABEL_15;
    }
    v12 = *(unsigned int *)(a1 + 40);
    if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
    {
      sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v12 + 1, 8u, (__int64)a5, a6);
      v12 = *(unsigned int *)(a1 + 40);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v12) = v9;
    v13 = *(_QWORD *)(a1 + 88);
    ++*(_DWORD *)(a1 + 40);
    if ( v9[1] >= v13 )
      v13 = v9[1];
    v9 += 3;
    *(_QWORD *)(a1 + 88) = v13;
  }
  while ( v11 != v9 );
LABEL_15:
  v9 = *(__int64 **)(a1 + 24);
  v10 = *(__int64 **)(a1 + 80);
  *(_QWORD *)(a1 + 16) = v9;
  if ( v9 == v10 )
  {
    *(_QWORD *)a1 = *(_QWORD *)(a1 + 8);
    *(_QWORD *)(a1 + 8) = *(_QWORD *)(a1 + 88);
    return;
  }
  if ( !*(_DWORD *)(a1 + 40) )
  {
    v11 = v9;
    goto LABEL_22;
  }
  v14 = *(_QWORD *)(a1 + 8);
  v11 = v9;
  if ( v14 != *v9 && (v9[2] & 4) == 0 )
  {
    *(_QWORD *)a1 = v14;
    *(_QWORD *)(a1 + 8) = *v9;
    return;
  }
LABEL_23:
  *(_QWORD *)a1 = v14;
  v15 = v9[1];
  v16 = v11 + 3;
  *(_QWORD *)(a1 + 24) = v11 + 3;
  *(_QWORD *)(a1 + 8) = v15;
  if ( (v9[2] & 4) != 0 )
  {
    if ( v10 != v16 )
    {
      while ( *v16 < v15 )
      {
        if ( (v16[2] & 4) == 0 )
        {
          *(_QWORD *)(a1 + 8) = *v16;
          return;
        }
        if ( v15 < v16[1] )
          v15 = v16[1];
        v16 += 3;
        *(_QWORD *)(a1 + 24) = v16;
        *(_QWORD *)(a1 + 8) = v15;
        if ( v10 == v16 )
          return;
      }
    }
  }
  else if ( v10 != v16 )
  {
    while ( v15 > *v16 )
    {
      if ( (v16[2] & 4) == 0 )
      {
        if ( v16[1] >= v15 )
          v15 = v16[1];
        *(_QWORD *)(a1 + 8) = v15;
      }
      v16 += 3;
      *(_QWORD *)(a1 + 24) = v16;
      if ( v10 == v16 )
        break;
      v15 = *(_QWORD *)(a1 + 8);
    }
  }
}
