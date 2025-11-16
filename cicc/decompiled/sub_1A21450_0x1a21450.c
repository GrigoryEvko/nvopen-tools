// Function: sub_1A21450
// Address: 0x1a21450
//
void __fastcall sub_1A21450(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5, int a6)
{
  __int64 v7; // rdx
  unsigned __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // rdx
  _QWORD *v13; // rdi
  _QWORD *v14; // rdx
  _QWORD *v15; // rdx
  char *v16; // rbx
  size_t v17; // rdx
  __int64 *v18; // r13
  __int64 *v19; // rsi
  __int64 *v20; // rbx
  __int64 v21; // rax
  unsigned __int64 v22; // rax
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  __int64 *v25; // rax

  v7 = *(unsigned int *)(a1 + 40);
  if ( (_DWORD)v7 )
  {
    v8 = *(_QWORD *)(a1 + 8);
    if ( v8 < *(_QWORD *)(a1 + 88) )
    {
      v9 = 8 * v7;
      v10 = *(_QWORD *)(a1 + 32);
      a5 = (_QWORD *)(v10 + v9);
      v11 = v9 >> 3;
      v12 = v9 >> 5;
      if ( v12 )
      {
        v13 = *(_QWORD **)(a1 + 32);
        v14 = (_QWORD *)(v10 + 32 * v12);
        while ( v8 < *(_QWORD *)(*v13 + 8LL) )
        {
          if ( v8 >= *(_QWORD *)(v13[1] + 8LL) )
          {
            ++v13;
            break;
          }
          if ( v8 >= *(_QWORD *)(v13[2] + 8LL) )
          {
            v13 += 2;
            break;
          }
          if ( v8 >= *(_QWORD *)(v13[3] + 8LL) )
          {
            v13 += 3;
            break;
          }
          v13 += 4;
          if ( v14 == v13 )
          {
            v11 = a5 - v13;
            goto LABEL_67;
          }
        }
LABEL_10:
        if ( a5 != v13 )
        {
          v15 = v13 + 1;
          if ( a5 != v13 + 1 )
          {
            while ( 1 )
            {
              if ( *(_QWORD *)(*v15 + 8LL) > v8 )
                *v13++ = *v15;
              if ( a5 == ++v15 )
                break;
              v8 = *(_QWORD *)(a1 + 8);
            }
            v10 = *(_QWORD *)(a1 + 32);
            v17 = v10 + 8LL * *(unsigned int *)(a1 + 40) - (_QWORD)a5;
            v16 = (char *)v13 + v17;
            if ( a5 != (_QWORD *)(v10 + 8LL * *(unsigned int *)(a1 + 40)) )
            {
              memmove(v13, a5, v17);
              v10 = *(_QWORD *)(a1 + 32);
            }
            goto LABEL_19;
          }
        }
LABEL_12:
        v16 = (char *)v13;
LABEL_19:
        *(_DWORD *)(a1 + 40) = (__int64)&v16[-v10] >> 3;
        goto LABEL_20;
      }
      v13 = *(_QWORD **)(a1 + 32);
LABEL_67:
      if ( v11 != 2 )
      {
        if ( v11 != 3 )
        {
          if ( v11 != 1 )
            goto LABEL_70;
          goto LABEL_75;
        }
        if ( v8 >= *(_QWORD *)(*v13 + 8LL) )
          goto LABEL_10;
        ++v13;
      }
      if ( v8 >= *(_QWORD *)(*v13 + 8LL) )
        goto LABEL_10;
      ++v13;
LABEL_75:
      if ( v8 >= *(_QWORD *)(*v13 + 8LL) )
        goto LABEL_10;
LABEL_70:
      v13 = a5;
      goto LABEL_12;
    }
    *(_DWORD *)(a1 + 40) = 0;
    *(_QWORD *)(a1 + 88) = 0;
  }
LABEL_20:
  v18 = *(__int64 **)(a1 + 16);
  v19 = *(__int64 **)(a1 + 80);
  if ( v18 == v19 )
    return;
  v20 = *(__int64 **)(a1 + 24);
  if ( v18 == v20 )
  {
    if ( *(_DWORD *)(a1 + 40) )
    {
      v23 = *(_QWORD *)(a1 + 8);
      goto LABEL_40;
    }
LABEL_39:
    v23 = *v18;
    goto LABEL_40;
  }
  do
  {
    while ( (v18[2] & 4) == 0 || *(_QWORD *)(a1 + 8) >= (unsigned __int64)v18[1] )
    {
      v18 += 3;
      if ( v20 == v18 )
        goto LABEL_31;
    }
    v21 = *(unsigned int *)(a1 + 40);
    if ( (unsigned int)v21 >= *(_DWORD *)(a1 + 44) )
    {
      sub_16CD150(a1 + 32, (const void *)(a1 + 48), 0, 8, (int)a5, a6);
      v21 = *(unsigned int *)(a1 + 40);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v21) = v18;
    v22 = *(_QWORD *)(a1 + 88);
    ++*(_DWORD *)(a1 + 40);
    if ( v18[1] >= v22 )
      v22 = v18[1];
    v18 += 3;
    *(_QWORD *)(a1 + 88) = v22;
  }
  while ( v20 != v18 );
LABEL_31:
  v18 = *(__int64 **)(a1 + 24);
  v19 = *(__int64 **)(a1 + 80);
  *(_QWORD *)(a1 + 16) = v18;
  if ( v18 == v19 )
  {
    *(_QWORD *)a1 = *(_QWORD *)(a1 + 8);
    *(_QWORD *)(a1 + 8) = *(_QWORD *)(a1 + 88);
    return;
  }
  if ( !*(_DWORD *)(a1 + 40) )
  {
    v20 = v18;
    goto LABEL_39;
  }
  v23 = *(_QWORD *)(a1 + 8);
  v20 = v18;
  if ( *v18 != v23 && (v18[2] & 4) == 0 )
  {
    *(_QWORD *)a1 = v23;
    *(_QWORD *)(a1 + 8) = *v18;
    return;
  }
LABEL_40:
  *(_QWORD *)a1 = v23;
  v24 = v18[1];
  v25 = v20 + 3;
  *(_QWORD *)(a1 + 24) = v20 + 3;
  *(_QWORD *)(a1 + 8) = v24;
  if ( (v18[2] & 4) != 0 )
  {
    if ( v25 != v19 )
    {
      while ( *v25 < v24 )
      {
        if ( (v25[2] & 4) == 0 )
        {
          *(_QWORD *)(a1 + 8) = *v25;
          return;
        }
        if ( v24 < v25[1] )
          v24 = v25[1];
        v25 += 3;
        *(_QWORD *)(a1 + 24) = v25;
        *(_QWORD *)(a1 + 8) = v24;
        if ( v25 == v19 )
          return;
      }
    }
  }
  else if ( v25 != v19 )
  {
    while ( *v25 < v24 )
    {
      if ( (v25[2] & 4) == 0 )
      {
        if ( v25[1] >= v24 )
          v24 = v25[1];
        *(_QWORD *)(a1 + 8) = v24;
      }
      v25 += 3;
      *(_QWORD *)(a1 + 24) = v25;
      if ( v25 == v19 )
        break;
      v24 = *(_QWORD *)(a1 + 8);
    }
  }
}
