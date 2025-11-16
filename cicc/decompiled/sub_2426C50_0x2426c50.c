// Function: sub_2426C50
// Address: 0x2426c50
//
unsigned __int64 *__fastcall sub_2426C50(
        unsigned __int64 *a1,
        unsigned __int64 *a2,
        unsigned __int64 *a3,
        unsigned __int64 *a4,
        unsigned __int64 *a5)
{
  unsigned __int64 *v8; // rbx
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // rcx
  unsigned int v11; // eax
  unsigned __int64 v12; // rdi
  __int64 v13; // r15
  __int64 v14; // r14
  unsigned __int64 *v15; // rbx
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rdi
  unsigned __int64 v19; // rdi
  __int64 v20; // r14
  unsigned __int64 *v21; // r12
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rdi

  if ( a1 == a2 )
  {
LABEL_10:
    v13 = (char *)a4 - (char *)a3;
    v14 = v13 >> 3;
    if ( v13 > 0 )
    {
      v15 = a5;
      do
      {
        v16 = *a3;
        *a3 = 0;
        v17 = *v15;
        *v15 = v16;
        if ( v17 )
          j_j___libc_free_0(v17);
        ++a3;
        ++v15;
        --v14;
      }
      while ( v14 );
      return (unsigned __int64 *)((char *)a5 + v13);
    }
    return a5;
  }
  v8 = a1;
  while ( a4 != a3 )
  {
    v9 = *v8;
    v10 = *a3;
    v11 = *(_DWORD *)(*v8 + 32);
    if ( *(_DWORD *)(*a3 + 32) == v11 )
    {
      if ( *(_DWORD *)(v10 + 36) >= *(_DWORD *)(v9 + 36) )
      {
LABEL_17:
        *v8 = 0;
        v19 = *a5;
        *a5 = v9;
        if ( v19 )
          j_j___libc_free_0(v19);
        ++v8;
        goto LABEL_9;
      }
    }
    else if ( *(_DWORD *)(*a3 + 32) >= v11 )
    {
      goto LABEL_17;
    }
    *a3 = 0;
    v12 = *a5;
    *a5 = v10;
    if ( v12 )
      j_j___libc_free_0(v12);
    ++a3;
LABEL_9:
    ++a5;
    if ( v8 == a2 )
      goto LABEL_10;
  }
  v13 = (char *)a2 - (char *)v8;
  v20 = a2 - v8;
  if ( (char *)a2 - (char *)v8 <= 0 )
    return a5;
  v21 = a5;
  do
  {
    v22 = *v8;
    *v8 = 0;
    v23 = *v21;
    *v21 = v22;
    if ( v23 )
      j_j___libc_free_0(v23);
    ++v8;
    ++v21;
    --v20;
  }
  while ( v20 );
  return (unsigned __int64 *)((char *)a5 + v13);
}
