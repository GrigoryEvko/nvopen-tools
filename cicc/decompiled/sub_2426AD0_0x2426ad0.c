// Function: sub_2426AD0
// Address: 0x2426ad0
//
unsigned __int64 *__fastcall sub_2426AD0(
        unsigned __int64 *a1,
        unsigned __int64 *a2,
        unsigned __int64 *a3,
        unsigned __int64 *a4,
        unsigned __int64 *a5)
{
  unsigned __int64 *i; // rbx
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rdx
  unsigned int v12; // esi
  unsigned __int64 v13; // rdi
  __int64 v14; // rdx
  unsigned __int64 *v15; // r14
  unsigned __int64 v16; // rcx
  unsigned __int64 v17; // rdi
  __int64 v18; // rcx
  __int64 v19; // r15
  __int64 v20; // r14
  unsigned __int64 *v21; // rbx
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rdi
  unsigned __int64 v25; // rdi
  __int64 v26; // [rsp+0h] [rbp-40h]
  __int64 v27; // [rsp+8h] [rbp-38h]

  for ( i = a1; a2 != i; ++a5 )
  {
    if ( a4 == a3 )
      break;
    v10 = *i;
    v11 = *a3;
    v12 = *(_DWORD *)(*i + 32);
    if ( *(_DWORD *)(*a3 + 32) == v12 )
    {
      if ( *(_DWORD *)(v11 + 36) >= *(_DWORD *)(v10 + 36) )
      {
LABEL_25:
        *i = 0;
        v25 = *a5;
        *a5 = v10;
        if ( v25 )
          j_j___libc_free_0(v25);
        ++i;
        continue;
      }
    }
    else if ( *(_DWORD *)(*a3 + 32) >= v12 )
    {
      goto LABEL_25;
    }
    *a3 = 0;
    v13 = *a5;
    *a5 = v11;
    if ( v13 )
      j_j___libc_free_0(v13);
    ++a3;
  }
  v26 = (char *)a2 - (char *)i;
  v14 = a2 - i;
  if ( (char *)a2 - (char *)i > 0 )
  {
    v15 = a5;
    do
    {
      v16 = *i;
      *i = 0;
      v17 = *v15;
      *v15 = v16;
      if ( v17 )
      {
        v27 = v14;
        j_j___libc_free_0(v17);
        v14 = v27;
      }
      ++i;
      ++v15;
      --v14;
    }
    while ( v14 );
    v18 = v26;
    if ( v26 <= 0 )
      v18 = 8;
    a5 = (unsigned __int64 *)((char *)a5 + v18);
  }
  v19 = (char *)a4 - (char *)a3;
  v20 = v19 >> 3;
  if ( v19 > 0 )
  {
    v21 = a5;
    do
    {
      v22 = *a3;
      *a3 = 0;
      v23 = *v21;
      *v21 = v22;
      if ( v23 )
        j_j___libc_free_0(v23);
      ++a3;
      ++v21;
      --v20;
    }
    while ( v20 );
    return (unsigned __int64 *)((char *)a5 + v19);
  }
  return a5;
}
