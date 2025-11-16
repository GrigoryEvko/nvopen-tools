// Function: sub_24A3760
// Address: 0x24a3760
//
unsigned __int64 *__fastcall sub_24A3760(
        unsigned __int64 *a1,
        unsigned __int64 *a2,
        unsigned __int64 *a3,
        unsigned __int64 *a4,
        unsigned __int64 *a5)
{
  unsigned __int64 *v8; // rbx
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rdi
  __int64 v13; // rdx
  unsigned __int64 *v14; // r14
  unsigned __int64 v15; // rcx
  unsigned __int64 v16; // rdi
  __int64 v17; // rcx
  __int64 v18; // r15
  __int64 v19; // r14
  unsigned __int64 *v20; // rbx
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rdi
  __int64 v24; // [rsp+0h] [rbp-40h]
  __int64 v25; // [rsp+8h] [rbp-38h]

  v8 = a1;
  if ( a2 != a1 )
  {
    while ( a4 != a3 )
    {
      v10 = *v8;
      v11 = *a3;
      if ( *(_QWORD *)(*a3 + 16) > *(_QWORD *)(*v8 + 16) )
      {
        *a3 = 0;
        v9 = *a5;
        *a5 = v11;
        if ( v9 )
          j_j___libc_free_0(v9);
        ++a3;
        ++a5;
        if ( a2 == v8 )
          break;
      }
      else
      {
        *v8 = 0;
        v12 = *a5;
        *a5 = v10;
        if ( v12 )
          j_j___libc_free_0(v12);
        ++v8;
        ++a5;
        if ( a2 == v8 )
          break;
      }
    }
  }
  v24 = (char *)a2 - (char *)v8;
  v13 = a2 - v8;
  if ( (char *)a2 - (char *)v8 > 0 )
  {
    v14 = a5;
    do
    {
      v15 = *v8;
      *v8 = 0;
      v16 = *v14;
      *v14 = v15;
      if ( v16 )
      {
        v25 = v13;
        j_j___libc_free_0(v16);
        v13 = v25;
      }
      ++v8;
      ++v14;
      --v13;
    }
    while ( v13 );
    v17 = v24;
    if ( v24 <= 0 )
      v17 = 8;
    a5 = (unsigned __int64 *)((char *)a5 + v17);
  }
  v18 = (char *)a4 - (char *)a3;
  v19 = v18 >> 3;
  if ( v18 > 0 )
  {
    v20 = a5;
    do
    {
      v21 = *a3;
      *a3 = 0;
      v22 = *v20;
      *v20 = v21;
      if ( v22 )
        j_j___libc_free_0(v22);
      ++a3;
      ++v20;
      --v19;
    }
    while ( v19 );
    return (unsigned __int64 *)((char *)a5 + v18);
  }
  return a5;
}
