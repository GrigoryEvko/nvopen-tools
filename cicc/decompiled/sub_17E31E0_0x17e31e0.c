// Function: sub_17E31E0
// Address: 0x17e31e0
//
char *__fastcall sub_17E31E0(__int64 *a1, __int64 *a2, __int64 *a3, __int64 *a4, __int64 *a5)
{
  __int64 *v8; // rbx
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rdi
  __int64 v13; // r15
  __int64 v14; // r14
  __int64 *v15; // rbx
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v19; // r14
  __int64 *v20; // r12
  __int64 v21; // rax
  __int64 v22; // rdi

  if ( a1 == a2 )
  {
LABEL_11:
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
          j_j___libc_free_0(v17, 40);
        ++a3;
        ++v15;
        --v14;
      }
      while ( v14 );
      return (char *)a5 + v13;
    }
  }
  else
  {
    v8 = a1;
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
          j_j___libc_free_0(v9, 40);
        ++a3;
        ++a5;
        if ( v8 == a2 )
          goto LABEL_11;
      }
      else
      {
        *v8 = 0;
        v12 = *a5;
        *a5 = v10;
        if ( v12 )
          j_j___libc_free_0(v12, 40);
        ++v8;
        ++a5;
        if ( v8 == a2 )
          goto LABEL_11;
      }
    }
    v13 = (char *)a2 - (char *)v8;
    v19 = a2 - v8;
    if ( (char *)a2 - (char *)v8 > 0 )
    {
      v20 = a5;
      do
      {
        v21 = *v8;
        *v8 = 0;
        v22 = *v20;
        *v20 = v21;
        if ( v22 )
          j_j___libc_free_0(v22, 40);
        ++v8;
        ++v20;
        --v19;
      }
      while ( v19 );
      return (char *)a5 + v13;
    }
  }
  return (char *)a5;
}
