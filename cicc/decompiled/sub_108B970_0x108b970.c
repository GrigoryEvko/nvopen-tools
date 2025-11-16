// Function: sub_108B970
// Address: 0x108b970
//
unsigned __int64 __fastcall sub_108B970(__int64 *a1, __int64 *a2)
{
  __int64 v3; // rdx
  unsigned __int64 result; // rax
  __int64 *v5; // r13
  __int64 v6; // rbx
  __int64 v7; // r12
  __int64 v8; // rdi
  __int64 v9; // rdi
  __int64 v10; // rbx
  __int64 j; // r12
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // r12
  __int64 v15; // rbx
  __int64 v16; // rdi
  __int64 v17; // rdi
  __int64 i; // r12
  __int64 v19; // rdi
  __int64 v20; // rdi

  v3 = a1[3];
  result = a2[3];
  v5 = (__int64 *)(v3 + 8);
  if ( v3 + 8 < result )
  {
    do
    {
      v6 = *v5;
      v7 = *v5 + 480;
      do
      {
        v8 = *(_QWORD *)(v6 + 64);
        if ( v8 != v6 + 80 )
          _libc_free(v8, a2);
        v9 = *(_QWORD *)(v6 + 32);
        if ( v9 != v6 + 48 )
          _libc_free(v9, a2);
        v6 += 96;
      }
      while ( v7 != v6 );
      result = a2[3];
      ++v5;
    }
    while ( result > (unsigned __int64)v5 );
    v3 = a1[3];
  }
  v10 = *a1;
  if ( result == v3 )
  {
    for ( i = *a2; i != v10; v10 += 96 )
    {
      v19 = *(_QWORD *)(v10 + 64);
      if ( v19 != v10 + 80 )
        _libc_free(v19, a2);
      v20 = *(_QWORD *)(v10 + 32);
      result = v10 + 48;
      if ( v20 != v10 + 48 )
        result = _libc_free(v20, a2);
    }
  }
  else
  {
    for ( j = a1[2]; j != v10; v10 += 96 )
    {
      v12 = *(_QWORD *)(v10 + 64);
      if ( v12 != v10 + 80 )
        _libc_free(v12, a2);
      v13 = *(_QWORD *)(v10 + 32);
      result = v10 + 48;
      if ( v13 != v10 + 48 )
        result = _libc_free(v13, a2);
    }
    v14 = *a2;
    v15 = a2[1];
    if ( *a2 != v15 )
    {
      do
      {
        v16 = *(_QWORD *)(v15 + 64);
        if ( v16 != v15 + 80 )
          _libc_free(v16, a2);
        v17 = *(_QWORD *)(v15 + 32);
        result = v15 + 48;
        if ( v17 != v15 + 48 )
          result = _libc_free(v17, a2);
        v15 += 96;
      }
      while ( v14 != v15 );
    }
  }
  return result;
}
