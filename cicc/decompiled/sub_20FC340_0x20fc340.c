// Function: sub_20FC340
// Address: 0x20fc340
//
__int64 __fastcall sub_20FC340(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  _QWORD *v5; // rax
  __int64 v6; // rdx
  _QWORD *v7; // rdi
  __int64 v8; // rcx
  __int64 v9; // rdx
  _QWORD *v10; // rdx

  v5 = *(_QWORD **)(a1 + 112);
  v6 = 8LL * *(unsigned int *)(a1 + 120);
  v7 = &v5[(unsigned __int64)v6 / 8];
  v8 = v6 >> 3;
  v9 = v6 >> 5;
  if ( v9 )
  {
    v10 = &v5[4 * v9];
    while ( a2 != *v5 )
    {
      if ( a2 == v5[1] )
      {
        LOBYTE(a5) = v7 != v5 + 1;
        return a5;
      }
      if ( a2 == v5[2] )
      {
        LOBYTE(a5) = v7 != v5 + 2;
        return a5;
      }
      if ( a2 == v5[3] )
      {
        LOBYTE(a5) = v7 != v5 + 3;
        return a5;
      }
      v5 += 4;
      if ( v5 == v10 )
      {
        v8 = v7 - v5;
        goto LABEL_11;
      }
    }
    goto LABEL_8;
  }
LABEL_11:
  if ( v8 != 2 )
  {
    if ( v8 != 3 )
    {
      a5 = 0;
      if ( v8 != 1 )
        return a5;
      goto LABEL_14;
    }
    if ( a2 == *v5 )
      goto LABEL_8;
    ++v5;
  }
  if ( a2 == *v5 )
    goto LABEL_8;
  ++v5;
LABEL_14:
  a5 = 0;
  if ( a2 == *v5 )
  {
LABEL_8:
    LOBYTE(a5) = v7 != v5;
    return a5;
  }
  return 0;
}
