// Function: sub_31A6470
// Address: 0x31a6470
//
__int64 __fastcall sub_31A6470(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  _QWORD *v5; // rax
  __int64 v6; // rdx
  _QWORD *v7; // rdi
  signed __int64 v8; // rdx
  _QWORD *v9; // rdx

  v5 = *(_QWORD **)(a1 + 112);
  v6 = 23LL * *(unsigned int *)(a1 + 120);
  v7 = &v5[v6];
  v8 = 0xD37A6F4DE9BD37A7LL * ((v6 * 8) >> 3);
  if ( v8 >> 2 )
  {
    v9 = &v5[92 * (v8 >> 2)];
    while ( a2 != v5[1] )
    {
      if ( a2 == v5[24] )
      {
        LOBYTE(a5) = v7 != v5 + 23;
        return a5;
      }
      if ( a2 == v5[47] )
      {
        LOBYTE(a5) = v7 != v5 + 46;
        return a5;
      }
      if ( a2 == v5[70] )
      {
        LOBYTE(a5) = v7 != v5 + 69;
        return a5;
      }
      v5 += 92;
      if ( v9 == v5 )
      {
        v8 = 0xD37A6F4DE9BD37A7LL * (v7 - v5);
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
    if ( a2 == v5[1] )
      goto LABEL_8;
    v5 += 23;
  }
  if ( a2 == v5[1] )
    goto LABEL_8;
  v5 += 23;
LABEL_14:
  a5 = 0;
  if ( a2 == v5[1] )
  {
LABEL_8:
    LOBYTE(a5) = v7 != v5;
    return a5;
  }
  return 0;
}
