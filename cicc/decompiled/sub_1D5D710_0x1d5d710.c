// Function: sub_1D5D710
// Address: 0x1d5d710
//
__int64 __fastcall sub_1D5D710(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  _DWORD *v5; // rax
  __int64 v6; // rcx
  int v7; // edx
  _DWORD *v8; // rdi
  __int64 v9; // rsi
  __int64 v10; // rcx
  _DWORD *v11; // rcx

  v5 = *(_DWORD **)(a1 + 408);
  v6 = 4LL * *(unsigned int *)(a1 + 416);
  v7 = *(_DWORD *)(a2 + 8) >> 8;
  v8 = &v5[(unsigned __int64)v6 / 4];
  v9 = v6 >> 2;
  v10 = v6 >> 4;
  if ( v10 )
  {
    v11 = &v5[4 * v10];
    while ( v7 != *v5 )
    {
      if ( v7 == v5[1] )
      {
        LOBYTE(a5) = v8 != v5 + 1;
        return a5;
      }
      if ( v7 == v5[2] )
      {
        LOBYTE(a5) = v8 != v5 + 2;
        return a5;
      }
      if ( v7 == v5[3] )
      {
        LOBYTE(a5) = v8 != v5 + 3;
        return a5;
      }
      v5 += 4;
      if ( v5 == v11 )
      {
        v9 = v8 - v5;
        goto LABEL_11;
      }
    }
    goto LABEL_8;
  }
LABEL_11:
  if ( v9 == 2 )
  {
LABEL_18:
    if ( v7 != *v5 )
    {
      ++v5;
LABEL_14:
      a5 = 0;
      if ( v7 != *v5 )
        return a5;
      goto LABEL_8;
    }
    goto LABEL_8;
  }
  if ( v9 != 3 )
  {
    a5 = 0;
    if ( v9 != 1 )
      return a5;
    goto LABEL_14;
  }
  if ( v7 != *v5 )
  {
    ++v5;
    goto LABEL_18;
  }
LABEL_8:
  LOBYTE(a5) = v8 != v5;
  return a5;
}
