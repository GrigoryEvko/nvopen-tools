// Function: sub_3700ED0
// Address: 0x3700ed0
//
__int64 __fastcall sub_3700ED0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // esi
  _DWORD *v8; // rax
  char v9; // r9
  _DWORD *v10; // rax
  _DWORD *i; // rdi
  unsigned int v12; // ecx
  unsigned int v13; // edx
  unsigned int v15; // edx

  v5 = *(_QWORD *)(a1 + 40);
  v6 = *(_QWORD *)(a1 + 48);
  if ( *(_QWORD *)(a1 + 56) )
  {
    v7 = 0;
    if ( !v5 )
    {
      a5 = 0;
      if ( !v6 )
        return a5;
    }
  }
  else if ( v6 )
  {
    v7 = 0;
    if ( !v5 )
      v7 = *(_DWORD *)(v6 + 56);
  }
  else
  {
    v7 = 0;
    if ( v5 )
      v7 = *(_DWORD *)(v5 + 56);
  }
  v8 = *(_DWORD **)a1;
  v9 = 0;
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) )
  {
    v15 = v8[1];
    v9 = 1;
    a5 = v15 + *v8 - v7;
    if ( v7 - *v8 >= v15 )
      a5 = 0;
  }
  v10 = v8 + 3;
  for ( i = &v10[3 * *(unsigned int *)(a1 + 8) - 3]; i != v10; v10 += 3 )
  {
    if ( *((_BYTE *)v10 + 8) )
    {
      v12 = v10[1];
      v13 = v12 + *v10 - v7;
      if ( v7 - *v10 >= v12 )
        v13 = 0;
      if ( v9 )
      {
        if ( a5 > v13 )
          a5 = v13;
      }
      else
      {
        a5 = v13;
      }
      v9 = 1;
    }
  }
  return a5;
}
