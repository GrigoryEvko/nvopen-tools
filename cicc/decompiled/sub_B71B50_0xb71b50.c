// Function: sub_B71B50
// Address: 0xb71b50
//
unsigned __int64 __fastcall sub_B71B50(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // r14
  unsigned __int64 result; // rax
  __int64 i; // rdx
  int v7; // esi
  unsigned __int64 *v8; // rdi
  __int64 v9; // rdx
  unsigned __int64 v10; // rcx
  unsigned __int64 *v11; // rdi
  __int64 v12; // rsi
  _QWORD *v13; // rax
  unsigned __int64 *v14; // rdx
  __int64 v15; // rdx

  v3 = *(unsigned int *)(a1 + 3428);
  result = *(unsigned int *)(a2 + 8);
  if ( v3 != result )
  {
    if ( v3 >= result )
    {
      if ( v3 > *(unsigned int *)(a2 + 12) )
      {
        sub_C8D5F0(a2, a2 + 16, *(unsigned int *)(a1 + 3428), 16);
        result = *(unsigned int *)(a2 + 8);
      }
      result = *(_QWORD *)a2 + 16 * result;
      for ( i = *(_QWORD *)a2 + 16 * v3; i != result; result += 16LL )
      {
        if ( result )
        {
          *(_QWORD *)result = 0;
          *(_QWORD *)(result + 8) = 0;
        }
      }
    }
    *(_DWORD *)(a2 + 8) = v3;
  }
  v7 = *(_DWORD *)(a1 + 3424);
  if ( v7 )
  {
    v8 = *(unsigned __int64 **)(a1 + 3416);
    result = *v8;
    if ( *v8 && result != -8 )
    {
      v10 = *(_QWORD *)(a1 + 3416);
    }
    else
    {
      result = (unsigned __int64)(v8 + 1);
      do
      {
        do
        {
          v9 = *(_QWORD *)result;
          v10 = result;
          result += 8LL;
        }
        while ( v9 == -8 );
      }
      while ( !v9 );
    }
    v11 = &v8[v7];
    while ( (unsigned __int64 *)v10 != v11 )
    {
      while ( 1 )
      {
        v12 = **(_QWORD **)v10;
        v13 = (_QWORD *)(*(_QWORD *)a2 + 16LL * *(unsigned int *)(*(_QWORD *)v10 + 8LL));
        *v13 = *(_QWORD *)v10 + 16LL;
        v14 = (unsigned __int64 *)(v10 + 8);
        v13[1] = v12;
        result = *(_QWORD *)(v10 + 8);
        if ( !result || result == -8 )
          break;
        v10 += 8LL;
        if ( v14 == v11 )
          return result;
      }
      result = v10 + 16;
      do
      {
        do
        {
          v15 = *(_QWORD *)result;
          v10 = result;
          result += 8LL;
        }
        while ( v15 == -8 );
      }
      while ( !v15 );
    }
  }
  return result;
}
