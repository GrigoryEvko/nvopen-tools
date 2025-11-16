// Function: sub_1603160
// Address: 0x1603160
//
unsigned __int64 __fastcall sub_1603160(__int64 *a1, __int64 a2)
{
  __int64 v2; // rdx
  unsigned __int64 result; // rax
  unsigned __int64 v5; // r13
  int v6; // r14d
  int v7; // ecx
  unsigned __int64 *v8; // rsi
  unsigned __int64 v9; // rdx
  unsigned __int64 *v10; // rdi
  __int64 v11; // rsi
  _QWORD *v12; // rax
  __int64 v13; // rdx
  __int64 i; // rdx

  v2 = *a1;
  result = *(unsigned int *)(a2 + 8);
  v5 = *(unsigned int *)(*a1 + 2684);
  v6 = *(_DWORD *)(*a1 + 2684);
  if ( v5 < result )
  {
LABEL_21:
    *(_DWORD *)(a2 + 8) = v6;
    v2 = *a1;
    goto LABEL_3;
  }
  if ( v5 > result )
  {
    if ( v5 > *(unsigned int *)(a2 + 12) )
    {
      sub_16CD150(a2, a2 + 16, *(unsigned int *)(*a1 + 2684), 16);
      result = *(unsigned int *)(a2 + 8);
    }
    result = *(_QWORD *)a2 + 16 * result;
    for ( i = 16 * v5 + *(_QWORD *)a2; i != result; result += 16LL )
    {
      if ( result )
      {
        *(_QWORD *)result = 0;
        *(_QWORD *)(result + 8) = 0;
      }
    }
    goto LABEL_21;
  }
LABEL_3:
  v7 = *(_DWORD *)(v2 + 2680);
  v8 = *(unsigned __int64 **)(v2 + 2672);
  if ( v7 )
  {
    result = *v8;
    v9 = *(_QWORD *)(v2 + 2672);
    if ( !*v8 || result == -8 )
    {
      do
      {
        do
        {
          result = *(_QWORD *)(v9 + 8);
          v9 += 8LL;
        }
        while ( result == -8 );
      }
      while ( !result );
    }
    v10 = &v8[v7];
    if ( v10 != (unsigned __int64 *)v9 )
    {
      while ( 1 )
      {
        v11 = **(_QWORD **)v9;
        v12 = (_QWORD *)(*(_QWORD *)a2 + 16LL * *(unsigned int *)(*(_QWORD *)v9 + 8LL));
        *v12 = *(_QWORD *)v9 + 16LL;
        v12[1] = v11;
        result = v9 + 8;
        v13 = *(_QWORD *)(v9 + 8);
        if ( v13 != -8 )
          goto LABEL_11;
        do
        {
          do
          {
            v13 = *(_QWORD *)(result + 8);
            result += 8LL;
          }
          while ( v13 == -8 );
LABEL_11:
          ;
        }
        while ( !v13 );
        if ( (unsigned __int64 *)result == v10 )
          return result;
        v9 = result;
      }
    }
  }
  return result;
}
