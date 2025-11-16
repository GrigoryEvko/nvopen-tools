// Function: sub_B6F6A0
// Address: 0xb6f6a0
//
unsigned __int64 __fastcall sub_B6F6A0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rcx
  unsigned __int64 result; // rax
  unsigned __int64 v5; // r14
  __int64 i; // rdx
  unsigned __int64 *v7; // rsi
  int v8; // ecx
  unsigned __int64 v9; // rdx
  unsigned __int64 *v10; // rdi
  __int64 v11; // rsi
  _QWORD *v12; // rax
  __int64 v13; // rdx

  v2 = *a1;
  result = *(unsigned int *)(a2 + 8);
  v5 = *(unsigned int *)(*a1 + 3212);
  if ( v5 != result )
  {
    if ( v5 >= result )
    {
      if ( v5 > *(unsigned int *)(a2 + 12) )
      {
        sub_C8D5F0(a2, a2 + 16, *(unsigned int *)(*a1 + 3212), 16);
        result = *(unsigned int *)(a2 + 8);
      }
      result = *(_QWORD *)a2 + 16 * result;
      for ( i = *(_QWORD *)a2 + 16 * v5; i != result; result += 16LL )
      {
        if ( result )
        {
          *(_QWORD *)result = 0;
          *(_QWORD *)(result + 8) = 0;
        }
      }
    }
    *(_DWORD *)(a2 + 8) = v5;
    v2 = *a1;
  }
  v7 = *(unsigned __int64 **)(v2 + 3200);
  v8 = *(_DWORD *)(v2 + 3208);
  if ( v8 )
  {
    result = *v7;
    v9 = (unsigned __int64)v7;
    if ( !*v7 || result == -8 )
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
    v10 = &v7[v8];
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
          goto LABEL_18;
        do
        {
          do
          {
            v13 = *(_QWORD *)(result + 8);
            result += 8LL;
          }
          while ( v13 == -8 );
LABEL_18:
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
