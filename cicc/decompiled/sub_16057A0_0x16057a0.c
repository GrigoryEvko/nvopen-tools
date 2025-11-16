// Function: sub_16057A0
// Address: 0x16057a0
//
unsigned __int64 __fastcall sub_16057A0(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // r14
  unsigned __int64 result; // rax
  int v6; // esi
  unsigned __int64 *v7; // rdi
  __int64 v8; // rdx
  unsigned __int64 v9; // rcx
  unsigned __int64 *v10; // rdi
  __int64 v11; // rsi
  _QWORD *v12; // rax
  unsigned __int64 *v13; // rdx
  __int64 v14; // rdx
  __int64 i; // rdx

  v3 = *(unsigned int *)(a1 + 2908);
  result = *(unsigned int *)(a2 + 8);
  if ( v3 < result )
  {
LABEL_23:
    *(_DWORD *)(a2 + 8) = v3;
    goto LABEL_3;
  }
  if ( v3 > result )
  {
    if ( v3 > *(unsigned int *)(a2 + 12) )
    {
      sub_16CD150(a2, a2 + 16, *(unsigned int *)(a1 + 2908), 16);
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
    goto LABEL_23;
  }
LABEL_3:
  v6 = *(_DWORD *)(a1 + 2904);
  if ( v6 )
  {
    v7 = *(unsigned __int64 **)(a1 + 2896);
    result = *v7;
    if ( *v7 && result != -8 )
    {
      v9 = *(_QWORD *)(a1 + 2896);
    }
    else
    {
      result = (unsigned __int64)(v7 + 1);
      do
      {
        do
        {
          v8 = *(_QWORD *)result;
          v9 = result;
          result += 8LL;
        }
        while ( v8 == -8 );
      }
      while ( !v8 );
    }
    v10 = &v7[v6];
    while ( (unsigned __int64 *)v9 != v10 )
    {
      while ( 1 )
      {
        v11 = **(_QWORD **)v9;
        v12 = (_QWORD *)(*(_QWORD *)a2 + 16LL * *(unsigned __int8 *)(*(_QWORD *)v9 + 8LL));
        *v12 = *(_QWORD *)v9 + 16LL;
        v13 = (unsigned __int64 *)(v9 + 8);
        v12[1] = v11;
        result = *(_QWORD *)(v9 + 8);
        if ( !result || result == -8 )
          break;
        v9 += 8LL;
        if ( v13 == v10 )
          return result;
      }
      result = v9 + 16;
      do
      {
        do
        {
          v14 = *(_QWORD *)result;
          v9 = result;
          result += 8LL;
        }
        while ( v14 == -8 );
      }
      while ( !v14 );
    }
  }
  return result;
}
