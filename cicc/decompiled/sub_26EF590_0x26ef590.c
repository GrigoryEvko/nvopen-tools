// Function: sub_26EF590
// Address: 0x26ef590
//
__int64 *__fastcall sub_26EF590(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 *v7; // rax
  __int64 v8; // r14
  __int64 *result; // rax
  __int64 v10; // rbx
  __int64 i; // r12
  __int64 *v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9

  v6 = a2;
  if ( !*(_BYTE *)(a2 + 28) )
    goto LABEL_21;
  v7 = *(__int64 **)(a2 + 8);
  a4 = *(unsigned int *)(a2 + 20);
  a3 = &v7[a4];
  if ( v7 == a3 )
  {
LABEL_20:
    if ( (unsigned int)a4 >= *(_DWORD *)(a2 + 16) )
    {
LABEL_21:
      a2 = a1;
      sub_C8CC70(v6, a1, (__int64)a3, a4, a5, a6);
      goto LABEL_6;
    }
    *(_DWORD *)(a2 + 20) = a4 + 1;
    *a3 = a1;
    ++*(_QWORD *)a2;
  }
  else
  {
    while ( a1 != *v7 )
    {
      if ( a3 == ++v7 )
        goto LABEL_20;
    }
  }
LABEL_6:
  v8 = *(_QWORD *)(a1 - 32);
  result = (__int64 *)(*(_DWORD *)(v8 + 4) & 0x7FFFFFF);
  if ( (*(_DWORD *)(v8 + 4) & 0x7FFFFFF) != 0 )
  {
    v10 = (unsigned int)((_DWORD)result - 1);
    for ( i = 0; ; ++i )
    {
      result = (__int64 *)sub_BD3990(*(unsigned __int8 **)(v8 + 32 * (i - (unsigned int)result)), a2);
      a2 = (__int64)result;
      if ( *(_BYTE *)result > 3u )
        goto LABEL_14;
      if ( *(_BYTE *)(v6 + 28) )
      {
        result = *(__int64 **)(v6 + 8);
        v13 = *(unsigned int *)(v6 + 20);
        v12 = &result[v13];
        if ( result != v12 )
        {
          while ( a2 != *result )
          {
            if ( v12 == ++result )
              goto LABEL_18;
          }
LABEL_14:
          if ( i == v10 )
            return result;
          goto LABEL_15;
        }
LABEL_18:
        if ( (unsigned int)v13 < *(_DWORD *)(v6 + 16) )
        {
          *(_DWORD *)(v6 + 20) = v13 + 1;
          *v12 = a2;
          ++*(_QWORD *)v6;
          goto LABEL_14;
        }
      }
      result = sub_C8CC70(v6, a2, (__int64)v12, v13, v14, v15);
      if ( i == v10 )
        return result;
LABEL_15:
      LODWORD(result) = *(_DWORD *)(v8 + 4) & 0x7FFFFFF;
    }
  }
  return result;
}
