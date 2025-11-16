// Function: sub_285BE30
// Address: 0x285be30
//
__int64 *__fastcall sub_285BE30(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v8; // r12
  __int64 *result; // rax
  __int64 *v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  char v14; // dl

  v6 = a4;
  v8 = (__int64)a3;
  if ( !a5 )
    goto LABEL_10;
  if ( *(_BYTE *)(a5 + 28) )
  {
    result = *(__int64 **)(a5 + 8);
    a3 = &result[*(unsigned int *)(a5 + 20)];
    if ( result == a3 )
      goto LABEL_10;
    while ( v8 != *result )
    {
      if ( a3 == ++result )
        goto LABEL_10;
    }
LABEL_7:
    *(_QWORD *)(a1 + 24) = -1;
    *(_QWORD *)(a1 + 32) = -1;
    *(_QWORD *)(a1 + 40) = -1;
    *(_QWORD *)(a1 + 48) = -1;
    return result;
  }
  result = sub_C8CA60(a5, (__int64)a3);
  if ( result )
    goto LABEL_7;
LABEL_10:
  if ( !*(_BYTE *)(v6 + 28) )
    goto LABEL_25;
  result = *(__int64 **)(v6 + 8);
  a4 = *(unsigned int *)(v6 + 20);
  a3 = &result[a4];
  if ( result != a3 )
  {
    while ( v8 != *result )
    {
      if ( a3 == ++result )
        goto LABEL_12;
    }
    return result;
  }
LABEL_12:
  if ( (unsigned int)a4 < *(_DWORD *)(v6 + 16) )
  {
    *(_DWORD *)(v6 + 20) = a4 + 1;
    *a3 = v8;
    ++*(_QWORD *)v6;
  }
  else
  {
LABEL_25:
    result = sub_C8CC70(v6, v8, (__int64)a3, a4, a5, a6);
    if ( !v14 )
      return result;
  }
  result = (__int64 *)sub_285BB00(a1, a2, v8, v6);
  if ( a5 && *(_DWORD *)(a1 + 28) == -1 )
  {
    if ( !*(_BYTE *)(a5 + 28) )
      return sub_C8CC70(a5, v8, (__int64)v10, v11, v12, v13);
    result = *(__int64 **)(a5 + 8);
    v11 = *(unsigned int *)(a5 + 20);
    v10 = &result[v11];
    if ( result == v10 )
    {
LABEL_28:
      if ( (unsigned int)v11 < *(_DWORD *)(a5 + 16) )
      {
        *(_DWORD *)(a5 + 20) = v11 + 1;
        *v10 = v8;
        ++*(_QWORD *)a5;
        return result;
      }
      return sub_C8CC70(a5, v8, (__int64)v10, v11, v12, v13);
    }
    while ( v8 != *result )
    {
      if ( v10 == ++result )
        goto LABEL_28;
    }
  }
  return result;
}
