// Function: sub_2ED2320
// Address: 0x2ed2320
//
__int64 *__fastcall sub_2ED2320(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v7; // rsi
  __int64 *v8; // rdx
  __int64 v9; // rcx
  __int64 **v10; // rax
  __int64 v11; // rcx
  int v12; // eax
  __int64 *result; // rax
  __int64 *v14; // rsi
  __int64 *v15; // rax
  int v16; // eax

  if ( !*(_BYTE *)(a1 + 76) )
  {
    v15 = sub_C8CA60(a1 + 48, a2);
    if ( !v15 )
    {
      v9 = *(unsigned int *)(a1 + 68);
      goto LABEL_17;
    }
    *v15 = -2;
    v16 = *(_DWORD *)(a1 + 72);
    ++*(_QWORD *)(a1 + 48);
    v9 = *(unsigned int *)(a1 + 68);
    v12 = v16 + 1;
    *(_DWORD *)(a1 + 72) = v12;
LABEL_7:
    if ( (_DWORD)v9 == v12 )
      goto LABEL_18;
LABEL_8:
    if ( !*(_BYTE *)(a1 + 28) )
      return sub_C8CC70(a1, a2, (__int64)v8, v9, a5, a6);
    result = *(__int64 **)(a1 + 8);
    v9 = *(unsigned int *)(a1 + 20);
    v8 = &result[v9];
    if ( v8 != result )
      goto LABEL_12;
LABEL_14:
    if ( (unsigned int)v9 < *(_DWORD *)(a1 + 16) )
    {
      *(_DWORD *)(a1 + 20) = v9 + 1;
      *v8 = a2;
      ++*(_QWORD *)a1;
      return result;
    }
    return sub_C8CC70(a1, a2, (__int64)v8, v9, a5, a6);
  }
  v7 = *(__int64 **)(a1 + 56);
  v8 = &v7[*(unsigned int *)(a1 + 68)];
  v9 = *(unsigned int *)(a1 + 68);
  v10 = (__int64 **)v7;
  if ( v7 != v8 )
  {
    while ( (__int64 *)a2 != *v10 )
    {
      if ( v8 == (__int64 *)++v10 )
        goto LABEL_17;
    }
    v11 = (unsigned int)(v9 - 1);
    *(_DWORD *)(a1 + 68) = v11;
    v8 = (__int64 *)v7[v11];
    *v10 = v8;
    v9 = *(unsigned int *)(a1 + 68);
    ++*(_QWORD *)(a1 + 48);
    v12 = *(_DWORD *)(a1 + 72);
    goto LABEL_7;
  }
LABEL_17:
  if ( (_DWORD)v9 != *(_DWORD *)(a1 + 72) )
    goto LABEL_8;
LABEL_18:
  if ( !*(_BYTE *)(a1 + 28) )
  {
    result = sub_C8CA60(a1, (__int64)&qword_4F82400);
    if ( result )
      return result;
    goto LABEL_8;
  }
  result = *(__int64 **)(a1 + 8);
  v14 = &result[*(unsigned int *)(a1 + 20)];
  v9 = *(unsigned int *)(a1 + 20);
  v8 = result;
  if ( result == v14 )
    goto LABEL_14;
  while ( (__int64 *)*v8 != &qword_4F82400 )
  {
    if ( v14 == ++v8 )
    {
LABEL_12:
      while ( a2 != *result )
      {
        if ( v8 == ++result )
          goto LABEL_14;
      }
      return result;
    }
  }
  return result;
}
