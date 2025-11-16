// Function: sub_2A73850
// Address: 0x2a73850
//
__int64 *__fastcall sub_2A73850(__int64 *a1, __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *result; // rax
  __int64 v7; // r14
  __int64 v10; // r13
  __int64 v11; // rbx
  __int64 v12; // rsi
  __int64 v13; // rdi
  __int64 *v14; // rdx
  char v15; // dl
  __int64 v16; // rax
  const void *v17; // [rsp+0h] [rbp-40h]

  result = (__int64 *)(a4 + 16);
  v7 = *(_QWORD *)(a2 + 16);
  v17 = (const void *)(a4 + 16);
  if ( v7 )
  {
    v10 = a4;
    while ( 1 )
    {
      v11 = *(_QWORD *)(v7 + 24);
      if ( v11 == a2 )
        goto LABEL_14;
      v12 = *(_QWORD *)(v11 + 40);
      v13 = *a1;
      if ( *(_BYTE *)(*a1 + 84) )
      {
        result = *(__int64 **)(v13 + 64);
        v14 = &result[*(unsigned int *)(v13 + 76)];
        if ( result == v14 )
          goto LABEL_14;
        while ( v12 != *result )
        {
          if ( v14 == ++result )
            goto LABEL_14;
        }
        if ( !*(_BYTE *)(a3 + 28) )
          goto LABEL_18;
        goto LABEL_10;
      }
      result = sub_C8CA60(v13 + 56, v12);
      if ( !result )
      {
LABEL_14:
        v7 = *(_QWORD *)(v7 + 8);
        if ( !v7 )
          return result;
      }
      else
      {
        if ( !*(_BYTE *)(a3 + 28) )
          goto LABEL_18;
LABEL_10:
        result = *(__int64 **)(a3 + 8);
        a4 = *(unsigned int *)(a3 + 20);
        v14 = &result[a4];
        if ( result != v14 )
        {
          while ( v11 != *result )
          {
            if ( v14 == ++result )
              goto LABEL_23;
          }
          goto LABEL_14;
        }
LABEL_23:
        if ( (unsigned int)a4 < *(_DWORD *)(a3 + 16) )
        {
          *(_DWORD *)(a3 + 20) = a4 + 1;
          *v14 = v11;
          ++*(_QWORD *)a3;
          goto LABEL_19;
        }
LABEL_18:
        result = sub_C8CC70(a3, v11, (__int64)v14, a4, a5, a6);
        if ( !v15 )
          goto LABEL_14;
LABEL_19:
        v16 = *(unsigned int *)(v10 + 8);
        a4 = *(unsigned int *)(v10 + 12);
        if ( v16 + 1 > a4 )
        {
          sub_C8D5F0(v10, v17, v16 + 1, 0x10u, a5, a6);
          v16 = *(unsigned int *)(v10 + 8);
        }
        result = (__int64 *)(*(_QWORD *)v10 + 16 * v16);
        *result = v11;
        result[1] = a2;
        ++*(_DWORD *)(v10 + 8);
        v7 = *(_QWORD *)(v7 + 8);
        if ( !v7 )
          return result;
      }
    }
  }
  return result;
}
