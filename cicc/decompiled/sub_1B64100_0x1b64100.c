// Function: sub_1B64100
// Address: 0x1b64100
//
__int64 *__fastcall sub_1B64100(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *result; // rax
  __int64 v5; // r14
  int v9; // r8d
  int v10; // r9d
  char v11; // dl
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // r15
  __int64 v15; // rsi
  __int64 *v16; // rdx
  __int64 *v17; // rcx
  __int64 v18; // rax
  __int64 *v19; // rsi
  unsigned int v20; // edi
  __int64 *v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // rsi
  __int64 *v24; // rdx
  const void *v25; // [rsp+0h] [rbp-50h]
  __int64 *v27; // [rsp+10h] [rbp-40h]

  result = (__int64 *)(a4 + 16);
  v5 = a1[1];
  v25 = (const void *)(a4 + 16);
  while ( v5 )
  {
    result = sub_1648700(v5);
    v14 = (__int64)result;
    if ( result == a1 )
      goto LABEL_6;
    v15 = result[5];
    v16 = *(__int64 **)(a2 + 72);
    result = *(__int64 **)(a2 + 64);
    if ( v16 == result )
    {
      v17 = &result[*(unsigned int *)(a2 + 84)];
      if ( result == v17 )
      {
        v24 = *(__int64 **)(a2 + 64);
      }
      else
      {
        do
        {
          if ( v15 == *result )
            break;
          ++result;
        }
        while ( v17 != result );
        v24 = v17;
      }
    }
    else
    {
      v27 = &v16[*(unsigned int *)(a2 + 80)];
      result = sub_16CC9F0(a2 + 56, v15);
      v17 = v27;
      if ( v15 == *result )
      {
        v22 = *(_QWORD *)(a2 + 72);
        if ( v22 == *(_QWORD *)(a2 + 64) )
          v23 = *(unsigned int *)(a2 + 84);
        else
          v23 = *(unsigned int *)(a2 + 80);
        v24 = (__int64 *)(v22 + 8 * v23);
      }
      else
      {
        v18 = *(_QWORD *)(a2 + 72);
        if ( v18 != *(_QWORD *)(a2 + 64) )
        {
          result = (__int64 *)(v18 + 8LL * *(unsigned int *)(a2 + 80));
          goto LABEL_12;
        }
        result = (__int64 *)(v18 + 8LL * *(unsigned int *)(a2 + 84));
        v24 = result;
      }
    }
    while ( v24 != result && (unsigned __int64)*result >= 0xFFFFFFFFFFFFFFFELL )
      ++result;
LABEL_12:
    if ( v17 != result )
    {
      result = *(__int64 **)(a3 + 8);
      if ( *(__int64 **)(a3 + 16) == result )
      {
        v19 = &result[*(unsigned int *)(a3 + 28)];
        v20 = *(_DWORD *)(a3 + 28);
        if ( result != v19 )
        {
          v21 = 0;
          while ( v14 != *result )
          {
            if ( *result == -2 )
              v21 = result;
            if ( v19 == ++result )
            {
              if ( !v21 )
                goto LABEL_37;
              v12 = a4;
              *v21 = v14;
              --*(_DWORD *)(a3 + 32);
              ++*(_QWORD *)a3;
              v13 = *(unsigned int *)(a4 + 8);
              if ( (unsigned int)v13 < *(_DWORD *)(a4 + 12) )
                goto LABEL_5;
              goto LABEL_22;
            }
          }
          goto LABEL_6;
        }
LABEL_37:
        if ( v20 < *(_DWORD *)(a3 + 24) )
        {
          *(_DWORD *)(a3 + 28) = v20 + 1;
          *v19 = v14;
          ++*(_QWORD *)a3;
LABEL_4:
          v12 = a4;
          v13 = *(unsigned int *)(a4 + 8);
          if ( (unsigned int)v13 >= *(_DWORD *)(a4 + 12) )
          {
LABEL_22:
            sub_16CD150(v12, v25, 0, 16, v9, v10);
            v13 = *(unsigned int *)(a4 + 8);
          }
LABEL_5:
          result = (__int64 *)(*(_QWORD *)a4 + 16 * v13);
          *result = v14;
          result[1] = (__int64)a1;
          ++*(_DWORD *)(a4 + 8);
          goto LABEL_6;
        }
      }
      result = sub_16CCBA0(a3, v14);
      if ( v11 )
        goto LABEL_4;
    }
LABEL_6:
    v5 = *(_QWORD *)(v5 + 8);
  }
  return result;
}
