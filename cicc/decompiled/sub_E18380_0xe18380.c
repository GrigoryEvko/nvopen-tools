// Function: sub_E18380
// Address: 0xe18380
//
__int64 __fastcall sub_E18380(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char *v6; // r12
  __int64 result; // rax
  char *v8; // r15
  __int64 v9; // rax
  __int64 v10; // rdx
  char *v11; // rcx
  char *v12; // rax
  char *v13; // rax
  __int64 n; // [rsp+8h] [rbp-38h]

  v6 = *(char **)(a1 + 8);
  if ( v6 == *(char **)(a1 + 16) )
  {
    v8 = *(char **)a1;
    n = (__int64)&v6[-*(_QWORD *)a1];
    if ( *(_QWORD *)a1 == a1 + 24 )
    {
      v12 = (char *)malloc(16 * (n >> 3), a2, n, 16 * (n >> 3), a5, a6);
      v10 = n;
      v11 = v12;
      if ( v12 )
      {
        if ( v6 != v8 )
        {
          v13 = (char *)memmove(v12, v8, n);
          v10 = n;
          v11 = v13;
        }
        *(_QWORD *)a1 = v11;
        goto LABEL_5;
      }
    }
    else
    {
      v9 = realloc(v8);
      v10 = n;
      *(_QWORD *)a1 = v9;
      v11 = (char *)v9;
      if ( v9 )
      {
LABEL_5:
        v6 = &v11[v10];
        *(_QWORD *)(a1 + 8) = &v11[v10];
        *(_QWORD *)(a1 + 16) = &v11[16 * (n >> 3)];
        goto LABEL_2;
      }
    }
    abort();
  }
LABEL_2:
  result = *a2;
  *(_QWORD *)(a1 + 8) = v6 + 8;
  *(_QWORD *)v6 = result;
  return result;
}
