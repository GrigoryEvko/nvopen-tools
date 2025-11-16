// Function: sub_8ECD10
// Address: 0x8ecd10
//
char *__fastcall sub_8ECD10(unsigned __int8 *a1, __int64 a2)
{
  unsigned __int8 *v2; // r12
  char *v4; // r9
  unsigned __int8 *v5; // r9
  __int64 v6; // rdx
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rcx
  char *v9; // r14
  int v10; // [rsp+0h] [rbp-30h] BYREF
  int v11; // [rsp+4h] [rbp-2Ch] BYREF
  _QWORD v12[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( *a1 == 111 )
  {
    if ( a1[1] != 110 )
      goto LABEL_4;
    v2 = a1 + 2;
    v4 = sub_8E6070(a1 + 2, &v10, &v11, v12, a2);
    if ( !v4 )
    {
      if ( !*(_DWORD *)(a2 + 24) )
      {
        ++*(_QWORD *)(a2 + 32);
        ++*(_QWORD *)(a2 + 48);
        *(_DWORD *)(a2 + 24) = 1;
      }
      return (char *)v2;
    }
    v2 += v11;
    if ( *(_QWORD *)(a2 + 32) )
    {
      if ( strcmp(v4, "cast") )
      {
LABEL_13:
        if ( !*(_DWORD *)(a2 + 24) && *v2 == 73 )
          return sub_8E9020(v2, a2);
        return (char *)v2;
      }
    }
    else
    {
      sub_8E5790("operator ", a2);
      if ( strcmp((const char *)v5, "cast") )
      {
        if ( !*(_QWORD *)(a2 + 32) )
          sub_8E5790(v5, a2);
        goto LABEL_13;
      }
    }
    v9 = sub_8E9FF0((__int64)v2, 0, 0, 0, 1u, a2);
    sub_8EB260(v2, 0, 0, a2);
    v2 = (unsigned __int8 *)v9;
    goto LABEL_13;
  }
  if ( *a1 != 100 || a1[1] != 110 )
  {
LABEL_4:
    v2 = sub_8E72C0(a1, 0, a2);
    if ( !*(_DWORD *)(a2 + 24) )
      goto LABEL_5;
    return (char *)v2;
  }
  if ( !*(_QWORD *)(a2 + 32) )
  {
    v6 = *(_QWORD *)(a2 + 8);
    v7 = v6 + 1;
    if ( !*(_DWORD *)(a2 + 28) )
    {
      v8 = *(_QWORD *)(a2 + 16);
      if ( v7 < v8 )
      {
        *(_BYTE *)(*(_QWORD *)a2 + v6) = 126;
        v7 = *(_QWORD *)(a2 + 8) + 1LL;
      }
      else
      {
        *(_DWORD *)(a2 + 28) = 1;
        if ( v8 )
        {
          *(_BYTE *)(*(_QWORD *)a2 + v8 - 1) = 0;
          v7 = *(_QWORD *)(a2 + 8) + 1LL;
        }
      }
    }
    *(_QWORD *)(a2 + 8) = v7;
  }
  if ( (unsigned int)a1[2] - 48 > 9 )
  {
    v2 = (unsigned __int8 *)sub_8E9FF0((__int64)(a1 + 2), 0, 0, 0, 1u, a2);
    sub_8EB260(a1 + 2, 0, 0, a2);
    return (char *)v2;
  }
  v2 = sub_8E72C0(a1 + 2, 0, a2);
  if ( *(_DWORD *)(a2 + 24) )
    return (char *)v2;
LABEL_5:
  if ( *v2 != 73 )
    return (char *)v2;
  return sub_8E9020(v2, a2);
}
