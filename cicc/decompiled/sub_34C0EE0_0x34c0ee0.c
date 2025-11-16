// Function: sub_34C0EE0
// Address: 0x34c0ee0
//
bool __fastcall sub_34C0EE0(__int64 a1, unsigned int *a2)
{
  __int64 v3; // rax
  char *v4; // rdi
  char *v5; // rdx
  int v6; // ecx
  bool result; // al
  unsigned __int64 v8; // rdi
  __int64 v9; // r13
  unsigned int v10; // ecx
  __int64 v11; // r14
  __int64 v12; // rbx
  __int64 v13; // rax
  bool v14; // cl
  __int64 v15; // rsi
  __int64 v16; // rsi
  int *v17; // rdi
  int *v18; // rax
  __int64 v19; // rax
  __int64 v20; // [rsp+8h] [rbp-38h]

  v20 = *(_QWORD *)(a1 + 72);
  if ( !v20 )
  {
    v3 = *(unsigned int *)(a1 + 8);
    v4 = *(char **)a1;
    v5 = &v4[4 * v3];
    v6 = *(_DWORD *)(a1 + 8);
    if ( v4 == v5 )
      return 0;
    while ( *(_DWORD *)v4 != *a2 )
    {
      v4 += 4;
      if ( v5 == v4 )
        return 0;
    }
    result = 0;
    if ( v5 != v4 )
    {
      if ( v5 != v4 + 4 )
      {
        memmove(v4, v4 + 4, v5 - (v4 + 4));
        v6 = *(_DWORD *)(a1 + 8);
      }
      result = 1;
      *(_DWORD *)(a1 + 8) = v6 - 1;
    }
    return result;
  }
  v8 = *(_QWORD *)(a1 + 48);
  v9 = a1 + 40;
  if ( v8 )
  {
    v10 = *a2;
    v11 = a1 + 40;
    v12 = v8;
    while ( 1 )
    {
      while ( *(_DWORD *)(v12 + 32) < v10 )
      {
        v12 = *(_QWORD *)(v12 + 24);
        if ( !v12 )
          goto LABEL_18;
      }
      v13 = *(_QWORD *)(v12 + 16);
      if ( *(_DWORD *)(v12 + 32) <= v10 )
        break;
      v11 = v12;
      v12 = *(_QWORD *)(v12 + 16);
      if ( !v13 )
      {
LABEL_18:
        v14 = v9 == v11;
        goto LABEL_19;
      }
    }
    v15 = *(_QWORD *)(v12 + 24);
    while ( v15 )
    {
      if ( v10 >= *(_DWORD *)(v15 + 32) )
      {
        v15 = *(_QWORD *)(v15 + 24);
      }
      else
      {
        v11 = v15;
        v15 = *(_QWORD *)(v15 + 16);
      }
    }
    while ( v13 )
    {
      while ( 1 )
      {
        v16 = *(_QWORD *)(v13 + 24);
        if ( v10 <= *(_DWORD *)(v13 + 32) )
          break;
        v13 = *(_QWORD *)(v13 + 24);
        if ( !v16 )
          goto LABEL_31;
      }
      v12 = v13;
      v13 = *(_QWORD *)(v13 + 16);
    }
LABEL_31:
    result = v9 == v11 && *(_QWORD *)(a1 + 56) == v12;
    if ( !result )
    {
      if ( v11 != v12 )
      {
        do
        {
          v17 = (int *)v12;
          v12 = sub_220EF30(v12);
          v18 = sub_220F330(v17, (_QWORD *)(a1 + 40));
          j_j___libc_free_0((unsigned __int64)v18);
          v19 = *(_QWORD *)(a1 + 72) - 1LL;
          *(_QWORD *)(a1 + 72) = v19;
        }
        while ( v12 != v11 );
        return v20 != v19;
      }
      return result;
    }
LABEL_20:
    sub_34BE7F0(v8);
    *(_QWORD *)(a1 + 56) = v9;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 64) = v9;
    *(_QWORD *)(a1 + 72) = 0;
    return 1;
  }
  v11 = a1 + 40;
  v14 = 1;
LABEL_19:
  result = v14 && *(_QWORD *)(a1 + 56) == v11;
  if ( result )
    goto LABEL_20;
  return result;
}
