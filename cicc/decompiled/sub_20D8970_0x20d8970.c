// Function: sub_20D8970
// Address: 0x20d8970
//
bool __fastcall sub_20D8970(__int64 a1, unsigned int *a2)
{
  int *v4; // rsi
  __int64 v5; // r8
  __int64 v6; // rdx
  unsigned int v7; // ecx
  int v8; // eax
  int *v9; // rdi
  bool result; // al
  __int64 v11; // rdi
  __int64 v12; // r13
  unsigned int v13; // ecx
  __int64 v14; // r14
  __int64 v15; // rbx
  __int64 v16; // rax
  bool v17; // cl
  __int64 v18; // rsi
  __int64 v19; // rsi
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // [rsp+8h] [rbp-38h]

  v23 = *(_QWORD *)(a1 + 72);
  if ( !v23 )
  {
    v4 = *(int **)a1;
    v5 = *(unsigned int *)(a1 + 8);
    v6 = *(_QWORD *)a1 + 4 * v5;
    if ( v6 == *(_QWORD *)a1 )
      return 0;
    v7 = *a2;
    while ( 1 )
    {
      v8 = *v4;
      v9 = v4++;
      if ( v8 == v7 )
        break;
      if ( (int *)v6 == v4 )
        return 0;
    }
    if ( (int *)v6 != v4 )
    {
      memmove(v9, v4, v6 - (_QWORD)v4);
      LODWORD(v5) = *(_DWORD *)(a1 + 8);
    }
    result = 1;
    *(_DWORD *)(a1 + 8) = v5 - 1;
    return result;
  }
  v11 = *(_QWORD *)(a1 + 48);
  v12 = a1 + 40;
  if ( v11 )
  {
    v13 = *a2;
    v14 = a1 + 40;
    v15 = v11;
    while ( 1 )
    {
      while ( *(_DWORD *)(v15 + 32) < v13 )
      {
        v15 = *(_QWORD *)(v15 + 24);
        if ( !v15 )
          goto LABEL_17;
      }
      v16 = *(_QWORD *)(v15 + 16);
      if ( *(_DWORD *)(v15 + 32) <= v13 )
        break;
      v14 = v15;
      v15 = *(_QWORD *)(v15 + 16);
      if ( !v16 )
      {
LABEL_17:
        v17 = v12 == v14;
        goto LABEL_18;
      }
    }
    v18 = *(_QWORD *)(v15 + 24);
    while ( v18 )
    {
      if ( v13 >= *(_DWORD *)(v18 + 32) )
      {
        v18 = *(_QWORD *)(v18 + 24);
      }
      else
      {
        v14 = v18;
        v18 = *(_QWORD *)(v18 + 16);
      }
    }
    while ( v16 )
    {
      while ( 1 )
      {
        v19 = *(_QWORD *)(v16 + 24);
        if ( v13 <= *(_DWORD *)(v16 + 32) )
          break;
        v16 = *(_QWORD *)(v16 + 24);
        if ( !v19 )
          goto LABEL_30;
      }
      v15 = v16;
      v16 = *(_QWORD *)(v16 + 16);
    }
LABEL_30:
    result = v12 == v14 && *(_QWORD *)(a1 + 56) == v15;
    if ( !result )
    {
      if ( v15 != v14 )
      {
        do
        {
          v20 = v15;
          v15 = sub_220EF30(v15);
          v21 = sub_220F330(v20, a1 + 40);
          j_j___libc_free_0(v21, 40);
          v22 = *(_QWORD *)(a1 + 72) - 1LL;
          *(_QWORD *)(a1 + 72) = v22;
        }
        while ( v15 != v14 );
        return v23 != v22;
      }
      return result;
    }
LABEL_19:
    sub_20D63D0(v11);
    *(_QWORD *)(a1 + 56) = v12;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 64) = v12;
    *(_QWORD *)(a1 + 72) = 0;
    return 1;
  }
  v14 = a1 + 40;
  v17 = 1;
LABEL_18:
  result = v17 && *(_QWORD *)(a1 + 56) == v14;
  if ( result )
    goto LABEL_19;
  return result;
}
