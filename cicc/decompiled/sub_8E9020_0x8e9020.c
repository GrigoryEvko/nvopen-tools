// Function: sub_8E9020
// Address: 0x8e9020
//
char *__fastcall sub_8E9020(_BYTE *a1, __int64 a2)
{
  char *v2; // r9
  bool v3; // zf
  int v4; // ebx
  char v5; // al
  __int64 v6; // rax
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rdx
  unsigned __int64 v12; // rax
  __int64 v14; // rdx
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rcx
  unsigned __int64 v17; // rcx

  v2 = a1 + 1;
  v3 = (*(_DWORD *)(a2 + 72) | (*a1 == 74)) == 0;
  v4 = *(_DWORD *)(a2 + 72) | (*a1 == 74);
  *(_DWORD *)(a2 + 72) = 0;
  if ( v3 && !*(_QWORD *)(a2 + 32) )
  {
    v14 = *(_QWORD *)(a2 + 8);
    v15 = v14 + 1;
    if ( !*(_DWORD *)(a2 + 28) )
    {
      v16 = *(_QWORD *)(a2 + 16);
      if ( v16 > v15 )
      {
        *(_BYTE *)(*(_QWORD *)a2 + v14) = 60;
        v15 = *(_QWORD *)(a2 + 8) + 1LL;
      }
      else
      {
        *(_DWORD *)(a2 + 28) = 1;
        if ( v16 )
        {
          *(_BYTE *)(*(_QWORD *)a2 + v16 - 1) = 0;
          v15 = *(_QWORD *)(a2 + 8) + 1LL;
        }
      }
    }
    *(_QWORD *)(a2 + 8) = v15;
  }
  if ( a1[1] != 69 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v2 = (char *)sub_8E8F40(v2, a2);
        v5 = *v2;
        if ( *v2 == 69 )
          goto LABEL_11;
        if ( *(_DWORD *)(a2 + 24) )
          goto LABEL_12;
        if ( v5 != 74 && (v5 != 73 || !dword_4D0425C) || v2[1] != 69 )
          break;
        if ( !*(_QWORD *)(a2 + 32) )
        {
          sub_8E5790((unsigned __int8 *)" ", a2);
          goto LABEL_10;
        }
      }
      if ( !*(_QWORD *)(a2 + 32) )
      {
        sub_8E5790((unsigned __int8 *)", ", a2);
LABEL_10:
        if ( *v2 == 69 )
          break;
      }
    }
  }
LABEL_11:
  ++v2;
LABEL_12:
  v6 = *(_QWORD *)(a2 + 32);
  if ( !v4 )
  {
    if ( v6 )
      return v2;
    v7 = *(_QWORD *)(a2 + 8);
    v8 = v7 + 1;
    if ( !*(_DWORD *)(a2 + 28) )
    {
      v9 = *(_QWORD *)(a2 + 16);
      if ( v9 > v8 )
      {
        *(_BYTE *)(*(_QWORD *)a2 + v7) = 62;
        v10 = *(_QWORD *)(a2 + 8) + 1LL;
        v6 = *(_QWORD *)(a2 + 32);
        goto LABEL_23;
      }
      *(_DWORD *)(a2 + 28) = 1;
      if ( v9 )
      {
        *(_BYTE *)(*(_QWORD *)a2 + v9 - 1) = 0;
        v10 = *(_QWORD *)(a2 + 8) + 1LL;
        v6 = *(_QWORD *)(a2 + 32);
LABEL_23:
        *(_QWORD *)(a2 + 8) = v10;
        goto LABEL_24;
      }
    }
    *(_QWORD *)(a2 + 8) = v8;
    v11 = *(_QWORD *)(a2 + 8);
    v12 = v11 + 1;
    if ( *(_DWORD *)(a2 + 28) )
      goto LABEL_26;
    goto LABEL_33;
  }
LABEL_24:
  if ( !v6 )
  {
    v11 = *(_QWORD *)(a2 + 8);
    v12 = v11 + 1;
    if ( *(_DWORD *)(a2 + 28) )
    {
LABEL_26:
      *(_QWORD *)(a2 + 8) = v12;
      return v2;
    }
LABEL_33:
    v17 = *(_QWORD *)(a2 + 16);
    if ( v17 > v12 )
    {
      *(_BYTE *)(*(_QWORD *)a2 + v11) = 32;
      ++*(_QWORD *)(a2 + 8);
      return v2;
    }
    *(_DWORD *)(a2 + 28) = 1;
    if ( v17 )
    {
      *(_BYTE *)(*(_QWORD *)a2 + v17 - 1) = 0;
      v12 = *(_QWORD *)(a2 + 8) + 1LL;
    }
    goto LABEL_26;
  }
  return v2;
}
