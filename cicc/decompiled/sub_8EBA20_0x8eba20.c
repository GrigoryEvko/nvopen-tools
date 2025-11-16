// Function: sub_8EBA20
// Address: 0x8eba20
//
unsigned __int8 *__fastcall sub_8EBA20(unsigned __int8 *a1, int a2, char a3, __int64 a4)
{
  unsigned __int8 *v4; // r14
  int v7; // r15d
  __int64 v8; // rax
  unsigned __int8 *v9; // r12
  int v10; // r13d
  __int64 v11; // rdx
  unsigned __int64 v12; // rax
  unsigned __int8 v13; // al
  __int64 v14; // rdx
  char *v15; // r15
  __int64 v16; // rdx
  unsigned __int64 v17; // rax
  unsigned __int64 v19; // rcx
  unsigned __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rsi
  unsigned __int64 v23; // rcx
  unsigned __int64 v24; // rax
  unsigned __int8 v25; // dl

  v4 = a1;
  v7 = a3 & 1;
  if ( (a3 & 1) == 0 )
  {
    v8 = *(_QWORD *)(a4 + 32);
    *(_QWORD *)(a4 + 32) = v8 + 1;
    if ( a2 )
    {
LABEL_3:
      *(_QWORD *)(a4 + 32) = v8;
      v9 = v4;
      goto LABEL_4;
    }
    v9 = (unsigned __int8 *)sub_8E9FF0((__int64)a1, 0, 0, 0, 1u, a4);
    sub_8EB260(a1, 0, 0, a4);
    v21 = *(_QWORD *)(a4 + 32);
    if ( v21 )
    {
      v8 = v21 - 1;
      v4 = v9;
      goto LABEL_3;
    }
    v22 = *(_QWORD *)(a4 + 8);
    v23 = v22 + 1;
    if ( *(_DWORD *)(a4 + 28) )
    {
      *(_QWORD *)(a4 + 8) = v23;
      v8 = 0;
      v4 = v9;
LABEL_69:
      --v8;
      goto LABEL_3;
    }
    goto LABEL_54;
  }
  if ( a2 )
  {
    v8 = *(_QWORD *)(a4 + 32);
    v9 = a1;
    goto LABEL_4;
  }
  v9 = (unsigned __int8 *)sub_8E9FF0((__int64)a1, 0, 0, 0, 1u, a4);
  sub_8EB260(a1, 0, 0, a4);
  v8 = *(_QWORD *)(a4 + 32);
  v21 = v8;
  if ( !v8 )
  {
    v22 = *(_QWORD *)(a4 + 8);
    v23 = v22 + 1;
    if ( *(_DWORD *)(a4 + 28) )
    {
      *(_QWORD *)(a4 + 8) = v23;
      goto LABEL_4;
    }
LABEL_54:
    v24 = *(_QWORD *)(a4 + 16);
    v4 = v9;
    if ( v24 > v23 )
    {
      *(_BYTE *)(*(_QWORD *)a4 + v22) = 32;
      v21 = *(_QWORD *)(a4 + 32);
      v23 = *(_QWORD *)(a4 + 8) + 1LL;
    }
    else
    {
      *(_DWORD *)(a4 + 28) = 1;
      if ( v24 )
      {
        *(_BYTE *)(*(_QWORD *)a4 + v24 - 1) = 0;
        v21 = *(_QWORD *)(a4 + 32);
        v23 = *(_QWORD *)(a4 + 8) + 1LL;
      }
    }
    *(_QWORD *)(a4 + 8) = v23;
    v8 = v21;
    if ( !v7 )
      goto LABEL_69;
LABEL_4:
    v10 = a3 & 2;
    if ( v10 )
      goto LABEL_5;
    goto LABEL_37;
  }
  v10 = a3 & 2;
  if ( v10 )
    goto LABEL_8;
LABEL_37:
  ++v8;
  v10 = 0;
  *(_QWORD *)(a4 + 32) = v8;
LABEL_5:
  if ( !v8 )
  {
    v11 = *(_QWORD *)(a4 + 8);
    v12 = v11 + 1;
    if ( !*(_DWORD *)(a4 + 28) )
    {
      v20 = *(_QWORD *)(a4 + 16);
      if ( v20 > v12 )
      {
        *(_BYTE *)(*(_QWORD *)a4 + v11) = 40;
        v12 = *(_QWORD *)(a4 + 8) + 1LL;
      }
      else
      {
        *(_DWORD *)(a4 + 28) = 1;
        if ( v20 )
        {
          *(_BYTE *)(*(_QWORD *)a4 + v20 - 1) = 0;
          v12 = *(_QWORD *)(a4 + 8) + 1LL;
        }
      }
    }
    *(_QWORD *)(a4 + 8) = v12;
  }
LABEL_8:
  v13 = *v9;
  if ( *v9 == 69 || !v13 || (v13 == 82 || v13 == 79) && v9[1] == 69 )
  {
    v14 = *(_QWORD *)(a4 + 32);
    if ( *(_DWORD *)(a4 + 24) )
      goto LABEL_30;
    goto LABEL_40;
  }
  if ( v13 == 118 && ((v25 = v9[1], v25 == 69) || !v25 || (v25 == 82 || v25 == 79) && v9[2] == 69) )
  {
    v14 = *(_QWORD *)(a4 + 32);
    ++v9;
  }
  else
  {
LABEL_13:
    if ( v13 == 122 )
    {
LABEL_14:
      v14 = *(_QWORD *)(a4 + 32);
      if ( !v14 )
      {
        sub_8E5790((unsigned __int8 *)"...", a4);
        v14 = *(_QWORD *)(a4 + 32);
      }
      v13 = v9[1];
      if ( !v13 || v13 == 69 )
      {
        ++v9;
        goto LABEL_28;
      }
      if ( (v13 == 79 || v13 == 82) && v9[2] == 69 )
      {
        ++v9;
        goto LABEL_22;
      }
      ++v9;
      if ( !*(_DWORD *)(a4 + 24) )
      {
LABEL_40:
        ++v14;
        ++*(_QWORD *)(a4 + 48);
        *(_DWORD *)(a4 + 24) = 1;
        *(_QWORD *)(a4 + 32) = v14;
      }
    }
    else
    {
      while ( 1 )
      {
        v15 = sub_8E9FF0((__int64)v9, 0, 0, 0, 1u, a4);
        sub_8EB260(v9, 0, 0, a4);
        v13 = *v15;
        v14 = *(_QWORD *)(a4 + 32);
        v9 = (unsigned __int8 *)v15;
LABEL_28:
        if ( v13 == 69 || !v13 )
          break;
LABEL_22:
        if ( (v13 == 82 || v13 == 79) && v9[1] == 69 || *(_DWORD *)(a4 + 24) )
          break;
        if ( v14 )
          goto LABEL_13;
        sub_8E5790((unsigned __int8 *)", ", a4);
        if ( *v9 == 122 )
          goto LABEL_14;
      }
    }
  }
LABEL_30:
  if ( !v14 )
  {
    v16 = *(_QWORD *)(a4 + 8);
    v17 = v16 + 1;
    if ( !*(_DWORD *)(a4 + 28) )
    {
      v19 = *(_QWORD *)(a4 + 16);
      if ( v19 > v17 )
      {
        *(_BYTE *)(*(_QWORD *)a4 + v16) = 41;
        v17 = *(_QWORD *)(a4 + 8) + 1LL;
      }
      else
      {
        *(_DWORD *)(a4 + 28) = 1;
        if ( v19 )
        {
          *(_BYTE *)(*(_QWORD *)a4 + v19 - 1) = 0;
          v17 = *(_QWORD *)(a4 + 8) + 1LL;
        }
      }
    }
    *(_QWORD *)(a4 + 8) = v17;
  }
  if ( !v10 )
    --*(_QWORD *)(a4 + 32);
  return v9;
}
