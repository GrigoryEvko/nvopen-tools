// Function: sub_8E6E80
// Address: 0x8e6e80
//
void __fastcall sub_8E6E80(char a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  int v5; // r9d
  int v6; // r11d
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rax
  int v12; // r10d
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rdx
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rcx

  if ( (a1 & 1) == 0 )
  {
    if ( (a1 & 2) == 0 )
    {
      if ( (a1 & 4) == 0 )
        return;
      v11 = *(_QWORD *)(a3 + 32);
LABEL_21:
      if ( v11 )
        return;
      goto LABEL_22;
    }
    v4 = *(_QWORD *)(a3 + 32);
    goto LABEL_4;
  }
  if ( *(_QWORD *)(a3 + 32) )
    return;
  sub_8E5790((unsigned __int8 *)"const", a3);
  if ( v12 )
  {
    if ( *(_QWORD *)(a3 + 32) )
      return;
    v13 = *(_QWORD *)(a3 + 8);
    v14 = v13 + 1;
    if ( !*(_DWORD *)(a3 + 28) )
    {
      v15 = *(_QWORD *)(a3 + 16);
      if ( v15 > v14 )
      {
        *(_BYTE *)(*(_QWORD *)a3 + v13) = 32;
        v16 = *(_QWORD *)(a3 + 8) + 1LL;
        v4 = *(_QWORD *)(a3 + 32);
        goto LABEL_35;
      }
      *(_DWORD *)(a3 + 28) = 1;
      if ( v15 )
      {
        *(_BYTE *)(*(_QWORD *)a3 + v15 - 1) = 0;
        v16 = *(_QWORD *)(a3 + 8) + 1LL;
        v4 = *(_QWORD *)(a3 + 32);
LABEL_35:
        *(_QWORD *)(a3 + 8) = v16;
LABEL_4:
        if ( v4 )
          return;
        goto LABEL_5;
      }
    }
    *(_QWORD *)(a3 + 8) = v14;
LABEL_5:
    sub_8E5790((unsigned __int8 *)"volatile", a3);
  }
  if ( v6 )
  {
    if ( *(_QWORD *)(a3 + 32) )
      return;
    v7 = *(_QWORD *)(a3 + 8);
    v8 = v7 + 1;
    if ( !*(_DWORD *)(a3 + 28) )
    {
      v9 = *(_QWORD *)(a3 + 16);
      if ( v9 > v8 )
      {
        *(_BYTE *)(*(_QWORD *)a3 + v7) = 32;
        v10 = *(_QWORD *)(a3 + 8) + 1LL;
        v11 = *(_QWORD *)(a3 + 32);
        goto LABEL_37;
      }
      *(_DWORD *)(a3 + 28) = 1;
      if ( v9 )
      {
        *(_BYTE *)(*(_QWORD *)a3 + v9 - 1) = 0;
        v10 = *(_QWORD *)(a3 + 8) + 1LL;
        v11 = *(_QWORD *)(a3 + 32);
LABEL_37:
        *(_QWORD *)(a3 + 8) = v10;
        goto LABEL_21;
      }
    }
    *(_QWORD *)(a3 + 8) = v8;
LABEL_22:
    sub_8E5790("__restrict__", a3);
  }
  if ( v5 && !*(_QWORD *)(a3 + 32) )
  {
    v17 = *(_QWORD *)(a3 + 8);
    v18 = v17 + 1;
    if ( !*(_DWORD *)(a3 + 28) )
    {
      v19 = *(_QWORD *)(a3 + 16);
      if ( v19 > v18 )
      {
        *(_BYTE *)(*(_QWORD *)a3 + v17) = 32;
        v18 = *(_QWORD *)(a3 + 8) + 1LL;
      }
      else
      {
        *(_DWORD *)(a3 + 28) = 1;
        if ( v19 )
        {
          *(_BYTE *)(*(_QWORD *)a3 + v19 - 1) = 0;
          v18 = *(_QWORD *)(a3 + 8) + 1LL;
        }
      }
    }
    *(_QWORD *)(a3 + 8) = v18;
  }
}
