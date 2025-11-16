// Function: sub_86C540
// Address: 0x86c540
//
void __fastcall sub_86C540(__int64 a1, __int64 a2, int a3)
{
  __int64 v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 i; // rcx
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rbx
  __int64 v17; // rdi
  __int64 v18; // rsi
  __int64 *v19; // rax
  __int64 *v20; // rcx
  __int64 v21; // rax
  unsigned __int8 v22; // [rsp+7h] [rbp-29h] BYREF
  __int64 v23[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( a3 )
  {
    v6 = *(_QWORD *)(a2 + 48);
    if ( v6 )
      sub_86C540(a1, v6, 1);
  }
  if ( (unsigned int)sub_86C240(*(_QWORD *)(a1 + 16), a2) )
    goto LABEL_32;
  v7 = *(_QWORD *)(a1 + 16);
  v8 = *(_QWORD *)(a2 + 16);
  if ( v7 == v8 )
  {
    if ( !a3 )
      goto LABEL_32;
LABEL_22:
    v14 = *(_QWORD *)a2;
    goto LABEL_23;
  }
  if ( v7 )
  {
    v9 = *(_QWORD *)(a1 + 16);
    while ( 1 )
    {
      v9 = *(_QWORD *)(v9 + 16);
      if ( !v9 )
        break;
      if ( v8 == v9 )
      {
        v19 = *(__int64 **)(a1 + 16);
        if ( a3 )
          goto LABEL_22;
        do
        {
          v20 = v19;
          v19 = (__int64 *)v19[2];
        }
        while ( (__int64 *)v8 != v19 );
        v14 = *v20;
        goto LABEL_23;
      }
    }
  }
  if ( !v8 )
  {
LABEL_14:
    v11 = *(_QWORD *)(v8 + 16);
    for ( i = *(_QWORD *)(a1 + 16); ; i = *(_QWORD *)(i + 16) )
    {
      if ( v11 )
      {
        if ( i == v11 )
        {
LABEL_48:
          if ( !a3 )
          {
            v11 = *(_QWORD *)(v7 + 16);
            v8 = *(_QWORD *)(a1 + 16);
          }
          while ( i != v11 )
          {
            v8 = v11;
            v11 = *(_QWORD *)(v11 + 16);
          }
          if ( a3 )
            v14 = **(_QWORD **)(v8 + 40);
          else
            v14 = *(_QWORD *)v8;
          goto LABEL_23;
        }
        v13 = *(_QWORD *)(v8 + 16);
        while ( 1 )
        {
          v13 = *(_QWORD *)(v13 + 16);
          if ( !v13 )
            break;
          if ( i == v13 )
            goto LABEL_48;
        }
      }
    }
  }
  v10 = *(_QWORD *)(a2 + 16);
  do
  {
    v10 = *(_QWORD *)(v10 + 16);
    if ( !v10 )
      goto LABEL_14;
  }
  while ( v7 != v10 );
  if ( a3 )
  {
    do
    {
      v21 = v8;
      v8 = *(_QWORD *)(v8 + 16);
    }
    while ( v7 != v8 );
    v14 = **(_QWORD **)(v21 + 40);
LABEL_23:
    if ( v14 )
    {
      v23[0] = 0;
      if ( !*(_BYTE *)(v14 + 32) )
      {
        while ( v7 )
        {
          if ( v7 != v14 )
          {
            v15 = v7;
            do
            {
              v15 = *(_QWORD *)(v15 + 16);
              if ( !v15 )
                goto LABEL_30;
            }
            while ( v15 != v14 );
          }
          v14 = *(_QWORD *)v14;
          if ( *(_BYTE *)(v14 + 32) )
            break;
        }
      }
LABEL_30:
      v22 = 3;
      sub_86BD50((__int64 *)v14, a1, (_DWORD *)(a2 + 24), v23, &v22);
      if ( v22 != 3 )
        sub_685910(v23[0], (FILE *)a1);
    }
  }
LABEL_32:
  if ( dword_4F077C4 == 2 )
  {
    v16 = *(_QWORD *)(a2 + 40);
    v17 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 80LL);
    v18 = *(_QWORD *)(v16 + 80);
    if ( v18 != v17 )
      v17 = sub_86BCA0(v17, v18);
    *(_QWORD *)(v16 + 80) = v17;
  }
  if ( dword_4D047EC )
  {
    if ( unk_4D047E8 )
      sub_86C360(a2, a1);
  }
}
