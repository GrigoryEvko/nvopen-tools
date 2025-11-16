// Function: sub_2EA4160
// Address: 0x2ea4160
//
__int64 __fastcall sub_2EA4160(__int64 a1, __int64 a2, __int64 **a3, char a4)
{
  __int64 *v4; // rcx
  __int64 *v5; // r12
  __int64 *v6; // r13
  __int64 *v7; // r15
  __int64 v8; // r14
  __int64 v9; // rdi
  __int64 v10; // rbx
  _QWORD *v11; // rax
  _QWORD *v12; // rsi
  __int64 v15; // [rsp+10h] [rbp-50h]
  __int64 v17; // [rsp+20h] [rbp-40h]
  char v18; // [rsp+2Fh] [rbp-31h]

  if ( a1 == a2 )
    return 0;
  v17 = a1;
  v15 = 0;
  v18 = a4 ^ 1;
  do
  {
    v4 = *(__int64 **)(*(_QWORD *)v17 + 112LL);
    v5 = &v4[*(unsigned int *)(*(_QWORD *)v17 + 120LL)];
    v6 = *a3;
    if ( v4 == v5 )
      goto LABEL_15;
    v7 = *(__int64 **)(*(_QWORD *)v17 + 112LL);
    v8 = 0;
    do
    {
      while ( 1 )
      {
        v9 = *v6;
        v10 = *v7;
        if ( *(_BYTE *)(*v6 + 84) )
        {
          v11 = *(_QWORD **)(v9 + 64);
          v12 = &v11[*(unsigned int *)(v9 + 76)];
          if ( v11 != v12 )
          {
            while ( v10 != *v11 )
            {
              if ( v12 == ++v11 )
                goto LABEL_18;
            }
            goto LABEL_10;
          }
        }
        else if ( sub_C8CA60(v9 + 56, *v7) )
        {
          goto LABEL_10;
        }
LABEL_18:
        if ( !v10 )
          goto LABEL_10;
        if ( !v8 )
          break;
        if ( v18 || v10 != v8 )
          return 0;
LABEL_10:
        if ( v5 == ++v7 )
          goto LABEL_11;
      }
      ++v7;
      v8 = v10;
    }
    while ( v5 != v7 );
LABEL_11:
    if ( !v8 )
      goto LABEL_15;
    if ( v15 )
    {
      if ( v8 != v15 || v18 )
        return 0;
    }
    else
    {
      v15 = v8;
    }
LABEL_15:
    v17 += 8;
  }
  while ( a2 != v17 );
  return v15;
}
