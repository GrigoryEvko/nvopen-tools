// Function: sub_D46690
// Address: 0xd46690
//
__int64 __fastcall sub_D46690(__int64 a1, __int64 a2, __int64 **a3, char a4)
{
  unsigned __int64 v4; // rax
  __int64 v5; // r13
  int v6; // r12d
  __int64 *v7; // r14
  unsigned int v8; // r15d
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rdx
  _QWORD *v13; // rax
  _QWORD *v14; // rcx
  __int64 v17; // [rsp+10h] [rbp-50h]
  __int64 v18; // [rsp+18h] [rbp-48h]
  __int64 v19; // [rsp+20h] [rbp-40h]
  char v20; // [rsp+2Fh] [rbp-31h]

  if ( a1 == a2 )
    return 0;
  v18 = a1;
  v17 = 0;
  v20 = a4 ^ 1;
  do
  {
    v4 = *(_QWORD *)(*(_QWORD *)v18 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v4 == *(_QWORD *)v18 + 48LL )
      goto LABEL_18;
    if ( !v4 )
      BUG();
    v5 = v4 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v4 - 24) - 30 > 0xA )
      goto LABEL_18;
    v6 = sub_B46E30(v5);
    v7 = *a3;
    if ( !v6 )
      goto LABEL_18;
    v8 = 0;
    v9 = 0;
    do
    {
      while ( 1 )
      {
        v10 = sub_B46EC0(v5, v8);
        v11 = *v7;
        v12 = v10;
        if ( *(_BYTE *)(*v7 + 84) )
        {
          v13 = *(_QWORD **)(v11 + 64);
          v14 = &v13[*(unsigned int *)(v11 + 76)];
          if ( v13 != v14 )
          {
            while ( v12 != *v13 )
            {
              if ( v14 == ++v13 )
                goto LABEL_22;
            }
            goto LABEL_13;
          }
        }
        else
        {
          v19 = v10;
          if ( sub_C8CA60(v11 + 56, v10) )
            goto LABEL_13;
          v12 = v19;
        }
LABEL_22:
        if ( !v12 )
          goto LABEL_13;
        if ( !v9 )
          break;
        if ( v20 || v12 != v9 )
          return 0;
LABEL_13:
        if ( v6 == ++v8 )
          goto LABEL_14;
      }
      ++v8;
      v9 = v12;
    }
    while ( v6 != v8 );
LABEL_14:
    if ( !v9 )
      goto LABEL_18;
    if ( v17 )
    {
      if ( v20 || v9 != v17 )
        return 0;
    }
    else
    {
      v17 = v9;
    }
LABEL_18:
    v18 += 8;
  }
  while ( a2 != v18 );
  return v17;
}
