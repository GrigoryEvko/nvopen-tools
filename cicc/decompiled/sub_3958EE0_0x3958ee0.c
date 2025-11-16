// Function: sub_3958EE0
// Address: 0x3958ee0
//
__int64 __fastcall sub_3958EE0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // r14
  __int64 v5; // rdx
  __int64 v7; // r15
  __int64 v8; // rbx
  __int64 i; // rsi
  __int64 v11; // rdx
  __int64 v13; // [rsp+8h] [rbp-38h]

  v3 = *((unsigned int *)a2 + 2);
  v4 = *a2 + 8 * v3;
  v13 = *a2;
  if ( v4 == *a2 )
    goto LABEL_15;
  v5 = *(_QWORD *)(v4 - 8);
  v7 = *a2;
  v8 = *a2 + 8 * v3;
  for ( i = v5; ; i = *(_QWORD *)(v4 - 8) )
  {
    if ( i == v5 )
      goto LABEL_7;
    if ( sub_15CCEE0(a1, i, v5) )
    {
      v5 = *(_QWORD *)(v8 - 8);
LABEL_7:
      v4 -= 8;
      if ( v7 == v4 )
        goto LABEL_8;
      continue;
    }
    if ( v4 == v7 )
      break;
    if ( v13 == v8 - 8 )
      goto LABEL_15;
    v7 = *a2;
    v4 = *a2 + 8LL * *((unsigned int *)a2 + 2);
    v5 = *(_QWORD *)(v8 - 16);
    if ( *a2 == v4 )
      goto LABEL_8;
    v8 -= 8;
  }
  v5 = *(_QWORD *)(v8 - 8);
LABEL_8:
  if ( v5 )
  {
LABEL_9:
    a3 = v5;
    if ( *(_BYTE *)(v5 + 16) == 77 )
    {
      v11 = v5 + 24;
      while ( *(_BYTE *)(v11 - 8) == 77 )
      {
        v11 = *(_QWORD *)(v11 + 8);
        if ( !v11 )
          BUG();
      }
      return v11 - 24;
    }
    return a3;
  }
LABEL_15:
  if ( a3 )
  {
    v5 = a3;
    goto LABEL_9;
  }
  return a3;
}
