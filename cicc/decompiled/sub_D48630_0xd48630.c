// Function: sub_D48630
// Address: 0xd48630
//
__int64 __fastcall sub_D48630(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v7; // rcx
  __int64 v8; // rdx
  unsigned int v9; // r13d
  __int64 v11; // r14
  __int64 v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rax

  v3 = **(_QWORD **)(a1 + 32);
  *a2 = 0;
  *a3 = 0;
  v4 = *(_QWORD *)(v3 + 16);
  if ( !v4 )
LABEL_19:
    BUG();
  while ( 1 )
  {
    v7 = *(_QWORD *)(v4 + 24);
    v4 = *(_QWORD *)(v4 + 8);
    if ( (unsigned __int8)(*(_BYTE *)v7 - 30) <= 0xAu )
      break;
    if ( !v4 )
      goto LABEL_19;
  }
  if ( !v4 )
  {
LABEL_12:
    *a3 = *(_QWORD *)(v7 + 40);
    return 0;
  }
  while ( (unsigned __int8)(**(_BYTE **)(v4 + 24) - 30) > 0xAu )
  {
    v4 = *(_QWORD *)(v4 + 8);
    if ( !v4 )
      goto LABEL_12;
  }
  *a3 = *(_QWORD *)(v7 + 40);
  v8 = *(_QWORD *)(v4 + 8);
  if ( v8 )
  {
    while ( 1 )
    {
      v7 = (unsigned int)**(unsigned __int8 **)(v8 + 24) - 30;
      if ( (unsigned __int8)(**(_BYTE **)(v8 + 24) - 30) <= 0xAu )
        break;
      v8 = *(_QWORD *)(v8 + 8);
      if ( !v8 )
        goto LABEL_15;
    }
    v9 = 0;
    *a2 = *(_QWORD *)(*(_QWORD *)(v4 + 24) + 40LL);
    return v9;
  }
LABEL_15:
  v11 = a1 + 56;
  v12 = *(_QWORD *)(*(_QWORD *)(v4 + 24) + 40LL);
  *a2 = v12;
  v9 = sub_B19060(a1 + 56, v12, v8, v7);
  if ( (_BYTE)v9 )
  {
    if ( !(unsigned __int8)sub_B19060(v11, *a3, v13, v14) )
    {
      v15 = *a2;
      *a2 = *a3;
      *a3 = v15;
      return v9;
    }
    return 0;
  }
  return sub_B19060(v11, *a3, v13, v14);
}
