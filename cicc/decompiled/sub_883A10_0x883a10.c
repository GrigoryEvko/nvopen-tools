// Function: sub_883A10
// Address: 0x883a10
//
_BOOL8 __fastcall sub_883A10(__int64 a1, __int64 a2, unsigned __int8 *a3)
{
  __int64 v3; // rcx
  __int64 i; // r14
  char v6; // al
  char v7; // dl
  __int64 v8; // rsi
  unsigned __int8 v10; // r12
  __int64 v11; // rdi
  __int64 v12; // rbx
  __int64 **v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // r15
  unsigned __int8 v16; // al
  char v17; // cl
  __int64 v18; // rdx

  v3 = a2;
  i = a1;
  v6 = *(_BYTE *)(a2 + 80);
  v7 = v6;
  if ( v6 == 16 )
  {
    v3 = **(_QWORD **)(a2 + 88);
    v7 = *(_BYTE *)(v3 + 80);
  }
  if ( v7 == 24 )
  {
    v3 = *(_QWORD *)(v3 + 88);
    v7 = *(_BYTE *)(v3 + 80);
  }
  if ( v7 == 17
    && *(_BYTE *)(a1 + 80) == 10
    && *(_QWORD *)(*(_QWORD *)(a1 + 88) + 240LL)
    && *(_QWORD *)(a1 + 64) != *(_QWORD *)(v3 + 64) )
  {
    for ( i = *(_QWORD *)(v3 + 88); ; i = *(_QWORD *)(i + 8) )
    {
      v17 = *(_BYTE *)(i + 80);
      v18 = i;
      if ( v17 == 16 )
      {
        v18 = **(_QWORD **)(i + 88);
        v17 = *(_BYTE *)(v18 + 80);
      }
      if ( v17 == 24 )
        v18 = *(_QWORD *)(v18 + 88);
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 96) + 32LL) == v18 )
        break;
    }
  }
  if ( v6 == 16 )
  {
    v11 = *(_QWORD *)(*(_QWORD *)(a2 + 88) + 8LL);
    v12 = *(_QWORD *)(v11 + 112);
    if ( v12 )
    {
      v13 = *(__int64 ***)(v11 + 112);
      if ( (*(_BYTE *)(v11 + 96) & 2) != 0 )
        v13 = sub_72B780(v11);
      v14 = (__int64)v13[1];
      v15 = *(_QWORD *)(a2 + 64);
      if ( a3 )
      {
        v10 = sub_883850(i, *(_QWORD *)(a2 + 64), v14, (__int64)v13, a2, 0);
        do
        {
          if ( !v10 )
            break;
          if ( (*(_BYTE *)(v12 + 24) & 2) == 0 )
          {
            v16 = sub_883850(i, v15, *(_QWORD *)(v12 + 8), v12, a2, 0);
            if ( v10 > v16 )
              v10 = v16;
          }
          v12 = *(_QWORD *)v12;
        }
        while ( v12 );
        goto LABEL_11;
      }
      if ( !(unsigned int)sub_883C30(i, v15, v14, v13, a2, 0) )
      {
        while ( (*(_BYTE *)(v12 + 24) & 2) != 0 || !(unsigned int)sub_883C30(i, v15, *(_QWORD *)(v12 + 8), v12, a2, 0) )
        {
          v12 = *(_QWORD *)v12;
          if ( !v12 )
            return 0;
        }
      }
      return 1;
    }
  }
  v8 = *(_QWORD *)(a2 + 64);
  if ( a3 )
  {
    v10 = sub_883850(i, v8, 0, 0, a2, 0);
LABEL_11:
    *a3 = v10;
    return 0;
  }
  return (unsigned int)sub_883C30(i, v8, 0, 0, a2, 0) != 0;
}
