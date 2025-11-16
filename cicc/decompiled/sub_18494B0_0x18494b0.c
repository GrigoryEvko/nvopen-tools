// Function: sub_18494B0
// Address: 0x18494b0
//
__int64 __fastcall sub_18494B0(__int64 a1)
{
  __int64 *v1; // rax
  __int64 v2; // r12
  unsigned int v3; // r9d
  __int64 v5; // rbx
  __int64 v6; // r14
  __int64 i; // r13
  unsigned __int8 v8; // al
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rcx
  unsigned __int64 v14; // rax
  __int64 v15; // rdi
  char v16; // al
  unsigned __int8 v17; // [rsp+Fh] [rbp-31h]

  v1 = *(__int64 **)(a1 + 80);
  v2 = *v1;
  if ( !*v1 )
    return 0;
  if ( sub_15E4F60(*v1) )
    return 0;
  v3 = sub_1560180(v2 + 112, 27);
  if ( (_BYTE)v3 )
    return 0;
  v5 = *(_QWORD *)(v2 + 80);
  v6 = v2 + 72;
  if ( v2 + 72 == v5 )
  {
    i = 0;
  }
  else
  {
    if ( !v5 )
      BUG();
    while ( 1 )
    {
      i = *(_QWORD *)(v5 + 24);
      if ( i != v5 + 16 )
        break;
      v5 = *(_QWORD *)(v5 + 8);
      if ( v6 == v5 )
        goto LABEL_12;
      if ( !v5 )
        BUG();
    }
  }
  while ( v5 != v6 )
  {
    if ( !i )
      BUG();
    v8 = *(_BYTE *)(i - 8);
    v9 = i - 24;
    if ( v8 > 0x17u )
    {
      if ( v8 == 78 )
      {
        v11 = v9 | 4;
LABEL_30:
        v12 = v11 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v11 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        {
          v13 = v12 - 24;
          v14 = v12 - 72;
          if ( (v11 & 4) != 0 )
            v14 = v13;
          v15 = *(_QWORD *)v14;
          if ( *(_BYTE *)(*(_QWORD *)v14 + 16LL) )
            return v3;
          if ( v2 == v15 )
            return v3;
          v17 = v3;
          v16 = sub_1560180(v15 + 112, 27);
          v3 = v17;
          if ( !v16 )
            return v3;
        }
        goto LABEL_20;
      }
      if ( v8 == 29 )
      {
        v11 = v9 & 0xFFFFFFFFFFFFFFFBLL;
        goto LABEL_30;
      }
    }
LABEL_20:
    for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v5 + 24) )
    {
      v10 = v5 - 24;
      if ( !v5 )
        v10 = 0;
      if ( i != v10 + 40 )
        break;
      v5 = *(_QWORD *)(v5 + 8);
      if ( v6 == v5 )
        goto LABEL_12;
      if ( !v5 )
        BUG();
    }
  }
LABEL_12:
  if ( (unsigned __int8)sub_1560180(v2 + 112, 27) )
  {
    return 0;
  }
  else
  {
    sub_15E0D50(v2, -1, 27);
    return 1;
  }
}
