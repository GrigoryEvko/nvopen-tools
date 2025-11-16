// Function: sub_735400
// Address: 0x735400
//
__int64 __fastcall sub_735400(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  _QWORD *i; // r14
  __int64 v4; // rbx
  __int64 j; // r15
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 k; // r15
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // r13
  __int64 v14; // rbx
  char v15; // al
  __int64 v16; // rax
  char v17; // al
  __int64 result; // rax
  __int64 v19; // rcx
  __int64 v20; // rdx

  v1 = sub_85EB10(a1);
  v2 = *(_QWORD *)(a1 + 168);
  for ( i = (_QWORD *)v1; v2; v2 = *(_QWORD *)(v2 + 112) )
  {
    if ( (*(_BYTE *)(v2 + 124) & 1) == 0 )
      sub_735400(*(_QWORD *)(v2 + 128));
  }
  v4 = *(_QWORD *)(a1 + 112);
  for ( j = 0; v4; *(_QWORD *)(v6 + 112) = 0 )
  {
    while ( 1 )
    {
      v6 = v4;
      v4 = *(_QWORD *)(v4 + 112);
      if ( *(char *)(v6 - 8) >= 0 )
        break;
      j = v6;
      if ( !v4 )
        goto LABEL_12;
    }
    sub_734EF0(v6);
    v7 = *(_QWORD *)(v6 + 112);
    if ( j )
      *(_QWORD *)(j + 112) = v7;
    else
      *(_QWORD *)(a1 + 112) = v7;
  }
LABEL_12:
  i[5] = j;
  if ( !*(_BYTE *)(a1 + 28) )
    sub_735030();
  v8 = *(_QWORD *)(a1 + 144);
  for ( k = 0; v8; *(_QWORD *)(v10 + 112) = 0 )
  {
    while ( 1 )
    {
      v10 = v8;
      v8 = *(_QWORD *)(v8 + 112);
      if ( *(char *)(v10 - 8) >= 0 )
        break;
      k = v10;
      if ( !v8 )
        goto LABEL_21;
    }
    sub_734AA0(v10);
    v11 = *(_QWORD *)(v10 + 112);
    if ( k )
      *(_QWORD *)(k + 112) = v11;
    else
      *(_QWORD *)(a1 + 144) = v11;
  }
LABEL_21:
  i[6] = k;
  if ( unk_4F07290 && *(char *)(unk_4F07290 - 8LL) >= 0 )
    unk_4F07290 = 0;
  v12 = *(_QWORD *)(a1 + 104);
  v13 = 0;
  while ( v12 )
  {
    v14 = v12;
    v12 = *(_QWORD *)(v12 + 112);
    v15 = *(_BYTE *)(v14 + 140);
    if ( v15 == 12 )
    {
      v16 = v14;
      do
      {
        if ( *(_QWORD *)(v16 + 8) )
          break;
        v16 = *(_QWORD *)(v16 + 160);
      }
      while ( *(_BYTE *)(v16 + 140) == 12 );
      if ( *(char *)(v16 - 8) >= 0 )
      {
LABEL_30:
        if ( v13 )
          *(_QWORD *)(v13 + 112) = v12;
        else
          *(_QWORD *)(a1 + 104) = v12;
        v17 = *(_BYTE *)(v14 + 140);
        *(_QWORD *)(v14 + 112) = 0;
        if ( (unsigned __int8)(v17 - 9) <= 2u )
        {
          if ( dword_4F077C4 == 2 )
            sub_734AF0(v14);
          *(_QWORD *)(v14 + 160) = 0;
          sub_725420(*(_QWORD *)(v14 + 168));
          *(_BYTE *)(*(_QWORD *)(v14 + 168) + 111LL) |= 4u;
        }
        continue;
      }
    }
    else
    {
      if ( *(char *)(v14 - 8) >= 0 )
        goto LABEL_30;
      if ( (unsigned __int8)(v15 - 9) <= 2u )
      {
        v13 = v14;
        sub_734E20(v14);
        continue;
      }
    }
    v13 = v14;
  }
  i[4] = v13;
  if ( !*(_BYTE *)(a1 + 28) )
    sub_86ADC0(a1);
  result = unk_4F07300;
  if ( unk_4F07300 )
  {
    v19 = 0;
    do
    {
      while ( 1 )
      {
        v20 = result;
        result = *(_QWORD *)(result + 112);
        if ( *(char *)(v20 - 8) >= 0 )
          break;
        v19 = v20;
        if ( !result )
          return result;
      }
      if ( v19 )
        *(_QWORD *)(v19 + 112) = result;
      else
        unk_4F07300 = result;
      *(_QWORD *)(v20 + 112) = 0;
    }
    while ( result );
  }
  return result;
}
