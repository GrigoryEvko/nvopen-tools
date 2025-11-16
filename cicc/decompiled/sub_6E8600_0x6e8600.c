// Function: sub_6E8600
// Address: 0x6e8600
//
__int64 __fastcall sub_6E8600(__int64 *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  char v7; // cl
  __int64 v8; // rax
  char v9; // dl
  char v10; // si
  __int64 v11; // rax
  char v12; // dl
  __int64 v13; // r14
  __int64 v14; // rbx
  unsigned __int8 v15; // al
  unsigned int v17; // ebx
  unsigned int v18; // esi
  unsigned __int8 v19; // bl
  char v20; // bl
  int v21; // r8d
  char v22; // al

  if ( a1 )
    a2 = *a1;
  if ( a3 )
    a4 = *a3;
  v7 = *(_BYTE *)(a2 + 140);
  if ( v7 == 12 )
  {
    v8 = a2;
    do
    {
      v8 = *(_QWORD *)(v8 + 160);
      v9 = *(_BYTE *)(v8 + 140);
    }
    while ( v9 == 12 );
  }
  else
  {
    v9 = *(_BYTE *)(a2 + 140);
  }
  if ( !v9 )
    goto LABEL_13;
  v10 = *(_BYTE *)(a4 + 140);
  if ( v10 == 12 )
  {
    v11 = a4;
    do
    {
      v11 = *(_QWORD *)(v11 + 160);
      v12 = *(_BYTE *)(v11 + 140);
    }
    while ( v12 == 12 );
  }
  else
  {
    v12 = *(_BYTE *)(a4 + 140);
  }
  if ( !v12 )
  {
LABEL_13:
    v13 = sub_72C930(a1);
    goto LABEL_14;
  }
  if ( v7 == 12 )
  {
    do
      a2 = *(_QWORD *)(a2 + 160);
    while ( *(_BYTE *)(a2 + 140) == 12 );
  }
  if ( v10 == 12 )
  {
    do
      a4 = *(_QWORD *)(a4 + 160);
    while ( *(_BYTE *)(a4 + 140) == 12 );
  }
  v17 = 14;
  if ( (unsigned int)sub_8D2A90(a2) )
    v17 = *(unsigned __int8 *)(a2 + 160);
  v18 = 14;
  if ( (unsigned int)sub_8D2A90(a4) )
    v18 = *(unsigned __int8 *)(a4 + 160);
  v19 = sub_6E55D0(v17, v18);
  if ( v19 == 14 )
  {
    if ( a1 )
      a2 = sub_6E8540((__int64)a1);
    else
      a2 = sub_8D6540(a2);
    while ( *(_BYTE *)(a2 + 140) == 12 )
      a2 = *(_QWORD *)(a2 + 160);
    if ( a3 )
      a4 = sub_6E8540((__int64)a3);
    else
      a4 = sub_8D6540(a4);
    while ( *(_BYTE *)(a4 + 140) == 12 )
      a4 = *(_QWORD *)(a4 + 160);
    v20 = 13;
    if ( (unsigned int)sub_8D2930(a2) )
      v20 = *(_BYTE *)(a2 + 160);
    v21 = sub_8D2930(a4);
    v22 = 13;
    if ( v21 )
      v22 = *(_BYTE *)(a4 + 160);
    if ( unk_4D04290 )
    {
      if ( v20 == 12 || v22 == 12 )
      {
        v13 = sub_72BA30(12);
        goto LABEL_14;
      }
      if ( v20 == 11 || v22 == 11 )
      {
        v13 = sub_72BA30(11);
        goto LABEL_14;
      }
    }
    if ( v20 != 10 && v22 != 10 )
    {
      if ( v20 != 9 && v22 != 9 )
      {
        if ( v20 != 8 && v22 != 8 )
        {
          if ( v20 != 7 && v22 != 7 )
          {
            if ( v20 == 6 || v22 == 6 )
              v13 = sub_72BA30(6);
            else
              v13 = sub_72BA30(5);
            goto LABEL_14;
          }
          if ( unk_4D0475C || unk_4F06B10 != unk_4F06B20 || v20 != 6 && v22 != 6 )
          {
            v13 = sub_72BA30(7);
            goto LABEL_14;
          }
        }
        v13 = sub_72BA30(8);
        goto LABEL_14;
      }
      if ( (unk_4F06B00 != unk_4F06B10 || v20 != 8 && v22 != 8) && (unk_4F06B00 != unk_4F06B20 || v20 != 6 && v22 != 6) )
      {
        v13 = sub_72BA30(9);
        goto LABEL_14;
      }
    }
    v13 = sub_72BA30(10);
    goto LABEL_14;
  }
  if ( (unsigned int)sub_8D2AF0(a2) || (unsigned int)sub_8D2AF0(a4) )
  {
    if ( (unsigned int)sub_8D2B20(a2) && (unsigned int)sub_8D2B20(a4) )
      v13 = sub_72C7D0(v19);
    else
      v13 = sub_72C6F0(v19);
  }
  else
  {
    v13 = sub_72C610(v19);
  }
LABEL_14:
  if ( !(unk_4D04548 | unk_4D04558) || !(unsigned int)sub_8D2930(v13) )
    return v13;
  v14 = v13;
  if ( *(_BYTE *)(v13 + 140) != 12 )
    goto LABEL_20;
  do
    v14 = *(_QWORD *)(v14 + 160);
  while ( *(_BYTE *)(v14 + 140) == 12 );
  if ( *(_BYTE *)(a2 + 140) != 12 )
    goto LABEL_23;
  do
  {
    a2 = *(_QWORD *)(a2 + 160);
LABEL_20:
    ;
  }
  while ( *(_BYTE *)(a2 + 140) == 12 );
  while ( *(_BYTE *)(a4 + 140) == 12 )
  {
    a4 = *(_QWORD *)(a4 + 160);
LABEL_23:
    ;
  }
  if ( !(unsigned int)sub_8D2930(a2) || *(_BYTE *)(a2 + 160) != *(_BYTE *)(v14 + 160) )
  {
    if ( !(unsigned int)sub_8D2930(a4) )
      return v13;
    v15 = *(_BYTE *)(v14 + 160);
    if ( v15 != *(_BYTE *)(a4 + 160) )
      return v13;
LABEL_29:
    if ( (*(_BYTE *)(a4 + 161) & 2) != 0 )
      return sub_72BC30(v15);
    return v13;
  }
  if ( (unsigned int)sub_8D2930(a4) )
  {
    v15 = *(_BYTE *)(v14 + 160);
    if ( v15 == *(_BYTE *)(a4 + 160) )
    {
      if ( (*(_BYTE *)(a2 + 161) & 2) == 0 )
        return v13;
      goto LABEL_29;
    }
  }
  if ( (*(_BYTE *)(a2 + 161) & 2) != 0 )
  {
    v15 = *(_BYTE *)(v14 + 160);
    return sub_72BC30(v15);
  }
  return v13;
}
