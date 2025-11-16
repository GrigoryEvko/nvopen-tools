// Function: sub_8C9400
// Address: 0x8c9400
//
void __fastcall sub_8C9400(__int64 a1, char a2)
{
  _QWORD *v2; // rbx
  __int64 v3; // rdi
  __int64 v4; // rax
  _QWORD *v5; // r13
  __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rax
  char v9; // al
  __int64 v10; // r13
  char v11; // al

  v2 = (_QWORD *)a1;
  if ( (*(_BYTE *)(a1 + 89) & 4) == 0 )
    goto LABEL_25;
  if ( a2 == 6
    && (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) <= 2u
    && *(_QWORD *)(a1 + 8)
    && *(char *)(a1 + 177) < 0
    && *(_QWORD *)(*(_QWORD *)(a1 + 168) + 168LL) )
  {
    goto LABEL_55;
  }
  v3 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
  if ( (unk_4D03FC4 || unk_4D03FC0 && (*(_BYTE *)(v3 + 89) & 4) != 0) && !*(_QWORD *)(v3 + 32) )
    sub_8C9400(v3, 6);
  if ( a2 == 59 )
    goto LABEL_35;
  v4 = sub_8C7030(v2[5]);
  v5 = (_QWORD *)v4;
  if ( !v4 )
  {
LABEL_25:
    if ( a2 == 11 )
    {
      if ( (*((_BYTE *)v2 + 195) & 1) != 0 && v2[30] )
        sub_8CC1D0(*v2);
      else
        sub_8CC270(v2);
LABEL_30:
      v5 = (_QWORD *)v2[4];
      if ( v5 )
        return;
      goto LABEL_51;
    }
    if ( (unsigned __int8)a2 <= 0xBu )
    {
      if ( a2 == 6 )
      {
        v9 = *((_BYTE *)v2 + 140);
        if ( (unsigned __int8)(v9 - 9) > 2u )
        {
          if ( v9 == 12 && *((_BYTE *)v2 + 184) == 10 )
          {
LABEL_57:
            v10 = *v2;
            sub_8C9360(*v2);
            if ( v2[4] )
              return;
            if ( *((_BYTE *)v2 + 140) == 12 )
              sub_8C9210(v10);
            else
              sub_8CCC20(v10);
            goto LABEL_42;
          }
LABEL_41:
          sub_8CC930(v2, (*((_BYTE *)v2 + 89) & 4) != 0);
LABEL_42:
          if ( v2[4] )
            return;
          sub_8CA0A0(v2, 1);
          goto LABEL_30;
        }
LABEL_55:
        if ( (*((_BYTE *)v2 + 177) & 0x10) != 0 && *(_QWORD *)(v2[21] + 168LL) )
          goto LABEL_57;
        goto LABEL_41;
      }
      if ( a2 == 7 )
      {
        sub_8CBDE0(v2);
        goto LABEL_30;
      }
LABEL_72:
      sub_721090();
    }
    if ( a2 == 28 )
    {
      sub_8CC480(v2);
      goto LABEL_30;
    }
    if ( a2 != 59 )
      goto LABEL_72;
LABEL_35:
    sub_8C88F0(v2, (*((_BYTE *)v2 + 89) & 4) != 0);
    goto LABEL_30;
  }
  if ( *(_QWORD *)(v4 + 32) )
  {
    if ( a2 == 6 )
    {
      v11 = *((_BYTE *)v2 + 140);
      if ( (unsigned __int8)(v11 - 9) > 2u )
      {
        if ( v11 == 12 && *((_BYTE *)v2 + 184) == 10 )
        {
          sub_8C9210(*v2);
          goto LABEL_14;
        }
      }
      else if ( (*((_BYTE *)v2 + 177) & 0x10) != 0 && *(_QWORD *)(v2[21] + 168LL) )
      {
        sub_8CCC20(*v2);
        goto LABEL_14;
      }
      sub_8CC930(v2, 1);
      goto LABEL_14;
    }
    if ( a2 == 11 && (*((_BYTE *)v2 + 195) & 1) != 0 && v2[30] )
      sub_8CC1D0(*v2);
  }
  else if ( (*(_BYTE *)(v4 + 177) & 0x10) != 0 && *(_QWORD *)(*(_QWORD *)(v4 + 168) + 168LL) )
  {
    sub_8CCC20(*(_QWORD *)v4);
  }
  else
  {
    sub_8CC930(v4, 0);
  }
LABEL_14:
  if ( v2[4] )
    return;
  v6 = *(_QWORD *)sub_8C7030(v2[5]);
  if ( v6 )
  {
    sub_8C9360(v6);
    v7 = *v2;
    if ( !*v2 )
      goto LABEL_18;
    goto LABEL_17;
  }
LABEL_51:
  v7 = *v2;
  if ( *v2 )
  {
LABEL_17:
    sub_8C9360(v7);
LABEL_18:
    if ( v2[4] )
      return;
  }
  sub_8C7090(a2, (__int64)v2);
  while ( (*((_BYTE *)v2 + 89) & 4) != 0 )
  {
    v8 = v2[5];
    v2 = *(_QWORD **)(v8 + 32);
    if ( v2 == v5 )
      break;
    if ( v2[4] )
      break;
    sub_8C7090(6, *(_QWORD *)(v8 + 32));
  }
}
